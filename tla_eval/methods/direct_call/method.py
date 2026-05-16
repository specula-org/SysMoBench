"""
Direct call method with a backend-driven validation/correction loop.

Generates a spec by prompting the model, extracts the spec (and optional
config) via the LanguageBackend's fence convention, runs the backend's
syntax validator, and on failure feeds the errors back to the model for
up to `max_correction_attempts` rounds. Language-neutral; backend choice
is per-construction.
"""

import logging
import tempfile
from pathlib import Path
from typing import Any, Dict, List

from ..base import TLAGenerationMethod, GenerationTask, GenerationOutput, format_prompt_template
from ...config import get_configured_model

logger = logging.getLogger(__name__)


class DirectCallMethod(TLAGenerationMethod):
    """Generation with a backend-driven syntax-feedback correction loop."""

    def __init__(self,
                 language: str = "TLA+",
                 max_correction_attempts: int = 3,
                 validation_timeout: int = 30):
        super().__init__("direct_call")
        from ...languages import get as _get_backend
        self.language = language
        self.backend = _get_backend(language)
        self.max_correction_attempts = max_correction_attempts
        self.validation_timeout = validation_timeout

    def generate(self, task: GenerationTask, model_name: str = None) -> GenerationOutput:
        try:
            model = get_configured_model(model_name)
            initial_prompt = self._create_prompt(task)

            from ...models.base import GenerationConfig
            gen_config = GenerationConfig(
                max_tokens=model.config.get('max_tokens'),
                temperature=model.config.get('temperature'),
                top_p=model.config.get('top_p'),
            )

            attempts: List[Dict[str, Any]] = []
            last_text: str = ""
            last_metadata: Dict[str, Any] = {}
            last_error: str = None
            prev_spec: str = ""
            prev_config: str = ""
            prev_errors: List[str] = []

            for attempt_idx in range(1, self.max_correction_attempts + 1):
                logger.info(f"[direct_call] Attempt {attempt_idx}/{self.max_correction_attempts}")

                if attempt_idx == 1:
                    result = model.generate_tla_specification(
                        task.source_code, initial_prompt, gen_config
                    )
                else:
                    correction_prompt = self._build_correction_prompt(
                        initial_prompt=initial_prompt,
                        source_code=task.source_code,
                        prev_spec=prev_spec,
                        prev_config=prev_config,
                        errors=prev_errors,
                        attempt_idx=attempt_idx,
                    )
                    result = model.generate_direct(correction_prompt, gen_config)

                if not result.success:
                    last_error = result.error_message
                    attempts.append({
                        "attempt": attempt_idx,
                        "api_success": False,
                        "error_message": result.error_message,
                    })
                    break

                last_text = result.generated_text
                last_metadata = result.metadata

                artifacts = self.backend.extract_artifacts(last_text)
                spec_content = artifacts.spec
                cfg_content = artifacts.config or ""

                try:
                    spec_ext = self.backend.spec_extension or ".spec"
                    module_basename = (task.spec_module or task.task_name) + spec_ext
                    with tempfile.TemporaryDirectory(prefix="direct_call_validate_") as td:
                        outcome = self.backend.validate_syntax(
                            spec=spec_content,
                            config=artifacts.config,
                            work_dir=Path(td),
                            timeout=self.validation_timeout,
                            spec_filename=module_basename,
                        )
                    validate_ok = outcome.success
                    errors = list(outcome.syntax_errors) + list(outcome.semantic_errors)
                except Exception as e:
                    logger.warning(f"[direct_call] {self.language} validation raised: {e}")
                    validate_ok = False
                    errors = [f"Validator raised: {e}"]

                attempts.append({
                    "attempt": attempt_idx,
                    "api_success": True,
                    # Keep the legacy key for downstream tooling that expects it.
                    "sany_success": validate_ok,
                    "validate_success": validate_ok,
                    "num_errors": len(errors),
                    "usage": last_metadata.get("usage"),
                    "finish_reason": last_metadata.get("finish_reason"),
                })

                if validate_ok:
                    logger.info(f"[direct_call] ✓ {self.language} validation passed on attempt {attempt_idx}")
                    break

                logger.info(f"[direct_call] ✗ {self.language} validation failed on attempt {attempt_idx} — {len(errors)} errors")

                if attempt_idx == self.max_correction_attempts:
                    logger.info("[direct_call] Max attempts reached; returning last response.")
                    break

                prev_spec = spec_content
                prev_config = cfg_content
                prev_errors = errors

            return GenerationOutput(
                tla_specification=last_text,
                method_name=self.name,
                task_name=task.task_name,
                metadata={
                    "model_info": model.get_model_info(),
                    "generation_metadata": last_metadata,
                    "prompt_template": "direct_call_with_correction",
                    "language": self.language,
                    "max_correction_attempts": self.max_correction_attempts,
                    "correction_attempts": attempts,
                    "final_sany_success": attempts[-1].get("sany_success", False) if attempts else False,
                    "final_validate_success": attempts[-1].get("validate_success", False) if attempts else False,
                    "attempts_used": len(attempts),
                },
                success=bool(last_text),
                error_message=last_error,
            )

        except Exception as e:
            logger.exception(f"[direct_call] Unexpected error: {e}")
            return GenerationOutput(
                tla_specification="",
                method_name=self.name,
                task_name=task.task_name,
                metadata={},
                success=False,
                error_message=str(e),
            )

    def _build_correction_prompt(
        self,
        initial_prompt: str,
        source_code: str,
        prev_spec: str,
        prev_config: str,
        errors: List[str],
        attempt_idx: int,
    ) -> str:
        shown = errors[:20]
        error_text = "\n".join(f"  - {e.strip()[:500]}" for e in shown)
        more = f"\n  ... and {len(errors) - len(shown)} more" if len(errors) > len(shown) else ""

        resolved_initial = initial_prompt.replace("{source_code}", source_code)

        spec_block = f"```{self.backend.fence_label}\n{prev_spec}\n```"
        if self.backend.config_fence_label:
            cfg_block = f"\n\n```{self.backend.config_fence_label}\n{prev_config}\n```"
        else:
            cfg_block = ""

        return f"""Your previous {self.language} specification failed validation. This is correction attempt {attempt_idx}. Regenerate the full specification, fixing the errors below.

## Original Task Instructions

{resolved_initial}

## Your Previous Output

{spec_block}{cfg_block}

## Validation Errors Found

{error_text}{more}

## Correction Instructions

1. Fix every error listed above.
2. Keep all mandatory action names from the original instructions (the naming convention is a hard contract).
3. {self.backend.fence_format_hint()}
"""

    def _create_prompt(self, task: GenerationTask) -> str:
        from ...tasks.loader import get_task_loader
        task_loader = get_task_loader()
        prompt_template = task_loader.get_task_prompt(
            task.task_name, self.name, language=self.language,
        )

        format_vars = {
            'language': task.language,
            'description': task.description,
            'system_type': task.system_type,
            'source_code': '{source_code}'
        }
        if task.extra_info:
            format_vars.update(task.extra_info)

        return format_prompt_template(prompt_template, format_vars)
