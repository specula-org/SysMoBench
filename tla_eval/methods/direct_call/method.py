"""
Direct call method implementation.

This method generates a TLA+ specification by prompting an LLM with the source
code, then runs SANY locally on the response. If SANY fails, the error list is
fed back to the model and the spec is regenerated, up to `max_correction_attempts`
times. This keeps the one-shot simplicity while adding a lightweight feedback
loop — the same "3-retry" behavior that CompositeEvaluator provides, but
inlined into generation so it works with the plain Phase 1 / 2 / 3 / 3b pipeline.
"""

import logging
import re
from typing import Any, Dict, List

from ..base import TLAGenerationMethod, GenerationTask, GenerationOutput, format_prompt_template
from ...config import get_configured_model
from ...core.verification.validators import TLAValidator

logger = logging.getLogger(__name__)

# Reused from the evaluator — identical regex for fenced-block extraction.
_TLA_FENCE = re.compile(r"```\s*tla\s*\n(.*?)\n```", re.DOTALL | re.IGNORECASE)
_CFG_FENCE = re.compile(r"```\s*cfg\s*\n(.*?)\n```", re.DOTALL | re.IGNORECASE)


class DirectCallMethod(TLAGenerationMethod):
    """
    Direct call method with SANY-feedback correction loop.

    Flow:
      1. Render task prompt + inject source code → send to model.
      2. Extract ```tla fenced block from response, run SANY on it.
      3. If SANY passes OR this was attempt N: return the latest response.
      4. Otherwise build a correction prompt (original context + failing spec
         + SANY errors) and re-ask the model. Goto 2.

    Notes on cost:
      - Each attempt is ONE API call (no proxy-side retries).
      - Azure Phase 0 is free; other providers pay N× per retry.
      - A model that cannot self-correct will cost max_correction_attempts calls.
    """

    def __init__(self, max_correction_attempts: int = 3, validation_timeout: int = 30):
        super().__init__("direct_call")
        self.max_correction_attempts = max_correction_attempts
        self.validator = TLAValidator(timeout=validation_timeout)

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
            prev_tla: str = ""
            prev_cfg: str = ""
            prev_errors: List[str] = []

            for attempt_idx in range(1, self.max_correction_attempts + 1):
                logger.info(f"[direct_call] Attempt {attempt_idx}/{self.max_correction_attempts}")

                if attempt_idx == 1:
                    # Initial generation — let the adapter substitute {source_code}.
                    result = model.generate_tla_specification(
                        task.source_code, initial_prompt, gen_config
                    )
                else:
                    # Correction — build a complete prompt, no further substitution.
                    correction_prompt = self._build_correction_prompt(
                        initial_prompt=initial_prompt,
                        source_code=task.source_code,
                        prev_tla=prev_tla,
                        prev_cfg=prev_cfg,
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
                    # API failure: stop (don't multiply costs with retries)
                    break

                last_text = result.generated_text
                last_metadata = result.metadata

                # Extract fenced blocks for SANY validation
                tla_match = _TLA_FENCE.search(last_text)
                cfg_match = _CFG_FENCE.search(last_text)
                tla_content = tla_match.group(1) if tla_match else last_text
                cfg_content = cfg_match.group(1) if cfg_match else ""

                # Run SANY locally
                try:
                    val_result = self.validator.validate_specification(tla_content)
                    sany_success = val_result.success
                    errors = list(val_result.syntax_errors) + list(val_result.semantic_errors)
                except Exception as e:
                    logger.warning(f"[direct_call] SANY validation raised: {e}")
                    sany_success = False
                    errors = [f"Validator raised: {e}"]

                attempts.append({
                    "attempt": attempt_idx,
                    "api_success": True,
                    "sany_success": sany_success,
                    "num_errors": len(errors),
                    "usage": last_metadata.get("usage"),
                    "finish_reason": last_metadata.get("finish_reason"),
                })

                if sany_success:
                    logger.info(f"[direct_call] ✓ SANY passed on attempt {attempt_idx}")
                    break

                logger.info(f"[direct_call] ✗ SANY failed on attempt {attempt_idx} — {len(errors)} errors")

                if attempt_idx == self.max_correction_attempts:
                    logger.info("[direct_call] Max attempts reached; returning last response.")
                    break

                # Stash state for next iteration's correction prompt
                prev_tla = tla_content
                prev_cfg = cfg_content
                prev_errors = errors

            return GenerationOutput(
                tla_specification=last_text,
                method_name=self.name,
                task_name=task.task_name,
                metadata={
                    "model_info": model.get_model_info(),
                    "generation_metadata": last_metadata,
                    "prompt_template": "direct_call_with_correction",
                    "max_correction_attempts": self.max_correction_attempts,
                    "correction_attempts": attempts,
                    "final_sany_success": attempts[-1].get("sany_success", False) if attempts else False,
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
        prev_tla: str,
        prev_cfg: str,
        errors: List[str],
        attempt_idx: int,
    ) -> str:
        # Cap error text to keep prompt within reasonable size
        shown = errors[:20]
        error_text = "\n".join(f"  - {e.strip()[:500]}" for e in shown)
        more = f"\n  ... and {len(errors) - len(shown)} more" if len(errors) > len(shown) else ""

        # Reuse the full original prompt (with source injected) so the model
        # still has the task context in view during correction.
        resolved_initial = initial_prompt.replace("{source_code}", source_code)

        return f"""Your previous TLA+ specification failed SANY validation. This is correction attempt {attempt_idx}. Regenerate the full specification, fixing the errors below.

## Original Task Instructions

{resolved_initial}

## Your Previous Output

```tla
{prev_tla}
```

```cfg
{prev_cfg}
```

## SANY Errors Found

{error_text}{more}

## Correction Instructions

1. Fix every error listed above. Common root causes:
   - Undeclared operators (e.g. `NULL`, `Min`, `Max`): either EXTENDS the module that provides them (e.g. `FiniteSetsExt` for `Min`/`Max`) or declare/define them explicitly.
   - Mismatched VARIABLES / CONSTANTS between the .tla and the .cfg.
   - Actions referenced in `Next` but never defined.
2. Keep all mandatory action names from the original instructions (the naming convention is a hard contract).
3. Output EXACTLY TWO fenced code blocks in the same format as before (```tla and ```cfg). Nothing outside the fences.
"""

    def _create_prompt(self, task: GenerationTask) -> str:
        """Create prompt for direct call method using task-specific prompt."""

        # Get task-specific prompt template (lazy import to avoid circular dependency)
        from ...tasks.loader import get_task_loader
        task_loader = get_task_loader()
        prompt_template = task_loader.get_task_prompt(task.task_name, self.name)

        # Prepare format variables
        format_vars = {
            'language': task.language,
            'description': task.description,
            'system_type': task.system_type,
            'source_code': '{source_code}'  # Keep this placeholder for the model adapter
        }

        # Add extra info if available
        if task.extra_info:
            format_vars.update(task.extra_info)

        # Format template with task information without interfering with literal braces
        return format_prompt_template(prompt_template, format_vars)

