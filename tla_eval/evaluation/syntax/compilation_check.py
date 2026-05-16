"""
Compilation Check Evaluator: Phase 1 syntax-level evaluation.

Language-neutral. Dispatches through the LanguageBackend selected by the
`language` constructor argument (default "TLA+"); the backend is responsible
for the language-specific parser/validator. Output shape and side effects
(SyntaxEvaluationResult population, output directory layout, generation-
usage JSON, error_statistics.yaml) are preserved across the refactor.
"""

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ...languages import get as get_backend
from ...languages.base import LanguageBackend
from ...models.base import GenerationResult
from ...utils.output_manager import get_output_manager
from ..base.evaluator import BaseEvaluator
from ..base.result_types import SyntaxEvaluationResult

logger = logging.getLogger(__name__)


def split_tla_and_cfg(raw: str):
    """Legacy helper kept for callers; delegates to the TLA+ backend extractor."""
    artifacts = get_backend("TLA+").extract_artifacts(raw)
    return artifacts.spec, artifacts.config


class CompilationCheckEvaluator(BaseEvaluator):
    """Phase 1 evaluator. Parses/type-checks the generated spec via the backend."""

    def __init__(self, language: str = "TLA+", validation_timeout: int = 30):
        super().__init__(timeout=validation_timeout)
        self.language = language
        self.backend: LanguageBackend = get_backend(language)

    def evaluate(self,
                 generation_result: GenerationResult,
                 task_name: str,
                 method_name: str,
                 model_name: str,
                 spec_module: str = None,
                 spec_file_path: Optional[str] = None,
                 config_file_path: Optional[str] = None) -> SyntaxEvaluationResult:
        logger.info(f"Evaluating compilation ({self.language}): {task_name}/{method_name}/{model_name}")

        output_manager = get_output_manager()
        output_dir = output_manager.create_experiment_dir(
            metric="compilation_check",
            task=task_name,
            method=method_name,
            model=model_name,
            language=self.language,
        )
        logger.info(f"Using output directory: {output_dir}")

        eval_result = SyntaxEvaluationResult(task_name, method_name, model_name)
        self._set_generation_result(eval_result, generation_result)

        spec_content: Optional[str] = None
        cfg_content: Optional[str] = None

        if spec_file_path and Path(spec_file_path).exists():
            logger.info(f"Using existing spec file: {spec_file_path}")
            try:
                with open(spec_file_path, "r", encoding="utf-8") as f:
                    spec_content = f.read()
            except Exception as e:
                logger.error(f"Failed to read spec file: {e}")
                self._record_failure(eval_result, f"Cannot read spec file: {e}")
                return eval_result
        else:
            if not generation_result.success:
                logger.warning(f"Generation failed, cannot proceed: {generation_result.error_message}")
                self._record_failure(eval_result, "Generation failed - no specification to validate",
                                     as_semantic=True)
                return eval_result
            artifacts = self.backend.extract_artifacts(generation_result.generated_text)
            spec_content = artifacts.spec
            cfg_content = artifacts.config

        eval_result.generated_specification = spec_content

        module_name = spec_module or task_name or "UnnamedModule"
        spec_ext = self.backend.spec_extension or ".spec"
        spec_path = output_dir / f"{module_name}{spec_ext}"
        try:
            with open(spec_path, "w", encoding="utf-8") as f:
                f.write(spec_content)
        except Exception as e:
            logger.error(f"Failed to write spec to disk: {e}")

        try:
            logger.debug(f"Starting {self.language} syntax validation...")
            outcome = self.backend.validate_syntax(
                spec=spec_content,
                config=cfg_content,
                work_dir=output_dir,
                timeout=self.timeout,
                spec_filename=spec_path.name,
            )
            logger.info(f"DEBUG: validate_syntax success={outcome.success}, "
                        f"syntax={len(outcome.syntax_errors)}, semantic={len(outcome.semantic_errors)}")
            logger.info(f"DEBUG: validate_syntax output (first 500 chars): {outcome.raw_output[:500]}")
            self._apply_outcome(eval_result, outcome)

            if outcome.success:
                logger.info("✓ Specification compiled successfully")
            else:
                err_count = len(outcome.syntax_errors) + len(outcome.semantic_errors)
                if err_count == 0:
                    logger.warning(f"Compilation success=False but no errors reported")
                else:
                    logger.warning(f"Compilation failed with {err_count} errors")
        except Exception as e:
            logger.error(f"Validation error: {e}")
            self._apply_outcome(
                eval_result,
                _failure_outcome(f"Validation error: {e}"),
            )

        if cfg_content and self.backend.config_extension:
            cfg_path = output_dir / f"{module_name}{self.backend.config_extension}"
            try:
                with open(cfg_path, "w", encoding="utf-8") as f:
                    f.write(cfg_content)
                logger.info(f"Saved config to: {cfg_path}")
            except Exception as e:
                logger.warning(f"Failed to write cfg: {e}")
        elif self.backend.config_fence_label and not cfg_content:
            logger.warning(
                f"Model did not emit a ```{self.backend.config_fence_label} fenced block — "
                f"no cfg written, downstream Phase 2/4 will fail"
            )

        # Phase 0 accounting — preserved from pre-refactor.
        gen_meta = getattr(generation_result, "metadata", {}) or {}
        nested = gen_meta.get("generation_metadata") or {}

        def _pick(key):
            return nested.get(key) if nested.get(key) is not None else gen_meta.get(key)

        generation_usage = {
            "model": _pick("model"),
            "litellm_model": _pick("litellm_model"),
            "provider": _pick("provider"),
            "finish_reason": _pick("finish_reason"),
            "latency_seconds": _pick("latency_seconds"),
            "usage": _pick("usage"),
            "response_id": _pick("response_id"),
            "attempts_used": gen_meta.get("attempts_used"),
            "final_sany_success": gen_meta.get("final_sany_success"),
            "correction_attempts": gen_meta.get("correction_attempts"),
        }
        try:
            import json as _json
            with open(output_dir / "generation_usage.json", "w", encoding="utf-8") as f:
                _json.dump(generation_usage, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.warning(f"Failed to save generation_usage.json: {e}")

        result_data = {
            "overall_success": eval_result.overall_success,
            "generation_successful": eval_result.generation_successful,
            "compilation_successful": eval_result.compilation_successful,
            "generation_time": eval_result.generation_time,
            "compilation_time": eval_result.compilation_time,
            "syntax_errors": eval_result.syntax_errors,
            "semantic_errors": eval_result.semantic_errors,
            "generation_error": eval_result.generation_error,
        }
        metadata = {
            "task_name": task_name,
            "method_name": method_name,
            "model_name": model_name,
            "metric": "compilation_check",
            "language": self.language,
            "spec_module": spec_module,
            "validation_timeout": self.timeout,
            "evaluation_timestamp": time.time(),
        }
        output_manager.save_result(output_dir, result_data, metadata)

        try:
            self.backend.finalize_run(
                work_dir=output_dir,
                task_name=task_name,
                method_name=method_name,
                model_name=model_name,
            )
        except Exception as e:
            logger.error(f"backend.finalize_run failed: {e}")

        logger.info(f"Evaluation complete: success={eval_result.overall_success}")
        return eval_result

    def evaluate_batch(self,
                       results: List[Tuple[GenerationResult, str, str, str]]
                       ) -> List[SyntaxEvaluationResult]:
        logger.info(f"Starting batch evaluation of {len(results)} results")
        out: List[SyntaxEvaluationResult] = []
        for i, (gr, task_name, method_name, model_name) in enumerate(results):
            logger.info(f"Processing batch item {i+1}/{len(results)}")
            try:
                out.append(self.evaluate(gr, task_name, method_name, model_name))
            except Exception as e:
                logger.error(f"Batch evaluation error for item {i+1}: {e}")
                er = SyntaxEvaluationResult(task_name, method_name, model_name)
                er.generation_error = f"Batch evaluation error: {e}"
                out.append(er)
        logger.info(f"Batch evaluation complete: {len(out)} results")
        return out

    # ---- helpers -----------------------------------------------------------

    def _set_generation_result(self, eval_result: SyntaxEvaluationResult,
                               generation_result: GenerationResult):
        eval_result.generation_successful = generation_result.success
        eval_result.generation_time = generation_result.metadata.get("latency_seconds", 0.0)
        eval_result.generated_specification = generation_result.generated_text
        if not generation_result.success:
            eval_result.generation_error = generation_result.error_message

    def _apply_outcome(self, eval_result: SyntaxEvaluationResult, outcome) -> None:
        eval_result.compilation_successful = outcome.success
        eval_result.compilation_time = outcome.elapsed_seconds
        eval_result.syntax_errors = list(outcome.syntax_errors)
        eval_result.semantic_errors = list(outcome.semantic_errors)
        eval_result.compilation_output = outcome.raw_output
        eval_result.compilation_errors = list(outcome.syntax_errors) + list(outcome.semantic_errors)
        eval_result.overall_success = (
            eval_result.generation_successful and eval_result.compilation_successful
        )

    def _record_failure(self, eval_result: SyntaxEvaluationResult, msg: str,
                        as_semantic: bool = False) -> None:
        self._apply_outcome(
            eval_result,
            _failure_outcome(msg, as_semantic=as_semantic),
        )

    def _calculate_summary(self, results: List[SyntaxEvaluationResult]) -> Dict[str, Any]:
        if not results:
            return {}
        total = len(results)
        gen_ok = sum(1 for r in results if r.generation_successful)
        compile_ok = sum(1 for r in results if r.compilation_successful)
        overall_ok = sum(1 for r in results if r.overall_success)
        gen_times = [r.generation_time for r in results if r.generation_time > 0]
        compile_times = [r.compilation_time for r in results if r.compilation_time > 0]
        total_errors = sum(len(r.compilation_errors) for r in results)
        return {
            "total_evaluations": total,
            "success_rates": {
                "generation": gen_ok / total if total else 0.0,
                "compilation": compile_ok / total if total else 0.0,
                "overall": overall_ok / total if total else 0.0,
            },
            "counts": {
                "generation_successful": gen_ok,
                "compilation_successful": compile_ok,
                "overall_successful": overall_ok,
            },
            "timing": {
                "avg_generation_time": sum(gen_times) / len(gen_times) if gen_times else 0.0,
                "avg_compilation_time": sum(compile_times) / len(compile_times) if compile_times else 0.0,
                "max_generation_time": max(gen_times) if gen_times else 0.0,
                "max_compilation_time": max(compile_times) if compile_times else 0.0,
            },
            "errors": {
                "total_compilation_errors": total_errors,
                "avg_errors_per_evaluation": total_errors / total if total else 0.0,
            },
        }

    def _get_evaluation_type(self) -> str:
        return "syntax_compilation_check"


def _failure_outcome(msg: str, as_semantic: bool = False):
    from ...languages.result_types import SyntaxOutcome
    return SyntaxOutcome(
        success=False,
        syntax_errors=[] if as_semantic else [msg],
        semantic_errors=[msg] if as_semantic else [],
        raw_output=msg,
        elapsed_seconds=0.0,
        error_message=msg,
    )


def create_compilation_check_evaluator(
    validation_timeout: int = 30,
    language: str = "TLA+",
) -> CompilationCheckEvaluator:
    return CompilationCheckEvaluator(language=language, validation_timeout=validation_timeout)
