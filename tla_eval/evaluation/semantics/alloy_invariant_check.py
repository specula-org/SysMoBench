"""Alloy invariant evaluator that injects template-based asserts."""

import json
import logging
import time
from pathlib import Path
from typing import Optional

from ...models.base import GenerationResult
from ...utils.output_manager import get_output_manager
from ..base.evaluator import BaseEvaluator
from ..base.result_types import SemanticEvaluationResult
from ...core.verification.alloy_runtime_executor import AlloyRuntimeExecutor

logger = logging.getLogger(__name__)


class AlloyInvariantTemplateManager:
    """Loads Alloy invariant templates from disk."""

    def __init__(self, templates_root: str = "data/alloy_invariant_templates"):
        self.templates_root = Path(templates_root)

    def load_template(self, task_name: str) -> str:
        template_file = self.templates_root / task_name / "invariants.als"

        if not template_file.exists():
            raise FileNotFoundError(
                f"Alloy invariant template not found for task '{task_name}'. "
                f"Expected: {template_file}"
            )

        return template_file.read_text(encoding="utf-8")


class AlloyInvariantCheckEvaluator(BaseEvaluator):
    """Evaluator for Phase 4 invariant checking on Alloy specs."""

    def __init__(self, validation_timeout: int = 60, templates_root: str = "data/alloy_invariant_templates"):
        super().__init__(timeout=validation_timeout)
        self.runtime_executor = AlloyRuntimeExecutor(timeout=validation_timeout)
        self.template_manager = AlloyInvariantTemplateManager(templates_root=templates_root)

    def evaluate(
        self,
        generation_result: GenerationResult,
        task_name: str,
        method_name: str,
        model_name: str,
        spec_module: str = None,
        spec_file_path: Optional[str] = None,
        config_file_path: Optional[str] = None,
    ) -> SemanticEvaluationResult:
        logger.info(f"Evaluating Alloy invariants: {task_name}/{method_name}/{model_name}")

        output_manager = get_output_manager()
        output_dir = output_manager.create_experiment_dir(
            metric="invariant_verification",
            task=task_name,
            method=method_name,
            model=model_name,
            language="alloy",
        )

        eval_result = SemanticEvaluationResult(task_name, method_name, model_name)

        if generation_result and generation_result.metadata:
            eval_result.generation_time = generation_result.metadata.get("latency_seconds", 0.0)

        base_spec_path: Optional[Path] = None
        if spec_file_path and Path(spec_file_path).exists():
            base_spec_path = Path(spec_file_path)
        elif generation_result and generation_result.generated_text:
            base_spec_path = output_dir / f"{task_name}.als"
            base_spec_path.write_text(generation_result.generated_text, encoding="utf-8")
        else:
            eval_result.model_checking_successful = False
            eval_result.model_checking_error = "No specification provided"
            eval_result.overall_success = False
            return eval_result

        eval_result.specification_file = str(base_spec_path)

        try:
            template_snippet = self.template_manager.load_template(task_name)
        except FileNotFoundError as exc:
            eval_result.model_checking_successful = False
            eval_result.model_checking_error = str(exc)
            eval_result.overall_success = False
            return eval_result

        final_spec_path = output_dir / f"{task_name}_with_invariants.als"
        base_content = base_spec_path.read_text(encoding="utf-8")
        snippet_header = (
            "\n\n// ---- SysMoBench Alloy Invariants (auto-inserted) ----\n"
        )
        final_spec_path.write_text(base_content + snippet_header + template_snippet, encoding="utf-8")

        logger.info(f"Appended invariant templates from {task_name} to spec: {final_spec_path}")

        start_time = time.time()
        runtime_result = self.runtime_executor.run(final_spec_path)
        eval_result.model_checking_time = time.time() - start_time

        if not runtime_result.get("success", False):
            eval_result.model_checking_successful = False
            eval_result.model_checking_error = runtime_result.get("error", "Runtime execution failed")
            eval_result.overall_success = False
            return eval_result

        invariant_commands = [
            cmd
            for cmd in runtime_result.get("commands", [])
            if str(cmd.get("type", "")).lower() == "check"
            and str(cmd.get("label", "")).upper().startswith("INV")
        ]

        if not invariant_commands:
            eval_result.model_checking_successful = False
            eval_result.model_checking_error = "No invariant check commands were executed"
            eval_result.overall_success = False
            eval_result.custom_data = {
                "total_commands": runtime_result.get("total_commands", 0),
                "invariants_total": 0,
            }
            return eval_result

        passed = [
            cmd for cmd in invariant_commands if str(cmd.get("result", "")).upper().startswith("PASS")
        ]
        failed = [
            cmd for cmd in invariant_commands if not str(cmd.get("result", "")).upper().startswith("PASS")
        ]

        eval_result.model_checking_successful = len(failed) == 0
        eval_result.model_checking_error = None if eval_result.model_checking_successful else "Invariant violations detected"
        eval_result.overall_success = eval_result.model_checking_successful

        eval_result.custom_data = {
            "invariants_total": len(invariant_commands),
            "invariants_passed": len(passed),
            "invariants_failed": len(failed),
            "invariant_results": invariant_commands,
            "total_commands": runtime_result.get("total_commands", 0),
            "spec_with_invariants": str(final_spec_path),
        }

        if eval_result.model_checking_successful:
            logger.info(f"All {len(invariant_commands)} Alloy invariants passed")
        else:
            logger.error(f"{len(failed)} Alloy invariants failed")

        results_file = output_dir / "evaluation_result.json"
        try:
            with open(results_file, "w", encoding="utf-8") as handle:
                json.dump(eval_result.to_dict(), handle, indent=2)
        except Exception as exc:  # pragma: no cover - best effort logging
            logger.warning(f"Failed to save invariant evaluation results: {exc}")

        return eval_result

    def _get_evaluation_type(self) -> str:
        return "alloy_invariant"

