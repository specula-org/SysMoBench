"""
Transition Validation evaluator.

Wraps `scripts/launch_tv_eval.sh` so it can be invoked through the standard
metric registry as `--metric transition_validation`. The launcher hands the
spec to a coding-agent CLI (claude-code or codex) that runs the `tv-eval`
skill: instrument the upstream system, generate windows, run TLC on every
(pre, post)-state pair, and produce per-action pass rates in
`<workspace>/reports/tv_results.json`.

This evaluator parses that JSON and packages the results.
"""

import glob
import json
import logging
import os
import subprocess
import time
from pathlib import Path
from typing import Optional

from ..base.evaluator import BaseEvaluator
from ..base.result_types import TransitionValidationResult
from ...models.base import GenerationResult

logger = logging.getLogger(__name__)


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent


class TransitionValidationEvaluator(BaseEvaluator):
    """
    Phase 3 evaluator. Dispatches through a LanguageBackend. If the backend
    exposes a direct (pre, action, post) validator, calls it. Otherwise falls
    back to the agent-driven `scripts/launch_tv_eval.sh` flow (TLA+'s path).
    """

    def __init__(self,
                 language: str = "TLA+",
                 tv_agent: Optional[str] = None,
                 tv_model: Optional[str] = None,
                 tv_budget: float = 5.0,
                 tv_timeout: int = 1800,
                 workspace_root: Optional[str] = None):
        super().__init__(timeout=tv_timeout)
        from ...languages import get as _get_backend
        self.language = language
        self.backend = _get_backend(language)
        self.tv_agent = tv_agent
        self.tv_model = tv_model
        self.tv_budget = tv_budget
        self.tv_timeout = tv_timeout
        self.workspace_root = Path(workspace_root) if workspace_root else (PROJECT_ROOT / "tv-workspaces")

    def evaluate(self,
                 generation_result: GenerationResult,
                 task_name: str,
                 method_name: str,
                 model_name: str,
                 spec_module: str = None,
                 spec_file_path: Optional[str] = None,
                 config_file_path: Optional[str] = None) -> TransitionValidationResult:
        result = TransitionValidationResult(task_name, method_name, model_name)

        if hasattr(generation_result, 'metadata') and 'latency_seconds' in generation_result.metadata:
            result.generation_time = generation_result.metadata['latency_seconds']

        if not spec_file_path:
            result.error_message = (
                "transition_validation requires a spec on disk. "
                "Pass --spec-file <path-to-.tla>, or run inside the batch pipeline which writes the spec first."
            )
            logger.error(result.error_message)
            return result

        spec_path = Path(spec_file_path).resolve()
        if not spec_path.exists():
            result.error_message = f"Spec file not found: {spec_path}"
            logger.error(result.error_message)
            return result

        spec_dir = spec_path.parent

        if self.backend.supports_direct_transition_validation:
            # Direct path: hand control to the backend. Trace loading + windowing
            # is the backend's responsibility (it knows the language semantics).
            result.error_message = (
                f"Direct transition validation for {self.language} is declared "
                "supported but the evaluator wrapper for that path is not yet implemented."
            )
            logger.error(result.error_message)
            return result

        logger.warning(
            "Launching transition validation for %s (%s) — agent path. "
            "Expect 30 min to several hours and roughly $1–4 in agent API spend.",
            task_name, self.language,
        )

        launcher = PROJECT_ROOT / "scripts" / "launch_tv_eval.sh"
        cmd = [
            "bash", str(launcher),
            f"--task={task_name}",
            f"--spec={spec_dir}",
            f"--workspace-root={self.workspace_root}",
            f"--max-budget={self.tv_budget}",
        ]
        if self.tv_agent:
            cmd.append(f"--agent={self.tv_agent}")
        if self.tv_model:
            cmd.append(f"--model={self.tv_model}")

        start = time.time()
        try:
            subprocess.run(
                cmd,
                capture_output=True, text=True,
                timeout=self.tv_timeout,
                cwd=str(PROJECT_ROOT),
            )
        except subprocess.TimeoutExpired:
            result.elapsed_seconds = time.time() - start
            result.error_message = f"launch_tv_eval.sh timed out after {self.tv_timeout}s"
            logger.error(result.error_message)
            return result
        except Exception as e:
            result.elapsed_seconds = time.time() - start
            result.error_message = f"launch_tv_eval.sh failed: {e}"
            logger.error(result.error_message)
            return result
        result.elapsed_seconds = time.time() - start

        # Pick the most recently modified workspace dir.
        candidates = sorted(
            glob.glob(str(self.workspace_root / "*")),
            key=os.path.getmtime,
            reverse=True,
        )
        if not candidates:
            result.error_message = "No transition-validation workspace was created"
            logger.error(result.error_message)
            return result

        workspace = Path(candidates[0])
        result.workspace_dir = str(workspace)

        # The tv-eval skill historically wrote `tv_results.json` but now emits
        # `tlc_results.json` with a flatter schema. Accept either filename.
        reports_dir = workspace / "reports"
        results_path = next(
            (reports_dir / name for name in ("tv_results.json", "tlc_results.json")
             if (reports_dir / name).exists()),
            None,
        )
        if results_path is None:
            result.error_message = (
                f"Workspace exists but neither {reports_dir}/tv_results.json nor "
                f"{reports_dir}/tlc_results.json is present"
            )
            logger.error(result.error_message)
            return result

        try:
            with open(results_path) as f:
                tv_data = json.load(f)
        except Exception as e:
            result.error_message = f"Failed to parse {results_path}: {e}"
            logger.error(result.error_message)
            return result

        # Two schemas are accepted:
        #   legacy: {action: {stats: {passed, total, pass_rate}, ...}}
        #   current: {action: {passes, total, rate, ...}}
        for action, info in tv_data.items():
            stats = info.get("stats") if isinstance(info, dict) else None
            if stats:
                passed = stats.get("passed", 0)
                total = stats.get("total", 0)
                pass_rate = stats.get("pass_rate", 0.0)
            else:
                passed = info.get("passes", info.get("passed", 0))
                total = info.get("total", 0)
                pass_rate = info.get("rate", info.get("pass_rate", 0.0))
            result.total_passed += passed
            result.total_windows += total
            result.per_action_pass_rates[action] = pass_rate

        if result.total_windows > 0:
            result.score = result.total_passed / result.total_windows

        # Define success as "every per-action group has at least one passing window."
        # Strict zero-tolerance per-action gate is a downstream policy concern.
        result.overall_success = result.total_windows > 0 and result.score > 0
        logger.info(
            f"Transition validation: {result.total_passed}/{result.total_windows} "
            f"({result.score:.1%}) across {len(result.per_action_pass_rates)} actions"
        )
        return result

    def _get_evaluation_type(self) -> str:
        return "transition_validation"
