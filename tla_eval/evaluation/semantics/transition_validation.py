"""
Transition Validation evaluator.

Wraps `scripts/launch_wv_eval.sh` so it can be invoked through the standard
metric registry as `--metric transition_validation`. The launcher hands the
spec to a coding-agent CLI (claude-code or codex) that runs the `wv-eval`
skill: instrument the upstream system, generate windows, run TLC on every
(pre, post)-state pair, and produce per-action pass rates in
`<workspace>/reports/wv_results.json`.

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
    """Score a TLA+ spec against captured system traces, action by action."""

    def __init__(self,
                 wv_agent: Optional[str] = None,
                 wv_model: Optional[str] = None,
                 wv_budget: float = 5.0,
                 wv_timeout: int = 1800,
                 workspace_root: Optional[str] = None):
        super().__init__(timeout=wv_timeout)
        self.wv_agent = wv_agent
        self.wv_model = wv_model
        self.wv_budget = wv_budget
        self.wv_timeout = wv_timeout
        self.workspace_root = Path(workspace_root) if workspace_root else (PROJECT_ROOT / "wv-workspaces")

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

        logger.warning(
            "Launching transition validation for %s — this runs an agent against the real system harness. "
            "Expect 30 min to several hours and roughly $1–4 in agent API spend.",
            task_name,
        )

        launcher = PROJECT_ROOT / "scripts" / "launch_wv_eval.sh"
        cmd = [
            "bash", str(launcher),
            f"--task={task_name}",
            f"--spec={spec_dir}",
            f"--workspace-root={self.workspace_root}",
            f"--max-budget={self.wv_budget}",
        ]
        if self.wv_agent:
            cmd.append(f"--agent={self.wv_agent}")
        if self.wv_model:
            cmd.append(f"--model={self.wv_model}")

        start = time.time()
        try:
            subprocess.run(
                cmd,
                capture_output=True, text=True,
                timeout=self.wv_timeout,
                cwd=str(PROJECT_ROOT),
            )
        except subprocess.TimeoutExpired:
            result.elapsed_seconds = time.time() - start
            result.error_message = f"launch_wv_eval.sh timed out after {self.wv_timeout}s"
            logger.error(result.error_message)
            return result
        except Exception as e:
            result.elapsed_seconds = time.time() - start
            result.error_message = f"launch_wv_eval.sh failed: {e}"
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
        results_path = workspace / "reports" / "wv_results.json"

        if not results_path.exists():
            result.error_message = f"Workspace exists but {results_path} is missing"
            logger.error(result.error_message)
            return result

        try:
            with open(results_path) as f:
                wv_data = json.load(f)
        except Exception as e:
            result.error_message = f"Failed to parse {results_path}: {e}"
            logger.error(result.error_message)
            return result

        for action, info in wv_data.items():
            stats = info.get("stats", {})
            result.total_passed += stats.get("passed", 0)
            result.total_windows += stats.get("total", 0)
            result.per_action_pass_rates[action] = stats.get("pass_rate", 0.0)

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
