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
    """Score a TLA+ spec against captured system traces, action by action."""

    def __init__(self,
                 tv_agent: Optional[str] = None,
                 tv_model: Optional[str] = None,
                 tv_budget: float = 5.0,
                 tv_timeout: int = 1800,
                 workspace_root: Optional[str] = None):
        super().__init__(timeout=tv_timeout)
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

        logger.warning(
            "Launching transition validation for %s — this runs an agent against the real system harness. "
            "Expect 30 min to several hours and roughly $1–4 in agent API spend.",
            task_name,
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

        # Snapshot existing workspaces BEFORE the harness runs, so we can
        # identify the one this run creates. Picking "most recent by mtime"
        # is wrong: if the harness fails before creating a workspace, it
        # selects a stale workspace from a previous task and reports a
        # spurious result.
        workspaces_before = set(glob.glob(str(self.workspace_root / "*")))

        start = time.time()
        try:
            proc = subprocess.run(
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

        # Pick the workspace this run created (not in the before-snapshot).
        new_workspaces = sorted(
            (set(glob.glob(str(self.workspace_root / "*"))) - workspaces_before),
            key=os.path.getmtime,
            reverse=True,
        )
        if not new_workspaces:
            # No new workspace — the harness failed before creating one.
            # Surface the launcher's own output so the real cause is visible.
            tail = ((proc.stderr or "") + (proc.stdout or ""))[-1500:]
            result.error_message = (
                "launch_tv_eval.sh created no workspace "
                f"(exit {proc.returncode}). Launcher output tail:\n{tail}"
            )
            logger.error(result.error_message)
            return result

        workspace = Path(new_workspaces[0])
        result.workspace_dir = str(workspace)
        results_path = workspace / "reports" / "tv_results.json"

        if not results_path.exists():
            result.error_message = f"Workspace exists but {results_path} is missing"
            logger.error(result.error_message)
            return result

        try:
            with open(results_path) as f:
                tv_data = json.load(f)
        except Exception as e:
            result.error_message = f"Failed to parse {results_path}: {e}"
            logger.error(result.error_message)
            return result

        for action, info in tv_data.items():
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
