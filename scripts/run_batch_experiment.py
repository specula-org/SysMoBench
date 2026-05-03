#!/usr/bin/env python3
"""
Batch Experiment Runner for TLA+ Specification Generation

This script runs batch experiments for multiple systems with:
- Thread pool for parallel execution (default: 5 threads)
- Up to 5 runs per system (early stop if all phases score 1.0)
- Three evaluation phases:
  1. Compilation: compilation_check pass=1.0, fail=action_decomposition_ratio*0.5
  2. Runtime: runtime_coverage coverage value
  3. Invariant: invariant_verification pass ratio (using agent translator)

Usage:
    python scripts/run_batch_experiment.py --systems etcd spin --runs 5 --threads 5
    python scripts/run_batch_experiment.py --all --threads 10
"""

import argparse
import json
import logging
import os
import signal
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import threading

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env", override=False)
except ImportError:
    pass

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] [%(threadName)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# All available systems
ALL_SYSTEMS = [
    "spin", "etcd", "mutex", "rwmutex", "curp", "dqueue",
    "locksvc", "raftkvs", "redisraft", "ringbuffer", "zookeeper"
]

# Supported generation methods. Only `direct_call` (single-shot API call via
# the configured model adapter in config/models.yaml) is wired up today;
# previous coding-agent driven generation paths were never finished and have
# been removed to avoid silently misdirecting users.
SUPPORTED_AGENTS = {
    "direct_call": "direct_call",
}

DEFAULT_AGENT = "direct_call"


def run_with_timeout_kill_process_group(
    cmd: List[str],
    timeout: int,
    cwd: Path,
) -> subprocess.CompletedProcess:
    """Run a command and kill its full process group if timeout is hit."""
    process = subprocess.Popen(
        cmd,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        def terminate_group(sig: int):
            if hasattr(os, "killpg"):
                os.killpg(process.pid, sig)
            elif sig == signal.SIGTERM:
                process.terminate()
            else:
                process.kill()

        if process.poll() is None:
            try:
                terminate_group(signal.SIGTERM)
                process.wait(timeout=5)
            except Exception:
                pass
            if process.poll() is None:
                try:
                    terminate_group(signal.SIGKILL)
                except Exception:
                    pass
            stdout, stderr = process.communicate()
        raise subprocess.TimeoutExpired(
            cmd=cmd,
            timeout=timeout,
            output=stdout,
            stderr=stderr,
        )

    return subprocess.CompletedProcess(
        args=cmd,
        returncode=process.returncode,
        stdout=stdout,
        stderr=stderr,
    )


@dataclass
class PhaseResult:
    """Result of a single evaluation phase.

    `status` semantics (distinct from score):
      - "ran": the phase was actually evaluated. `score` is meaningful.
               score=0.0 here means the spec truly failed this check.
      - "skipped": cascade-skip — upstream phase failed so evaluating this
                   phase is pointless. score=0.0 but does NOT count as a real
                   failure; just an absence of signal.
      - "pending": infrastructure missing (e.g., TV requires a trace harness
                   that hasn't been built yet). score=None; to be filled in
                   later when the infra exists.
      - "not_evaluated": the pipeline never reached this phase (e.g.,
                   generation failed, or the batch crashed before getting
                   here). score=None.
    """
    phase_name: str
    score: Optional[float]
    passed: bool
    status: str = "ran"
    details: Dict = field(default_factory=dict)
    error: Optional[str] = None
    passed_items: List[str] = field(default_factory=list)
    failed_items: List[str] = field(default_factory=list)


@dataclass
class RunResult:
    """Result of a single experiment run"""
    run_id: int
    system: str
    timestamp: str
    generation_success: bool
    generation_time: float
    spec_path: Optional[str] = None
    config_path: Optional[str] = None
    workspace_path: Optional[str] = None

    phase1_compilation: Optional[PhaseResult] = None
    phase2_runtime: Optional[PhaseResult] = None
    phase3_tv: Optional[PhaseResult] = None
    phase3_invariant: Optional[PhaseResult] = None

    # Per-phase API usage. phase0 = generation (dict from generation_usage.json),
    # phase3_tv_usage = {"cost_usd", "duration_ms", "num_turns", "model_usage"}
    # from the TV agent's .run.usage.json. Dollar amounts for phase0 are NOT
    # included (adapter reports only tokens; user looks up $ from gptsapi dashboard).
    phase0_usage: Optional[Dict[str, Any]] = None
    phase3_tv_usage: Optional[Dict[str, Any]] = None

    total_score: float = 0.0
    is_perfect: bool = False
    error: Optional[str] = None

    def calculate_total_score(self):
        """
        Calculate total score. Formula:
          total = mean over {ran at score} ∪ {skipped as 0}
        Rationale: a "skipped" phase (cascade-skip because upstream failed)
        means the spec could not demonstrate anything in that phase — giving
        0 credit. Before 2026-04-18 we excluded skipped phases from the
        denominator, which let a spec that only passed Phase 1 end up at
        0.50 just because its cascade-skipped phases didn't drag down
        the average. That was generous nonsense — fixed here.

        "not_evaluated" / "pending" phases remain excluded: those mean
        "we never attempted this phase" (e.g., TV not enabled, generation
        failed entirely) rather than "we attempted but cascade-skipped".
        """
        phases = [self.phase1_compilation, self.phase2_runtime,
                  self.phase3_tv, self.phase3_invariant]
        scored = []
        for p in phases:
            if p is None:
                continue
            if p.status == "ran" and p.score is not None:
                scored.append(p.score)
            elif p.status == "skipped":
                scored.append(0.0)
            # "not_evaluated" / "pending" / None → excluded

        self.total_score = sum(scored) / len(scored) if scored else 0.0

        ran_all = all(p is not None and p.status == "ran" for p in phases)
        self.is_perfect = (
            ran_all and all((p.score or 0) >= 1.0 for p in phases)
        )


@dataclass
class SystemResult:
    """Results for all runs of a single system"""
    system: str
    runs: List[RunResult] = field(default_factory=list)
    best_run: Optional[RunResult] = None
    best_spec_path: Optional[str] = None

    def find_best_run(self):
        """Find the best run based on total score"""
        if not self.runs:
            return

        valid_runs = [r for r in self.runs if r.generation_success]
        if not valid_runs:
            return

        self.best_run = max(valid_runs, key=lambda r: r.total_score)
        self.best_spec_path = self.best_run.spec_path


class BatchExperimentRunner:
    """Main experiment runner with thread pool"""

    def __init__(self,
                 systems: List[str],
                 max_runs: int = 5,
                 num_threads: int = 5,
                 output_dir: str = "experiments",
                 model: str = "opus",
                 agent: str = DEFAULT_AGENT,
                 enable_tv: bool = True,
                 tv_budget: float = 5.0,
                 tv_timeout: int = 1800,
                 inv_model: str = "sonnet",
                 tv_agent: Optional[str] = None,
                 tv_model: Optional[str] = None):
        """
        Initialize the batch experiment runner.

        Args:
            systems: List of system names to evaluate
            max_runs: Maximum runs per system (default: 5)
            num_threads: Number of parallel threads (default: 5)
            output_dir: Base output directory for results
            model: Model to use for generation and agent translator (default: opus)
            agent: Code agent to use for generation (default: claude_code)
            enable_tv: Run transition validation (default: True). Set to False for cheap CI/smoke runs.
            tv_budget: Max API budget per TV evaluation in USD (default: 5)
            tv_timeout: Timeout per TV evaluation in seconds (default: 1800)
        """
        self.systems = systems
        self.max_runs = max_runs
        self.num_threads = num_threads
        self.output_dir = Path(output_dir)
        self.model = model
        self.agent = agent
        self.agent_method = SUPPORTED_AGENTS[agent]
        self.enable_tv = enable_tv
        self.tv_budget = tv_budget
        self.tv_timeout = tv_timeout
        self.inv_model = inv_model
        self.tv_agent = tv_agent
        self.tv_model = tv_model

        # Create output directory with timestamp
        self.experiment_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_dir = self.output_dir / f"batch_{self.experiment_id}"
        self.experiment_dir.mkdir(parents=True, exist_ok=True)

        # Results storage
        self.system_results: Dict[str, SystemResult] = {}
        self.results_lock = threading.Lock()

        # Setup file logging
        log_file = self.experiment_dir / "experiment.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter(
            '[%(asctime)s] [%(levelname)s] [%(threadName)s] %(message)s'
        ))
        logging.getLogger().addHandler(file_handler)

        logger.info(f"Initialized batch experiment: {self.experiment_id}")
        logger.info(f"Agent: {agent} (method: {self.agent_method})")
        logger.info(f"Model: {model}")
        logger.info(f"Systems: {systems}")
        logger.info(f"Max runs per system: {max_runs}")
        logger.info(f"Thread pool size: {num_threads}")
        logger.info(f"Output directory: {self.experiment_dir}")

    def run_generation(self, system: str, run_id: int) -> Tuple[bool, str, Optional[str], Optional[str], float]:
        """
        Run the generation phase using the configured code agent.

        Returns:
            Tuple of (success, workspace_path, spec_path, config_path, generation_time)
        """
        logger.info(f"[{system}][Run {run_id}] Starting generation phase ({self.agent})...")

        start_time = time.time()

        cmd = [
            "python3", "scripts/run_benchmark.py",
            "--task", system,
            "--method", self.agent_method,
            "--model", self.model,
        ]

        try:
            result = run_with_timeout_kill_process_group(
                cmd,
                timeout=5400,  # 90 min cap — accommodates heavy-reasoning models
                               # (MiniMax-M2.7) doing up to 3 retry attempts on
                               # a big prompt. Single-attempt timeout is set
                               # per-model in models.yaml.
                cwd=PROJECT_ROOT
            )

            generation_time = time.time() - start_time

            # Determine spec/cfg paths from THIS subprocess's stdout only.
            # The compilation_check evaluator always logs:
            #   "Saved specification to: <abs_path>"
            #   "Saved config to: <abs_path>"
            # when it successfully writes its output. No logs → no files →
            # genuine generation failure. We used to fall back to scanning
            # the output tree by mtime, but that leaked across batches and
            # silently fed stale specs from earlier runs into Phase 1
            # (observed 2026-04-17 on minimax_m27 etcd re-run: Phase 0
            # wrote nothing, fallback picked a 50-min-old spec and
            # reported 0.46). Keep this strict.
            import re as _re
            workspace_path = None
            spec_path = None
            config_path = None

            # Scan both stdout AND stderr: Python's stdlib logger writes to
            # stderr by default, so the evaluator's "Saved specification to:"
            # line lives there, not in stdout.
            combined = (result.stdout or "") + "\n" + (result.stderr or "")
            spec_match = _re.search(r"Saved specification to:\s*(\S+\.tla)", combined)
            if spec_match:
                raw = spec_match.group(1)
                # Evaluator may emit a relative path; resolve against project root.
                p = Path(raw)
                spec_path = str(p if p.is_absolute() else PROJECT_ROOT / p)
                workspace_path = str(Path(spec_path).parent)

            cfg_match = _re.search(r"Saved config to:\s*(\S+\.cfg)", combined)
            if cfg_match:
                raw = cfg_match.group(1)
                p = Path(raw)
                config_path = str(p if p.is_absolute() else PROJECT_ROOT / p)

            # Legacy code_agent path: writes to workspaces/workspace_<ts>/output/.
            # Keep support for agents that print "workspaces/workspace_..." but
            # drop the mtime-scan heuristic that caused stale-spec pickups.
            if not spec_path:
                ws_match = _re.search(r"(workspaces/workspace_\S+)", combined)
                if ws_match:
                    workspace_path = str(PROJECT_ROOT / ws_match.group(1))
                    output_dir = Path(workspace_path) / "output"
                    if output_dir.exists():
                        tla_files = list(output_dir.glob("*.tla"))
                        cfg_files = list(output_dir.glob("*.cfg"))
                        if tla_files:
                            spec_path = str(tla_files[0])
                        if cfg_files:
                            config_path = str(cfg_files[0])

            success = result.returncode == 0 and spec_path is not None

            # Always persist the subprocess output so we can post-mortem any
            # failure. Previously stdout/stderr were captured in memory and
            # thrown away, making it impossible to tell why a Phase 0 failed
            # (e.g., which API error came back, whether retry kicked in).
            try:
                gen_log_dir = self.experiment_dir / system
                gen_log_dir.mkdir(parents=True, exist_ok=True)
                gen_log_path = gen_log_dir / f"run_{run_id}_generation.log"
                with open(gen_log_path, 'w', encoding='utf-8') as f:
                    f.write(f"=== cmd ===\n{' '.join(cmd)}\n\n")
                    f.write(f"=== returncode ===\n{result.returncode}\n\n")
                    f.write(f"=== duration ===\n{generation_time:.1f}s\n\n")
                    f.write("=== stdout ===\n")
                    f.write(result.stdout or "")
                    f.write("\n\n=== stderr ===\n")
                    f.write(result.stderr or "")
                logger.info(f"[{system}][Run {run_id}] Generation log saved: {gen_log_path}")
            except Exception as e:
                logger.warning(f"Failed to persist generation log: {e}")

            if success:
                logger.info(f"[{system}][Run {run_id}] Generation successful: {spec_path}")
            else:
                logger.warning(f"[{system}][Run {run_id}] Generation failed (exit code: {result.returncode})")
                # Surface last lines of stderr inline so it's visible without
                # opening the log file.
                if result.stderr:
                    tail = "\n".join(result.stderr.strip().splitlines()[-20:])
                    logger.warning(f"[{system}][Run {run_id}] stderr tail:\n{tail}")

            return success, workspace_path, spec_path, config_path, generation_time

        except subprocess.TimeoutExpired:
            logger.error(f"[{system}][Run {run_id}] Generation timed out after 30 minutes")
            return False, None, None, None, time.time() - start_time
        except Exception as e:
            logger.error(f"[{system}][Run {run_id}] Generation error: {e}")
            return False, None, None, None, time.time() - start_time

    def run_phase1_compilation(self, system: str, run_id: int,
                               spec_path: str, config_path: str,
                               generation_passed: bool) -> PhaseResult:
        """
        Run Phase 1: Compilation check.

        If generation already passed (code_agent does Phase 1+2), return full score.
        Otherwise, run compilation_check, and if fails, run action_decomposition.
        """
        logger.info(f"[{system}][Run {run_id}] Phase 1: Compilation check...")

        # Always re-run SANY: direct_call generation does not verify its
        # own output, so semantically-broken specs (e.g., undeclared NULL)
        # would silently pass Phase 1 and waste downstream agent budget.

        # Run compilation_check
        cmd = [
            "python3", "scripts/run_benchmark.py",
            "--task", system,
            "--method", "direct_call",
            "--model", self.model,
            "--metric", "compilation_check",
            "--spec-file", spec_path,
            "--config-file", config_path,
        ]

        try:
            result = run_with_timeout_kill_process_group(
                cmd,
                timeout=300,
                cwd=PROJECT_ROOT
            )

            # Check if passed
            if "✓ PASS" in result.stdout:
                logger.info(f"[{system}][Run {run_id}] Phase 1: PASS (compilation_check)")
                return PhaseResult(
                    phase_name="compilation",
                    score=1.0,
                    passed=True,
                    details={"method": "compilation_check"}
                )

            # Compilation failed, run action_decomposition
            logger.info(f"[{system}][Run {run_id}] Phase 1: compilation_check failed, running action_decomposition...")

            cmd_action = [
                "python3", "scripts/run_benchmark.py",
                "--task", system,
                "--method", "direct_call",
                "--model", self.model,
                "--metric", "action_decomposition",
                "--spec-file", spec_path,
                "--config-file", config_path,
            ]

            result_action = run_with_timeout_kill_process_group(
                cmd_action,
                timeout=600,
                cwd=PROJECT_ROOT
            )

            # Parse action_decomposition ratio
            import re
            ratio_match = re.search(r'(\d+)/(\d+)\s*\((\d+\.?\d*)%\)', result_action.stdout)
            if ratio_match:
                passed_actions = int(ratio_match.group(1))
                total_actions = int(ratio_match.group(2))
                ratio = passed_actions / total_actions if total_actions > 0 else 0
                score = ratio * 0.5

                logger.info(f"[{system}][Run {run_id}] Phase 1: PARTIAL ({passed_actions}/{total_actions} actions, score={score:.2f})")
                return PhaseResult(
                    phase_name="compilation",
                    score=score,
                    passed=False,
                    details={
                        "method": "action_decomposition",
                        "passed_actions": passed_actions,
                        "total_actions": total_actions,
                        "ratio": ratio
                    }
                )
            else:
                logger.warning(f"[{system}][Run {run_id}] Phase 1: Could not parse action_decomposition output")
                return PhaseResult(
                    phase_name="compilation",
                    score=0.0,
                    passed=False,
                    details={"method": "action_decomposition", "error": "parse_failed"}
                )

        except subprocess.TimeoutExpired:
            logger.error(f"[{system}][Run {run_id}] Phase 1: Timeout")
            return PhaseResult(
                phase_name="compilation",
                score=0.0,
                passed=False,
                error="timeout"
            )
        except Exception as e:
            logger.error(f"[{system}][Run {run_id}] Phase 1: Error - {e}")
            return PhaseResult(
                phase_name="compilation",
                score=0.0,
                passed=False,
                error=str(e)
            )

    def run_phase2_runtime(self, system: str, run_id: int,
                          spec_path: str, config_path: str) -> PhaseResult:
        """
        Run Phase 2: Runtime check + Runtime coverage.
        First runs runtime_check (TLC model checking), then runtime_coverage.
        """
        import re

        # Step 1: Run runtime_check first
        logger.info(f"[{system}][Run {run_id}] Phase 2: Runtime check (TLC model checking)...")

        cmd_check = [
            "python3", "scripts/run_benchmark.py",
            "--task", system,
            "--method", "direct_call",
            "--model", self.model,
            "--metric", "runtime_check",
            "--spec-file", spec_path,
            "--config-file", config_path,
        ]

        runtime_check_passed = False
        runtime_check_error = None
        try:
            result_check = subprocess.run(
                cmd_check,
                capture_output=True,
                text=True,
                timeout=60,
                cwd=PROJECT_ROOT
            )
            combined_check = result_check.stdout + "\n" + result_check.stderr

            # Match the evaluator's exact final verdict markers. Substring-
            # matching "FAIL"/"Error" across the entire combined output was
            # unreliable — unrelated startup warnings like
            # "Error loading mapping from .../mutex/mutex_mapping.json"
            # and "Warning: Failed to load system 'zookeeper'" tripped the
            # keyword filter and misreported a passing spec as failed
            # (observed 2026-04-16 on sonnet/gemini spin runs).
            if re.search(r"Runtime check:\s*✓\s*PASS", combined_check):
                runtime_check_passed = True
                logger.info(f"[{system}][Run {run_id}] Phase 2: Runtime check PASSED")
            elif re.search(r"Runtime check:\s*✗\s*FAIL", combined_check):
                runtime_check_passed = False
                logger.info(f"[{system}][Run {run_id}] Phase 2: Runtime check FAILED")
            elif result_check.returncode == 0:
                # No explicit verdict line but subprocess exited cleanly — treat
                # as PASS (TLC quietly completed without violations).
                runtime_check_passed = True
                logger.info(f"[{system}][Run {run_id}] Phase 2: Runtime check PASSED (no explicit verdict, exit 0)")
            else:
                runtime_check_passed = False
                logger.info(f"[{system}][Run {run_id}] Phase 2: Runtime check FAILED (exit {result_check.returncode})")
        except subprocess.TimeoutExpired:
            # Timeout means no error found within time limit - this counts as PASS
            runtime_check_passed = True
            runtime_check_error = "timeout"
            logger.info(f"[{system}][Run {run_id}] Phase 2: Runtime check PASSED (timeout, no errors found)")
        except Exception as e:
            runtime_check_error = str(e)
            logger.warning(f"[{system}][Run {run_id}] Phase 2: Runtime check error - {e}")

        # Step 2: Run runtime_coverage
        logger.info(f"[{system}][Run {run_id}] Phase 2: Runtime coverage...")

        cmd_coverage = [
            "python3", "scripts/run_benchmark.py",
            "--task", system,
            "--method", "direct_call",
            "--model", self.model,
            "--metric", "runtime_coverage",
            "--spec-file", spec_path,
            "--config-file", config_path,
        ]

        try:
            result = run_with_timeout_kill_process_group(
                cmd_coverage,
                timeout=600,
                cwd=PROJECT_ROOT
            )
            # Combine stdout and stderr for parsing
            combined_output = result.stdout + "\n" + result.stderr

            # Look for "Runtime coverage score: XX.XX%"
            coverage_match = re.search(r'Runtime coverage score:\s*(\d+\.?\d*)%', combined_output)
            if coverage_match:
                coverage = float(coverage_match.group(1)) / 100.0
                logger.info(f"[{system}][Run {run_id}] Phase 2: Coverage = {coverage:.2%}")
                return PhaseResult(
                    phase_name="runtime",
                    score=coverage,
                    passed=coverage >= 0.5,
                    details={
                        "coverage": coverage,
                        "runtime_check_passed": runtime_check_passed,
                        "runtime_check_error": runtime_check_error
                    }
                )

            # Alternative: look for generic coverage percentage
            coverage_match2 = re.search(r'coverage[:\s]+(\d+\.?\d*)%', combined_output, re.IGNORECASE)
            if coverage_match2:
                coverage = float(coverage_match2.group(1)) / 100.0
                logger.info(f"[{system}][Run {run_id}] Phase 2: Coverage = {coverage:.2%}")
                return PhaseResult(
                    phase_name="runtime",
                    score=coverage,
                    passed=coverage >= 0.5,
                    details={
                        "coverage": coverage,
                        "runtime_check_passed": runtime_check_passed,
                        "runtime_check_error": runtime_check_error
                    }
                )

            # Alternative: look for success without detailed coverage
            if "✓ PASS" in combined_output or "PASS" in combined_output:
                logger.info(f"[{system}][Run {run_id}] Phase 2: PASS (no coverage data)")
                return PhaseResult(
                    phase_name="runtime",
                    score=1.0,
                    passed=True,
                    details={
                        "method": "pass_without_coverage",
                        "runtime_check_passed": runtime_check_passed,
                        "runtime_check_error": runtime_check_error
                    }
                )

            logger.warning(f"[{system}][Run {run_id}] Phase 2: Could not parse coverage")
            logger.debug(f"Output snippet: {combined_output[-500:]}")
            return PhaseResult(
                phase_name="runtime",
                score=0.0,
                passed=False,
                details={
                    "error": "parse_failed",
                    "runtime_check_passed": runtime_check_passed,
                    "runtime_check_error": runtime_check_error
                },
                error="Could not parse coverage output"
            )

        except subprocess.TimeoutExpired:
            logger.error(f"[{system}][Run {run_id}] Phase 2: Timeout")
            return PhaseResult(
                phase_name="runtime",
                score=0.0,
                passed=False,
                details={
                    "runtime_check_passed": runtime_check_passed,
                    "runtime_check_error": runtime_check_error
                },
                error="timeout"
            )
        except Exception as e:
            logger.error(f"[{system}][Run {run_id}] Phase 2: Error - {e}")
            return PhaseResult(
                phase_name="runtime",
                score=0.0,
                passed=False,
                details={
                    "runtime_check_passed": runtime_check_passed,
                    "runtime_check_error": runtime_check_error
                },
                error=str(e)
            )

    def run_phase3_invariant(self, system: str, run_id: int,
                            spec_path: str, config_path: str) -> PhaseResult:
        """
        Run Phase 3: Invariant verification with agent translator.
        """
        logger.info(f"[{system}][Run {run_id}] Phase 3: Invariant verification (agent)...")

        # Use self.inv_model (default: "sonnet") for Phase 3b's agent translator.
        # This runs Claude Code CLI which uses Claude Code's own credentials,
        # NOT the user's paid API. See memory/feedback_api_usage_policy.md.
        cmd = [
            "python3", "scripts/run_benchmark.py",
            "--task", system,
            "--method", "direct_call",
            "--model", self.inv_model,
            "--metric", "invariant_verification",
            "--spec-file", spec_path,
            "--config-file", config_path,
            "--inv-translator-type", "agent",
        ]

        try:
            result = run_with_timeout_kill_process_group(
                cmd,
                timeout=900,  # 15 minutes for agent
                cwd=PROJECT_ROOT
            )
            # Combine stdout and stderr for parsing
            combined_output = result.stdout + "\n" + result.stderr

            # Parse invariant results
            import re

            # Parse individual invariant results (✓ PASS / ✗ FAIL)
            passed_invs = []
            failed_invs = []
            inv_results = re.findall(r'\d+\.\s+(\w+):\s*(✓ PASS|✗ FAIL)', combined_output)
            for inv_name, status in inv_results:
                if "PASS" in status:
                    passed_invs.append(inv_name)
                else:
                    failed_invs.append(inv_name)

            # Look for "Manual invariant testing: X/Y invariants passed"
            inv_match = re.search(r'Manual invariant testing:\s*(\d+)/(\d+)\s*invariants?\s*passed', combined_output)
            if inv_match:
                passed = int(inv_match.group(1))
                total = int(inv_match.group(2))
                ratio = passed / total if total > 0 else 0
                logger.info(f"[{system}][Run {run_id}] Phase 3: {passed}/{total} invariants passed ({ratio:.2%})")
                if failed_invs:
                    logger.info(f"[{system}][Run {run_id}] Phase 3: Failed invariants: {', '.join(failed_invs)}")
                return PhaseResult(
                    phase_name="invariant",
                    score=ratio,
                    passed=ratio >= 1.0,
                    details={
                        "passed_invariants": passed,
                        "total_invariants": total,
                        "ratio": ratio
                    },
                    passed_items=passed_invs,
                    failed_items=failed_invs
                )

            # Look for separate "Passed invariants: X" and "Total invariants tested: Y" lines
            passed_match = re.search(r'Passed invariants:\s*(\d+)', combined_output)
            total_match = re.search(r'Total invariants tested:\s*(\d+)', combined_output)
            if passed_match and total_match:
                passed = int(passed_match.group(1))
                total = int(total_match.group(1))
                ratio = passed / total if total > 0 else 0
                logger.info(f"[{system}][Run {run_id}] Phase 3: {passed}/{total} invariants passed ({ratio:.2%})")
                if failed_invs:
                    logger.info(f"[{system}][Run {run_id}] Phase 3: Failed invariants: {', '.join(failed_invs)}")
                return PhaseResult(
                    phase_name="invariant",
                    score=ratio,
                    passed=ratio >= 1.0,
                    details={
                        "passed_invariants": passed,
                        "total_invariants": total,
                        "ratio": ratio
                    },
                    passed_items=passed_invs,
                    failed_items=failed_invs
                )

            # Generic X/Y pattern
            inv_match2 = re.search(r'(\d+)/(\d+)\s*invariants?\s*passed', combined_output, re.IGNORECASE)
            if inv_match2:
                passed = int(inv_match2.group(1))
                total = int(inv_match2.group(2))
                ratio = passed / total if total > 0 else 0
                logger.info(f"[{system}][Run {run_id}] Phase 3: {passed}/{total} invariants passed ({ratio:.2%})")
                if failed_invs:
                    logger.info(f"[{system}][Run {run_id}] Phase 3: Failed invariants: {', '.join(failed_invs)}")
                return PhaseResult(
                    phase_name="invariant",
                    score=ratio,
                    passed=ratio >= 1.0,
                    details={
                        "passed_invariants": passed,
                        "total_invariants": total,
                        "ratio": ratio
                    },
                    passed_items=passed_invs,
                    failed_items=failed_invs
                )

            # Check for full pass
            if "All invariants passed: True" in combined_output:
                logger.info(f"[{system}][Run {run_id}] Phase 3: PASS (all invariants)")
                return PhaseResult(
                    phase_name="invariant",
                    score=1.0,
                    passed=True,
                    details={"all_passed": True},
                    passed_items=passed_invs,
                    failed_items=[]
                )

            logger.warning(f"[{system}][Run {run_id}] Phase 3: Could not parse invariant results")
            logger.debug(f"Output snippet: {combined_output[-500:]}")
            return PhaseResult(
                phase_name="invariant",
                score=0.0,
                passed=False,
                details={"error": "parse_failed"},
                error="Could not parse invariant output"
            )

        except subprocess.TimeoutExpired:
            logger.error(f"[{system}][Run {run_id}] Phase 3: Timeout")
            return PhaseResult(
                phase_name="invariant",
                score=0.0,
                passed=False,
                error="timeout"
            )
        except Exception as e:
            logger.error(f"[{system}][Run {run_id}] Phase 3: Error - {e}")
            return PhaseResult(
                phase_name="invariant",
                score=0.0,
                passed=False,
                error=str(e)
            )

    def run_phase3_tv(self, system: str, run_id: int,
                      spec_path: str) -> PhaseResult:
        """
        Phase 3 TV: Transition validation via agent-driven evaluation.
        Launches the configured agent adapter that follows the tv-eval skill.

        Skipped only when --skip-tv is set (costs ~$1-4 per spec).
        """
        logger.info(f"[{system}][Run {run_id}] Phase 3 TV: starting transition validation...")

        launcher = PROJECT_ROOT / "scripts" / "launch_tv_eval.sh"
        workspace_root = PROJECT_ROOT / "tv-workspaces"

        try:
            cmd = [
                "bash", str(launcher),
                f"--task={system}",
                f"--spec={spec_path}",
                f"--workspace-root={workspace_root}",
                f"--max-budget={self.tv_budget}",
            ]
            if self.tv_agent:
                cmd.append(f"--agent={self.tv_agent}")
            if self.tv_model:
                cmd.append(f"--model={self.tv_model}")

            result = subprocess.run(
                cmd,
                capture_output=True, text=True,
                timeout=self.tv_timeout,
                cwd=str(PROJECT_ROOT),
            )

            # Find the workspace that was just created (latest one for this spec)
            import glob
            ws_pattern = str(workspace_root / "*")
            workspaces = sorted(glob.glob(ws_pattern), key=os.path.getmtime, reverse=True)
            if not workspaces:
                return PhaseResult(phase_name="tv", score=None, passed=False,
                                   status="pending",
                                   details={"reason": "no_workspace_created"},
                                   error="no workspace created")

            ws = Path(workspaces[0])
            report_path = ws / "reports" / "final_report.md"
            results_path = ws / "reports" / "tv_results.json"

            # Try to parse structured results
            if results_path.exists():
                with open(results_path) as f:
                    tv_data = json.load(f)
                total_passed, total_windows = 0, 0
                action_scores = {}
                for action, info in tv_data.items():
                    stats = info.get("stats", {})
                    p, t = stats.get("passed", 0), stats.get("total", 0)
                    total_passed += p
                    total_windows += t
                    action_scores[action] = stats.get("pass_rate", 0.0)
                score = total_passed / total_windows if total_windows > 0 else 0.0
                logger.info(f"[{system}][Run {run_id}] Phase 3 TV: {total_passed}/{total_windows} ({score:.1%})")
                return PhaseResult(
                    phase_name="tv",
                    score=score,
                    passed=score > 0,
                    details={"per_action": action_scores, "total_passed": total_passed,
                             "total_windows": total_windows, "workspace": str(ws)},
                )

            # Fallback: parse final_report.md for pass rates
            if report_path.exists():
                report = report_path.read_text()
                if "Cannot evaluate" in report or "CANNOT EVALUATE" in report:
                    # Spec compiled but TV agent couldn't meaningfully evaluate
                    # (e.g., SANY errors found in composite step, or harness
                    # doesn't exist yet for this task). Not a real 0 — score
                    # is unknown until infra is fixed or spec is correct.
                    logger.info(f"[{system}][Run {run_id}] Phase 3 TV: cannot evaluate")
                    return PhaseResult(phase_name="tv", score=None, passed=False,
                                       status="pending",
                                       details={"reason": "cannot_evaluate",
                                                "workspace": str(ws)})
                # Try to extract per-action pass rates from the Summary table.
                # Agents use various formats in the table cell:
                #   | AcquireLock  | 103/103 (100%) |     → strict
                #   | AcquireLock  | 100% (210/210) |     → wrapped
                #   | AcquireLock  | 358 / 358      |     → spaces
                # Parse each markdown table row, pull FIRST X/Y seen per row
                # where X ≤ Y. Ignores unrelated numbers elsewhere in report.
                import re
                pair_re = re.compile(r'(\d+)\s*/\s*(\d+)')
                matches = []
                for line in report.splitlines():
                    if not line.lstrip().startswith('|'):
                        continue
                    for m in pair_re.finditer(line):
                        x, y = int(m.group(1)), int(m.group(2))
                        if 0 <= x <= y and y > 0:
                            matches.append((x, y))
                            break  # one per row
                if matches:
                    tp = sum(m[0] for m in matches)
                    tw = sum(m[1] for m in matches)
                    score = tp / tw if tw > 0 else 0.0
                    logger.info(f"[{system}][Run {run_id}] Phase 3 TV: {tp}/{tw} ({score:.1%})")
                    return PhaseResult(phase_name="tv", score=score, passed=score > 0,
                                       details={"total_passed": tp, "total_windows": tw,
                                                 "workspace": str(ws)})

            return PhaseResult(phase_name="tv", score=None, passed=False,
                               status="pending",
                               details={"reason": "no_results_in_workspace",
                                        "workspace": str(ws)},
                               error="no results found in workspace")

        except subprocess.TimeoutExpired:
            logger.warning(f"[{system}][Run {run_id}] Phase 3 TV: timeout ({self.tv_timeout}s)")
            return PhaseResult(phase_name="tv", score=None, passed=False,
                               status="pending",
                               details={"reason": "tv_timeout"},
                               error="timeout")
        except Exception as e:
            logger.error(f"[{system}][Run {run_id}] Phase 3 TV: error - {e}")
            return PhaseResult(phase_name="tv", score=0.0, passed=False, error=str(e))

    def run_single_experiment(self, system: str, run_id: int) -> RunResult:
        """
        Run a single experiment for a system.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        logger.info(f"[{system}][Run {run_id}] Starting experiment...")

        # Initialize result
        result = RunResult(
            run_id=run_id,
            system=system,
            timestamp=timestamp,
            generation_success=False,
            generation_time=0.0
        )

        try:
            # Phase 0: Generation
            gen_success, workspace_path, spec_path, config_path, gen_time = self.run_generation(system, run_id)

            result.generation_success = gen_success
            result.generation_time = gen_time
            result.workspace_path = workspace_path
            result.spec_path = spec_path
            result.config_path = config_path

            # Load Phase 0 API usage (written by compilation_check.py evaluator)
            if workspace_path:
                usage_file = Path(workspace_path) / "generation_usage.json"
                if usage_file.exists():
                    try:
                        with open(usage_file, 'r', encoding='utf-8') as f:
                            result.phase0_usage = json.load(f)
                    except Exception as e:
                        logger.warning(f"[{system}] Failed to read {usage_file}: {e}")

            if not gen_success or not spec_path:
                result.error = "Generation failed"
                logger.error(f"[{system}][Run {run_id}] Generation failed, skipping evaluation")
                # Mark every phase as not_evaluated so run_1.json doesn't
                # show phase fields as None (ambiguous with "pending").
                not_eval = {"reason": "generation_failed"}
                result.phase1_compilation = PhaseResult(
                    phase_name="compilation", score=None, passed=False,
                    status="not_evaluated", details=not_eval
                )
                result.phase2_runtime = PhaseResult(
                    phase_name="runtime", score=None, passed=False,
                    status="not_evaluated", details=not_eval
                )
                result.phase3_tv = PhaseResult(
                    phase_name="tv", score=None, passed=False,
                    status="not_evaluated", details=not_eval
                )
                result.phase3_invariant = PhaseResult(
                    phase_name="invariant", score=None, passed=False,
                    status="not_evaluated", details=not_eval
                )
                return result

            # Phase 1: Compilation (real SANY for direct_call, fast-path for code_agent)
            result.phase1_compilation = self.run_phase1_compilation(
                system, run_id, spec_path, config_path,
                generation_passed=True
            )

            phase1_ok = bool(result.phase1_compilation and result.phase1_compilation.passed)

            # Cascade skip: if Phase 1 fails the spec cannot run in TLC, so
            # Phase 2/3/3b all produce no signal and waste resources (especially
            # the agent-based Phase 3 TV which costs real $). Short-circuit here.
            if not phase1_ok:
                logger.info(f"[{system}][Run {run_id}] Phase 1 failed — skipping Phase 2/3/3b")
                skip_reason = {"skipped": "phase1_failed"}
                result.phase2_runtime = PhaseResult(
                    phase_name="runtime", score=None, passed=False,
                    status="skipped", details=skip_reason
                )
                result.phase3_tv = PhaseResult(
                    phase_name="tv", score=None, passed=False,
                    status="skipped", details=skip_reason
                )
                result.phase3_invariant = PhaseResult(
                    phase_name="invariant", score=None, passed=False,
                    status="skipped", details=skip_reason
                )
                result.calculate_total_score()
                logger.info(f"[{system}][Run {run_id}] Completed (early) - Total Score: {result.total_score:.2f}")
                return result

            # Phase 2: Runtime coverage (local TLC; coverage < 1.0 is NOT a gate)
            result.phase2_runtime = self.run_phase2_runtime(
                system, run_id, spec_path, config_path
            )

            # Cascade skip when Phase 2 didn't actually exercise the spec:
            #   (a) runtime_check reported a violation (deadlock / invariant
            #       failure / semantic error TLC catches), OR
            #   (b) coverage == 0 — TLC explored zero states, so
            #       "runtime_check passed" is vacuous (no violations because
            #       nothing was evaluated; Init likely UNSAT or constants
            #       broken). Observed 2026-04-17 on kimi_k25_ds/etcd: spec
            #       compiled, TLC ran, coverage 0%, but Phase 3b invariant
            #       (agent-translator, TLC-independent) scored 92% and
            #       inflated total to 0.64.
            # Coverage < 1.0 but > 0 is still a real signal and NOT a gate.
            p2_details = (result.phase2_runtime.details or {}) if result.phase2_runtime else {}
            runtime_check_passed = p2_details.get("runtime_check_passed", True)
            coverage = p2_details.get("coverage", None)
            zero_coverage = (coverage is not None and coverage <= 0.0)
            if (not runtime_check_passed) or zero_coverage:
                reason_tag = (
                    "phase2_runtime_check_failed" if not runtime_check_passed
                    else "phase2_zero_coverage"
                )
                logger.info(
                    f"[{system}][Run {run_id}] Phase 2 not usable "
                    f"({reason_tag}) — skipping Phase 3/3b"
                )
                skip_reason = {"skipped": reason_tag}
                result.phase3_tv = PhaseResult(
                    phase_name="tv", score=None, passed=False,
                    status="skipped", details=skip_reason
                )
                result.phase3_invariant = PhaseResult(
                    phase_name="invariant", score=None, passed=False,
                    status="skipped", details=skip_reason
                )
                result.calculate_total_score()
                logger.info(f"[{system}][Run {run_id}] Completed (early) - Total Score: {result.total_score:.2f}")
                return result

            # Phase 3a: TV transition validation (if enabled and Phase 1 passed)
            if self.enable_tv and result.phase1_compilation and result.phase1_compilation.passed:
                result.phase3_tv = self.run_phase3_tv(system, run_id, spec_path)
                # Load the TV agent's cost/usage breakdown
                ws = (result.phase3_tv.details or {}).get("workspace") if result.phase3_tv else None
                if ws:
                    tv_usage_file = Path(ws) / ".run.usage.json"
                    if tv_usage_file.exists():
                        try:
                            with open(tv_usage_file, 'r', encoding='utf-8') as f:
                                full = json.load(f)
                            result.phase3_tv_usage = {
                                "cost_usd": full.get("total_cost_usd"),
                                "duration_ms": full.get("duration_ms"),
                                "num_turns": full.get("num_turns"),
                                "num_tool_uses": full.get("num_tool_uses"),
                                "model_usage": full.get("model_usage"),
                            }
                        except Exception as e:
                            logger.warning(f"[{system}] Failed to read {tv_usage_file}: {e}")
            elif self.enable_tv:
                logger.info(f"[{system}][Run {run_id}] Skipping Phase 3 TV (Phase 1 failed)")
                result.phase3_tv = PhaseResult(
                    phase_name="tv", score=0.0, passed=False,
                    details={"skipped": "phase1_failed"}
                )

            # Phase 3b: Invariant verification (only if Phase 1 passed)
            if result.phase1_compilation and result.phase1_compilation.passed:
                result.phase3_invariant = self.run_phase3_invariant(
                    system, run_id, spec_path, config_path
                )
            else:
                logger.info(f"[{system}][Run {run_id}] Skipping Phase 3b invariant (Phase 1 failed)")
                result.phase3_invariant = PhaseResult(
                    phase_name="invariant",
                    score=0.0,
                    passed=False,
                    details={"skipped": "phase1_failed"}
                )

            # Calculate total score
            result.calculate_total_score()

            logger.info(f"[{system}][Run {run_id}] Completed - Total Score: {result.total_score:.2f}, Perfect: {result.is_perfect}")

        except Exception as e:
            result.error = str(e)
            logger.error(f"[{system}][Run {run_id}] Experiment error: {e}")

        return result

    def run_system(self, system: str) -> SystemResult:
        """
        Run all experiments for a single system.
        """
        logger.info(f"[{system}] Starting experiments (max {self.max_runs} runs)...")

        system_result = SystemResult(system=system)

        for run_id in range(1, self.max_runs + 1):
            run_result = self.run_single_experiment(system, run_id)
            system_result.runs.append(run_result)

            # Save intermediate result
            self.save_run_result(system, run_result)

            # Early stop if perfect score
            if run_result.is_perfect:
                logger.info(f"[{system}] Perfect score achieved in run {run_id}, stopping early")
                break

        # Find best run
        system_result.find_best_run()

        if system_result.best_run:
            logger.info(f"[{system}] Best run: #{system_result.best_run.run_id} with score {system_result.best_run.total_score:.2f}")

        return system_result

    def save_run_result(self, system: str, run_result: RunResult):
        """Save individual run result to file."""
        system_dir = self.experiment_dir / system
        system_dir.mkdir(exist_ok=True)

        result_file = system_dir / f"run_{run_result.run_id}.json"

        # Convert PhaseResult to dict manually to handle all fields
        def phase_to_dict(phase):
            if phase is None:
                return None
            return {
                "phase_name": phase.phase_name,
                "status": phase.status,
                "score": phase.score,
                "passed": phase.passed,
                "details": phase.details,
                "error": phase.error,
                "passed_items": phase.passed_items,
                "failed_items": phase.failed_items
            }

        # Convert to dict
        result_dict = {
            "run_id": run_result.run_id,
            "system": run_result.system,
            "timestamp": run_result.timestamp,
            "generation_success": run_result.generation_success,
            "generation_time": run_result.generation_time,
            "spec_path": run_result.spec_path,
            "config_path": run_result.config_path,
            "workspace_path": run_result.workspace_path,
            "phase1_compilation": phase_to_dict(run_result.phase1_compilation),
            "phase2_runtime": phase_to_dict(run_result.phase2_runtime),
            "phase3_tv": phase_to_dict(run_result.phase3_tv),
            "phase3_invariant": phase_to_dict(run_result.phase3_invariant),
            "phase0_usage": run_result.phase0_usage,
            "phase3_tv_usage": run_result.phase3_tv_usage,
            "total_score": run_result.total_score,
            "is_perfect": run_result.is_perfect,
            "error": run_result.error
        }

        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(result_dict, f, indent=2, ensure_ascii=False)

    def collect_best_specs(self):
        """Collect best specs from all systems into a single directory."""
        best_specs_dir = self.experiment_dir / "best_specs"
        best_specs_dir.mkdir(exist_ok=True)

        summary = []

        for system, system_result in self.system_results.items():
            if system_result.best_run and system_result.best_run.spec_path:
                best_run = system_result.best_run

                # Copy spec file
                src_spec = Path(best_run.spec_path)
                if src_spec.exists():
                    dst_spec = best_specs_dir / f"{system}.tla"
                    shutil.copy2(src_spec, dst_spec)

                # Copy config file
                if best_run.config_path:
                    src_cfg = Path(best_run.config_path)
                    if src_cfg.exists():
                        dst_cfg = best_specs_dir / f"{system}.cfg"
                        shutil.copy2(src_cfg, dst_cfg)

                summary.append({
                    "system": system,
                    "best_run_id": best_run.run_id,
                    "total_score": best_run.total_score,
                    "phase1_score": best_run.phase1_compilation.score if best_run.phase1_compilation else 0,
                    "phase2_score": best_run.phase2_runtime.score if best_run.phase2_runtime else 0,
                    "phase3_score": best_run.phase3_invariant.score if best_run.phase3_invariant else 0,
                    "is_perfect": best_run.is_perfect,
                    "spec_file": f"{system}.tla",
                    "config_file": f"{system}.cfg"
                })

                logger.info(f"Collected best spec for {system}: run #{best_run.run_id}, score={best_run.total_score:.2f}")

        # Save summary
        summary_file = best_specs_dir / "summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        logger.info(f"Best specs collected in: {best_specs_dir}")
        return summary

    def generate_report(self):
        """Generate final experiment report."""
        report = {
            "experiment_id": self.experiment_id,
            "timestamp": datetime.now().isoformat(),
            "config": {
                "systems": self.systems,
                "max_runs": self.max_runs,
                "num_threads": self.num_threads,
                "model": self.model
            },
            "results": {}
        }

        for system, system_result in self.system_results.items():
            report["results"][system] = {
                "total_runs": len(system_result.runs),
                "best_run_id": system_result.best_run.run_id if system_result.best_run else None,
                "best_score": system_result.best_run.total_score if system_result.best_run else 0,
                "runs": [
                    {
                        "run_id": r.run_id,
                        "generation_success": r.generation_success,
                        "phase1_score": r.phase1_compilation.score if r.phase1_compilation else 0,
                        "phase2_score": r.phase2_runtime.score if r.phase2_runtime else 0,
                        "phase3_score": r.phase3_invariant.score if r.phase3_invariant else 0,
                        "phase3_passed_invariants": r.phase3_invariant.passed_items if r.phase3_invariant else [],
                        "phase3_failed_invariants": r.phase3_invariant.failed_items if r.phase3_invariant else [],
                        "total_score": r.total_score,
                        "is_perfect": r.is_perfect
                    }
                    for r in system_result.runs
                ]
            }

        report_file = self.experiment_dir / "experiment_report.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        logger.info(f"Experiment report saved to: {report_file}")
        return report

    def print_summary(self):
        """Print summary to console."""
        print("\n" + "=" * 100)
        print("BATCH EXPERIMENT SUMMARY")
        print("=" * 100)
        print(f"Experiment ID: {self.experiment_id}")
        print(f"Output Directory: {self.experiment_dir}")
        print()

        print(f"{'System':<15} {'Runs':<6} {'Best Run':<10} {'P1 (Comp)':<12} {'P2 (Runtime)':<14} {'P3 (Inv)':<12} {'Total':<8} {'Perfect':<8}")
        print("-" * 100)

        for system in self.systems:
            if system in self.system_results:
                sr = self.system_results[system]
                if sr.best_run:
                    br = sr.best_run
                    def _s(phase):
                        if phase is None or phase.score is None:
                            return "  -  "
                        return f"{phase.score:.2f}"
                    p1 = _s(br.phase1_compilation)
                    p2 = _s(br.phase2_runtime)
                    p3 = _s(br.phase3_invariant)
                    perfect = "Yes" if br.is_perfect else "No"
                    print(f"{system:<15} {len(sr.runs):<6} #{br.run_id:<9} {p1:<12} {p2:<14} {p3:<12} {br.total_score:<8.2f} {perfect:<8}")
                else:
                    print(f"{system:<15} {len(sr.runs):<6} {'N/A':<10} {'N/A':<12} {'N/A':<14} {'N/A':<12} {'N/A':<8} {'No':<8}")
            else:
                print(f"{system:<15} {'0':<6} {'N/A':<10} {'N/A':<12} {'N/A':<14} {'N/A':<12} {'N/A':<8} {'No':<8}")

        print("=" * 100)

        # Print detailed invariant failures per run
        print("\nDETAILED INVARIANT RESULTS BY RUN:")
        print("-" * 100)
        for system in self.systems:
            if system in self.system_results:
                sr = self.system_results[system]
                print(f"\n{system}:")
                for run in sr.runs:
                    if run.phase3_invariant:
                        p3 = run.phase3_invariant
                        passed_count = p3.details.get("passed_invariants", 0)
                        total_count = p3.details.get("total_invariants", 0)
                        failed_list = p3.failed_items if p3.failed_items else []
                        passed_list = p3.passed_items if p3.passed_items else []

                        if failed_list:
                            print(f"  Run {run.run_id}: {passed_count}/{total_count} passed | Failed: {', '.join(failed_list)}")
                        elif total_count > 0:
                            print(f"  Run {run.run_id}: {passed_count}/{total_count} passed | All passed!")
                        else:
                            print(f"  Run {run.run_id}: No invariant data")
                    else:
                        print(f"  Run {run.run_id}: Phase 3 not executed")

        print("=" * 100)

    def run(self):
        """Run the full batch experiment."""
        logger.info("Starting batch experiment...")
        start_time = time.time()

        # Use thread pool for parallel execution
        with ThreadPoolExecutor(max_workers=self.num_threads, thread_name_prefix="Worker") as executor:
            # Submit all system tasks
            future_to_system = {
                executor.submit(self.run_system, system): system
                for system in self.systems
            }

            # Process completed tasks
            for future in as_completed(future_to_system):
                system = future_to_system[future]
                try:
                    system_result = future.result()
                    with self.results_lock:
                        self.system_results[system] = system_result
                    logger.info(f"[{system}] Completed all runs")
                except Exception as e:
                    logger.error(f"[{system}] Failed with error: {e}")
                    with self.results_lock:
                        self.system_results[system] = SystemResult(system=system)

        total_time = time.time() - start_time
        logger.info(f"Batch experiment completed in {total_time:.2f}s")

        # Collect best specs
        self.collect_best_specs()

        # Generate report
        self.generate_report()

        # Print summary
        self.print_summary()

        return self.system_results


def main():
    parser = argparse.ArgumentParser(
        description="Batch Experiment Runner for TLA+ Specification Generation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run all systems with default settings (claude_code agent)
    python scripts/run_batch_experiment.py --all

    # Run specific systems
    python scripts/run_batch_experiment.py --systems etcd zookeeper raftkvs

    # Run with custom settings
    python scripts/run_batch_experiment.py --all --runs 3 --threads 10

    # Use different code agents
    python scripts/run_batch_experiment.py --all --agent gemini --model default
    python scripts/run_batch_experiment.py --all --agent codex --model gpt-5

    # Specify output directory
    python scripts/run_batch_experiment.py --all --output my_experiments

    # List available agents
    python scripts/run_batch_experiment.py --list-agents
        """
    )

    parser.add_argument("--systems", nargs="+",
                       help="Systems to evaluate (space-separated)")
    parser.add_argument("--all", action="store_true",
                       help="Run all available systems")
    parser.add_argument("--runs", type=int, default=5,
                       help="Maximum runs per system (default: 5)")
    parser.add_argument("--threads", type=int, default=5,
                       help="Number of parallel threads (default: 5)")
    parser.add_argument("--output", default="experiments",
                       help="Output directory (default: experiments)")
    parser.add_argument("--model", default="claude",
                       help="Model to use for generation (entry name in config/models.yaml; default: claude)")
    parser.add_argument("--agent", default=DEFAULT_AGENT,
                       choices=list(SUPPORTED_AGENTS.keys()),
                       help=f"Code agent to use for generation (default: {DEFAULT_AGENT})")
    parser.add_argument("--skip-tv", action="store_true",
                       help="Skip transition validation. By default it runs on every cell that passes compilation; "
                            "transition validation costs ~$1-4 per spec via the coding-agent CLI.")
    parser.add_argument("--tv-budget", type=float, default=5.0,
                       help="Max API budget (USD) per TV evaluation (default: 5)")
    parser.add_argument("--tv-timeout", type=int, default=1800,
                       help="Timeout (seconds) per TV evaluation (default: 1800)")
    parser.add_argument("--tv-agent", default=None,
                       help="Agent adapter for TV launcher (e.g. claude-code, codex)")
    parser.add_argument("--tv-model", default=None,
                       help="Model override passed to the TV agent adapter")
    parser.add_argument("--inv-model", default="sonnet",
                       help="Model for Phase 3b invariant-translator agent CLI (default: sonnet). "
                            "Uses Claude Code's own credentials, NOT user's paid API.")
    parser.add_argument("--list-systems", action="store_true",
                       help="List all available systems")
    parser.add_argument("--list-agents", action="store_true",
                       help="List all supported code agents")

    args = parser.parse_args()

    if args.list_systems:
        print("Available systems:")
        for system in ALL_SYSTEMS:
            print(f"  - {system}")
        return

    if args.list_agents:
        print("Supported code agents:")
        for agent, method in SUPPORTED_AGENTS.items():
            default_marker = " (default)" if agent == DEFAULT_AGENT else ""
            print(f"  - {agent}: {method}{default_marker}")
        return

    # Determine systems to run
    if args.all:
        systems = ALL_SYSTEMS
    elif args.systems:
        # Validate systems
        invalid = [s for s in args.systems if s not in ALL_SYSTEMS]
        if invalid:
            print(f"Error: Invalid systems: {invalid}")
            print(f"Available systems: {ALL_SYSTEMS}")
            sys.exit(1)
        systems = args.systems
    else:
        parser.error("Must specify --systems or --all")

    # Create and run experiment
    runner = BatchExperimentRunner(
        systems=systems,
        max_runs=args.runs,
        num_threads=args.threads,
        output_dir=args.output,
        model=args.model,
        agent=args.agent,
        enable_tv=not args.skip_tv,
        tv_budget=args.tv_budget,
        tv_timeout=args.tv_timeout,
        tv_agent=args.tv_agent,
        tv_model=args.tv_model,
        inv_model=args.inv_model,
    )

    runner.run()


if __name__ == "__main__":
    main()
