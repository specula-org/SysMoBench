"""
Evaluator module that wraps SysMoBench's TLAValidator and TLCRunner.

This module provides a unified interface for validating TLA+ specifications
through two phases:
- Phase 1: Syntax validation using SANY parser
- Phase 2: Runtime model checking using TLC
"""

import json
import os
import sys
import tempfile
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tla_eval.core.verification.validators import TLAValidator
from tla_eval.evaluation.semantics.runtime_check import TLCRunner


@dataclass
class PhaseResult:
    """Result of a single validation phase."""
    success: bool
    output_file: str
    error_summary: Optional[str] = None


@dataclass
class EvaluationResult:
    """Complete evaluation result for an attempt."""
    success: bool
    attempt: int
    remaining_attempts: int
    phase1_syntax: Optional[PhaseResult] = None
    phase2_runtime: Optional[PhaseResult] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = {
            "success": self.success,
            "attempt": self.attempt,
            "remaining_attempts": self.remaining_attempts,
        }
        if self.phase1_syntax:
            result["phase1_syntax"] = asdict(self.phase1_syntax)
        else:
            result["phase1_syntax"] = None
        if self.phase2_runtime:
            result["phase2_runtime"] = asdict(self.phase2_runtime)
        else:
            result["phase2_runtime"] = None
        return result


class SpecEvaluator:
    """
    Evaluator that wraps SysMoBench's validation and model checking tools.

    This class manages the evaluation of TLA+ specifications through two phases:
    1. Syntax validation using TLAValidator (SANY parser)
    2. Runtime model checking using TLCRunner

    All artifacts are saved to a structured output directory for analysis.
    """

    def __init__(
        self,
        task_name: str,
        spec_module: str,
        output_dir: str,
        run_timestamp: str,
        syntax_timeout: int = 30,
        runtime_timeout: int = 300,
    ):
        """
        Initialize the evaluator.

        Args:
            task_name: Name of the evaluation task
            spec_module: TLA+ module name expected in the specification
            output_dir: Base output directory for saving artifacts
            run_timestamp: Timestamp string for this run session
            syntax_timeout: Timeout in seconds for syntax validation (default: 30)
            runtime_timeout: Timeout in seconds for TLC model checking (default: 300)
        """
        self.task_name = task_name
        self.spec_module = spec_module
        self.output_dir = Path(output_dir)
        self.run_timestamp = run_timestamp
        self.syntax_timeout = syntax_timeout
        self.runtime_timeout = runtime_timeout

        # Initialize validators
        self.syntax_validator = TLAValidator(timeout=syntax_timeout)
        self.tlc_runner = TLCRunner(timeout=runtime_timeout)

        # Create base output directory for this run
        self.run_dir = self.output_dir / "submissions" / task_name / run_timestamp
        self.run_dir.mkdir(parents=True, exist_ok=True)

    def evaluate(
        self,
        spec_content: str,
        config_content: str,
        attempt: int,
        max_attempts: int,
    ) -> EvaluationResult:
        """
        Evaluate a TLA+ specification through both phases.

        Args:
            spec_content: The TLA+ specification content
            config_content: The TLC configuration file content
            attempt: Current attempt number (1-indexed)
            max_attempts: Maximum allowed attempts

        Returns:
            EvaluationResult containing success status and output file paths
        """
        remaining = max_attempts - attempt
        attempt_dir = self.run_dir / f"attempt_{attempt}"
        attempt_dir.mkdir(parents=True, exist_ok=True)

        # Save submitted spec and config
        # Note: TLC requires the file name to match the module name
        spec_file = attempt_dir / f"{self.spec_module}.tla"
        config_file = attempt_dir / f"{self.spec_module}.cfg"
        spec_file.write_text(spec_content)
        config_file.write_text(config_content)

        # Phase 1: Syntax validation
        phase1_result = self._run_phase1(spec_content, attempt_dir)

        if not phase1_result.success:
            result = EvaluationResult(
                success=False,
                attempt=attempt,
                remaining_attempts=remaining,
                phase1_syntax=phase1_result,
                phase2_runtime=None,
            )
            self._save_result(attempt_dir, result)
            return result

        # Phase 2: Runtime model checking
        phase2_result = self._run_phase2(str(spec_file), str(config_file), attempt_dir)

        result = EvaluationResult(
            success=phase2_result.success,
            attempt=attempt,
            remaining_attempts=remaining,
            phase1_syntax=phase1_result,
            phase2_runtime=phase2_result,
        )
        self._save_result(attempt_dir, result)
        return result

    def _run_phase1(self, spec_content: str, attempt_dir: Path) -> PhaseResult:
        """
        Run Phase 1: Syntax validation using SANY.

        Args:
            spec_content: The TLA+ specification content
            attempt_dir: Directory to save output files

        Returns:
            PhaseResult with validation outcome
        """
        phase1_log = attempt_dir / "phase1.log"

        try:
            validation_result = self.syntax_validator.validate_specification(
                tla_content=spec_content,
                module_name=self.spec_module,
                task_name=self.task_name,
                context="compilation",
            )

            # Build output log content
            log_content = f"=== Phase 1: Syntax Validation (SANY) ===\n"
            log_content += f"Module: {self.spec_module}\n"
            log_content += f"Timestamp: {datetime.now().isoformat()}\n"
            log_content += f"Success: {validation_result.success}\n"
            log_content += f"Compilation Time: {validation_result.compilation_time:.2f}s\n"
            log_content += f"\n=== Output ===\n{validation_result.output}\n"

            if validation_result.syntax_errors:
                log_content += f"\n=== Syntax Errors ===\n"
                for err in validation_result.syntax_errors:
                    log_content += f"- {err}\n"

            if validation_result.semantic_errors:
                log_content += f"\n=== Semantic Errors ===\n"
                for err in validation_result.semantic_errors:
                    log_content += f"- {err}\n"

            phase1_log.write_text(log_content)

            error_summary = None
            if not validation_result.success:
                # Extract concise error summary from output
                error_summary = self._extract_error_summary(validation_result.output)

            return PhaseResult(
                success=validation_result.success,
                output_file=str(phase1_log),
                error_summary=error_summary,
            )

        except Exception as e:
            error_msg = f"Phase 1 validation failed with exception: {str(e)}"
            log_content = f"=== Phase 1: Syntax Validation (SANY) ===\n"
            log_content += f"ERROR: {error_msg}\n"
            phase1_log.write_text(log_content)

            return PhaseResult(
                success=False,
                output_file=str(phase1_log),
                error_summary=error_msg,
            )

    def _run_phase2(
        self,
        spec_file: str,
        config_file: str,
        attempt_dir: Path,
    ) -> PhaseResult:
        """
        Run Phase 2: Runtime model checking using TLC.

        Args:
            spec_file: Path to the TLA+ specification file
            config_file: Path to the TLC configuration file
            attempt_dir: Directory to save output files

        Returns:
            PhaseResult with model checking outcome
        """
        phase2_log = attempt_dir / "phase2.log"

        try:
            success, output, exit_code = self.tlc_runner.run_model_checking(
                spec_file=spec_file,
                config_file=config_file,
                record_stats=False,  # Don't record to global stats
                use_deadlock_flag=True,
            )

            # Build output log content
            log_content = f"=== Phase 2: Runtime Model Checking (TLC) ===\n"
            log_content += f"Spec File: {spec_file}\n"
            log_content += f"Config File: {config_file}\n"
            log_content += f"Timestamp: {datetime.now().isoformat()}\n"
            log_content += f"Success: {success}\n"
            log_content += f"Exit Code: {exit_code}\n"
            log_content += f"\n=== TLC Output ===\n{output}\n"

            phase2_log.write_text(log_content)

            error_summary = None
            if not success:
                # Try to extract a meaningful error summary from output
                if "Invariant" in output and "violated" in output:
                    error_summary = "Invariant violation detected"
                elif "Deadlock" in output:
                    error_summary = "Deadlock detected"
                elif exit_code != 0:
                    error_summary = f"TLC exited with code {exit_code}"

            return PhaseResult(
                success=success,
                output_file=str(phase2_log),
                error_summary=error_summary,
            )

        except Exception as e:
            error_msg = f"Phase 2 model checking failed with exception: {str(e)}"
            log_content = f"=== Phase 2: Runtime Model Checking (TLC) ===\n"
            log_content += f"ERROR: {error_msg}\n"
            phase2_log.write_text(log_content)

            return PhaseResult(
                success=False,
                output_file=str(phase2_log),
                error_summary=error_msg,
            )

    def _save_result(self, attempt_dir: Path, result: EvaluationResult) -> None:
        """Save the evaluation result as JSON."""
        result_file = attempt_dir / "result.json"
        result_file.write_text(json.dumps(result.to_dict(), indent=2))

    def _extract_error_summary(self, output: str) -> str:
        """
        Extract a concise error summary from SANY output.

        Args:
            output: Full SANY output text

        Returns:
            A concise error message
        """
        lines = output.split('\n')

        # Look for common error patterns
        for i, line in enumerate(lines):
            if "***Parse Error***" in line:
                # Get the next line which usually contains the error description
                if i + 1 < len(lines):
                    return f"Parse Error: {lines[i + 1].strip()}"
                return "Parse Error"

            if "Encountered" in line and "at line" in line:
                return line.strip()

            if "Unknown operator" in line:
                return line.strip()

            if "Could not find module" in line:
                return line.strip()

            if "Circular dependency" in line:
                return line.strip()

        # Count total errors
        error_count = output.count("*** Errors:")
        if error_count > 0:
            return f"Syntax validation failed with {error_count} error(s)"

        return "Syntax validation failed"

    def save_summary(self, results: list[EvaluationResult]) -> None:
        """
        Save a summary of all attempts for this run.

        Args:
            results: List of all evaluation results from this run
        """
        summary = {
            "task_name": self.task_name,
            "spec_module": self.spec_module,
            "run_timestamp": self.run_timestamp,
            "total_attempts": len(results),
            "final_success": results[-1].success if results else False,
            "attempts": [r.to_dict() for r in results],
        }

        summary_file = self.run_dir / "summary.json"
        summary_file.write_text(json.dumps(summary, indent=2))
