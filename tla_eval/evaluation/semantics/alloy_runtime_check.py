"""
Alloy Runtime Check Evaluator: Executes run/check commands in Alloy specifications.

This evaluator uses AlloyRuntime to execute all run/check commands and reports results.
"""

import logging
import subprocess
import time
import json
import re
from pathlib import Path
from typing import Optional, Dict, List

from ...models.base import GenerationResult
from ...utils.output_manager import get_output_manager
from ..base.evaluator import BaseEvaluator
from ..base.result_types import SemanticEvaluationResult

logger = logging.getLogger(__name__)


class AlloyRuntimeCheckEvaluator(BaseEvaluator):
    """
    Evaluator for Alloy runtime checking (executing run/check commands).
    """

    def __init__(self, validation_timeout: int = 60):
        """
        Initialize Alloy runtime check evaluator.

        Args:
            validation_timeout: Timeout for runtime checking in seconds
        """
        super().__init__(timeout=validation_timeout)
        self.alloy_jar = Path("lib/alloy.jar")
        self.runtime_class = "AlloyRuntime"

        if not self.alloy_jar.exists():
            raise FileNotFoundError(f"Alloy JAR not found: {self.alloy_jar}")

    def evaluate(self,
                generation_result: GenerationResult,
                task_name: str,
                method_name: str,
                model_name: str,
                spec_module: str = None,
                spec_file_path: Optional[str] = None,
                config_file_path: Optional[str] = None) -> SemanticEvaluationResult:
        """
        Evaluate Alloy specification runtime checking.

        Args:
            generation_result: Result from Alloy generation
            task_name: Name of the task
            method_name: Name of the generation method
            model_name: Name of the model used
            spec_module: Not used for Alloy
            spec_file_path: Optional path to existing .als file
            config_file_path: Not used for Alloy

        Returns:
            SemanticEvaluationResult with runtime checking metrics
        """
        logger.info(f"Evaluating Alloy runtime: {task_name}/{method_name}/{model_name}")

        # Create output directory
        output_manager = get_output_manager()
        output_dir = output_manager.create_experiment_dir(
            metric="runtime_check",
            task=task_name,
            method=method_name,
            model=model_name,
            language="alloy"
        )
        logger.info(f"Using output directory: {output_dir}")

        # Create evaluation result
        eval_result = SemanticEvaluationResult(task_name, method_name, model_name)

        # Get Alloy content
        if spec_file_path and Path(spec_file_path).exists():
            logger.info(f"Using existing spec file: {spec_file_path}")
            alloy_file = Path(spec_file_path)
            eval_result.specification_file = str(alloy_file)
        elif generation_result and generation_result.generated_text:
            logger.info("Using generated Alloy specification")
            # Save to output directory
            alloy_file = output_dir / f"{task_name}.als"
            alloy_file.write_text(generation_result.generated_text, encoding='utf-8')
            eval_result.specification_file = str(alloy_file)
        else:
            logger.error("No valid spec file or generation result provided")
            eval_result.model_checking_successful = False
            eval_result.model_checking_error = "No specification provided"
            eval_result.overall_success = False
            return eval_result

        # Execute runtime checking
        logger.info("Starting Alloy runtime checking")
        start_time = time.time()

        runtime_result = self._execute_runtime(alloy_file)

        eval_result.model_checking_time = time.time() - start_time
        eval_result.model_checking_successful = runtime_result['success']

        if runtime_result['success']:
            eval_result.states_explored = runtime_result.get('total_commands', 0)
            eval_result.model_checking_error = None

            # Store command results in custom_data
            eval_result.custom_data = {
                'total_commands': runtime_result.get('total_commands', 0),
                'successful_commands': runtime_result.get('successful_commands', 0),
                'failed_commands': runtime_result.get('failed_commands', 0),
                'command_results': runtime_result.get('commands', [])
            }

            logger.info(f"Runtime check passed: {runtime_result['successful_commands']}/{runtime_result['total_commands']} commands succeeded")
        else:
            eval_result.model_checking_error = runtime_result.get('error', 'Unknown error')
            logger.error(f"Runtime check failed: {eval_result.model_checking_error}")

        eval_result.overall_success = eval_result.model_checking_successful

        # Save results
        results_file = output_dir / "evaluation_result.json"
        try:
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(eval_result.to_dict(), f, indent=2)
            logger.info(f"Saved evaluation results to {results_file}")
        except Exception as e:
            logger.warning(f"Failed to save results: {e}")

        return eval_result

    def _execute_runtime(self, spec_file: Path) -> Dict:
        """
        Execute Alloy runtime checker.

        Args:
            spec_file: Path to .als file

        Returns:
            Dictionary with execution results
        """
        classpath = f"{self.alloy_jar}:{self.alloy_jar.parent}"

        cmd = [
            "java",
            "-cp", classpath,
            self.runtime_class,
            str(spec_file.resolve()),
            "--timeout", str(self.timeout)
        ]

        logger.info(f"Running: {' '.join(cmd)}")

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout + 10  # Extra buffer
            )

            return self._parse_runtime_output(result.returncode, result.stdout, result.stderr)

        except subprocess.TimeoutExpired:
            logger.error(f"Runtime check timeout after {self.timeout}s")
            return {
                'success': False,
                'error': f"Timeout after {self.timeout}s"
            }

        except Exception as e:
            logger.error(f"Failed to run runtime checker: {e}")
            return {
                'success': False,
                'error': f"Cannot run runtime checker: {e}"
            }

    def _parse_runtime_output(self, returncode: int, stdout: str, stderr: str) -> Dict:
        """
        Parse output from AlloyRuntime.

        Args:
            returncode: Exit code
            stdout: Standard output
            stderr: Standard error

        Returns:
            Dictionary with parsed results
        """
        commands = []
        total_commands = 0
        successful_commands = 0
        failed_commands = 0

        if returncode == 0 or returncode == 1:
            # Parse command results from stdout
            lines = stdout.split('\n')
            current_command = None

            for line in lines:
                line = line.strip()

                if line.startswith('COMMANDS:'):
                    total_commands = int(line.split(':')[1].strip())

                elif line.startswith('=== COMMAND_'):
                    if current_command:
                        commands.append(current_command)
                    current_command = {}

                elif line.startswith('LABEL:') and current_command is not None:
                    current_command['label'] = line.split(':', 1)[1].strip()

                elif line.startswith('TYPE:') and current_command is not None:
                    current_command['type'] = line.split(':', 1)[1].strip()

                elif line.startswith('RESULT:') and current_command is not None:
                    result_str = line.split(':', 1)[1].strip()
                    current_command['result'] = result_str
                    if 'PASS' in result_str:
                        successful_commands += 1
                    else:
                        failed_commands += 1

                elif line.startswith('STATUS:') and current_command is not None:
                    current_command['status'] = line.split(':', 1)[1].strip()

                elif line.startswith('EXEC_TIME:') and current_command is not None:
                    time_str = line.split(':', 1)[1].strip().replace('ms', '')
                    current_command['exec_time_ms'] = int(time_str)

            if current_command:
                commands.append(current_command)

            return {
                'success': returncode == 0,
                'total_commands': total_commands,
                'successful_commands': successful_commands,
                'failed_commands': failed_commands,
                'commands': commands
            }

        else:
            # Internal error
            return {
                'success': False,
                'error': f"Runtime checker internal error (code {returncode}): {stderr}"
            }

    def _get_evaluation_type(self) -> str:
        """Return the evaluation type identifier"""
        return "alloy_runtime"
