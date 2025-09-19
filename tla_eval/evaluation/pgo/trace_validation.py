"""
Trace Validation Evaluator: System consistency evaluation for TLA+ specifications.

This evaluator implements a generic trace validation pipeline that works with
any system that provides trace generation and conversion implementations:

1. System-specific trace generation
2. LLM-based configuration generation for trace validation
3. System-specific trace format conversion 
4. TLC verification of traces against converted specifications

The system-specific logic is delegated to modules in tla_eval/core/trace_generation/{system}/
"""

import os
import subprocess
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

from ...core.trace_generation.registry import get_system, get_available_systems, is_system_supported
from ...core.spec_processing import SpecTraceGenerator, generate_config_from_tla
from ...core.verification import TLCRunner
from ..base.evaluator import BaseEvaluator
from ..base.result_types import ConsistencyEvaluationResult


class PGoTraceValidationEvaluator(BaseEvaluator):
    """
    Trace Validation Evaluator: System consistency evaluation.
    
    This evaluator implements a generic trace validation workflow that works
    with any system implementation:
    1. **System-specific Trace Generation**: Uses system modules for trace generation
    2. **Config Generation**: LLM-based YAML configuration from TLA+ specs
    3. **System-specific Conversion**: Uses system modules for trace format conversion
    4. **TLC Verification**: Trace validation against converted specifications
    """
    
    def __init__(self, 
                 spec_dir: str = "data/spec",
                 traces_dir: str = "data/sys_traces",
                 timeout: int = 600,
                 model_name: str = None,
                 max_workers: int = 4):
        """
        Initialize trace validation evaluator.
        
        Args:
            spec_dir: Directory containing TLA+ specifications
            traces_dir: Base directory to store generated traces (system subdirs created automatically)
            timeout: Timeout for evaluation operations in seconds
            model_name: Name of the model to use for specTrace generation (if None, uses default)
            max_workers: Maximum number of worker threads for concurrent trace validation
        """
        super().__init__(timeout=timeout)
        self.spec_dir = Path(spec_dir)
        self.traces_dir = Path(traces_dir)
        self.model_name = model_name
        self.max_workers = max_workers

        self.pgo_exe = Path(__file__).parent / "pgo.jar"
        
        # Ensure base traces directory exists
        self.traces_dir.mkdir(parents=True, exist_ok=True)
        
    def evaluate(self, task_name: str, config: Dict[str, Any], spec_file_path: str, config_file_path: str) -> ConsistencyEvaluationResult:
        """
        Run trace validation evaluation for a given task.
        
        Args:
            task_name: Name of the task/system (e.g., "etcd", "asterinas")
            config: Configuration parameters for trace generation
            
        Returns:
            ConsistencyEvaluationResult with evaluation results
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            src_dir = Path(f"tla_eval/core/trace_generation/{task_name}")
            shutil.copytree(src=src_dir / "traces_found", dst=temp_dir, dirs_exist_ok=True)
            traces_dirs = list(Path(temp_dir).iterdir())
            for traces_dir in traces_dirs:
                subprocess.run([
                    "java", "-jar", self.pgo_exe, "tracegen",
                    src_dir / f"{task_name}.tla",
                    "--noall-paths", "--cfg-file", config_file_path, traces_dir,
                ], check=True)

                # patch out cfg parts
                cfg_path_tmp = Path(traces_dir) / f"{task_name}Validate.cfg"
                cfg_lines = cfg_path_tmp.read_text().splitlines()
                def cfg_lines_pred(line):
                    if line.startswith("SPECIFICATION"):
                        return line == "SPECIFICATION __Spec"
                    elif line.startswith("INIT "):
                        return False
                    elif line.startswith("NEXT "):
                        return False
                    else:
                        return True
                cfg_lines = filter(cfg_lines_pred, cfg_lines)
                cfg_lines = list(cfg_lines) + (src_dir / f"{task_name}Validate.cfg").read_text().splitlines()
                cfg_path_tmp.write_text('\n'.join(cfg_lines))

                # overwrite TLA+
                (Path(traces_dir) / f"{task_name}.tla").write_text(Path(spec_file_path).read_text())
        
            print(f"generated validation setups for {len(traces_dirs)} traces")

            for traces_dir in traces_dirs:
                subprocess.run([
                    "java", "-jar", self.pgo_exe, "tlc",
                    "--dfs", Path(traces_dir) / f"{task_name}Validate.tla",
                ], check=True)

        print("oh hey there!")
        raise Exception("TODO")
        # start_time = datetime.now()
        
        # print(f"Starting trace validation evaluation for task: {task_name}")
        # print(f"Configuration: {config}")
        
        # # Create evaluation result
        # result = ConsistencyEvaluationResult(task_name, "trace_validation", "system")
        
        # # Check if system is supported
        # if not is_system_supported(task_name):
        #     result.trace_generation_error = f"System '{task_name}' is not supported. Available systems: {list(get_available_systems().keys())}"
        #     return result
        
        # # Get system implementation
        # system_module = get_system(task_name)
        # if not system_module:
        #     result.trace_generation_error = f"Failed to load system module for '{task_name}'"
        #     return result
        
        # try:
        #     # Create system-specific traces directory
        #     system_traces_dir = self.traces_dir / task_name
        #     system_traces_dir.mkdir(parents=True, exist_ok=True)
            
        #     # Get number of traces to generate from config
        #     num_traces = config.get('num_traces', 20)
        #     timestamp = start_time.strftime("%Y%m%d_%H%M%S")
            
        #     # Step 1: Generate multiple runtime traces using system-specific implementation
        #     print(f"Step 1: Generating {num_traces} runtime traces using system-specific implementation...")
        #     trace_results = self._generate_multiple_system_traces(system_module, config, system_traces_dir, task_name, timestamp, num_traces)
            
        #     result.trace_generation_time = (datetime.now() - start_time).total_seconds()
        #     result.trace_generation_successful = all(tr["success"] for tr in trace_results)
            
        #     if not result.trace_generation_successful:
        #         failed_traces = [tr["error"] for tr in trace_results if not tr["success"]]
        #         result.trace_generation_error = f"Failed to generate some traces: {failed_traces}"
        #         return result
            
        #     result.generated_trace_count = sum(tr["event_count"] for tr in trace_results)
        #     result.raw_trace_files = [tr["trace_file"] for tr in trace_results]
        #     total_events = sum(tr["event_count"] for tr in trace_results)
        #     print(f"Step 1 completed: {len(trace_results)} traces generated with {total_events} total events")
            
        #     # Step 2: Generate specTrace.tla from TLA+ spec (LLM + static analysis)
        #     print("Step 2: Generating specTrace.tla from TLA+ spec...")
        #     step2_start = datetime.now()
        #     spectrace_result = self._generate_spectrace_from_tla(task_name, timestamp, spec_file_path, config_file_path, self.model_name)
            
        #     print(f"DEBUG: Step 2 result: {spectrace_result}")
            
        #     if not spectrace_result["success"]:
        #         print(f"ERROR: Step 2 failed with error: {spectrace_result['error']}")
        #         result.trace_generation_error = spectrace_result["error"]
        #         return result
            
        #     result.specification_files = [spectrace_result.get("config_file", "")]
        #     print("Step 2 completed: specTrace.tla and specTrace.cfg generated")
            
        #     # Step 3: Convert multiple sys_traces to spec-compatible format using system-specific implementation
        #     print(f"Step 3: Converting {len(trace_results)} sys_traces to spec-compatible format using system-specific implementation...")
        #     step3_start = datetime.now()
            
        #     conversion_results = self._convert_multiple_system_traces(system_module, trace_results, task_name, timestamp)
            
        #     print(f"DEBUG: Step 3 results: {len(conversion_results)} conversions attempted")
            
        #     result.trace_conversion_time = (datetime.now() - step3_start).total_seconds()
        #     result.trace_conversion_successful = all(cr["success"] for cr in conversion_results)
            
        #     if not result.trace_conversion_successful:
        #         failed_conversions = [cr["error"] for cr in conversion_results if not cr["success"]]
        #         print(f"ERROR: Step 3 failed with errors: {failed_conversions}")
        #         result.trace_conversion_error = f"Failed to convert some traces: {failed_conversions}"
        #         return result
            
        #     result.converted_trace_files = [cr["output_file"] for cr in conversion_results]
        #     total_input_events = sum(cr["input_events"] for cr in conversion_results)
        #     total_output_transitions = sum(cr["output_transitions"] for cr in conversion_results)
        #     print(f"Step 3 completed: Converted {total_input_events} events to {total_output_transitions} transitions across {len(conversion_results)} traces")
            
        #     # Step 4: Run TLC verification concurrently for multiple traces
        #     print(f"Step 4: Running TLC verification for {len(conversion_results)} traces using {self.max_workers} workers...")
        #     step4_start = datetime.now()
        #     verification_results = self._run_concurrent_tlc_verification(conversion_results, spectrace_result["output_dir"])
            
        #     print(f"DEBUG: Step 4 results: {len(verification_results)} verifications completed")
            
        #     result.trace_validation_time = (datetime.now() - step4_start).total_seconds()
        #     result.trace_validation_successful = all(vr["success"] for vr in verification_results)
        #     result.validated_events = sum(cr['output_transitions'] for cr in conversion_results)
            
        #     if not result.trace_validation_successful:
        #         failed_verifications = [vr.get("error", "TLC verification failed") for vr in verification_results if not vr["success"]]
        #         print(f"ERROR: Step 4 failed with errors: {failed_verifications}")
        #         result.trace_validation_error = f"Failed to verify some traces: {failed_verifications}"
        #     else:
        #         successful_count = sum(1 for vr in verification_results if vr["success"])
        #         print(f"Step 4 completed: Successfully verified {successful_count}/{len(verification_results)} traces")
            
        #     # Update overall success
        #     result.overall_success = (
        #         result.trace_generation_successful and
        #         result.trace_conversion_successful and
        #         result.trace_validation_successful
        #     )
            
        #     if result.overall_success:
        #         print("✓ Trace validation evaluation: PASS")
        #     else:
        #         print("✗ Trace validation evaluation: FAIL")
            
        #     return result
            
        # except Exception as e:
        #     result.trace_validation_error = f"Trace validation evaluation failed: {str(e)}"
        #     return result
    
    def get_evaluation_name(self) -> str:
        """Get the name of this evaluation method."""
        return "pgo_trace_validation"
    
    def get_supported_tasks(self):
        """Get list of tasks supported by this evaluator."""
        return ["dqueue"]
    
    def get_default_config(self, system_name: str = None) -> Dict[str, Any]:
        return {}
    
    def get_available_scenarios(self, system_name: str = None) -> Dict[str, Dict[str, Any]]:
        """
        Get available predefined scenarios.
        
        Args:
            system_name: Optional system name to get system-specific scenarios
            
        Returns:
            Dictionary mapping scenario names to their configurations
        """
        # if system_name and is_system_supported(system_name):
        #     system_module = get_system(system_name)
        #     if system_module:
        #         trace_generator = system_module.get_trace_generator()
        #         return trace_generator.get_available_scenarios()
        
        return {}
    
    def _get_evaluation_type(self) -> str:
        """Return the evaluation type identifier"""
        return "consistency_pgo_trace_validation"


# Convenience function for backward compatibility
def create_trace_validation_evaluator(
    spec_dir: str = "data/spec",
    traces_dir: str = "data/sys_traces",
    timeout: int = 600,
    model_name: str = None,
    max_workers: int = 4
) -> PGoTraceValidationEvaluator:
    """
    Factory function to create a trace validation evaluator.
    
    Args:
        spec_dir: Directory containing TLA+ specifications
        traces_dir: Base directory to store generated traces
        timeout: Timeout for evaluation operations in seconds
        model_name: Name of the model to use for specTrace generation
        max_workers: Maximum number of worker threads for concurrent trace validation
        
    Returns:
        TraceValidationEvaluator instance
    """
    return PGoTraceValidationEvaluator(spec_dir, traces_dir, timeout, model_name, max_workers)
