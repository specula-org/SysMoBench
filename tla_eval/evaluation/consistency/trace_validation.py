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
import yaml
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


class TraceValidationEvaluator(BaseEvaluator):
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
        
        # Ensure base traces directory exists
        self.traces_dir.mkdir(parents=True, exist_ok=True)
        
    def evaluate(self, task_name: str, config: Dict[str, Any], spec_file_path: str = None, config_file_path: str = None) -> ConsistencyEvaluationResult:
        """
        Run trace validation evaluation for a given task.
        
        Args:
            task_name: Name of the task/system (e.g., "etcd", "asterinas")
            config: Configuration parameters for trace generation
            
        Returns:
            ConsistencyEvaluationResult with evaluation results
        """
        start_time = datetime.now()
        
        print(f"Starting trace validation evaluation for task: {task_name}")
        print(f"Configuration: {config}")
        
        # Create evaluation result
        result = ConsistencyEvaluationResult(task_name, "trace_validation", "system")
        
        # Check if system is supported
        if not is_system_supported(task_name):
            result.trace_generation_error = f"System '{task_name}' is not supported. Available systems: {list(get_available_systems().keys())}"
            return result
        
        # Get system implementation
        system_module = get_system(task_name)
        if not system_module:
            result.trace_generation_error = f"Failed to load system module for '{task_name}'"
            return result
        
        try:
            # Create system-specific traces directory
            system_traces_dir = self.traces_dir / task_name
            system_traces_dir.mkdir(parents=True, exist_ok=True)
            
            # Get number of traces to generate from config
            num_traces = config.get('num_traces', 20)
            timestamp = start_time.strftime("%Y%m%d_%H%M%S")
            
            # Step 1: Generate multiple runtime traces using system-specific implementation
            print(f"Step 1: Generating {num_traces} runtime traces using system-specific implementation...")
            trace_results = self._generate_multiple_system_traces(system_module, config, system_traces_dir, task_name, timestamp, num_traces)
            
            result.trace_generation_time = (datetime.now() - start_time).total_seconds()
            result.trace_generation_successful = all(tr["success"] for tr in trace_results)
            
            if not result.trace_generation_successful:
                failed_traces = [tr["error"] for tr in trace_results if not tr["success"]]
                result.trace_generation_error = f"Failed to generate some traces: {failed_traces}"
                return result
            
            result.generated_trace_count = sum(tr["event_count"] for tr in trace_results)
            result.raw_trace_files = [tr["trace_file"] for tr in trace_results]
            total_events = sum(tr["event_count"] for tr in trace_results)
            print(f"Step 1 completed: {len(trace_results)} traces generated with {total_events} total events")
            
            # Step 2: Generate specTrace.tla from TLA+ spec (LLM + static analysis)
            print("Step 2: Generating specTrace.tla from TLA+ spec...")
            step2_start = datetime.now()
            spectrace_result = self._generate_spectrace_from_tla(task_name, timestamp, spec_file_path, config_file_path, self.model_name)
            
            print(f"DEBUG: Step 2 result: {spectrace_result}")
            
            if not spectrace_result["success"]:
                print(f"ERROR: Step 2 failed with error: {spectrace_result['error']}")
                result.trace_generation_error = spectrace_result["error"]
                return result
            
            result.specification_files = [spectrace_result.get("config_file", "")]
            print("Step 2 completed: specTrace.tla and specTrace.cfg generated")
            
            # Step 3: Convert multiple sys_traces to spec-compatible format using system-specific implementation
            print(f"Step 3: Converting {len(trace_results)} sys_traces to spec-compatible format using system-specific implementation...")
            step3_start = datetime.now()
            
            conversion_results = self._convert_multiple_system_traces(system_module, trace_results, task_name, timestamp)
            
            print(f"DEBUG: Step 3 results: {len(conversion_results)} conversions attempted")
            
            result.trace_conversion_time = (datetime.now() - step3_start).total_seconds()
            result.trace_conversion_successful = all(cr["success"] for cr in conversion_results)
            
            if not result.trace_conversion_successful:
                failed_conversions = [cr["error"] for cr in conversion_results if not cr["success"]]
                print(f"ERROR: Step 3 failed with errors: {failed_conversions}")
                result.trace_conversion_error = f"Failed to convert some traces: {failed_conversions}"
                return result
            
            result.converted_trace_files = [cr["output_file"] for cr in conversion_results]
            total_input_events = sum(cr["input_events"] for cr in conversion_results)
            total_output_transitions = sum(cr["output_transitions"] for cr in conversion_results)
            print(f"Step 3 completed: Converted {total_input_events} events to {total_output_transitions} transitions across {len(conversion_results)} traces")
            
            # Step 4: Run TLC verification concurrently for multiple traces
            print(f"Step 4: Running TLC verification for {len(conversion_results)} traces using {self.max_workers} workers...")
            step4_start = datetime.now()
            verification_results = self._run_concurrent_tlc_verification(conversion_results, spectrace_result["output_dir"])
            
            print(f"DEBUG: Step 4 results: {len(verification_results)} verifications completed")
            
            result.trace_validation_time = (datetime.now() - step4_start).total_seconds()
            result.trace_validation_successful = all(vr["success"] for vr in verification_results)
            result.validated_events = sum(cr['output_transitions'] for cr in conversion_results)
            
            if not result.trace_validation_successful:
                failed_verifications = [vr.get("error", "TLC verification failed") for vr in verification_results if not vr["success"]]
                print(f"ERROR: Step 4 failed with errors: {failed_verifications}")
                result.trace_validation_error = f"Failed to verify some traces: {failed_verifications}"
            else:
                successful_count = sum(1 for vr in verification_results if vr["success"])
                print(f"Step 4 completed: Successfully verified {successful_count}/{len(verification_results)} traces")
            
            # Update overall success
            result.overall_success = (
                result.trace_generation_successful and
                result.trace_conversion_successful and
                result.trace_validation_successful
            )
            
            if result.overall_success:
                print("✓ Trace validation evaluation: PASS")
            else:
                print("✗ Trace validation evaluation: FAIL")
            
            return result
            
        except Exception as e:
            result.trace_validation_error = f"Trace validation evaluation failed: {str(e)}"
            return result
    
    def _generate_system_trace(self, system_module, config: Dict[str, Any], trace_path: Path) -> Dict[str, Any]:
        """
        Generate runtime trace using system-specific implementation.
        
        Args:
            system_module: System implementation module
            config: Configuration for trace generation
            trace_path: Path where trace file should be saved
            
        Returns:
            Dictionary with generation results
        """
        try:
            trace_generator = system_module.get_trace_generator()
            return trace_generator.generate_trace(config, trace_path)
            
        except Exception as e:
            return {
                "success": False,
                "error": f"System trace generation failed: {str(e)}"
            }
    
    def _generate_multiple_system_traces(self, system_module, config: Dict[str, Any], 
                                       system_traces_dir: Path, task_name: str, 
                                       timestamp: str, num_traces: int) -> List[Dict[str, Any]]:
        """
        Generate multiple runtime traces using system-specific implementation.
        
        Args:
            system_module: System implementation module
            config: Configuration for trace generation (will include num_traces)
            system_traces_dir: Directory to store trace files
            task_name: Name of the task/system
            timestamp: Timestamp for file naming
            num_traces: Number of traces requested
            
        Returns:
            List of dictionaries with generation results for each trace
        """
        try:
            trace_generator = system_module.get_trace_generator()
            name_prefix = f"{task_name}_trace_{timestamp}"
            
            # Let the trace generator know how many traces we want
            trace_config = config.copy()
            trace_config['num_traces'] = num_traces
            
            # Generate traces
            print(f"  Generating {num_traces} traces...")
            all_trace_results = trace_generator.generate_traces(
                config=trace_config,
                output_dir=system_traces_dir,
                name_prefix=name_prefix
            )
            
            # Check trace count and handle accordingly
            actual_count = len(all_trace_results)
            if actual_count < num_traces:
                raise ValueError(f"Generator produced {actual_count} traces, but {num_traces} were requested")
            elif actual_count > num_traces:
                print(f"  Generator produced {actual_count} traces, taking first {num_traces}")
                trace_results = all_trace_results[:num_traces]
            else:
                trace_results = all_trace_results
            
            # Report results
            for i, result in enumerate(trace_results):
                if result["success"]:
                    print(f"  Generated trace {i+1}: {result['event_count']} events")
                else:
                    print(f"  Failed to generate trace {i+1}: {result.get('error', 'Unknown error')}")
            
            return trace_results
            
        except Exception as e:
            return [{"success": False, "error": f"Trace generation failed: {str(e)}"}] * num_traces
    
    def _convert_system_trace(self, system_module, input_trace_path: Path, system_name: str, timestamp: str) -> Dict[str, Any]:
        """
        Convert system trace to TLA+ specification-compatible format using system-specific implementation.
        
        Args:
            system_module: System implementation module
            input_trace_path: Path to system-generated trace file
            system_name: Name of the system
            timestamp: Timestamp for output file naming
            
        Returns:
            Dictionary with conversion results
        """
        try:
            # Create output directory for converted traces
            converted_traces_dir = Path("data/traces") / system_name
            converted_traces_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate output path for converted trace
            converted_trace_path = converted_traces_dir / f"{system_name}_converted_{timestamp}.ndjson"
            
            trace_converter = system_module.get_trace_converter()
            return trace_converter.convert_trace(input_path=input_trace_path, output_path=converted_trace_path)
                
        except Exception as e:
            return {
                "success": False,
                "error": f"System trace conversion failed: {str(e)}"
            }
    
    def _convert_multiple_system_traces(self, system_module, trace_results: List[Dict[str, Any]], 
                                      system_name: str, timestamp: str) -> List[Dict[str, Any]]:
        """
        Convert multiple system traces to TLA+ specification-compatible format.
        
        Args:
            system_module: System implementation module
            trace_results: List of trace generation results
            system_name: Name of the system
            timestamp: Timestamp for output file naming
            
        Returns:
            List of dictionaries with conversion results for each trace
        """
        conversion_results = []
        
        for i, trace_result in enumerate(trace_results):
            if not trace_result["success"]:
                # Skip failed trace generation
                conversion_results.append({
                    "success": False,
                    "error": f"Cannot convert failed trace: {trace_result['error']}"
                })
                continue
            
            try:
                input_trace_path = Path(trace_result["trace_file"])
                
                # Create output directory for converted traces
                converted_traces_dir = Path("data/traces") / system_name
                converted_traces_dir.mkdir(parents=True, exist_ok=True)
                
                # Generate unique output path for each converted trace
                converted_trace_path = converted_traces_dir / f"{system_name}_converted_{timestamp}_{i+1}.ndjson"
                
                print(f"  Converting trace {i+1}/{len(trace_results)}: {input_trace_path.name}")
                trace_converter = system_module.get_trace_converter()
                conversion_result = trace_converter.convert_trace(input_path=input_trace_path, output_path=converted_trace_path)
                conversion_results.append(conversion_result)
                
                if conversion_result["success"]:
                    print(f"  Converted trace {i+1}: {conversion_result['input_events']} events -> {conversion_result['output_transitions']} transitions")
                else:
                    print(f"  Failed to convert trace {i+1}: {conversion_result['error']}")
                    
            except Exception as e:
                conversion_results.append({
                    "success": False,
                    "error": f"System trace conversion failed: {str(e)}"
                })
                print(f"  Failed to convert trace {i+1}: {str(e)}")
        
        return conversion_results
    
    def _generate_spectrace_from_tla(self, task_name: str, timestamp: str, spec_file_path: str = None, config_file_path: str = None, model_name: str = None) -> Dict[str, Any]:
        """
        Generate specTrace.tla and specTrace.cfg from existing TLA+ specification.
        
        This step is generic across all systems - it converts TLA+ specs to trace format.
        
        Args:
            task_name: Name of the task
            timestamp: Timestamp for file naming
            spec_file_path: Optional path to specific TLA+ spec file
            config_file_path: Optional path to specific config file
            model_name: Name of the model to use for generation
            
        Returns:
            Dictionary with generation results
        """
        try:
            # Use user-specified files if provided, otherwise find files by task name
            if spec_file_path:
                spec_file = Path(spec_file_path)
                if not spec_file.exists():
                    return {
                        "success": False,
                        "error": f"Specified TLA+ specification file not found: {spec_file_path}"
                    }
                print(f"Using user-specified TLA+ specification: {spec_file}")
            else:
                # Find TLA+ specification file by task name
                spec_files = list(Path(self.spec_dir).glob(f"{task_name}/*.tla"))
                if not spec_files:
                    return {
                        "success": False,
                        "error": f"No TLA+ specification found for task: {task_name}"
                    }
                spec_file = spec_files[0]  # Use first found spec file
                print(f"Using TLA+ specification: {spec_file}")
            
            if config_file_path:
                cfg_file = Path(config_file_path)
                if not cfg_file.exists():
                    return {
                        "success": False,
                        "error": f"Specified CFG configuration file not found: {config_file_path}"
                    }
                print(f"Using user-specified CFG configuration: {cfg_file}")
            else:
                # Find corresponding CFG file by task name
                cfg_files = list(Path(self.spec_dir).glob(f"{task_name}/*.cfg"))
                if not cfg_files:
                    return {
                        "success": False,
                        "error": f"No CFG configuration found for task: {task_name}"
                    }
                cfg_file = cfg_files[0]  # Use first found cfg file
                print(f"Using CFG configuration: {cfg_file}")
            
            # Generate configuration using LLM
            print(f"DEBUG: Calling generate_config_from_tla with spec_file={spec_file}, cfg_file={cfg_file}, model_name={model_name}")
            try:
                config_data = generate_config_from_tla(str(spec_file), str(cfg_file), model_name)
                print(f"DEBUG: LLM config generation successful, got config keys: {list(config_data.keys()) if config_data else 'None'}")
            except Exception as e:
                print(f"ERROR: LLM config generation failed: {str(e)}")
                return {
                    "success": False,
                    "error": f"LLM config generation failed: {str(e)}"
                }
            
            # Save configuration for debugging
            config_path = self.spec_dir / task_name / f"trace_config_{timestamp}.yaml"
            with open(config_path, 'w', encoding='utf-8') as f:
                yaml.dump(config_data, f, default_flow_style=False)
            
            print(f"Generated trace configuration: {config_path}")
            
            # Output to the correct spec directory, not traces directory
            output_dir = self.spec_dir / task_name
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Copy common TLA+ files first
            common_dir = Path("data/spec/common")
            if common_dir.exists():
                print(f"  Copying common TLA+ files to {output_dir}")
                for common_file in common_dir.iterdir():
                    if common_file.is_file():
                        dest_file = output_dir / common_file.name
                        shutil.copy2(common_file, dest_file)
                        print(f"    Copied: {common_file.name}")
            
            # Generate trace validation files
            generator = SpecTraceGenerator(config_data)
            files = generator.generate_files(str(output_dir))
            
            print(f"  Generated specTrace files in: {output_dir}")
            
            return {
                "success": True,
                "config_data": config_data,
                "config_file": str(config_path),
                "output_dir": str(output_dir),
                "files": files
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"specTrace generation failed: {str(e)}"
            }
    
    def _run_tlc_verification(self, trace_path: Path, spec_dir: str) -> Dict[str, Any]:
        """
        Run TLC verification of trace against converted specification.
        
        This step is generic across all systems.
        
        Args:
            trace_path: Path to the trace file
            spec_dir: Directory containing specTrace.tla and specTrace.cfg
            
        Returns:
            Dictionary with verification results
        """
        try:
            spec_dir_path = Path(spec_dir)
            
            # Create a symbolic link for the trace file in the spec directory
            # This ensures TLC can find the trace file relative to the spec
            trace_link = spec_dir_path / "trace.ndjson" 
            if trace_link.exists():
                trace_link.unlink()  # Remove existing link
            
            # Create relative path to trace file
            try:
                trace_link.symlink_to(trace_path.resolve())
                print(f"Created trace symlink: {trace_link} -> {trace_path}")
            except OSError:
                # If symlink fails, copy the file instead
                shutil.copy2(trace_path, trace_link)
                print(f"Copied trace file to: {trace_link}")
            
            # Run TLC verification
            tlc_runner = TLCRunner()
            return tlc_runner.run_verification(trace_path, spec_dir)
            
        except Exception as e:
            return {
                "success": False,
                "error": f"TLC setup failed: {str(e)}"
            }
    
    def _run_concurrent_tlc_verification(self, conversion_results: List[Dict[str, Any]], 
                                       spec_dir: str) -> List[Dict[str, Any]]:
        """
        Run TLC verification concurrently for multiple traces.
        
        Args:
            conversion_results: List of trace conversion results
            spec_dir: Directory containing specTrace.tla and specTrace.cfg
            
        Returns:
            List of dictionaries with verification results for each trace
        """
        verification_results = []
        
        def verify_single_trace(i: int, conversion_result: Dict[str, Any]) -> tuple:
            """Helper function to verify a single trace."""
            if not conversion_result["success"]:
                return i, {
                    "success": False,
                    "error": f"Cannot verify failed conversion: {conversion_result['error']}"
                }
            
            try:
                trace_path = Path(conversion_result["output_file"])
                print(f"  Verifying trace {i+1}/{len(conversion_results)}: {trace_path.name}")
                
                result = self._run_tlc_verification(trace_path, spec_dir)
                
                if result["success"]:
                    print(f"  ✓ Verification {i+1} passed")
                else:
                    print(f"  ✗ Verification {i+1} failed: {result.get('error', 'Unknown error')}")
                
                return i, result
                
            except Exception as e:
                error_msg = f"TLC verification failed: {str(e)}"
                print(f"  ✗ Verification {i+1} failed: {error_msg}")
                return i, {
                    "success": False,
                    "error": error_msg
                }
        
        # Run verifications concurrently
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all verification tasks
            future_to_index = {
                executor.submit(verify_single_trace, i, conversion_result): i 
                for i, conversion_result in enumerate(conversion_results)
            }
            
            # Initialize results list with None placeholders
            verification_results = [None] * len(conversion_results)
            
            # Collect results as they complete
            for future in as_completed(future_to_index):
                try:
                    index, result = future.result()
                    verification_results[index] = result
                except Exception as e:
                    index = future_to_index[future]
                    verification_results[index] = {
                        "success": False,
                        "error": f"Verification task failed: {str(e)}"
                    }
        
        return verification_results
    
    def get_evaluation_name(self) -> str:
        """Get the name of this evaluation method."""
        return "trace_validation"
    
    def get_supported_tasks(self):
        """Get list of tasks supported by this evaluator."""
        return list(get_available_systems().keys())
    
    def get_default_config(self, system_name: str = None) -> Dict[str, Any]:
        """
        Get default configuration for trace validation evaluation.
        
        Args:
            system_name: Optional system name to get system-specific defaults
            
        Returns:
            Default configuration dictionary
        """
        if system_name and is_system_supported(system_name):
            system_module = get_system(system_name)
            if system_module:
                trace_generator = system_module.get_trace_generator()
                return trace_generator.get_default_config()
        
        # Generic defaults if system not specified or not found
        return {
            "duration_seconds": 60,
            "scenario": "normal_operation"
        }
    
    def get_available_scenarios(self, system_name: str = None) -> Dict[str, Dict[str, Any]]:
        """
        Get available predefined scenarios.
        
        Args:
            system_name: Optional system name to get system-specific scenarios
            
        Returns:
            Dictionary mapping scenario names to their configurations
        """
        if system_name and is_system_supported(system_name):
            system_module = get_system(system_name)
            if system_module:
                trace_generator = system_module.get_trace_generator()
                return trace_generator.get_available_scenarios()
        
        return {}
    
    def _get_evaluation_type(self) -> str:
        """Return the evaluation type identifier"""
        return "consistency_trace_validation"


# Convenience function for backward compatibility
def create_trace_validation_evaluator(
    spec_dir: str = "data/spec",
    traces_dir: str = "data/sys_traces",
    timeout: int = 600,
    model_name: str = None,
    max_workers: int = 4
) -> TraceValidationEvaluator:
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
    return TraceValidationEvaluator(spec_dir, traces_dir, timeout, model_name, max_workers)