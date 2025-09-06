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
from typing import Dict, Any
from datetime import datetime

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
                 timeout: int = 600):
        """
        Initialize trace validation evaluator.
        
        Args:
            spec_dir: Directory containing TLA+ specifications
            traces_dir: Base directory to store generated traces (system subdirs created automatically)
            timeout: Timeout for evaluation operations in seconds
        """
        super().__init__(timeout=timeout)
        self.spec_dir = Path(spec_dir)
        self.traces_dir = Path(traces_dir)
        
        # Ensure base traces directory exists
        self.traces_dir.mkdir(parents=True, exist_ok=True)
        
    def evaluate(self, task_name: str, config: Dict[str, Any]) -> ConsistencyEvaluationResult:
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
            
            # Generate trace file name
            timestamp = start_time.strftime("%Y%m%d_%H%M%S")
            trace_filename = f"{task_name}_trace_{timestamp}.ndjson"
            trace_path = system_traces_dir / trace_filename
            
            # Step 1: Generate runtime trace using system-specific implementation
            print("Step 1: Generating runtime trace using system-specific implementation...")
            trace_result = self._generate_system_trace(system_module, config, trace_path)
            
            result.trace_generation_time = (datetime.now() - start_time).total_seconds()
            result.trace_generation_successful = trace_result["success"]
            
            if not trace_result["success"]:
                result.trace_generation_error = trace_result["error"]
                return result
            
            result.generated_trace_count = trace_result["event_count"]
            result.raw_trace_files = [trace_result["trace_file"]]
            print(f"Step 1 completed: {trace_result['event_count']} events generated")
            
            # Step 2: Generate specTrace.tla from TLA+ spec (LLM + static analysis)
            print("Step 2: Generating specTrace.tla from TLA+ spec...")
            step2_start = datetime.now()
            spectrace_result = self._generate_spectrace_from_tla(task_name, timestamp)
            
            print(f"DEBUG: Step 2 result: {spectrace_result}")
            
            if not spectrace_result["success"]:
                print(f"ERROR: Step 2 failed with error: {spectrace_result['error']}")
                result.trace_generation_error = spectrace_result["error"]
                return result
            
            result.specification_files = [spectrace_result.get("config_file", "")]
            print("Step 2 completed: specTrace.tla and specTrace.cfg generated")
            
            # Step 3: Convert sys_trace to spec-compatible format using system-specific implementation
            print("Step 3: Converting sys_trace to spec-compatible format using system-specific implementation...")
            step3_start = datetime.now()
            actual_trace_path = Path(trace_result["trace_file"])
            trace_conversion_result = self._convert_system_trace(system_module, actual_trace_path, task_name, timestamp)
            
            print(f"DEBUG: Step 3 result: {trace_conversion_result}")
            
            result.trace_conversion_time = (datetime.now() - step3_start).total_seconds()
            result.trace_conversion_successful = trace_conversion_result["success"]
            
            if not trace_conversion_result["success"]:
                print(f"ERROR: Step 3 failed with error: {trace_conversion_result['error']}")
                result.trace_conversion_error = trace_conversion_result["error"]
                return result
            
            result.converted_trace_files = [trace_conversion_result["output_file"]]
            print(f"Step 3 completed: Converted {trace_conversion_result['input_events']} events to {trace_conversion_result['output_transitions']} transitions")
            
            # Step 4: Run TLC verification
            print("Step 4: Running TLC verification...")
            step4_start = datetime.now()
            verification_result = self._run_tlc_verification(Path(trace_conversion_result["output_file"]), spectrace_result["output_dir"])
            
            print(f"DEBUG: Step 4 result: {verification_result}")
            
            result.trace_validation_time = (datetime.now() - step4_start).total_seconds()
            result.trace_validation_successful = verification_result["success"]
            result.validated_events = trace_conversion_result['output_transitions']
            
            if not verification_result["success"]:
                print(f"ERROR: Step 4 failed with error: {verification_result.get('error', 'TLC verification failed')}")
                result.trace_validation_error = verification_result.get("error", "TLC verification failed")
            
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
    
    def _generate_spectrace_from_tla(self, task_name: str, timestamp: str) -> Dict[str, Any]:
        """
        Generate specTrace.tla and specTrace.cfg from existing TLA+ specification.
        
        This step is generic across all systems - it converts TLA+ specs to trace format.
        
        Args:
            task_name: Name of the task
            timestamp: Timestamp for file naming
            
        Returns:
            Dictionary with generation results
        """
        try:
            # Find TLA+ specification file
            spec_files = list(self.spec_dir.glob(f"{task_name}/*.tla"))
            if not spec_files:
                return {
                    "success": False,
                    "error": f"No TLA+ specification found for task: {task_name}"
                }
            
            spec_file = spec_files[0]  # Use first found spec file
            print(f"Using TLA+ specification: {spec_file}")
            
            # Find corresponding CFG file
            cfg_files = list(self.spec_dir.glob(f"{task_name}/*.cfg"))
            if not cfg_files:
                return {
                    "success": False,
                    "error": f"No CFG configuration found for task: {task_name}"
                }
            
            cfg_file = cfg_files[0]  # Use first found cfg file
            print(f"Using CFG configuration: {cfg_file}")
            
            # Generate configuration using LLM
            print(f"DEBUG: Calling generate_config_from_tla with spec_file={spec_file}, cfg_file={cfg_file}")
            try:
                config_data = generate_config_from_tla(str(spec_file), str(cfg_file), "my_claude")
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
    timeout: int = 600
) -> TraceValidationEvaluator:
    """
    Factory function to create a trace validation evaluator.
    
    Args:
        spec_dir: Directory containing TLA+ specifications
        traces_dir: Base directory to store generated traces
        timeout: Timeout for evaluation operations in seconds
        
    Returns:
        GenericTraceValidationEvaluator instance
    """
    return TraceValidationEvaluator(spec_dir, traces_dir, timeout)