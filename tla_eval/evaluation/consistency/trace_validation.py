"""
Trace Validation Evaluator: System consistency evaluation for TLA+ specifications.

This evaluator implements the complete trace validation pipeline:
1. Real trace generation from etcd raft clusters
2. LLM-based configuration generation for trace validation
3. Static analysis conversion of TLA+ specs to trace format  
4. TLC verification of traces against converted specifications
"""

import os
import subprocess
import tempfile
import yaml
import shutil
from pathlib import Path
from typing import Dict, Any
from datetime import datetime

from ...core.trace_generation.etcd.cluster import RaftCluster, FileTraceLogger
from ...core.trace_generation.etcd.event_driver import RandomEventDriver
from ...core.spec_processing import SpecTraceGenerator, generate_config_from_tla
from ...core.spec_processing.trace_converter import TraceConverter
from ...core.verification import TLCRunner
from ..base.evaluator import BaseEvaluator
from ..base.result_types import ConsistencyEvaluationResult


class TraceValidationEvaluator(BaseEvaluator):
    """
    Trace Validation Evaluator: System consistency evaluation.
    
    This evaluator implements the complete trace validation workflow:
    1. **Trace Generation**: Real etcd raft cluster trace generation
    2. **Config Generation**: LLM-based YAML configuration from TLA+ specs
    3. **Spec Conversion**: Static analysis conversion to trace-validation format
    4. **TLC Verification**: Trace validation against converted specifications
    """
    
    def __init__(self, 
                 spec_dir: str = "data/spec",
                 traces_dir: str = "data/sys_traces/etcd",
                 raft_repo_dir: str = "data/repositories/raft",
                 timeout: int = 600):
        """
        Initialize trace validation evaluator.
        
        Args:
            spec_dir: Directory containing TLA+ specifications
            traces_dir: Directory to store generated traces
            raft_repo_dir: Directory containing etcd raft repository
            timeout: Timeout for evaluation operations in seconds
        """
        super().__init__(timeout=timeout)
        self.spec_dir = Path(spec_dir)
        self.traces_dir = Path(traces_dir)
        self.raft_repo_dir = Path(raft_repo_dir)
        
        # Ensure directories exist
        self.traces_dir.mkdir(parents=True, exist_ok=True)
        
    def evaluate(self, task_name: str, config: Dict[str, Any]) -> ConsistencyEvaluationResult:
        """
        Run trace validation evaluation for a given task.
        
        Args:
            task_name: Name of the task (e.g., "etcd")
            config: Configuration parameters for trace generation
            
        Returns:
            ConsistencyEvaluationResult with evaluation results
        """
        start_time = datetime.now()
        
        print(f"Starting trace validation evaluation for task: {task_name}")
        print(f"Configuration: {config}")
        
        # Create evaluation result
        result = ConsistencyEvaluationResult(task_name, "trace_validation", "system")
        
        try:
            # Generate trace file name
            timestamp = start_time.strftime("%Y%m%d_%H%M%S")
            trace_filename = f"{task_name}_trace_{timestamp}.ndjson"
            trace_path = self.traces_dir / trace_filename
            
            # Step 1: Generate runtime trace using real etcd raft cluster
            print("Step 1: Generating runtime trace...")
            trace_result = self._generate_real_trace(task_name, config, trace_path)
            
            result.trace_generation_time = (datetime.now() - start_time).total_seconds()
            result.trace_generation_successful = trace_result["success"]
            
            if not trace_result["success"]:
                result.trace_generation_error = trace_result["error"]
                return result
            
            result.generated_trace_count = trace_result["event_count"]
            result.raw_trace_files = [str(trace_path)]
            print(f"Step 1 completed: {trace_result['event_count']} events generated")
            
            # Step 2: Generate specTrace.tla from TLA+ spec (LLM + static analysis)
            print("Step 2: Generating specTrace.tla from TLA+ spec...")
            step2_start = datetime.now()
            spectrace_result = self._generate_spectrace_from_tla(task_name, timestamp)
            
            if not spectrace_result["success"]:
                result.trace_generation_error = spectrace_result["error"]
                return result
            
            result.specification_files = [spectrace_result.get("config_file", "")]
            print("Step 2 completed: specTrace.tla and specTrace.cfg generated")
            
            # Step 3: Convert sys_trace to spec-compatible format
            print("Step 3: Converting sys_trace to spec-compatible format...")
            step3_start = datetime.now()
            # Use the actual trace file from Step 1 result
            actual_trace_path = Path(trace_result["trace_file"])
            trace_conversion_result = self._convert_trace_to_spec_format(actual_trace_path, timestamp)
            
            result.trace_conversion_time = (datetime.now() - step3_start).total_seconds()
            result.trace_conversion_successful = trace_conversion_result["success"]
            
            if not trace_conversion_result["success"]:
                result.trace_conversion_error = trace_conversion_result["error"]
                return result
            
            result.converted_trace_files = [trace_conversion_result["converted_trace"]]
            print(f"Step 3 completed: Converted {trace_conversion_result['input_events']} events to {trace_conversion_result['output_transitions']} transitions")
            
            # Step 4: Run TLC verification
            print("Step 4: Running TLC verification...")
            step4_start = datetime.now()
            verification_result = self._run_tlc_verification(Path(trace_conversion_result["converted_trace"]), spectrace_result["output_dir"])
            
            result.trace_validation_time = (datetime.now() - step4_start).total_seconds()
            result.trace_validation_successful = verification_result["success"]
            result.validated_events = trace_conversion_result['output_transitions']
            
            if not verification_result["success"]:
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
    
    def _generate_real_trace(self, task_name: str, config: Dict[str, Any], trace_path: Path) -> Dict[str, Any]:
        """
        Generate runtime trace using real etcd raft cluster.
        
        Args:
            task_name: Name of the task
            config: Configuration for trace generation
            trace_path: Path where trace file should be saved
            
        Returns:
            Dictionary with generation results
        """
        try:
            # Extract configuration parameters with defaults
            node_count = config.get("node_count", 3)
            duration = config.get("duration_seconds", 60)
            client_qps = config.get("client_qps", 10.0)
            fault_rate = config.get("fault_rate", 0.1)
            scenario = config.get("scenario", "normal_operation")
            
            print(f"Generating real trace: {node_count} nodes, {duration}s duration, {client_qps} QPS")
            
            # Initialize real etcd raft cluster
            trace_logger = FileTraceLogger(str(trace_path))
            cluster = RaftCluster(node_count, trace_logger)
            
            # Initialize event driver with scenario
            driver = RandomEventDriver(cluster, config)
            driver.set_scenario(scenario)
            
            # Start cluster
            cluster.start()
            
            # Run trace generation
            start_time = datetime.now()
            driver.run_scenario(duration)
            generation_duration = (datetime.now() - start_time).total_seconds()
            
            # Stop cluster and finalize trace
            cluster.stop()
            event_count = trace_logger.get_event_count()
            
            print(f"Generated {event_count} events in {generation_duration:.2f}s")
            
            return {
                "success": True,
                "trace_file": str(trace_path),
                "event_count": event_count,
                "duration": generation_duration,
                "cluster_size": node_count
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Trace generation failed: {str(e)}"
            }
    
    def _generate_spectrace_from_tla(self, task_name: str, timestamp: str) -> Dict[str, Any]:
        """
        Generate specTrace.tla and specTrace.cfg from existing TLA+ specification.
        
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
            
            # Read TLA+ specification
            with open(spec_file, 'r', encoding='utf-8') as f:
                tla_content = f.read()
            
            # Generate configuration using LLM
            config_data = generate_config_from_tla(tla_content, task_name, "gpt-4")
            
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
    
    def _convert_trace_to_spec_format(self, sys_trace_path: Path, timestamp: str) -> Dict[str, Any]:
        """
        Convert system trace to TLA+ specification-compatible format.
        
        This converts the raw system trace (NDJSON) to the format expected
        by TLA+ specifications for trace validation.
        
        Args:
            sys_trace_path: Path to system-generated trace file
            timestamp: Timestamp for output file naming
            
        Returns:
            Dictionary with conversion results
        """
        try:
            # Create output directory for converted traces
            converted_traces_dir = Path("data/traces/etcd")
            converted_traces_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate output path for converted trace
            converted_trace_path = converted_traces_dir / f"etcd_converted_{timestamp}.ndjson"
            
            print(f"Converting trace from {sys_trace_path} to {converted_trace_path}")
            
            # Initialize trace converter
            converter = TraceConverter()
            
            # Perform conversion
            result = converter.convert_trace(
                input_trace_path=str(sys_trace_path),
                output_trace_path=str(converted_trace_path)
            )
            
            if result["success"]:
                return {
                    "success": True,
                    "input_events": result["input_events"],
                    "output_transitions": result["output_transitions"],
                    "converted_trace": result["output_file"]
                }
            else:
                return {
                    "success": False,
                    "error": result["error"]
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": f"Trace conversion failed: {str(e)}"
            }
    
    def _run_tlc_verification(self, trace_path: Path, spec_dir: str) -> Dict[str, Any]:
        """
        Run TLC verification of trace against converted specification.
        
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
        return ["etcd"]  # Currently only supports etcd raft
    
    def get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for trace validation evaluation."""
        return {
            "node_count": 3,
            "duration_seconds": 60,
            "client_qps": 10.0,
            "fault_rate": 0.1,
            "scenario": "normal_operation",
            "enable_network_faults": True,
            "enable_node_restart": True
        }
    
    def get_available_scenarios(self) -> Dict[str, Dict[str, Any]]:
        """Get available predefined scenarios."""
        # Create a dummy driver to get scenario configs
        dummy_cluster = None
        driver = RandomEventDriver(dummy_cluster, {})
        
        scenarios = {}
        scenario_names = ["normal_operation", "light_faults", "heavy_faults", 
                         "high_load", "partition_focused"]
        
        for name in scenario_names:
            scenarios[name] = driver.get_scenario_config(name)
            
        return scenarios
    
    def _get_evaluation_type(self) -> str:
        """Return the evaluation type identifier"""
        return "consistency_trace_validation"


# Convenience function for backward compatibility
def create_trace_validation_evaluator(
    spec_dir: str = "data/spec",
    traces_dir: str = "data/sys_traces/etcd",
    raft_repo_dir: str = "data/repositories/raft",
    timeout: int = 600
) -> TraceValidationEvaluator:
    """
    Factory function to create a trace validation evaluator.
    
    Args:
        spec_dir: Directory containing TLA+ specifications
        traces_dir: Directory to store generated traces
        raft_repo_dir: Directory containing etcd raft repository
        timeout: Timeout for evaluation operations in seconds
        
    Returns:
        TraceValidationEvaluator instance
    """
    return TraceValidationEvaluator(spec_dir, traces_dir, raft_repo_dir, timeout)