"""
Phase 3 Evaluator: Complete Trace Generation and Validation Pipeline

This module implements the complete Phase 3 evaluation pipeline:
1. Real trace generation from etcd raft clusters
2. LLM-based configuration generation for trace validation
3. Static analysis conversion of TLA+ specs to trace format  
4. TLC verification of traces against converted specifications
"""

import os
import subprocess
import tempfile
import yaml
from pathlib import Path
from typing import Dict, Any
from datetime import datetime

from ...core.trace_generation.etcd.cluster import RaftCluster, FileTraceLogger
from ...core.trace_generation.etcd.event_driver import RandomEventDriver
from ...core.spec_processing import SpecTraceGenerator, generate_config_from_tla
from ...core.verification import TLCRunner


class Phase3Evaluator:
    """
    Complete Phase 3 evaluator with 4-step pipeline.
    
    This evaluator implements the complete Phase 3 workflow:
    1. **Trace Generation**: Real etcd raft cluster trace generation
    2. **Config Generation**: LLM-based YAML configuration from TLA+ specs
    3. **Spec Conversion**: Static analysis conversion to trace-validation format
    4. **TLC Verification**: Trace validation against converted specifications
    """
    
    def __init__(self, 
                 spec_dir: str = "data/spec",
                 traces_dir: str = "data/sys_traces/etcd",
                 raft_repo_dir: str = "data/repositories/raft"):
        """
        Initialize Phase 3 evaluator.
        
        Args:
            spec_dir: Directory containing TLA+ specifications
            traces_dir: Directory to store generated traces
            raft_repo_dir: Directory containing etcd raft repository
        """
        self.spec_dir = Path(spec_dir)
        self.traces_dir = Path(traces_dir)
        self.raft_repo_dir = Path(raft_repo_dir)
        
        # Ensure directories exist
        self.traces_dir.mkdir(parents=True, exist_ok=True)
        
    def evaluate(self, task_name: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run Phase 3 evaluation for a given task.
        
        Args:
            task_name: Name of the task (e.g., "etcd")
            config: Configuration parameters for trace generation
            
        Returns:
            Dictionary containing evaluation results
        """
        start_time = datetime.now()
        
        print(f"Starting Phase 3 evaluation for task: {task_name}")
        print(f"Configuration: {config}")
        
        try:
            # Generate trace file name
            timestamp = start_time.strftime("%Y%m%d_%H%M%S")
            trace_filename = f"{task_name}_trace_{timestamp}.ndjson"
            trace_path = self.traces_dir / trace_filename
            
            # Step 1: Generate runtime trace using real etcd raft cluster
            print("Step 1: Generating runtime trace...")
            trace_result = self._generate_real_trace(task_name, config, trace_path)
            
            if not trace_result["success"]:
                return {
                    "task_name": task_name,
                    "method_name": "phase3_complete_pipeline",
                    "success": False,
                    "error": trace_result["error"],
                    "step_failed": "trace_generation",
                    "duration": (datetime.now() - start_time).total_seconds()
                }
            
            print(f"Step 1 completed: {trace_result['event_count']} events generated")
            
            # Step 2: Generate specTrace.tla from TLA+ spec (LLM + static analysis)
            print("Step 2: Generating specTrace.tla from TLA+ spec...")
            spectrace_result = self._generate_spectrace_from_tla(task_name, timestamp)
            
            if not spectrace_result["success"]:
                return {
                    "task_name": task_name,
                    "method_name": "phase3_complete_pipeline",
                    "success": False,
                    "error": spectrace_result["error"],
                    "step_failed": "spectrace_generation",
                    "duration": (datetime.now() - start_time).total_seconds()
                }
            
            print("Step 2 completed: specTrace.tla and specTrace.cfg generated")
            
            # Step 3: Convert sys_trace to spec-compatible format
            print("Step 3: Converting sys_trace to spec-compatible format...")
            # TODO: Implement trace conversion from sys_trace to spec format
            trace_conversion_result = {"success": True, "converted_trace": str(trace_path)}
            
            # Step 4: Run TLC verification
            print("Step 4: Running TLC verification...")
            verification_result = self._run_tlc_verification(trace_path, spectrace_result["output_dir"])
            
            # Compile final result
            result = {
                "task_name": task_name,
                "method_name": "phase3_complete_pipeline",
                "success": verification_result["success"],
                "trace_file": str(trace_path),
                "trace_events": trace_result["event_count"],
                "cluster_nodes": config.get("node_count", 3),
                "generation_duration": trace_result["duration"],
                "total_duration": (datetime.now() - start_time).total_seconds(),
                "config_file": spectrace_result.get("config_file", ""),
                "spec_trace_files": spectrace_result.get("files", {}),
                "verification_result": verification_result["result"] if verification_result["success"] else "FAILED"
            }
            
            # Add detailed information for each step
            if "generator_output" in trace_result:
                result["step1_details"] = trace_result["generator_output"]
            
            if "config_data" in spectrace_result:
                result["step2_details"] = f"Generated config with {len(spectrace_result['config_data'])} sections"
            
            result["step3_details"] = f"Trace conversion placeholder - TODO: implement trace converter"
            
            if verification_result["success"]:
                result["step4_details"] = verification_result.get("details", "")
            else:
                result["error"] = verification_result.get("error", "TLC verification failed")
                result["verification_error"] = verification_result.get("details", "")
                
            return result
            
        except Exception as e:
            return {
                "task_name": task_name,
                "method_name": "phase3_complete_pipeline", 
                "success": False,
                "error": f"Phase 3 evaluation failed: {str(e)}",
                "duration": (datetime.now() - start_time).total_seconds()
            }
    
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
            
            # Create event driver with scenario configuration
            driver = RandomEventDriver(cluster, config)
            if scenario != "custom":
                # Use predefined scenario configuration
                scenario_config = driver.get_scenario_config(scenario)
                driver = RandomEventDriver(cluster, scenario_config)
            
            # Start the cluster
            if not cluster.start_cluster():
                return {
                    "success": False,
                    "error": "Failed to start real etcd raft cluster"
                }
            
            try:
                # Run the event driver to generate realistic trace
                driver_result = driver.run_for_duration(duration)
                
                if driver_result["success"]:
                    return {
                        "success": True,
                        "event_count": driver_result["event_count"],
                        "duration": driver_result["duration"],
                        "trace_file": driver_result["trace_file"],
                        "generator_output": driver_result.get("generator_output", "")
                    }
                else:
                    return {
                        "success": False,
                        "error": f"Event driver failed: {driver_result.get('error', 'Unknown error')}"
                    }
                
            finally:
                # Always cleanup cluster
                cluster.stop_cluster()
                trace_logger.close()
                
        except Exception as e:
            return {
                "success": False,
                "error": f"Real trace generation failed: {str(e)}"
            }
    
    def _generate_spectrace_from_tla(self, task_name: str, timestamp: str) -> Dict[str, Any]:
        """
        Generate specTrace.tla and specTrace.cfg from TLA+ spec using LLM + static analysis.
        
        This combines:
        1. LLM-based YAML configuration generation from TLA+ spec
        2. Static analysis conversion to specTrace.tla format
        
        Args:
            task_name: Name of the task
            timestamp: Timestamp for output directory
            
        Returns:
            Dictionary with generation results
        """
        try:
            # Locate TLA+ and CFG files for the task
            task_dir = self.spec_dir / task_name
            tla_files = list(task_dir.glob("*.tla"))
            cfg_files = list(task_dir.glob("*.cfg"))
            
            if not tla_files:
                return {
                    "success": False,
                    "error": f"No TLA+ files found in {task_dir}"
                }
            
            if not cfg_files:
                return {
                    "success": False,
                    "error": f"No CFG files found in {task_dir}"
                }
            
            tla_file = str(tla_files[0])  # Use first TLA file found
            cfg_file = str(cfg_files[0])  # Use first CFG file found
            
            print(f"Using TLA+ file: {tla_file}")
            print(f"Using CFG file: {cfg_file}")
            
            # Step 2a: Generate configuration using LLM
            print("  2a: Generating YAML configuration with LLM...")
            config_data = generate_config_from_tla(tla_file, cfg_file)
            
            # Save YAML configuration for reference
            config_filename = f"trace_config_{task_name}.yaml"
            config_path = self.traces_dir / config_filename
            
            with open(config_path, 'w') as f:
                yaml.dump(config_data, f, default_flow_style=False)
            
            print(f"  Generated YAML config: {config_path}")
            
            # Step 2b: Convert to specTrace.tla using static analysis
            print("  2b: Converting to specTrace.tla with static analysis...")
            output_dir = self.traces_dir / f"spec_trace_{timestamp}"
            output_dir.mkdir(parents=True, exist_ok=True)
            
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
        
        Args:
            trace_path: Path to the trace file
            spec_dir: Directory containing specTrace.tla and specTrace.cfg
            
        Returns:
            Dictionary with verification results
        """
        tlc_runner = TLCRunner()
        return tlc_runner.run_verification(trace_path, spec_dir)
    
    def get_evaluation_name(self) -> str:
        """Get the name of this evaluation method."""
        return "phase3_complete_pipeline"
    
    def get_supported_tasks(self):
        """Get list of tasks supported by this evaluator."""
        return ["etcd"]  # Currently only supports etcd raft
    
    def get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for Phase 3 evaluation."""
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