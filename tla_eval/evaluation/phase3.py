"""
Phase 3 Evaluator: Real Trace Generation and Validation

This module implements Phase 3 evaluation using real etcd raft clusters
to generate runtime traces, then validates them against TLA+ specifications.
"""

import os
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, Any
from datetime import datetime

from ..trace_generator.etcd.cluster import RaftCluster, FileTraceLogger
from ..trace_generator.etcd.event_driver import RandomEventDriver


class Phase3Evaluator:
    """
    Phase 3 evaluator for real trace generation and validation.
    
    This evaluator:
    1. Sets up a real etcd raft cluster using rafttest.InteractionEnv
    2. Generates runtime traces through random operations and fault injection
    3. Validates traces against TLA+ specifications using existing tools
    4. Returns comprehensive evaluation results
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
            trace_result = self._generate_real_trace(task_name, config, trace_path)
            
            if not trace_result["success"]:
                return {
                    "task_name": task_name,
                    "method_name": "phase3_real_trace_generation",
                    "success": False,
                    "error": trace_result["error"],
                    "duration": (datetime.now() - start_time).total_seconds()
                }
            
            print(f"Trace generation successful: {trace_result['event_count']} events")
            
            # Step 2: Validate trace against TLA+ specification
            validation_result = self._validate_trace_with_tla(task_name, trace_path)
            
            # Compile final result
            result = {
                "task_name": task_name,
                "method_name": "phase3_real_trace_generation",
                "success": validation_result["success"],
                "trace_file": str(trace_path),
                "trace_events": trace_result["event_count"],
                "cluster_nodes": config.get("node_count", 3),
                "generation_duration": trace_result["duration"],
                "total_duration": (datetime.now() - start_time).total_seconds(),
                "validation_result": validation_result["result"] if validation_result["success"] else "FAILED"
            }
            
            # Add detailed information
            if "generator_output" in trace_result:
                result["generator_details"] = trace_result["generator_output"]
            
            if validation_result["success"]:
                result["validation_details"] = validation_result.get("details", "")
            else:
                result["error"] = validation_result.get("error", "Trace validation failed")
                result["validation_error"] = validation_result.get("details", "")
                
            return result
            
        except Exception as e:
            return {
                "task_name": task_name,
                "method_name": "phase3_real_trace_generation", 
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
    
    def _validate_trace_with_tla(self, task_name: str, trace_path: Path) -> Dict[str, Any]:
        """
        Validate generated trace against TLA+ specification using etcd's validation tools.
        
        Args:
            task_name: Name of the task
            trace_path: Path to the trace file
            
        Returns:
            Dictionary with validation results
        """
        try:
            # Use the existing TLA+ validation tools from etcd raft
            tla_dir = self.raft_repo_dir / "tla"
            
            if not tla_dir.exists():
                return {
                    "success": False,
                    "error": f"TLA+ directory not found: {tla_dir}"
                }
            
            # Use the etcd raft validation script
            validate_script = tla_dir / "validate.sh"
            if not validate_script.exists():
                return {
                    "success": False,
                    "error": f"TLA+ validation script not found: {validate_script}"
                }
            
            print(f"Running TLA+ trace validation using etcd's tools...")
            
            # Run the validation script with our trace file
            cmd = [
                "bash", str(validate_script),
                str(trace_path)  # Pass our trace file
            ]
            
            # Set working directory to TLA directory
            result = subprocess.run(
                cmd,
                cwd=str(tla_dir),
                capture_output=True,
                text=True,
                timeout=600  # 10 minute timeout for TLA+ validation
            )
            
            if result.returncode == 0:
                return {
                    "success": True,
                    "result": "PASS",
                    "details": "Real trace validation completed successfully using etcd TLA+ tools",
                    "output": result.stdout
                }
            else:
                return {
                    "success": False,
                    "result": "FAILED",
                    "error": f"TLA+ validation failed with return code {result.returncode}",
                    "details": result.stderr,
                    "output": result.stdout
                }
                
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "error": "TLA+ trace validation timed out after 10 minutes"
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"TLA+ trace validation error: {str(e)}"
            }
    
    def get_evaluation_name(self) -> str:
        """Get the name of this evaluation method."""
        return "phase3_real_trace_generation"
    
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