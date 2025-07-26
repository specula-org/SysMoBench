"""
Cluster management for etcd raft trace generation.

This module provides a Python interface to the Go-based trace generator
that uses real etcd raft components.
"""

import json
import subprocess
import tempfile
import os
from pathlib import Path
from typing import Dict, Any, Optional
import time


class RaftCluster:
    """
    Python interface to etcd raft cluster using the Go trace generator.
    
    This class manages the lifecycle of a real etcd raft cluster through
    the Go-based trace generator that uses rafttest.InteractionEnv.
    """
    
    def __init__(self, 
                 node_count: int = 3,
                 trace_logger: Optional['FileTraceLogger'] = None):
        """
        Initialize raft cluster manager.
        
        Args:
            node_count: Number of nodes in the cluster
            trace_logger: Optional trace logger (will be created if None)
        """
        self.node_count = node_count
        self.trace_logger = trace_logger
        self.process = None
        self.trace_file = None
        
        # Path to the Go trace generator
        current_dir = Path(__file__).parent
        self.generator_dir = current_dir
        self.binary_path = self.generator_dir / "trace_generator"
        
    def start_cluster(self) -> bool:
        """
        Start the raft cluster.
        
        Returns:
            True if cluster started successfully, False otherwise
        """
        try:
            # Ensure the Go binary is built
            if not self._build_generator():
                return False
                
            print(f"Raft cluster with {self.node_count} nodes ready for operations")
            return True
            
        except Exception as e:
            print(f"Failed to start cluster: {e}")
            return False
    
    def stop_cluster(self):
        """Stop the raft cluster and cleanup resources."""
        if self.process and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        
        self.process = None
    
    def generate_trace(self, 
                      duration_seconds: int = 60,
                      client_qps: float = 10.0,
                      fault_rate: float = 0.1,
                      output_file: str = None) -> Dict[str, Any]:
        """
        Generate a runtime trace by running the cluster for specified duration.
        
        Args:
            duration_seconds: How long to run the cluster
            client_qps: Client operations per second
            fault_rate: Rate of fault injection (0.0 to 1.0)
            output_file: Path for trace output (auto-generated if None)
            
        Returns:
            Dictionary with generation results
        """
        if output_file is None:
            timestamp = int(time.time())
            output_file = f"/home/ubuntu/LLM_Gen_TLA_benchmark_framework/data/sys_traces/etcd/etcd_trace_{timestamp}.ndjson"
        
        # Build command arguments
        cmd = [
            str(self.binary_path),
            "-nodes", str(self.node_count),
            "-duration", str(duration_seconds), 
            "-qps", str(client_qps),
            "-fault-rate", str(fault_rate),
            "-output", output_file
        ]
        
        start_time = time.time()
        
        try:
            print(f"Running trace generation: {' '.join(cmd)}")
            
            # Run the Go trace generator
            result = subprocess.run(
                cmd,
                cwd=str(self.generator_dir),
                capture_output=True,
                text=True,
                timeout=duration_seconds + 30  # Extra time for setup/cleanup
            )
            
            end_time = time.time()
            
            if result.returncode == 0:
                # Count events in generated trace
                event_count = self._count_trace_events(output_file)
                
                return {
                    "success": True,
                    "trace_file": output_file,
                    "event_count": event_count,
                    "duration": end_time - start_time,
                    "generator_output": result.stdout
                }
            else:
                return {
                    "success": False,
                    "error": f"Trace generator failed with return code {result.returncode}",
                    "stderr": result.stderr,
                    "stdout": result.stdout
                }
                
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "error": f"Trace generation timed out after {duration_seconds + 30} seconds"
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to run trace generator: {str(e)}"
            }
    
    def _build_generator(self) -> bool:
        """
        Build the Go trace generator if needed.
        
        Returns:
            True if build successful, False otherwise
        """
        try:
            # Check if binary already exists and is recent
            if (self.binary_path.exists() and 
                self.binary_path.stat().st_mtime > time.time() - 3600):  # 1 hour
                return True
            
            print("Building Go trace generator...")
            
            # Update go.mod with correct relative path to raft repository
            self._update_go_mod_paths()
            
            # Run go mod tidy first
            subprocess.run(
                ["go", "mod", "tidy"],
                cwd=str(self.generator_dir),
                check=True,
                capture_output=True
            )
            
            # Build with TLA tracing enabled
            build_result = subprocess.run([
                "go", "build", 
                "-tags", "with_tla",
                "-o", str(self.binary_path),
                "./cmd/trace_generator.go"
            ], 
            cwd=str(self.generator_dir),
            capture_output=True,
            text=True
            )
            
            if build_result.returncode != 0:
                print(f"Build failed: {build_result.stderr}")
                return False
                
            print("Go trace generator built successfully")
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"Failed to build trace generator: {e}")
            return False
        except Exception as e:
            print(f"Unexpected error during build: {e}")
            return False
    
    def _count_trace_events(self, trace_file: str) -> int:
        """Count number of events in trace file."""
        try:
            if not os.path.exists(trace_file):
                return 0
                
            count = 0
            with open(trace_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:  # Skip empty lines
                        count += 1
            return count
        except Exception:
            return 0
    
    def _update_go_mod_paths(self):
        """Update go.mod file with correct relative paths that work for any user."""
        go_mod_path = self.generator_dir / "go.mod"
        
        # Calculate relative path from generator directory to raft repository
        # From: tla_eval/core/trace_generation/etcd/
        # To: data/repositories/raft
        relative_raft_path = "../../../../data/repositories/raft"
        
        # Read current go.mod content
        with open(go_mod_path, 'r') as f:
            content = f.read()
        
        # Replace the raft repository path with correct relative path
        import re
        # Match the replace line and update it
        pattern = r'replace go\.etcd\.io/raft/v3 => .*'
        replacement = f'replace go.etcd.io/raft/v3 => {relative_raft_path}'
        
        updated_content = re.sub(pattern, replacement, content)
        
        # Write updated content back
        with open(go_mod_path, 'w') as f:
            f.write(updated_content)
        
        print(f"Updated go.mod with raft path: {relative_raft_path}")


class FileTraceLogger:
    """
    Simple file-based trace logger for compatibility.
    
    Note: The actual tracing is handled by the Go program,
    this is just for interface compatibility.
    """
    
    def __init__(self, output_file: str):
        """
        Initialize file trace logger.
        
        Args:
            output_file: Path to output trace file
        """
        self.output_file = output_file
        
    def close(self):
        """Close the trace logger."""
        pass  # Nothing to close in this implementation