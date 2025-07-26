"""
TLC Runner Module

This module handles running TLC model checker for trace validation.
"""

import os
import subprocess
from pathlib import Path
from typing import Dict, Any

class TLCRunner:
    """
    Handles TLC model checker execution for trace validation.
    """
    
    def __init__(self, tla_tools_jar: str = None):
        if tla_tools_jar is None:
            # Use absolute path to jar file in project root
            from pathlib import Path
            project_root = Path(__file__).parent.parent.parent.parent
            tla_tools_jar = str(project_root / "lib" / "tla2tools.jar")
        self.tla_tools_jar = tla_tools_jar
        self.tlc_available = self._check_tlc_availability()
    
    def _check_tlc_availability(self) -> bool:
        """Check if TLC tools are available."""
        return os.path.exists(self.tla_tools_jar)
    
    def run_verification(self, trace_path: Path, spec_dir: str) -> Dict[str, Any]:
        """
        Run TLC verification of trace against converted specification.
        
        Args:
            trace_path: Path to the trace file
            spec_dir: Directory containing specTrace.tla and specTrace.cfg
            
        Returns:
            Dictionary with verification results
        """
        try:
            # Check if TLC is available
            if not self.tlc_available:
                return {
                    "success": False,
                    "error": f"TLC tools not found at {self.tla_tools_jar}. TLC verification skipped.",
                    "result": "SKIPPED",
                    "details": "TLC tools are not installed. Please install TLA+ tools to enable verification."
                }
            
            spec_dir_path = Path(spec_dir)
            tla_file = spec_dir_path / "specTrace.tla"
            cfg_file = spec_dir_path / "specTrace.cfg"
            
            if not tla_file.exists() or not cfg_file.exists():
                return {
                    "success": False,
                    "error": f"Missing specTrace files in {spec_dir}"
                }
            
            print(f"Running TLC verification...")
            print(f"Trace file: {trace_path}")
            print(f"Spec directory: {spec_dir}")
            
            # Prepare environment variables for TLC
            env = os.environ.copy()
            # Set TRACE_PATH to the absolute path of the converted trace file
            env["TRACE_PATH"] = str(trace_path.resolve())
            print(f"Set TRACE_PATH environment variable to: {trace_path.resolve()}")
            
            # Run TLC with the generated specification
            cmd = [
                "java", "-cp", self.tla_tools_jar,
                "tlc2.TLC",
                "-config", str(cfg_file),
                str(tla_file)
            ]
            
            result = subprocess.run(
                cmd,
                cwd=str(spec_dir_path),
                capture_output=True,
                text=True,
                env=env,
                timeout=600  # 10 minute timeout
            )
            
            # Check if TLC succeeded
            if result.returncode == 0:
                return {
                    "success": True,
                    "result": "PASS",
                    "details": "TLC verification completed successfully",
                    "output": result.stdout
                }
            else:
                return {
                    "success": False,
                    "result": "FAILED",
                    "error": f"TLC verification failed with return code {result.returncode}",
                    "details": result.stderr,
                    "output": result.stdout
                }
                
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "error": "TLC verification timed out after 10 minutes"
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"TLC verification error: {str(e)}"
            }

__all__ = ['TLCRunner']