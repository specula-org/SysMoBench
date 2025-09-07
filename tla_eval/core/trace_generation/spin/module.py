"""
Asterinas SpinLock system implementation for trace generation and conversion.

This module implements the system-specific interfaces for Asterinas SpinLock
trace generation and format conversion using real kernel tests.
"""

import subprocess
import json
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any
from datetime import datetime

from ..base import TraceGenerator, TraceConverter, SystemModule


class SpinLockTraceGenerator(TraceGenerator):
    """Asterinas SpinLock-specific trace generator implementation."""
    
    def generate_trace(self, config: Dict[str, Any], output_path: Path) -> Dict[str, Any]:
        """
        Generate runtime trace using real Asterinas SpinLock kernel tests.
        
        Args:
            config: Configuration for trace generation
            output_path: Path where trace file should be saved
            
        Returns:
            Dictionary with generation results
        """
        try:
            # Extract configuration parameters with defaults
            duration = config.get("duration_seconds", 10)
            thread_count = config.get("thread_count", 3)
            iterations = config.get("iterations_per_thread", 5)
            contention_level = config.get("contention_level", "medium")
            
            print(f"Generating SpinLock trace: {thread_count} threads, {iterations} iterations each")
            
            # Path to Asterinas repository
            asterinas_path = config.get("asterinas_path", "asterinas")
            
            # Start trace generation
            start_time = datetime.now()
            result = self._run_asterinas_spinlock_test(asterinas_path, output_path, config)
            generation_duration = (datetime.now() - start_time).total_seconds()
            
            if result["success"]:
                return {
                    "success": True,
                    "trace_file": str(output_path),
                    "event_count": result["event_count"],
                    "duration": generation_duration,
                    "metadata": {
                        "thread_count": thread_count,
                        "iterations_per_thread": iterations,
                        "contention_level": contention_level,
                        "sync_primitive": "spinlock"
                    }
                }
            else:
                return {
                    "success": False,
                    "error": result["error"],
                    "duration": generation_duration
                }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"SpinLock trace generation failed: {str(e)}"
            }
    
    def _run_asterinas_spinlock_test(self, asterinas_path: str, output_path: Path, config: Dict[str, Any]) -> Dict[str, Any]:
        """Run the actual Asterinas SpinLock kernel test with TLA+ tracing."""
        try:
            # Change to Asterinas directory
            original_cwd = Path.cwd()
            asterinas_path_obj = Path(asterinas_path)
            
            if not asterinas_path_obj.exists():
                return {
                    "success": False,
                    "error": f"Asterinas path does not exist: {asterinas_path}"
                }
            
            # Create temporary file for capturing trace output
            with tempfile.NamedTemporaryFile(mode='w+', suffix='.log', delete=False) as trace_file:
                trace_temp_path = trace_file.name
            
            try:
                # Change to Asterinas directory for build
                import os
                os.chdir(asterinas_path)
                
                # Run kernel test with TLA+ tracing enabled
                cmd = [
                    "timeout", "60",  # 60 second timeout
                    "make", "ktest", 
                    f"FEATURES=tla-trace"
                ]
                
                print(f"Running command: {' '.join(cmd)}")
                
                # Run the test and capture output
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=120  # 2 minute timeout
                )
                
                # Parse trace output from stdout/stderr
                trace_events = self._parse_trace_output(result.stdout + result.stderr)
                
                # Write trace events to output file
                with open(output_path, 'w') as f:
                    for event in trace_events:
                        f.write(json.dumps(event) + '\n')
                
                event_count = len(trace_events)
                
                if result.returncode == 0 and event_count > 0:
                    print(f"Successfully generated {event_count} trace events")
                    return {
                        "success": True,
                        "event_count": event_count,
                        "trace_output": result.stdout[:1000],  # First 1000 chars for debug
                        "trace_errors": result.stderr[:1000] if result.stderr else ""
                    }
                else:
                    return {
                        "success": False,
                        "error": f"Test failed with return code {result.returncode}. Stdout: {result.stdout[:500]}. Stderr: {result.stderr[:500]}",
                        "event_count": event_count
                    }
                    
            finally:
                # Always change back to original directory
                os.chdir(original_cwd)
                # Clean up temp file
                try:
                    Path(trace_temp_path).unlink(missing_ok=True)
                except:
                    pass
                    
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "error": "Asterinas kernel test timed out"
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to run Asterinas test: {str(e)}"
            }
    
    def _parse_trace_output(self, output: str) -> list:
        """Parse TLA+ trace events from kernel test output."""
        events = []
        
        for line in output.split('\n'):
            line = line.strip()
            
            # Look for JSON trace events
            if line.startswith('{"seq":') and '"action":' in line and '"lock":' in line:
                try:
                    event = json.loads(line)
                    # Validate expected fields
                    if all(field in event for field in ['seq', 'lock', 'actor', 'action']):
                        events.append(event)
                except json.JSONDecodeError:
                    continue
        
        # Sort by sequence number to ensure correct ordering
        events.sort(key=lambda x: x.get('seq', 0))
        return events
    
    def get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for SpinLock trace generation."""
        return {
            "duration_seconds": 10,
            "thread_count": 3,
            "iterations_per_thread": 5,
            "contention_level": "medium",
            "asterinas_path": "data/repositories/asterinas",
            "enable_tla_trace": True
        }
    
    def get_available_scenarios(self) -> Dict[str, Dict[str, Any]]:
        """Get available predefined scenarios for SpinLock testing."""
        return {
            "light_contention": {
                "thread_count": 2,
                "iterations_per_thread": 3,
                "duration_seconds": 5,
                "contention_level": "light"
            },
            "medium_contention": {
                "thread_count": 3,
                "iterations_per_thread": 5,
                "duration_seconds": 10,
                "contention_level": "medium"
            },
            "heavy_contention": {
                "thread_count": 5,
                "iterations_per_thread": 8,
                "duration_seconds": 15,
                "contention_level": "heavy"
            },
            "stress_test": {
                "thread_count": 8,
                "iterations_per_thread": 10,
                "duration_seconds": 30,
                "contention_level": "extreme"
            }
        }


class SpinLockTraceConverter(TraceConverter):
    """SpinLock-specific trace converter implementation."""
    
    def convert_trace(self, input_path: Path, output_path: Path) -> Dict[str, Any]:
        """
        Convert SpinLock trace to TLA+ specification-compatible format.
        
        Args:
            input_path: Path to the raw SpinLock trace file
            output_path: Path where converted trace should be saved
            
        Returns:
            Dictionary with conversion results
        """
        try:
            print(f"Converting SpinLock trace from {input_path} to {output_path}")
            
            # Read input trace events
            events = []
            with open(input_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            event = json.loads(line)
                            events.append(event)
                        except json.JSONDecodeError:
                            continue
            
            if not events:
                return {
                    "success": False,
                    "error": "No valid trace events found in input file"
                }
            
            # Convert to TLA+ format
            tla_transitions = self._convert_to_tla_format(events)
            
            # Write converted trace
            with open(output_path, 'w') as f:
                for transition in tla_transitions:
                    f.write(json.dumps(transition) + '\n')
            
            return {
                "success": True,
                "input_events": len(events),
                "output_transitions": len(tla_transitions),
                "output_file": str(output_path)
            }
                
        except Exception as e:
            return {
                "success": False,
                "error": f"SpinLock trace conversion failed: {str(e)}"
            }
    
    def _convert_to_tla_format(self, events: list) -> list:
        """Convert raw SpinLock events to TLA+ state transitions."""
        transitions = []
        
        # Map of action names to TLA+ equivalents
        action_map = {
            "TryAcquireBlocking": "TryLock",
            "TryAcquireNonBlocking": "TryLockNonBlocking", 
            "AcquireSuccess": "LockAcquired",
            "AcquireFailNonBlocking": "TryLockFailed",
            "Spinning": "Waiting",
            "Release": "Unlock"
        }
        
        for event in events:
            tla_event = {
                "step": event.get("seq", 0),
                "actor": f"Thread{event.get('actor', 0)}",
                "lock": f"Lock{event.get('lock', 0)}",
                "action": action_map.get(event.get('action', ''), event.get('action', '')),
                "timestamp": event.get("timestamp", ""),
                "metadata": {
                    "original_action": event.get('action', ''),
                    "sync_primitive": "SpinLock"
                }
            }
            transitions.append(tla_event)
        
        return transitions


class SpinLockSystemModule(SystemModule):
    """Complete SpinLock system implementation."""
    
    def __init__(self):
        self._trace_generator = SpinLockTraceGenerator()
        self._trace_converter = SpinLockTraceConverter()
    
    def get_trace_generator(self) -> TraceGenerator:
        """Get the SpinLock trace generator."""
        return self._trace_generator
    
    def get_trace_converter(self) -> TraceConverter:
        """Get the SpinLock trace converter."""
        return self._trace_converter
    
    def get_system_name(self) -> str:
        """Get the system name identifier."""
        return "spin"


def get_system() -> SystemModule:
    """
    Factory function to get the SpinLock system implementation.
    
    This function is called by the system registry to load this system.
    
    Returns:
        SpinLockSystemModule instance
    """
    return SpinLockSystemModule()