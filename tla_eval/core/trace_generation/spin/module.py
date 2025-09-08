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
        Generate runtime traces using pre-generated SpinLock traces from Asterinas.
        
        Args:
            config: Configuration for trace generation
            output_path: Base path where trace files should be saved
            
        Returns:
            Dictionary with generation results including list of traces
        """
        try:
            # Extract configuration parameters with defaults
            trace_ids = config.get("trace_ids", [1])  # List of trace IDs to generate
            batch_size = config.get("batch_size", 20)  # How many traces to generate
            scenario_type = config.get("scenario_type", "all")  # "all", "specific", or scenario name
            
            # Handle different generation modes
            if scenario_type == "all":
                # Generate all 20 traces
                trace_ids = list(range(1, 21))
            elif isinstance(trace_ids, int):
                # Single trace ID provided
                trace_ids = [trace_ids]
            elif not trace_ids:
                # Default to first batch_size traces
                trace_ids = list(range(1, min(batch_size + 1, 21)))
            
            print(f"Generating {len(trace_ids)} SpinLock traces: {trace_ids}")
            
            # Start batch generation
            start_time = datetime.now()
            results = self._load_multiple_traces(trace_ids, output_path, config)
            generation_duration = (datetime.now() - start_time).total_seconds()
            
            if results["success"]:
                # For backward compatibility with single-trace evaluators
                # Provide both batch and single-trace compatible formats
                total_events = sum(trace.get("event_count", 0) for trace in results["traces"])
                
                # Create a combined trace file if multiple traces exist
                if len(results["traces"]) == 1:
                    # Single trace mode - direct compatibility
                    single_trace = results["traces"][0]
                    primary_trace_file = single_trace["trace_file"]
                else:
                    # Multiple traces mode - create a combined file for compatibility
                    combined_trace_path = output_path.parent / f"{output_path.stem}_combined.jsonl"
                    self._create_combined_trace_file(results["traces"], combined_trace_path)
                    primary_trace_file = str(combined_trace_path)
                
                return {
                    "success": True,
                    # Batch interface (new)
                    "traces": results["traces"],
                    "total_traces": len(results["traces"]),
                    "total_events": total_events,
                    # Single trace interface (backward compatibility)
                    "event_count": total_events,
                    "trace_file": primary_trace_file,
                    # Common fields
                    "duration": generation_duration,
                    "metadata": {
                        "batch_size": len(trace_ids),
                        "sync_primitive": "spinlock", 
                        "source": "asterinas_kernel",
                        "generation_mode": scenario_type
                    }
                }
            else:
                return {
                    "success": False,
                    "error": results["error"],
                    "duration": generation_duration,
                    "partial_results": results.get("traces", [])
                }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"SpinLock trace generation failed: {str(e)}"
            }
    
    def _load_multiple_traces(self, trace_ids: list, base_output_path: Path, config: Dict[str, Any]) -> Dict[str, Any]:
        """Load multiple pre-generated SpinLock traces from the data directory."""
        try:
            traces = []
            errors = []
            
            # Ensure output directory exists
            if base_output_path.is_file():
                output_dir = base_output_path.parent
                base_name = base_output_path.stem
            else:
                output_dir = base_output_path
                base_name = "spin_trace"
                output_dir.mkdir(parents=True, exist_ok=True)
            
            for trace_id in trace_ids:
                try:
                    # Create individual output file
                    output_file = output_dir / f"{base_name}_{trace_id:02d}.jsonl"
                    
                    # Load individual trace
                    result = self._load_pregenerated_trace(trace_id, output_file, config)
                    
                    if result["success"]:
                        traces.append({
                            "trace_id": trace_id,
                            "trace_file": str(output_file),
                            "event_count": result["event_count"],
                            "scenario_type": result.get("scenario_type", "Unknown"),
                            "source_file": result.get("source_file", "")
                        })
                        print(f"Successfully loaded trace {trace_id} with {result['event_count']} events")
                    else:
                        errors.append(f"Trace {trace_id}: {result['error']}")
                        print(f"Failed to load trace {trace_id}: {result['error']}")
                        
                except Exception as e:
                    error_msg = f"Trace {trace_id}: {str(e)}"
                    errors.append(error_msg)
                    print(f"Error loading trace {trace_id}: {str(e)}")
            
            if traces:
                return {
                    "success": True,
                    "traces": traces,
                    "errors": errors if errors else []
                }
            else:
                return {
                    "success": False,
                    "error": f"No traces could be loaded. Errors: {'; '.join(errors)}",
                    "traces": []
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to load multiple traces: {str(e)}",
                "traces": []
            }
    
    def _create_combined_trace_file(self, traces: list, combined_path: Path) -> None:
        """Create a combined trace file from multiple individual trace files."""
        try:
            with open(combined_path, 'w') as combined_file:
                # Add header
                combined_file.write("# Combined trace file from multiple SpinLock traces\n")
                
                global_seq = 0
                for trace_info in traces:
                    trace_file = Path(trace_info["trace_file"])
                    if trace_file.exists():
                        combined_file.write(f"# === TRACE_{trace_info['trace_id']}: {trace_info['scenario_type']} ===\n")
                        
                        with open(trace_file, 'r') as f:
                            for line in f:
                                line = line.strip()
                                if line and not line.startswith('#'):
                                    try:
                                        # Parse and renumber sequence for global ordering
                                        import json
                                        event = json.loads(line)
                                        event['seq'] = global_seq
                                        event['original_trace_id'] = trace_info['trace_id']
                                        combined_file.write(json.dumps(event) + '\n')
                                        global_seq += 1
                                    except json.JSONDecodeError:
                                        continue
                        
                        combined_file.write(f"# === End of TRACE_{trace_info['trace_id']} ===\n")
                        
        except Exception as e:
            print(f"Warning: Failed to create combined trace file: {e}")
    
    def _load_pregenerated_trace(self, trace_id, output_path: Path, config: Dict[str, Any]) -> Dict[str, Any]:
        """Load a pre-generated SpinLock trace from the data directory."""
        try:
            import random
            
            # Handle random trace selection
            if trace_id == "random":
                trace_id = random.randint(1, 20)
                print(f"Randomly selected trace {trace_id}")
            
            # Ensure trace_id is within valid range
            trace_id = max(1, min(20, int(trace_id)))
            
            # Path to pre-generated traces (go up to project root, then to data)
            project_root = Path(__file__).parent.parent.parent.parent.parent  # Go up to project root
            traces_dir = project_root / "data" / "sys_traces" / "spin"
            trace_file = traces_dir / f"trace_{trace_id:02d}.jsonl"
            summary_file = traces_dir / "traces_summary.json"
            
            if not trace_file.exists():
                return {
                    "success": False,
                    "error": f"Pre-generated trace file not found: {trace_file}"
                }
            
            # Load trace events
            trace_events = []
            scenario_type = "Unknown"
            
            with open(trace_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line.startswith('#'):
                        # Extract scenario type from header
                        if 'SpinLock' in line:
                            scenario_type = line.split('SpinLock', 1)[1].strip()
                        continue
                    elif line:
                        try:
                            event = json.loads(line)
                            trace_events.append(event)
                        except json.JSONDecodeError:
                            continue
            
            # Copy trace to output file
            shutil.copy2(trace_file, output_path)
            
            event_count = len(trace_events)
            
            if event_count > 0:
                print(f"Successfully loaded {event_count} trace events from {trace_file}")
                return {
                    "success": True,
                    "event_count": event_count,
                    "scenario_type": scenario_type,
                    "source_file": str(trace_file)
                }
            else:
                return {
                    "success": False,
                    "error": f"No valid trace events found in {trace_file}",
                    "event_count": 0
                }
                    
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to load pre-generated trace: {str(e)}"
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
            "trace_ids": list(range(1, 21)),  # All 20 pre-generated traces
            "batch_size": 20,  # Generate all traces by default
            "scenario_type": "all",  # Generate all scenarios
            "source": "asterinas_kernel",
            "enable_tla_trace": True
        }
    
    def get_available_scenarios(self) -> Dict[str, Dict[str, Any]]:
        """Get available predefined scenarios for SpinLock testing."""
        return {
            "all_traces": {
                "trace_ids": list(range(1, 21)),
                "scenario_type": "all",
                "description": "All 20 pre-generated SpinLock traces from Asterinas kernel"
            },
            "basic_operations": {
                "trace_ids": [4, 8, 12, 16, 20],  # Basic Lock Operations scenarios
                "scenario_type": "Basic Lock Operations",
                "description": "Simple lock/unlock sequences with multiple threads and locks"
            },
            "try_lock_mixed": {
                "trace_ids": [1, 5, 9, 13, 17],  # Try Lock Mixed Operations scenarios
                "scenario_type": "Try Lock Mixed Operations", 
                "description": "Mix of blocking and non-blocking lock attempts"
            },
            "multi_contention": {
                "trace_ids": [2, 6, 10, 14, 18],  # Multi Lock Contention scenarios
                "scenario_type": "Multi Lock Contention",
                "description": "Multiple threads competing for the same lock"
            },
            "complex_patterns": {
                "trace_ids": [3, 7, 11, 15, 19],  # Complex Lock Patterns scenarios
                "scenario_type": "Complex Lock Patterns",
                "description": "Complex sequences with multiple locks and varied operations"
            },
            "sample_set": {
                "trace_ids": [1, 3, 5, 8, 12, 15, 18, 20],  # Representative sample
                "scenario_type": "sample",
                "description": "Representative sample of different scenario types"
            },
            "single_trace": {
                "trace_ids": [1],  # Single trace for testing
                "scenario_type": "single",
                "description": "Single trace for quick testing"
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