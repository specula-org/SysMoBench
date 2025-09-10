"""
SpinLock Trace Converter Implementation

Configuration-based implementation for SpinLock trace conversion.
Based on the etcd trace converter pattern but adapted for SpinLock traces.
"""

import json
import os
from typing import Dict, Any, List, Optional
from pathlib import Path
from collections import defaultdict


class SpinLockTraceConverterImpl:
    """
    Configuration-based implementation for SpinLock trace conversion.
    
    This handles the conversion from raw SpinLock traces (JSONL format)
    to the format expected by TLA+ specifications for validation.
    """
    
    def __init__(self, mapping_file: str = None):
        """
        Initialize trace converter with mapping configuration.
        
        Args:
            mapping_file: Path to JSON mapping file
        """
        if mapping_file is None:
            # Look in data/convertor/spin first, fallback to module directory
            data_mapping = "data/convertor/spin/spin_mapping.json"
            module_mapping = os.path.join(os.path.dirname(__file__), "spin_mapping.json")
            
            if os.path.exists(data_mapping):
                mapping_file = data_mapping
            else:
                mapping_file = module_mapping
        
        self.mapping_file = mapping_file
        self.mapping = self._load_mapping()
        
    def _load_mapping(self) -> Dict[str, Any]:
        """Load mapping configuration from JSON file."""
        try:
            with open(self.mapping_file, 'r') as f:
                mapping = json.load(f)
                print(f"Loaded SpinLock mapping configuration from: {self.mapping_file}")
                return mapping
        except Exception as e:
            print(f"Failed to load mapping file {self.mapping_file}: {e}")
            return self._get_default_mapping()
    
    def _get_default_mapping(self) -> Dict[str, Any]:
        """Get default mapping configuration if file loading fails."""
        return {
            "config": {
                "Threads": ["Thread0", "Thread1", "Thread2", "Thread3"],
                "Locks": ["Lock0", "Lock1", "Lock2"],
                "MaxSteps": 1000,
                "States": ["unlocked", "locked"],
                "Nil": "Nil"
            },
            "events": {
                "TryAcquireBlocking": "TryLock",
                "TryAcquireNonBlocking": "TryLockNonBlocking",
                "AcquireSuccess": "LockAcquired",
                "Release": "Unlock",
                "default": "Step"
            },
            "variables": {
                "lockState": {
                    "system_path": ["state"],
                    "value_mapping": {"unlocked": "Unlocked", "locked": "Locked"},
                    "default_value": "Unlocked"
                },
                "owner": {
                    "system_path": ["actor"],
                    "default_value": "Nil"
                }
            },
            "thread_mapping": {"0": "Thread0", "1": "Thread1", "2": "Thread2", "3": "Thread3"},
            "lock_mapping": {"0": "Lock0", "1": "Lock1", "2": "Lock2"}
        }
    
    def _extract_value_from_event(self, event: Dict[str, Any], path: List[str], default_value: Any = None) -> Any:
        """Extract value from event using dot notation path."""
        current = event
        try:
            for key in path:
                if isinstance(current, dict) and key in current:
                    current = current[key]
                else:
                    return default_value
            return current
        except:
            return default_value
    
    def _map_thread_id(self, thread_id: str) -> str:
        """Map system thread ID to TLA+ thread name."""
        thread_mapping = self.mapping.get("thread_mapping", {})
        return thread_mapping.get(str(thread_id), f"Thread{thread_id}")
    
    def _map_lock_id(self, lock_id: str) -> str:
        """Map system lock ID to TLA+ lock name."""
        lock_mapping = self.mapping.get("lock_mapping", {})
        return lock_mapping.get(str(lock_id), f"Lock{lock_id}")
    
    def _map_event_name(self, event_name: str) -> str:
        """Map system event name to TLA+ action name."""
        events_mapping = self.mapping.get("events", {})
        return events_mapping.get(event_name, events_mapping.get("default", "Step"))
    
    def _map_variable_value(self, var_config: Dict[str, Any], raw_value: Any, context: Dict[str, Any]) -> Any:
        """Map system variable value to TLA+ format."""
        # Apply value mapping if configured
        if "value_mapping" in var_config and raw_value is not None:
            value_mapping = var_config["value_mapping"]
            if str(raw_value) in value_mapping:
                return value_mapping[str(raw_value)]
        
        # Handle special cases
        if raw_value is None:
            return var_config.get("default_value", "Nil")
        
        # Convert numeric strings to numbers where appropriate
        if isinstance(raw_value, str) and raw_value.isdigit():
            return int(raw_value)
            
        return raw_value
    
    def _build_initial_state(self) -> Dict[str, Dict[str, Any]]:
        """Build initial state with default values for all locks."""
        state = defaultdict(lambda: defaultdict(dict))
        
        lock_mapping = self.mapping.get("lock_mapping", {})
        variables_config = self.mapping.get("variables", {})
        
        # Initialize state for all known locks
        for lock_id, lock_name in lock_mapping.items():
            for var_name, var_config in variables_config.items():
                state[var_name][lock_name] = var_config.get("default_value", "Nil")
        
        return state
    
    def _update_state_with_event(self, state: Dict[str, Dict[str, Any]], event: Dict[str, Any]) -> None:
        """Update state incrementally with a single event."""
        lock_id = event.get("lock", "0")
        lock_name = self._map_lock_id(lock_id)
        thread_id = event.get("thread", event.get("actor", "0"))
        thread_name = self._map_thread_id(thread_id)
        
        variables_config = self.mapping.get("variables", {})
        
        # Ensure lock exists in state
        if lock_name not in state.get("lockState", {}):
            for var_name, var_config in variables_config.items():
                state[var_name][lock_name] = var_config.get("default_value", "Nil")
        
        # Update each variable based on the event
        for var_name, var_config in variables_config.items():
            system_path = var_config.get("system_path", [])
            raw_value = self._extract_value_from_event(event, system_path)
            
            if raw_value is not None:
                context = {"thread": thread_name, "lock": lock_name, "event": event}
                mapped_value = self._map_variable_value(var_config, raw_value, context)
                
                # Special handling for owner variable
                if var_name == "owner":
                    action = event.get("action", "")
                    if action in ["AcquireSuccess", "LockAcquired"]:
                        state[var_name][lock_name] = thread_name
                    elif action in ["Release", "Unlock"]:
                        state[var_name][lock_name] = "Nil"
                    else:
                        state[var_name][lock_name] = mapped_value
                else:
                    state[var_name][lock_name] = mapped_value
    
    def convert_trace(self, input_trace_path: str, output_trace_path: str, config: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Convert trace from system format to spec format.
        
        Args:
            input_trace_path: Path to input trace file (JSONL)
            output_trace_path: Path for output trace file
            config: Configuration for conversion
            
        Returns:
            Dictionary with conversion results
        """
        try:
            print(f"Converting SpinLock trace: {input_trace_path} -> {output_trace_path}")
            
            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_trace_path), exist_ok=True)
            
            # Read input trace
            events = []
            with open(input_trace_path, 'r') as f:
                for line_num, line in enumerate(f):
                    line = line.strip()
                    if line and not line.startswith('#'):
                        try:
                            event = json.loads(line)
                            events.append(event)
                        except json.JSONDecodeError as e:
                            print(f"Skipping invalid JSON line {line_num}: {line[:50]}... Error: {e}")
                            continue
            
            if not events:
                return {
                    "success": False,
                    "error": "No valid events found in input trace"
                }
            
            print(f"Processing {len(events)} trace events...")
            
            # Convert to TLA+ format
            output_lines = []
            
            # First line: configuration
            config_line = json.dumps(self.mapping.get("config", {}))
            output_lines.append(config_line)
            
            # Process events to generate state transitions
            state = self._build_initial_state()
            
            for event_num, event in enumerate(events):
                # Update state incrementally with current event
                self._update_state_with_event(state, event)
                
                # Map event name to TLA+ action
                event_name = event.get("action", "Step")
                tla_action = self._map_event_name(event_name)
                
                # Build output event
                output_event = {}
                
                # Add all variables with their current state
                for var_name, lock_values in state.items():
                    output_event[var_name] = [{
                        "op": "Update",
                        "path": [],
                        "args": [dict(lock_values)]  # Convert to dict for JSON serialization
                    }]
                
                # Add event metadata
                output_event["event"] = tla_action
                output_event["step"] = event.get("seq", event_num)
                output_event["thread"] = self._map_thread_id(str(event.get("thread", event.get("actor", "0"))))
                output_event["lock"] = self._map_lock_id(str(event.get("lock", "0")))
                
                output_lines.append(json.dumps(output_event))
            
            # Write output trace
            with open(output_trace_path, 'w') as f:
                for line in output_lines:
                    f.write(line + '\n')
            
            print(f"Conversion complete: {len(events)} events -> {len(output_lines) - 1} transitions")
            
            return {
                "success": True,
                "input_events": len(events),
                "output_transitions": len(output_lines) - 1,  # Exclude config line
                "output_file": output_trace_path
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"SpinLock trace conversion failed: {str(e)}"
            }