"""
Trace Converter Module

This module handles conversion of system traces to specification-compatible formats.
This is Step 3 in the Phase 3 pipeline - converting sys_traces to spec-acceptable format.
"""

import json
import os
from typing import Dict, Any, List, Optional
from pathlib import Path
from collections import defaultdict

class TraceConverter:
    """
    Converts system traces to specification-compatible formats.
    
    This handles the conversion from raw system traces (NDJSON format)
    to the format expected by TLA+ specifications for validation.
    """
    
    def __init__(self, mapping_file: str = None):
        """
        Initialize trace converter with mapping configuration.
        
        Args:
            mapping_file: Path to JSON mapping file
        """
        if mapping_file is None:
            mapping_file = os.path.join(os.path.dirname(__file__), "etcd_mapping.json")
        
        self.mapping_file = mapping_file
        self.mapping = self._load_mapping()
        
    def _load_mapping(self) -> Dict[str, Any]:
        """Load mapping configuration from JSON file."""
        try:
            with open(self.mapping_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Failed to load mapping file {self.mapping_file}: {e}")
            return {}
    
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
    
    def _map_node_id(self, node_id: str) -> str:
        """Map system node ID to TLA+ node name."""
        node_mapping = self.mapping.get("node_mapping", {})
        return node_mapping.get(str(node_id), f"n{node_id}")
    
    def _map_event_name(self, event_name: str) -> str:
        """Map system event name to TLA+ action name."""
        events_mapping = self.mapping.get("events", {})
        return events_mapping.get(event_name, events_mapping.get("default", "Step"))
    
    def _map_variable_value(self, var_config: Dict[str, Any], raw_value: Any, node_id: str) -> Any:
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
    
    def _build_state_snapshot(self, events: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """Build complete state snapshot from events up to current point."""
        state = defaultdict(dict)
        
        # Initialize with default values
        node_mapping = self.mapping.get("node_mapping", {})
        variables_config = self.mapping.get("variables", {})
        
        for node_id, node_name in node_mapping.items():
            for var_name, var_config in variables_config.items():
                state[var_name][node_name] = var_config.get("default_value")
        
        # Apply events in order to build current state
        for event in events:
            node_id = event.get("nid", "1")
            node_name = self._map_node_id(node_id)
            
            # Update each variable based on the event
            for var_name, var_config in variables_config.items():
                system_path = var_config.get("system_path", [])
                raw_value = self._extract_value_from_event(event, system_path)
                
                if raw_value is not None:
                    mapped_value = self._map_variable_value(var_config, raw_value, node_id)
                    state[var_name][node_name] = mapped_value
        
        return dict(state)
    
    def _build_initial_state(self) -> Dict[str, Dict[str, Any]]:
        """Build initial state with default values."""
        state = defaultdict(dict)
        
        node_mapping = self.mapping.get("node_mapping", {})
        variables_config = self.mapping.get("variables", {})
        
        for node_id, node_name in node_mapping.items():
            for var_name, var_config in variables_config.items():
                state[var_name][node_name] = var_config.get("default_value")
        
        return state
    
    def _update_state_with_event(self, state: Dict[str, Dict[str, Any]], event: Dict[str, Any]) -> None:
        """Update state incrementally with a single event."""
        node_id = event.get("nid", "1")
        node_name = self._map_node_id(node_id)
        
        variables_config = self.mapping.get("variables", {})
        
        # Update each variable based on the event
        for var_name, var_config in variables_config.items():
            system_path = var_config.get("system_path", [])
            raw_value = self._extract_value_from_event(event, system_path)
            
            if raw_value is not None:
                mapped_value = self._map_variable_value(var_config, raw_value, node_id)
                state[var_name][node_name] = mapped_value
    
    def convert_trace(self, input_trace_path: str, output_trace_path: str, config: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Convert trace from system format to spec format.
        
        Args:
            input_trace_path: Path to input trace file (NDJSON)
            output_trace_path: Path for output trace file
            config: Configuration for conversion
            
        Returns:
            Dictionary with conversion results
        """
        try:
            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_trace_path), exist_ok=True)
            
            # Read input trace
            events = []
            with open(input_trace_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            event = json.loads(line)
                            events.append(event)
                        except json.JSONDecodeError as e:
                            print(f"Skipping invalid JSON line: {line[:100]}... Error: {e}")
                            continue
            
            if not events:
                return {
                    "success": False,
                    "error": "No valid events found in input trace"
                }
            
            # Convert to TLA+ format
            output_lines = []
            
            # First line: configuration
            config_line = json.dumps(self.mapping.get("config", {}))
            output_lines.append(config_line)
            
            # Process events to generate state transitions with incremental state updates
            # Initialize state once
            state = self._build_initial_state()
            
            processed_events = []
            for event in events:
                # Update state incrementally with current event
                self._update_state_with_event(state, event)
                
                # Map event name to TLA+ action
                event_name = event.get("name", "Step")
                tla_action = self._map_event_name(event_name)
                
                # Build output line
                output_event = {}
                
                # Add all variables with their current state
                for var_name, node_values in state.items():
                    output_event[var_name] = [{
                        "op": "Update",
                        "path": [],
                        "args": [dict(node_values)]  # Convert to dict for JSON serialization
                    }]
                
                # Add event name
                output_event["event"] = tla_action
                
                output_lines.append(json.dumps(output_event))
                processed_events.append(event)
            
            # Write output trace
            with open(output_trace_path, 'w') as f:
                for line in output_lines:
                    f.write(line + '\n')
            
            return {
                "success": True,
                "input_events": len(events),
                "output_transitions": len(output_lines) - 1,  # Exclude config line
                "output_file": output_trace_path
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Trace conversion failed: {str(e)}"
            }

__all__ = ['TraceConverter']