"""
RwMutex Trace Converter Implementation

Configuration-based implementation for RwMutex trace conversion.
Based on the Mutex trace converter pattern but adapted for RwMutex read-write semantics.
"""

import json
import os
from typing import Dict, Any, List, Optional
from pathlib import Path
from collections import defaultdict


class RwMutexTraceConverterImpl:
    """
    Configuration-based implementation for RwMutex trace conversion.
    
    This handles the conversion from raw RwMutex traces (JSONL format)
    to the format expected by TLA+ specifications for validation.
    
    RwMutex has more complex semantics than regular Mutex:
    - Multiple readers can hold the lock simultaneously
    - Only one writer can hold the lock exclusively
    - Upgradeable readers can be upgraded to writers
    """
    
    def __init__(self, mapping_file: str = None):
        """
        Initialize trace converter with mapping configuration.
        
        Args:
            mapping_file: Path to JSON mapping file
        """
        if mapping_file is None:
            # Look in data/convertor/rwmutex first, fallback to module directory
            data_mapping = "data/convertor/rwmutex/rwmutex_mapping.json"
            module_mapping = os.path.join(os.path.dirname(__file__), "rwmutex_mapping.json")
            
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
                print(f"Loaded RwMutex mapping configuration from: {self.mapping_file}")
                return mapping
        except Exception as e:
            print(f"Error loading mapping from {self.mapping_file}: {e}")
            # Return default mapping if file not found
            return self._get_default_mapping()
    
    def _get_default_mapping(self) -> Dict[str, Any]:
        """Get default mapping configuration if file not found."""
        return {
            "config": {
                "Threads": ["Thread0", "Thread1", "Thread2", "Thread3"],
                "RwMutexes": ["RwMutex0", "RwMutex1", "RwMutex2", "RwMutex3"],
                "MaxSteps": 1000
            },
            "events": {
                "TryReadLock": "RequestReadLock",
                "ReadLockAcquired": "ReadLockAcquired",
                "ReadUnlock": "ReleaseReadLock",
                "TryWriteLock": "RequestWriteLock", 
                "WriteLockAcquired": "WriteLockAcquired",
                "WriteUnlock": "ReleaseWriteLock",
                "TryUpgradeableReadLock": "RequestUpgradeableReadLock",
                "UpgradeableReadLockAcquired": "UpgradeableReadLockAcquired",
                "UpgradeableReadUnlock": "ReleaseUpgradeableReadLock",
                "UpgradeToWriteLock": "UpgradeToWriteLock",
                "UpgradeSuccess": "UpgradeSuccess",
                "DowngradeToUpgradeableRead": "DowngradeToUpgradeableRead"
            }
        }
    
    def convert_trace(self, input_file: str, output_file: str) -> Dict[str, Any]:
        """
        Convert raw rwmutex trace to TLA+ format.
        
        Args:
            input_file: Path to input JSONL trace file
            output_file: Path to output NDJSON trace file
            
        Returns:
            Dictionary with conversion results
        """
        try:
            print(f"Converting rwmutex trace: {input_file} -> {output_file}")
            
            # Read and parse input trace
            input_events = self._read_input_trace(input_file)
            if not input_events:
                return {
                    "success": False,
                    "error": "No events found in input trace",
                    "input_file": input_file,
                    "output_file": output_file,
                    "input_events": 0,
                    "output_transitions": 0
                }
            
            # Convert events to TLA+ format
            output_transitions = self._convert_events(input_events)
            
            # Write output trace
            self._write_output_trace(output_transitions, output_file)
            
            print(f"Conversion successful: {len(input_events)} input events -> {len(output_transitions)} output transitions")
            
            return {
                "success": True,
                "input_file": input_file,
                "output_file": output_file,
                "input_events": len(input_events),
                "output_transitions": len(output_transitions)
            }
            
        except Exception as e:
            error_msg = f"Error converting trace {input_file}: {str(e)}"
            print(error_msg)
            return {
                "success": False,
                "error": error_msg,
                "input_file": input_file,
                "output_file": output_file,
                "input_events": 0,
                "output_transitions": 0
            }
    
    def _read_input_trace(self, input_file: str) -> List[Dict[str, Any]]:
        """Read and parse input trace file."""
        events = []
        
        try:
            with open(input_file, 'r') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    
                    # Skip comments and empty lines
                    if not line or line.startswith('#'):
                        continue
                    
                    try:
                        event = json.loads(line)
                        events.append(event)
                    except json.JSONDecodeError as e:
                        print(f"Warning: Invalid JSON on line {line_num} in {input_file}: {e}")
                        continue
                        
        except FileNotFoundError:
            print(f"Error: Input file not found: {input_file}")
            return []
        except Exception as e:
            print(f"Error reading input file {input_file}: {e}")
            return []
        
        print(f"Read {len(events)} events from {input_file}")
        return events
    
    def _convert_events(self, events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert input events to TLA+ format."""
        transitions = []
        
        # Add configuration header
        config = self.mapping.get("config", {})
        header = {
            "Threads": config.get("Threads", ["Thread0", "Thread1", "Thread2", "Thread3"]),
            "RwMutexes": config.get("RwMutexes", ["RwMutex0", "RwMutex1", "RwMutex2", "RwMutex3"]),
            "MaxSteps": config.get("MaxSteps", 1000),
            "States": ["unlocked", "read_locked", "write_locked", "upgradeable_locked"],
            "Nil": "Nil"
        }
        transitions.append(header)
        
        # Initialize rwmutex state tracking
        rwmutex_states = defaultdict(lambda: "unlocked")
        rwmutex_readers = defaultdict(set)  # Set of thread names holding read locks
        rwmutex_writer = defaultdict(lambda: "Nil")  # Thread name holding write lock
        rwmutex_upgradeable = defaultdict(lambda: "Nil")  # Thread name holding upgradeable lock
        rwmutex_sequences = defaultdict(lambda: 0)
        
        event_mapping = self.mapping.get("events", {})
        
        for i, event in enumerate(events):
            # Extract event fields
            seq = event.get('seq', i)
            thread_id = event.get('thread', 0)
            rwmutex_id = event.get('rwmutex', 0)
            action = event.get('action', '')
            state = event.get('state', '')
            
            # Map thread ID to thread name
            thread_name = f"Thread{thread_id}"
            rwmutex_name = f"RwMutex{rwmutex_id}"
            
            # Map action to TLA+ event
            tla_event = event_mapping.get(action, action)
            
            # Update state based on action
            if action == "ReadLockAcquired":
                rwmutex_readers[rwmutex_name].add(thread_name)
                rwmutex_states[rwmutex_name] = "read_locked"
                rwmutex_sequences[rwmutex_name] += 1
            elif action == "ReadUnlock":
                rwmutex_readers[rwmutex_name].discard(thread_name)
                if not rwmutex_readers[rwmutex_name] and rwmutex_upgradeable[rwmutex_name] == "Nil":
                    rwmutex_states[rwmutex_name] = "unlocked"
                rwmutex_sequences[rwmutex_name] += 1
            elif action == "WriteLockAcquired":
                rwmutex_writer[rwmutex_name] = thread_name
                rwmutex_states[rwmutex_name] = "write_locked"
                rwmutex_sequences[rwmutex_name] += 1
            elif action == "WriteUnlock":
                rwmutex_writer[rwmutex_name] = "Nil"
                rwmutex_states[rwmutex_name] = "unlocked"
                rwmutex_sequences[rwmutex_name] += 1
            elif action == "UpgradeableReadLockAcquired":
                rwmutex_upgradeable[rwmutex_name] = thread_name
                if rwmutex_readers[rwmutex_name]:
                    rwmutex_states[rwmutex_name] = "read_locked"
                else:
                    rwmutex_states[rwmutex_name] = "upgradeable_locked"
                rwmutex_sequences[rwmutex_name] += 1
            elif action == "UpgradeableReadUnlock":
                rwmutex_upgradeable[rwmutex_name] = "Nil"
                if rwmutex_readers[rwmutex_name]:
                    rwmutex_states[rwmutex_name] = "read_locked"
                else:
                    rwmutex_states[rwmutex_name] = "unlocked"
                rwmutex_sequences[rwmutex_name] += 1
            elif action == "UpgradeSuccess":
                # Upgrade from upgradeable to write
                rwmutex_upgradeable[rwmutex_name] = "Nil"
                rwmutex_writer[rwmutex_name] = thread_name
                rwmutex_states[rwmutex_name] = "write_locked"
                rwmutex_sequences[rwmutex_name] += 1
            elif action == "DowngradeToUpgradeableRead":
                # Downgrade from write to upgradeable
                rwmutex_writer[rwmutex_name] = "Nil"
                rwmutex_upgradeable[rwmutex_name] = thread_name
                rwmutex_states[rwmutex_name] = "upgradeable_locked"
                rwmutex_sequences[rwmutex_name] += 1
            
            # Create state snapshot for all rwmutexes
            all_rwmutex_states = {}
            all_rwmutex_readers = {}
            all_rwmutex_writers = {}
            all_rwmutex_upgradeables = {}
            all_rwmutex_sequences = {}
            
            # Include states for all rwmutexes mentioned in config
            for rwmutex_name_config in config.get("RwMutexes", [f"RwMutex{j}" for j in range(4)]):
                all_rwmutex_states[rwmutex_name_config] = rwmutex_states[rwmutex_name_config]
                all_rwmutex_readers[rwmutex_name_config] = list(rwmutex_readers[rwmutex_name_config])
                all_rwmutex_writers[rwmutex_name_config] = rwmutex_writer[rwmutex_name_config]
                all_rwmutex_upgradeables[rwmutex_name_config] = rwmutex_upgradeable[rwmutex_name_config]
                all_rwmutex_sequences[rwmutex_name_config] = rwmutex_sequences[rwmutex_name_config]
            
            # Create transition record
            transition = {
                "rwmutexState": [{"op": "Update", "path": [], "args": [all_rwmutex_states]}],
                "readers": [{"op": "Update", "path": [], "args": [all_rwmutex_readers]}],
                "writer": [{"op": "Update", "path": [], "args": [all_rwmutex_writers]}],
                "upgradeable": [{"op": "Update", "path": [], "args": [all_rwmutex_upgradeables]}],
                "sequence": [{"op": "Update", "path": [], "args": [all_rwmutex_sequences]}],
                "event": tla_event,
                "step": seq,
                "thread": thread_name,
                "rwmutex": rwmutex_name,
                "lock_type": self._get_lock_type(action)
            }
            
            transitions.append(transition)
        
        return transitions
    
    def _get_lock_type(self, action: str) -> str:
        """Determine if action is read, write, or upgradeable related."""
        if any(read_action in action for read_action in ['Read', 'read']) and 'Upgradeable' not in action:
            return "read"
        elif any(write_action in action for write_action in ['Write', 'write']):
            return "write"
        elif 'Upgradeable' in action or 'Upgrade' in action:
            return "upgradeable"
        else:
            return "unknown"
    
    def _write_output_trace(self, transitions: List[Dict[str, Any]], output_file: str) -> None:
        """Write converted transitions to output file."""
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        with open(output_file, 'w') as f:
            for transition in transitions:
                f.write(json.dumps(transition) + '\n')
        
        print(f"Wrote {len(transitions)} transitions to {output_file}")
    
    def get_mapping_info(self) -> Dict[str, Any]:
        """Get information about the current mapping configuration."""
        return {
            "mapping_file": self.mapping_file,
            "config": self.mapping.get("config", {}),
            "events": self.mapping.get("events", {}),
            "total_event_mappings": len(self.mapping.get("events", {}))
        }