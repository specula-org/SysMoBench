#!/usr/bin/env python3
"""
Basic usage example of the TLA+ evaluation framework.

This example demonstrates how to:
1. Initialize a model adapter
2. Load a benchmark task  
3. Generate TLA+ specifications
4. Evaluate the results
"""

import os
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from tla_eval.models import get_model_adapter
from tla_eval.tasks import load_task
from tla_eval.evaluation import evaluate_specification


def main():
    """Demonstrate basic usage of the framework."""
    
    # Example 1: Using OpenAI GPT-4
    print("=== Example 1: OpenAI GPT-4 ===")
    
    # Check if API key is available
    if not os.getenv("OPENAI_API_KEY"):
        print("Warning: OPENAI_API_KEY not found. Skipping OpenAI example.")
    else:
        # Initialize model adapter
        model_config = {
            "type": "api",
            "provider": "openai", 
            "model_name": "gpt-4",
            "max_tokens": 2048,
            "temperature": 0.1
        }
        
        model = get_model_adapter("openai_gpt4", **model_config)
        print(f"Initialized model: {model.get_model_info()}")
    
    # Example 2: Load a benchmark task
    print("\n=== Example 2: Load benchmark task ===")
    
    try:
        task = load_task("etcd")
        print(f"Loaded task: {task.name}")
        print(f"Description: {task.description}")
        print(f"Difficulty: {task.difficulty}")
    except Exception as e:
        print(f"Could not load task: {e}")
    
    # Example 3: Generate TLA+ specification (mock)
    print("\n=== Example 3: Generate specification ===")
    
    sample_code = """
    // etcd key-value operations
    func (s *EtcdServer) Put(key, value string) error {
        // Acquire lock
        s.mutex.Lock()
        defer s.mutex.Unlock()
        
        // Store in state machine
        s.store[key] = value
        
        // Replicate to followers
        return s.replicate(PUT, key, value)
    }
    """
    
    print(f"Sample source code:\n{sample_code}")
    print("Generated TLA+ specification: [Implementation pending]")
    
    # Example 4: Evaluation metrics
    print("\n=== Example 4: Evaluation metrics ===")
    
    sample_spec = """
    VARIABLES store, mutex
    
    Put(key, value) ==
        /\ mutex = "unlocked"
        /\ mutex' = "locked" 
        /\ store' = [store EXCEPT ![key] = value]
        /\ mutex'' = "unlocked"
    """
    
    print(f"Sample TLA+ specification:\n{sample_spec}")
    print("Evaluation results: [Implementation pending]")
    
    print("\n=== Framework ready for implementation! ===")


if __name__ == "__main__":
    main()