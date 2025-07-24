#!/usr/bin/env python3
"""
TLA+ Generation Benchmark Runner

This script runs TLA+ generation benchmarks using different methods
on various source code tasks, with models configured in config files.

Usage:
    python scripts/run_benchmark.py --method direct_call --task etcd
    python scripts/run_benchmark.py --method direct_call --task etcd --model my_claude
    python scripts/run_benchmark.py --method agent_based --task all --config config/my_models.yaml
"""

import argparse
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from tla_eval.methods import DirectCallMethod
from tla_eval.tasks import load_task, get_task_loader
from tla_eval.config import get_config_manager


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run TLA+ generation benchmark"
    )
    
    # Method selection
    parser.add_argument(
        "--method", 
        type=str,
        required=True,
        choices=["direct_call", "agent_based"],
        help="TLA+ generation method to use"
    )
    
    # Task selection
    parser.add_argument(
        "--task", 
        type=str,
        required=True,
        help="Task to run (e.g., 'etcd', 'raft', or 'all')"
    )
    
    # Model selection (optional, uses default from config)
    parser.add_argument(
        "--model",
        type=str,
        help="Model name from config file (optional, uses default)"
    )
    
    # Configuration file
    parser.add_argument(
        "--config",
        type=str, 
        default="config/models.yaml",
        help="Path to model configuration file"
    )
    
    # Output options
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/results",
        help="Directory to save results"
    )
    
    # Execution options
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without executing"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true", 
        help="Enable verbose output"
    )
    
    return parser.parse_args()


def get_generation_method(method_name: str):
    """Get generation method instance by name."""
    if method_name == "direct_call":
        return DirectCallMethod()
    elif method_name == "agent_based":
        # TODO: Implement agent-based method
        raise NotImplementedError("Agent-based method not implemented yet")
    else:
        raise ValueError(f"Unknown method: {method_name}")


def run_single_task(method, task_name: str, model_name: str = None, verbose: bool = False):
    """Run benchmark on a single task."""
    
    try:
        # Load task
        print(f"Loading task: {task_name}")
        task = load_task(task_name)
        
        if verbose:
            print(f"Task info:")
            print(f"  System: {task.description}")
            print(f"  Type: {task.system_type}")
            print(f"  Language: {task.language}")
            print(f"  Source code length: {len(task.source_code)} chars")
        
        # Generate TLA+ specification
        print(f"Generating TLA+ specification using {method.name} method...")
        result = method.generate(task, model_name)
        
        # Display results
        if result.success:
            print(f"✓ Generation successful!")
            print(f"\nGenerated TLA+ Specification:")
            print("=" * 50)
            print(result.tla_specification)
            print("=" * 50)
            
            if verbose:
                print(f"\nMetadata:")
                for key, value in result.metadata.items():
                    print(f"  {key}: {value}")
        else:
            print(f"✗ Generation failed: {result.error_message}")
            return False
            
    except Exception as e:
        print(f"Error running task {task_name}: {e}")
        return False
    
    return True


def main():
    """Main entry point."""
    args = parse_args()
    
    # Show configuration in dry-run mode
    if args.dry_run:
        print("DRY RUN MODE - No actual generation will be performed")
        print(f"Method: {args.method}")
        print(f"Task: {args.task}")
        print(f"Model: {args.model or 'default'}")
        print(f"Config: {args.config}")
        
        # Show available tasks and models
        try:
            config_manager = get_config_manager(args.config)
            task_loader = get_task_loader()
            
            print(f"\nAvailable models: {config_manager.list_available_models()}")
            print(f"Default model: {config_manager.get_default_model_name()}")
            print(f"Available tasks: {task_loader.list_available_tasks()}")
            
        except Exception as e:
            print(f"Error loading configuration: {e}")
        
        return
    
    try:
        # Initialize configuration
        config_manager = get_config_manager(args.config)
        print(f"Loaded configuration from: {args.config}")
        
        if args.verbose:
            print(f"Available models: {config_manager.list_available_models()}")
            print(f"Using model: {args.model or config_manager.get_default_model_name()}")
        
        # Initialize method
        method = get_generation_method(args.method)
        print(f"Using method: {method.name}")
        
        if args.verbose:
            method_info = method.get_method_info()
            print(f"Method info: {method_info}")
        
        # Run benchmark
        if args.task == "all":
            # Run all available tasks
            task_loader = get_task_loader()
            tasks = task_loader.list_available_tasks()
            
            if not tasks:
                print("No tasks available")
                sys.exit(1)
            
            print(f"Running {len(tasks)} tasks...")
            success_count = 0
            
            for task_name in tasks:
                print(f"\n{'='*20} {task_name} {'='*20}")
                if run_single_task(method, task_name, args.model, args.verbose):
                    success_count += 1
            
            print(f"\nCompleted: {success_count}/{len(tasks)} tasks successful")
            
        else:
            # Run single task
            success = run_single_task(method, args.task, args.model, args.verbose)
            if not success:
                sys.exit(1)
        
        print("Benchmark completed successfully!")
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()