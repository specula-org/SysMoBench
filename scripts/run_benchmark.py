#!/usr/bin/env python3
"""
Main entry point for running TLA+ generation benchmarks.

This script provides a command-line interface for evaluating LLMs on TLA+ 
specification generation tasks.

Usage:
    python scripts/run_benchmark.py --model openai_gpt4 --task etcd
    python scripts/run_benchmark.py --model anthropic_claude3 --task all
    python scripts/run_benchmark.py --config config/custom_experiment.yaml
"""

import argparse
import sys
import os
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from tla_eval.models import get_model_adapter
from tla_eval.evaluation import Evaluator
from tla_eval.utils import load_config, setup_logging


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run TLA+ generation benchmark evaluation"
    )
    
    # Model configuration
    parser.add_argument(
        "--model", 
        type=str, 
        help="Model identifier (e.g., openai_gpt4, anthropic_claude3)"
    )
    parser.add_argument(
        "--model-args", 
        type=str, 
        help="Model-specific arguments as key=value pairs"
    )
    
    # Task configuration
    parser.add_argument(
        "--task", 
        type=str, 
        default="all",
        help="Task to evaluate (etcd, raft, or all)"
    )
    
    # Configuration file
    parser.add_argument(
        "--config", 
        type=str,
        help="Path to configuration file"
    )
    
    # Output options
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default="data/results",
        help="Directory to save results"
    )
    parser.add_argument(
        "--save-specs", 
        action="store_true",
        help="Save generated TLA+ specifications"
    )
    
    # Execution options
    parser.add_argument(
        "--dry-run", 
        action="store_true",
        help="Print what would be done without executing"
    )
    parser.add_argument(
        "--verbose", 
        "-v", 
        action="store_true",
        help="Enable verbose logging"
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    # Setup logging
    setup_logging(verbose=args.verbose)
    
    if args.dry_run:
        print("DRY RUN MODE - No actual evaluation will be performed")
        print(f"Would evaluate model: {args.model}")
        print(f"Would run task: {args.task}")
        print(f"Would save results to: {args.output_dir}")
        return
    
    try:
        # Load configuration
        if args.config:
            config = load_config(args.config)
        else:
            # Use default configuration
            config = load_config("config/default.yaml")
        
        # TODO: Implement actual evaluation logic
        print("Benchmark framework is ready!")
        print("Implementation coming soon...")
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()