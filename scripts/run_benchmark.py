#!/usr/bin/env python3
"""
TLA+ Generation Benchmark Runner

This is the unified benchmark runner for evaluating LLM performance on 
TLA+ specification generation from real-world distributed systems.

Supports different evaluation phases:
- Phase 1: Compilation checking (syntax/semantic errors)
- Phase 2: Runtime checking (model checking capabilities) [Future]  
- Phase 3: Consistency checking (specification correctness) [Future]

Usage Examples:
    # Phase 1: Compilation checking (default)
    python scripts/run_benchmark.py --task etcd --method direct_call --model my_yunwu
    
    # Batch evaluation with multiple combinations
    python scripts/run_benchmark.py --tasks etcd raft --methods direct_call agent_based --models gpt-4 claude-3 --output results/
    
    # Specify evaluation phase explicitly
    python scripts/run_benchmark.py --phase 1 --task etcd --method direct_call --model my_yunwu
    
    # List available options
    python scripts/run_benchmark.py --list-tasks
    python scripts/run_benchmark.py --list-methods
    python scripts/run_benchmark.py --list-models
"""

import argparse
import sys
import time
import json
from pathlib import Path
from typing import List, Optional

# Add project root to Python path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from tla_eval.config import get_configured_model, get_config_manager
from tla_eval.tasks.loader import get_task_loader
from tla_eval.methods import get_method
from tla_eval.models.base import GenerationConfig
from tla_eval.utils import validate_tla_tools_setup
from tla_eval.utils.repository_manager import setup_task_repository

# Import evaluators
from tla_eval.evaluation import Phase1Evaluator, Phase2Evaluator, Phase3Evaluator

import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


def validate_prerequisites(phase: int = 1):
    """Validate that all prerequisites are met for running the benchmark."""
    logger.info("Validating prerequisites...")
    
    if phase == 1:
        # Check TLA+ tools setup for compilation checking
        validation = validate_tla_tools_setup()
        
        if not validation["ready"]:
            logger.error("TLA+ tools are not properly set up!")
            
            if not validation["java_available"]:
                logger.error("Java is not available. Please install Java.")
            
            if not validation["tla_tools_exists"]:
                logger.error("tla2tools.jar not found. Run 'python3 -m scripts.setup_tools' to download it.")
            
            return False
    
    logger.info("✓ All prerequisites validated")
    return True


def run_single_benchmark(task_name: str, method_name: str, model_name: str, 
                        phase: int = 1,
                        source_file: Optional[str] = None,
                        generation_config: Optional[GenerationConfig] = None) -> dict:
    """
    Run benchmark for a single task/method/model combination.
    
    Args:
        task_name: Name of the task
        method_name: Name of the generation method  
        model_name: Name of the model
        phase: Evaluation phase (1=compilation, 2=runtime, 3=consistency)
        source_file: Optional specific source file
        generation_config: Optional generation configuration
        
    Returns:
        Dictionary with benchmark results
    """
    logger.info(f"Running Phase {phase} benchmark: {task_name}/{method_name}/{model_name}")
    
    try:
        # Load task
        task_loader = get_task_loader()
        task = task_loader.load_task(task_name, source_file)
        logger.info(f"Loaded task: {task.task_name} ({task.system_type})")
        
        # Setup repository if needed for Phase 3
        if phase == 3:
            logger.info("Setting up repository for Phase 3 trace generation...")
            try:
                # Get task configuration to check for patch requirements
                task_config = task_loader.get_task_info(task_name)
                repo_path = setup_task_repository(task_config)
                logger.info(f"Repository setup completed: {repo_path}")
            except Exception as e:
                logger.warning(f"Repository setup failed (continuing anyway): {e}")
        
        # Get prompt for this method
        prompt_template = task_loader.get_task_prompt(task_name, method_name)
        logger.info(f"Loaded prompt template ({len(prompt_template)} chars)")
        
        # Load model
        model = get_configured_model(model_name)
        logger.info(f"Loaded model: {model.model_name}")
        
        # Load method
        method = get_method(method_name)
        logger.info(f"Using method: {method_name}")
        
        # Generate TLA+ specification
        logger.info("Starting TLA+ generation...")
        start_time = time.time()
        
        # Note: methods use their own prompt templates and load models internally
        generation_output = method.generate(task, model_name)
        
        generation_time = time.time() - start_time
        logger.info(f"Generation completed in {generation_time:.2f}s")
        
        # Convert to GenerationResult format expected by evaluator
        from tla_eval.models.base import GenerationResult
        generation_result = GenerationResult(
            generated_text=generation_output.tla_specification,
            metadata={
                'method': generation_output.method_name,
                'latency_seconds': generation_time,
                **generation_output.metadata
            },
            timestamp=time.time(),
            success=generation_output.success,
            error_message=generation_output.error_message
        )
        
        # Evaluate based on phase
        evaluation_result = None
        
        if phase == 1:
            # Phase 1: Compilation checking
            evaluator = Phase1Evaluator()
            evaluation_result = evaluator.evaluate_generation(
                generation_result, task_name, method_name, model_name, task.spec_module
            )
            logger.info(f"Phase 1 evaluation: {'✓ PASS' if evaluation_result.overall_success else '✗ FAIL'}")
            
        elif phase == 2:
            # Phase 2: Model checking with TLC
            evaluator = Phase2Evaluator()
            
            # Get specification file path from Phase 1 or user input
            spec_file_path = f"data/spec/{task_name}/{task.spec_module}.tla"
            if not Path(spec_file_path).exists():
                logger.error(f"Specification file not found: {spec_file_path}")
                return {"success": False, "error": f"Specification file not found: {spec_file_path}"}
            
            evaluation_result = evaluator.evaluate_specification(
                spec_file_path, task_name, method_name, model_name
            )
            logger.info(f"Phase 2 evaluation: {'✓ PASS' if evaluation_result.get('success', False) else '✗ FAIL'}")
            
        elif phase == 3:
            # Phase 3: Trace generation and validation
            evaluator = Phase3Evaluator()
            
            # Use default configuration for Phase 3
            phase3_config = evaluator.get_default_config()
            
            evaluation_result = evaluator.evaluate(task_name, phase3_config)
            logger.info(f"Phase 3 evaluation: {'✓ PASS' if evaluation_result.get('success', False) else '✗ FAIL'}")
        else:
            raise ValueError(f"Unknown evaluation phase: {phase}")
        
        return {
            "success": True,
            "phase": phase,
            "evaluation_result": evaluation_result,
            "error": None
        }
        
    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        return {
            "success": False,
            "phase": phase,
            "evaluation_result": None,
            "error": str(e)
        }


def run_batch_benchmark(tasks: List[str], methods: List[str], models: List[str],
                       phase: int = 1, output_dir: str = "results", 
                       generation_config: Optional[GenerationConfig] = None):
    """
    Run benchmark for multiple task/method/model combinations.
    
    Args:
        tasks: List of task names
        methods: List of method names
        models: List of model names
        phase: Evaluation phase
        output_dir: Output directory for results
        generation_config: Optional generation configuration
    """
    logger.info(f"Starting Phase {phase} batch benchmark: {len(tasks)} tasks × {len(methods)} methods × {len(models)} models")
    
    total_combinations = len(tasks) * len(methods) * len(models)
    current = 0
    
    results = []
    
    for task_name in tasks:
        for method_name in methods:
            for model_name in models:
                current += 1
                logger.info(f"Progress: {current}/{total_combinations}")
                
                result = run_single_benchmark(
                    task_name, method_name, model_name, phase,
                    generation_config=generation_config
                )
                
                if result["success"]:
                    results.append(result["evaluation_result"])
                else:
                    logger.error(f"Failed combination: {task_name}/{method_name}/{model_name}: {result['error']}")
    
    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    timestamp = int(time.time())
    results_file = output_path / f"phase{phase}_results_{timestamp}.json"
    
    # Save results based on phase
    if phase == 1 and results:
        evaluator = Phase1Evaluator()
        evaluator.save_results(results, str(results_file), include_specifications=True)
    elif phase == 2 and results:
        evaluator = Phase2Evaluator()
        evaluator.save_results(results, str(results_file))
    else:
        # Generic JSON save for other phases
        output_data = {
            "evaluation_type": f"phase{phase}",
            "total_evaluations": len(results),
            "timestamp": time.time(),
            "results": [r.to_dict() if hasattr(r, 'to_dict') else r for r in results]
        }
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Batch benchmark complete. Results saved to: {results_file}")
    
    # Print summary
    if results and phase == 1:
        total = len(results)
        successful = sum(1 for r in results if r.overall_success)
        logger.info(f"Summary: {successful}/{total} ({successful/total*100:.1f}%) successful")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="TLA+ Generation Benchmark Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single benchmark (Phase 1 compilation checking)
  python3 scripts/run_benchmark.py --task etcd --method direct_call --model my_yunwu
  
  # Batch benchmark with multiple combinations
  python3 scripts/run_benchmark.py --tasks etcd raft --methods direct_call agent_based --models gpt-4 claude-3 --output results/
  
  # Specify evaluation phase explicitly  
  python3 scripts/run_benchmark.py --phase 1 --task etcd --method direct_call --model my_yunwu
  
  # List available options
  python3 scripts/run_benchmark.py --list-tasks
  python3 scripts/run_benchmark.py --list-methods
  python3 scripts/run_benchmark.py --list-models
        """
    )
    
    # Evaluation phase selection
    parser.add_argument("--phase", type=int, default=1, choices=[1, 2, 3],
                       help="Evaluation phase: 1=compilation, 2=runtime, 3=consistency (default: 1)")
    
    # Input selection
    parser.add_argument("--task", help="Single task name")
    parser.add_argument("--method", help="Single method name")  
    parser.add_argument("--model", help="Single model name")
    parser.add_argument("--source-file", help="Specific source file within task")
    
    # Batch mode
    parser.add_argument("--tasks", nargs="+", help="Multiple task names")
    parser.add_argument("--methods", nargs="+", help="Multiple method names")
    parser.add_argument("--models", nargs="+", help="Multiple model names")
    parser.add_argument("--output", default="results", help="Output directory (default: results)")
    
    # Generation parameters
    parser.add_argument("--max-tokens", type=int, default=4096, help="Maximum tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.1, help="Generation temperature")
    
    # Listing options
    parser.add_argument("--list-tasks", action="store_true", help="List available tasks")
    parser.add_argument("--list-methods", action="store_true", help="List available methods")
    parser.add_argument("--list-models", action="store_true", help="List available models")
    
    args = parser.parse_args()
    
    # Handle listing options
    if args.list_tasks:
        task_loader = get_task_loader()
        tasks = task_loader.list_available_tasks()
        print("Available tasks:")
        for task in tasks:
            print(f"  - {task}")
        return
    
    if args.list_methods:
        from tla_eval.methods import list_available_methods
        methods = list_available_methods()
        print("Available methods:")
        for method in methods:
            print(f"  - {method}")
        return
    
    if args.list_models:
        config_manager = get_config_manager()
        models = config_manager.list_available_models()
        print("Available models:")
        for model in models:
            print(f"  - {model}")
        return
    
    # Validate prerequisites
    if not validate_prerequisites(args.phase):
        sys.exit(1)
    
    # Create generation config
    generation_config = GenerationConfig(
        max_tokens=args.max_tokens,
        temperature=args.temperature
    )
    
    # Determine run mode
    single_mode = args.task and args.method and args.model
    batch_mode = args.tasks and args.methods and args.models
    
    if single_mode:
        # Single benchmark
        result = run_single_benchmark(
            args.task, args.method, args.model, args.phase,
            source_file=args.source_file,
            generation_config=generation_config
        )
        
        if result["success"]:
            eval_result = result["evaluation_result"]
            
            if args.phase == 1:
                print(f"\nPhase 1 Results: {'✓ PASS' if eval_result.overall_success else '✗ FAIL'}")
                print(f"Generation time: {eval_result.generation_time:.2f}s")
                print(f"Compilation time: {eval_result.compilation_time:.2f}s")
                print(f"Syntax errors: {len(eval_result.syntax_errors)}")
                print(f"Semantic errors: {len(eval_result.semantic_errors)}")
                
                if not eval_result.overall_success:
                    if not eval_result.generation_successful:
                        print(f"Generation error: {eval_result.generation_error}")
                    if not eval_result.compilation_successful:
                        print(f"Compilation errors: {eval_result.syntax_errors + eval_result.semantic_errors}")
                        
            elif args.phase == 2:
                print(f"\nPhase 2 Results: {'✓ PASS' if eval_result.overall_success else '✗ FAIL'}")
                print(f"Invariant generation time: {eval_result.invariant_generation_time:.2f}s")
                print(f"Config generation time: {eval_result.config_generation_time:.2f}s")
                print(f"Model checking time: {eval_result.model_checking_time:.2f}s")
                print(f"States explored: {eval_result.states_explored}")
                print(f"Invariant violations: {len(eval_result.invariant_violations)}")
                print(f"Deadlock found: {eval_result.deadlock_found}")
                
                if not eval_result.overall_success:
                    if not eval_result.invariant_generation_successful:
                        print(f"Invariant generation error: {eval_result.invariant_generation_error}")
                    if not eval_result.config_generation_successful:
                        print(f"Config generation error: {eval_result.config_generation_error}")
                    if not eval_result.model_checking_successful:
                        print(f"Model checking error: {eval_result.model_checking_error}")
                    if eval_result.invariant_violations:
                        print(f"Violations: {eval_result.invariant_violations}")
                        
                # Show file locations
                print(f"Specification: {eval_result.specification_file}")
                if eval_result.config_file_path:
                    print(f"Config file: {eval_result.config_file_path}")
            
        else:
            print(f"Benchmark failed: {result['error']}")
            sys.exit(1)
    
    elif batch_mode:
        # Batch benchmark
        run_batch_benchmark(
            args.tasks, args.methods, args.models, args.phase, args.output,
            generation_config=generation_config
        )
    
    else:
        parser.error("Must specify either single mode (--task --method --model) or batch mode (--tasks --methods --models)")


if __name__ == "__main__":
    main()