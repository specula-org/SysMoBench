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

# Import evaluators and metric registry
from tla_eval.evaluation import (
    CompilationCheckEvaluator, 
    InvariantVerificationEvaluator, 
    TraceValidationEvaluator
)
from tla_eval.evaluation.base import (
    get_available_metrics,
    get_available_dimensions, 
    create_evaluator,
    get_metric_registry
)

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
                        evaluation_type: str = "syntax",
                        metric: Optional[str] = None,
                        phase: Optional[int] = None,  # Legacy support
                        source_file: Optional[str] = None,
                        generation_config: Optional[GenerationConfig] = None,
                        **metric_params) -> dict:
    """
    Run benchmark for a single task/method/model combination.
    
    Args:
        task_name: Name of the task
        method_name: Name of the generation method  
        model_name: Name of the model
        evaluation_type: Evaluation type ("syntax", "semantics", "consistency")
        metric: Specific metric to run (if None, uses default for evaluation_type)
        phase: Legacy evaluation phase (1=syntax, 2=semantics, 3=consistency)
        source_file: Optional specific source file
        generation_config: Optional generation configuration
        metric_params: Additional parameters for specific metrics
        
    Returns:
        Dictionary with benchmark results
    """
    # Handle legacy phase parameter
    if phase is not None:
        phase_to_type = {1: "syntax", 2: "semantics", 3: "consistency"}
        evaluation_type = phase_to_type.get(phase, evaluation_type)
        logger.warning(f"Using legacy --phase {phase} parameter, consider using --evaluation-type {evaluation_type}")
    
    # Determine metric to use
    if metric is None:
        # Use default metric for each dimension
        default_metrics = {
            "syntax": "compilation_check",
            "semantics": "invariant_verification", 
            "consistency": "trace_validation"
        }
        metric = default_metrics.get(evaluation_type)
        if metric is None:
            raise ValueError(f"No default metric for evaluation type: {evaluation_type}")
    
    # Validate metric exists and matches dimension
    registry = get_metric_registry()
    try:
        metric_info = registry.get_metric(metric)
        if metric_info.dimension != evaluation_type:
            logger.warning(f"Metric '{metric}' belongs to dimension '{metric_info.dimension}', but evaluation-type is '{evaluation_type}'. Using metric's dimension.")
            evaluation_type = metric_info.dimension
    except ValueError as e:
        available_metrics = get_available_metrics(evaluation_type)
        raise ValueError(f"Unknown metric '{metric}' for dimension '{evaluation_type}'. Available metrics: {available_metrics}") from e
    
    logger.info(f"Running {evaluation_type} evaluation with metric '{metric}': {task_name}/{method_name}/{model_name}")
    
    try:
        # Load task
        task_loader = get_task_loader()
        task = task_loader.load_task(task_name, source_file)
        logger.info(f"Loaded task: {task.task_name} ({task.system_type})")
        
        # Setup repository if needed for consistency evaluation
        if evaluation_type == "consistency":
            logger.info("Setting up repository for consistency evaluation...")
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
        
        # Create evaluator using metric registry
        evaluator = create_evaluator(metric, **metric_params)
        
        # Evaluate based on metric type
        evaluation_result = None
        
        if metric == "compilation_check":
            # Syntax evaluation: Compilation checking
            evaluation_result = evaluator.evaluate(
                generation_result, task_name, method_name, model_name, task.spec_module
            )
            logger.info(f"Compilation check: {'✓ PASS' if evaluation_result.overall_success else '✗ FAIL'}")
            
        elif metric == "invariant_verification":
            # Semantics evaluation: Model checking with TLC
            # Use the generated specification from generation_result
            if not generation_result.success:
                logger.error("Cannot perform invariant verification: TLA+ generation failed")
                return {"success": False, "error": "TLA+ generation failed"}
            
            evaluation_result = evaluator.evaluate(
                generation_result, task_name, method_name, model_name, task.spec_module
            )
            logger.info(f"Invariant verification: {'✓ PASS' if evaluation_result.overall_success else '✗ FAIL'}")
            
        elif metric == "trace_validation":
            # Consistency evaluation: Trace generation and validation
            # Use default configuration for consistency evaluation
            consistency_config = evaluator.get_default_config()
            
            evaluation_result = evaluator.evaluate(task_name, consistency_config)
            logger.info(f"Trace validation: {'✓ PASS' if evaluation_result.overall_success else '✗ FAIL'}")
        else:
            # For future metrics, use generic interface
            try:
                # Try to call with standard parameters first
                if hasattr(evaluator, 'evaluate'):
                    if metric_info.dimension == "syntax":
                        evaluation_result = evaluator.evaluate(
                            generation_result, task_name, method_name, model_name, task.spec_module
                        )
                    elif metric_info.dimension == "semantics":
                        if not generation_result.success:
                            return {"success": False, "error": "TLA+ generation failed"}
                        evaluation_result = evaluator.evaluate(
                            generation_result, task_name, method_name, model_name, task.spec_module
                        )
                    elif metric_info.dimension == "consistency":
                        consistency_config = evaluator.get_default_config() if hasattr(evaluator, 'get_default_config') else {}
                        evaluation_result = evaluator.evaluate(task_name, consistency_config)
                    else:
                        raise ValueError(f"Unknown dimension: {metric_info.dimension}")
                else:
                    raise ValueError(f"Evaluator for metric '{metric}' does not have an evaluate method")
                    
                logger.info(f"Metric '{metric}': {'✓ PASS' if evaluation_result.overall_success else '✗ FAIL'}")
                
            except Exception as e:
                logger.error(f"Error running metric '{metric}': {e}")
                return {"success": False, "error": f"Error running metric '{metric}': {e}"}
        
        return {
            "success": True,
            "evaluation_type": evaluation_type,
            "metric": metric,
            "evaluation_result": evaluation_result,
            "error": None
        }
        
    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        return {
            "success": False,
            "evaluation_type": evaluation_type,
            "metric": metric,
            "evaluation_result": None,
            "error": str(e)
        }


def run_batch_benchmark(tasks: List[str], methods: List[str], models: List[str],
                       evaluation_type: str = "syntax", metric: Optional[str] = None,
                       phase: Optional[int] = None, output_dir: str = "results", 
                       generation_config: Optional[GenerationConfig] = None,
                       **metric_params):
    """
    Run benchmark for multiple task/method/model combinations.
    
    Args:
        tasks: List of task names
        methods: List of method names
        models: List of model names
        evaluation_type: Evaluation type ("syntax", "semantics", "consistency")
        phase: Legacy evaluation phase (deprecated)
        output_dir: Output directory for results
        generation_config: Optional generation configuration
    """
    # Handle legacy phase parameter
    if phase is not None:
        phase_to_type = {1: "syntax", 2: "semantics", 3: "consistency"}
        evaluation_type = phase_to_type.get(phase, evaluation_type)
        logger.warning(f"Using legacy --phase {phase} parameter, consider using --evaluation-type {evaluation_type}")
    
    logger.info(f"Starting {evaluation_type} batch benchmark: {len(tasks)} tasks × {len(methods)} methods × {len(models)} models")
    
    total_combinations = len(tasks) * len(methods) * len(models)
    current = 0
    
    results = []
    
    for task_name in tasks:
        for method_name in methods:
            for model_name in models:
                current += 1
                logger.info(f"Progress: {current}/{total_combinations}")
                
                result = run_single_benchmark(
                    task_name, method_name, model_name, evaluation_type, metric,
                    generation_config=generation_config, **metric_params
                )
                
                if result["success"]:
                    results.append(result["evaluation_result"])
                else:
                    logger.error(f"Failed combination: {task_name}/{method_name}/{model_name}: {result['error']}")
    
    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    timestamp = int(time.time())
    results_file = output_path / f"{evaluation_type}_results_{timestamp}.json"
    
    # Save results based on evaluation type
    if evaluation_type == "syntax" and results:
        evaluator = CompilationCheckEvaluator()
        evaluator.save_results(results, str(results_file), include_specifications=True)
    elif evaluation_type == "semantics" and results:
        evaluator = InvariantVerificationEvaluator()
        evaluator.save_results(results, str(results_file))
    elif evaluation_type == "consistency" and results:
        evaluator = TraceValidationEvaluator()
        evaluator.save_results(results, str(results_file))
    else:
        # Generic JSON save for other evaluation types
        output_data = {
            "evaluation_type": evaluation_type,
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
  # Single benchmark (default syntax evaluation with compilation_check metric)
  python3 scripts/run_benchmark.py --task etcd --method direct_call --model my_yunwu
  
  # Specify specific metric
  python3 scripts/run_benchmark.py --task etcd --method direct_call --model gpt-4 --metric invariant_verification
  
  # Use metric-specific parameters
  python3 scripts/run_benchmark.py --task etcd --method direct_call --model gpt-4 --metric pass_at_k --k 5
  
  # Run consistency evaluation with progressive granularity
  python3 scripts/run_benchmark.py --task etcd --method direct_call --model gpt-4 --metric progressive_granularity --level 2
  
  # Batch benchmark with specific metric
  python3 scripts/run_benchmark.py --tasks etcd --methods direct_call --models gpt-4 claude-3 --metric compilation_check --output results/
  
  # List available options
  python3 scripts/run_benchmark.py --list-tasks
  python3 scripts/run_benchmark.py --list-methods  
  python3 scripts/run_benchmark.py --list-models
  python3 scripts/run_benchmark.py --list-metrics
  python3 scripts/run_benchmark.py --list-metrics-for syntax
        """
    )
    
    # Evaluation type and metric selection
    parser.add_argument("--evaluation-type", default="syntax", 
                       choices=get_available_dimensions(),
                       help="Evaluation type: syntax=compilation, semantics=model-checking, consistency=trace-validation (default: syntax)")
    parser.add_argument("--metric", 
                       help="Specific metric to run (if not specified, uses default for evaluation-type)")
    parser.add_argument("--phase", type=int, choices=[1, 2, 3],
                       help="Legacy evaluation phase: 1=syntax, 2=semantics, 3=consistency (deprecated, use --evaluation-type)")
    
    # Metric-specific parameters
    parser.add_argument("--k", type=int, 
                       help="Number of attempts for pass@k metrics")
    parser.add_argument("--level", type=int,
                       help="Granularity level for progressive metrics")
    
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
    parser.add_argument("--max-tokens", type=int, default=640000, help="Maximum tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.1, help="Generation temperature")
    
    # Listing options
    parser.add_argument("--list-tasks", action="store_true", help="List available tasks")
    parser.add_argument("--list-methods", action="store_true", help="List available methods")
    parser.add_argument("--list-models", action="store_true", help="List available models")
    parser.add_argument("--list-metrics", action="store_true", help="List available metrics")
    parser.add_argument("--list-metrics-for", help="List metrics for specific evaluation type")
    
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
    
    if args.list_metrics:
        registry = get_metric_registry()
        metrics = registry.list_metrics()
        print("Available metrics:")
        current_dimension = None
        for metric in metrics:
            if metric.dimension != current_dimension:
                current_dimension = metric.dimension
                print(f"\n{current_dimension.title()} dimension:")
            print(f"  - {metric.name}: {metric.description}")
        return
    
    if args.list_metrics_for:
        registry = get_metric_registry()
        try:
            metrics = registry.list_metrics(args.list_metrics_for)
            print(f"Available metrics for {args.list_metrics_for}:")
            for metric in metrics:
                print(f"  - {metric.name}: {metric.description}")
        except ValueError:
            print(f"Unknown evaluation type: {args.list_metrics_for}")
            print(f"Available evaluation types: {get_available_dimensions()}")
        return
    
    # Handle legacy phase parameter
    evaluation_type = args.evaluation_type
    if args.phase is not None:
        phase_to_type = {1: "syntax", 2: "semantics", 3: "consistency"}
        evaluation_type = phase_to_type.get(args.phase, evaluation_type)
        logger.warning(f"Using legacy --phase {args.phase} parameter, consider using --evaluation-type {evaluation_type}")
    
    # Validate prerequisites
    legacy_phase = {"syntax": 1, "semantics": 2, "consistency": 3}.get(evaluation_type, 1)
    if not validate_prerequisites(legacy_phase):
        sys.exit(1)
    
    # Create generation config
    generation_config = GenerationConfig(
        max_tokens=args.max_tokens,
        temperature=args.temperature
    )
    
    # Determine run mode
    single_mode = args.task and args.method and args.model
    batch_mode = args.tasks and args.methods and args.models
    
    # Collect metric-specific parameters
    metric_params = {}
    if args.k is not None:
        metric_params['k'] = args.k
    if args.level is not None:
        metric_params['level'] = args.level
    
    if single_mode:
        # Single benchmark
        result = run_single_benchmark(
            args.task, args.method, args.model, evaluation_type, args.metric,
            source_file=args.source_file,
            generation_config=generation_config,
            **metric_params
        )
        
        if result["success"]:
            eval_result = result["evaluation_result"]
            
            # Determine evaluation type from the result object type instead of variable
            from tla_eval.evaluation.base.result_types import SyntaxEvaluationResult, SemanticEvaluationResult
            
            if isinstance(eval_result, SyntaxEvaluationResult):
                print(f"\nSyntax Evaluation Results: {'✓ PASS' if eval_result.overall_success else '✗ FAIL'}")
                print(f"Generation time: {eval_result.generation_time:.2f}s")
                print(f"Compilation time: {eval_result.compilation_time:.2f}s")
                print(f"Syntax errors: {len(eval_result.syntax_errors)}")
                print(f"Semantic errors: {len(eval_result.semantic_errors)}")
                
                if not eval_result.overall_success:
                    if not eval_result.generation_successful:
                        print(f"Generation error: {eval_result.generation_error}")
                    if not eval_result.compilation_successful:
                        print(f"Compilation errors: {eval_result.syntax_errors + eval_result.semantic_errors}")
                        
            elif isinstance(eval_result, SemanticEvaluationResult):
                print(f"\nSemantics Evaluation Results: {'✓ PASS' if eval_result.overall_success else '✗ FAIL'}")
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
                    
            elif evaluation_type == "consistency":
                print(f"\nConsistency Evaluation Results: {'✓ PASS' if eval_result.overall_success else '✗ FAIL'}")
                print(f"Trace generation time: {eval_result.trace_generation_time:.2f}s")
                print(f"Trace conversion time: {eval_result.trace_conversion_time:.2f}s")
                print(f"Trace validation time: {eval_result.trace_validation_time:.2f}s")
                print(f"Generated trace count: {eval_result.generated_trace_count}")
                print(f"Validated events: {eval_result.validated_events}")
                
                if not eval_result.overall_success:
                    if not eval_result.trace_generation_successful:
                        print(f"Trace generation error: {eval_result.trace_generation_error}")
                    if not eval_result.trace_conversion_successful:
                        print(f"Trace conversion error: {eval_result.trace_conversion_error}")
                    if not eval_result.trace_validation_successful:
                        print(f"Trace validation error: {eval_result.trace_validation_error}")
                        
                # Show file locations
                if eval_result.raw_trace_files:
                    print(f"Raw traces: {', '.join(eval_result.raw_trace_files)}")
                if eval_result.converted_trace_files:
                    print(f"Converted traces: {', '.join(eval_result.converted_trace_files)}")
                if eval_result.specification_files:
                    print(f"Specifications: {', '.join(eval_result.specification_files)}")
            
        else:
            print(f"Benchmark failed: {result['error']}")
            sys.exit(1)
    
    elif batch_mode:
        # Batch benchmark
        run_batch_benchmark(
            args.tasks, args.methods, args.models, evaluation_type, args.metric,
            output_dir=args.output, generation_config=generation_config,
            **metric_params
        )
    
    else:
        parser.error("Must specify either single mode (--task --method --model) or batch mode (--tasks --methods --models)")


if __name__ == "__main__":
    main()