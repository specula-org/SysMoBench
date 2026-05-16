#!/usr/bin/env python3
"""
Single-cell SysMoBench runner.

Runs one (task, method, model, metric) combination and prints results.
Phase 1 (compilation/syntax) and Phase 2 (runtime/semantics) live here;
Phase 3 (transition validation) is launched via scripts/launch_tv_eval.sh,
and Phase 4 (invariant verification) is selected by --metric invariant_verification.

For batch evaluation across many models/systems, use scripts/run_batch_experiment.py.

Usage:
    python3 scripts/run_benchmark.py --task etcd --method direct_call --model claude --metric compilation_check
    python3 scripts/run_benchmark.py --task etcd --method direct_call --model claude --metric runtime_check --tlc-timeout 120
    python3 scripts/run_benchmark.py --task etcd --method direct_call --model claude --metric coverage --spec-file path/to/spec.tla
    python3 scripts/run_benchmark.py --list-metrics
"""

import argparse
import sys
import time
import inspect
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env", override=False)
except ImportError:
    pass

from tla_eval.config import get_configured_model, get_config_manager
from tla_eval.tasks.loader import get_task_loader
from tla_eval.methods import get_method
from tla_eval.utils import validate_tla_tools_setup
from tla_eval.evaluation.base import create_evaluator, get_metric_registry

import logging

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


def _display_evaluation_results(eval_result):
    """Print a unified summary across SyntaxEvaluationResult / SemanticEvaluationResult / TransitionValidationResult."""
    from tla_eval.evaluation.base.result_types import (
        SyntaxEvaluationResult,
        SemanticEvaluationResult,
        TransitionValidationResult,
    )

    if isinstance(eval_result, SyntaxEvaluationResult):
        print(f"\nSyntax Evaluation Results: {'✓ PASS' if eval_result.overall_success else '✗ FAIL'}")
        print(f"Generation time: {eval_result.generation_time:.2f}s")
        print(f"Compilation time: {eval_result.compilation_time:.2f}s")
        print(f"Syntax errors: {len(eval_result.syntax_errors)}")
        print(f"Semantic errors: {len(eval_result.semantic_errors)}")

        if hasattr(eval_result, 'total_actions') and eval_result.total_actions:
            success_rate = getattr(eval_result, 'action_success_rate', 0.0)
            successful = getattr(eval_result, 'successful_actions', 0)
            total = eval_result.total_actions
            print(f"Action success rate: {successful}/{total} ({success_rate:.1%})")

            vars_added = getattr(eval_result, 'total_variables_added', 0)
            funcs_added = getattr(eval_result, 'total_functions_added', 0)
            recovery_attempts = getattr(eval_result, 'total_recovery_attempts', 0)
            if vars_added or funcs_added or recovery_attempts:
                print(f"Recovery statistics: {vars_added} variables, {funcs_added} functions added, {recovery_attempts} attempts")

        if hasattr(eval_result, 'output_directory') and eval_result.output_directory:
            print(f"Results saved to: {eval_result.output_directory}")

        if not eval_result.overall_success:
            if not eval_result.generation_successful:
                print(f"Generation error: {eval_result.generation_error}")
            if not eval_result.compilation_successful:
                all_errors = eval_result.syntax_errors + eval_result.semantic_errors
                if all_errors:
                    print("Compilation errors:")
                    for i, error in enumerate(all_errors, 1):
                        print(f"  {i}. {error}")
                        print()

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

        print(f"Specification: {eval_result.specification_file}")
        if eval_result.config_file_path:
            print(f"Config file: {eval_result.config_file_path}")

    elif isinstance(eval_result, TransitionValidationResult):
        print(f"\nTransition Validation Results: {'✓ PASS' if eval_result.overall_success else '✗ FAIL'}")
        print(f"Score: {eval_result.score:.1%} ({eval_result.total_passed}/{eval_result.total_windows} windows)")
        print(f"Elapsed: {eval_result.elapsed_seconds:.0f}s")
        if eval_result.per_action_pass_rates:
            print("Per-action pass rates:")
            for action, rate in sorted(eval_result.per_action_pass_rates.items()):
                print(f"  - {action}: {rate:.1%}")
        if eval_result.workspace_dir:
            print(f"Workspace: {eval_result.workspace_dir}")
        if eval_result.error_message:
            print(f"Error: {eval_result.error_message}")


# Per-metric whitelist: which CLI metric_params each evaluator accepts.
METRIC_PARAM_WHITELIST = {
    "compilation_check": {"tlc_timeout"},
    "action_decomposition": {"tlc_timeout", "keep_temp_files"},
    "runtime_check": {"tlc_timeout"},
    "coverage": {"tlc_timeout", "coverage_interval"},
    "runtime_coverage": {
        "num_simulations", "simulation_depth", "traces_per_simulation",
        "tlc_timeout", "coverage_interval",
    },
    "invariant_verification": {
        "tlc_timeout", "templates_dir", "translator_type", "agent_timeout",
    },
    "transition_validation": {
        "tv_agent", "tv_model", "tv_budget", "tv_timeout", "workspace_root",
    },
}


def filter_metric_params(metric: str, params: dict) -> dict:
    """Drop params not relevant to `metric`, with a warning."""
    allowed = METRIC_PARAM_WHITELIST.get(metric, set())
    filtered = {k: v for k, v in params.items() if k in allowed}

    ignored = set(params.keys()) - allowed
    if ignored:
        logger.warning(f"Metric '{metric}' does not support parameters: {sorted(ignored)}")
        if allowed:
            logger.info(f"Metric '{metric}' supports these parameters: {sorted(allowed)}")
        else:
            logger.info(f"Metric '{metric}' does not accept any metric-specific parameters")

    return filtered


def validate_prerequisites() -> bool:
    """Verify TLA+ tools (java + tla2tools.jar) are installed."""
    logger.info("Validating prerequisites...")
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


def _call_evaluator_with_files(evaluator, generation_result, task_name, method_name, model_name,
                              spec_module, spec_file=None, config_file=None):
    """Invoke evaluator.evaluate, passing spec_module/spec_file_path/config_file_path only if its signature accepts them."""
    evaluate_method = getattr(evaluator, 'evaluate', None)
    if not evaluate_method:
        raise ValueError(f"Evaluator {type(evaluator).__name__} does not have an evaluate method")

    sig = inspect.signature(evaluate_method)
    args = [generation_result, task_name, method_name, model_name]
    kwargs = {}

    if 'spec_module' in sig.parameters:
        if len(sig.parameters) > 4:
            args.append(spec_module)
        else:
            kwargs['spec_module'] = spec_module

    if 'spec_file_path' in sig.parameters and spec_file:
        kwargs['spec_file_path'] = spec_file
    if 'config_file_path' in sig.parameters and config_file:
        kwargs['config_file_path'] = config_file

    return evaluate_method(*args, **kwargs)


def run_single_benchmark(task_name: str, method_name: str, model_name: str,
                        metric: Optional[str] = None,
                        source_file: Optional[str] = None,
                        traces_folder: Optional[str] = None,
                        spec_file: Optional[str] = None,
                        config_file: Optional[str] = None,
                        language: str = "TLA+",
                        **metric_params) -> dict:
    """
    Run one (task, method, model, metric) cell.

    If `spec_file` is given, generation is skipped and the existing spec is evaluated.
    Otherwise the model generates a spec via `method`, then the metric's evaluator runs on it.
    """
    if metric is None:
        metric = "compilation_check"

    registry = get_metric_registry()
    try:
        registry.get_metric(metric)
    except ValueError as e:
        available = sorted({m.name for m in registry.list_metrics()})
        raise ValueError(f"Unknown metric '{metric}'. Available metrics: {available}") from e

    logger.info(f"Running metric '{metric}': {task_name}/{method_name}/{model_name}")

    try:
        task_loader = get_task_loader()
        task = task_loader.load_task(task_name, source_file, traces_folder)
        logger.info(f"Loaded task: {task.task_name} ({task.system_type})")

        prompt_template = task_loader.get_task_prompt(task_name, method_name, language=language)
        logger.info(f"Loaded prompt template ({len(prompt_template)} chars, language={language})")

        from tla_eval.models.base import GenerationResult

        if spec_file:
            logger.info(f"Using existing specification file: {spec_file}")
            if config_file:
                logger.info(f"Using existing config file: {config_file}")

            if not Path(spec_file).exists():
                logger.error(f"Specified spec file does not exist: {spec_file}")
                return {"success": False, "error": f"Spec file not found: {spec_file}"}
            if config_file and not Path(config_file).exists():
                logger.error(f"Specified config file does not exist: {config_file}")
                return {"success": False, "error": f"Config file not found: {config_file}"}

            try:
                with open(spec_file, 'r', encoding='utf-8') as f:
                    spec_content = f.read()
            except Exception as e:
                logger.error(f"Failed to read spec file {spec_file}: {e}")
                return {"success": False, "error": f"Failed to read spec file: {e}"}

            generation_result = GenerationResult(
                generated_text=spec_content,
                metadata={
                    'method': method_name,
                    'latency_seconds': 0.0,
                    'using_existing_files': True,
                    'spec_file': spec_file,
                    'config_file': config_file,
                },
                timestamp=time.time(),
                success=True,
                error_message=None,
            )

        else:
            get_configured_model(model_name)  # validate model exists in config
            logger.info(f"Loaded model: {model_name}")

            method = get_method(method_name, language=language)
            logger.info(f"Using method: {method_name} (language={language})")

            start_time = time.time()
            generation_output = method.generate(task, model_name)
            generation_time = time.time() - start_time
            logger.info(f"Generation completed in {generation_time:.2f}s")

            generation_result = GenerationResult(
                generated_text=generation_output.tla_specification,
                metadata={
                    'method': generation_output.method_name,
                    'latency_seconds': generation_time,
                    **generation_output.metadata,
                },
                timestamp=time.time(),
                success=generation_output.success,
                error_message=generation_output.error_message,
            )

        # Syntax metrics still produce a useful score from a broken spec; everything else needs a parseable one.
        if metric not in {"compilation_check", "action_decomposition"} and not generation_result.success and not spec_file:
            logger.error(f"Cannot evaluate '{metric}': TLA+ generation failed and no --spec-file provided")
            return {"success": False, "error": "TLA+ generation failed"}

        filtered_params = filter_metric_params(metric, metric_params)
        evaluator = create_evaluator(metric, language=language, **filtered_params)

        evaluation_result = _call_evaluator_with_files(
            evaluator, generation_result, task_name, method_name, model_name,
            task.spec_module, spec_file, config_file,
        )
        logger.info(f"Metric '{metric}': {'✓ PASS' if evaluation_result.overall_success else '✗ FAIL'}")

        return {
            "success": True,
            "metric": metric,
            "evaluation_result": evaluation_result,
            "error": None,
        }

    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        return {
            "success": False,
            "metric": metric,
            "evaluation_result": None,
            "error": str(e),
        }


def main():
    parser = argparse.ArgumentParser(
        description="SysMoBench single-cell runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 scripts/run_benchmark.py --task etcd --method direct_call --model claude
  python3 scripts/run_benchmark.py --task etcd --method direct_call --model claude --metric runtime_check
  python3 scripts/run_benchmark.py --task etcd --method direct_call --model claude --metric coverage --spec-file path/to/spec.tla
  python3 scripts/run_benchmark.py --list-metrics
        """,
    )

    parser.add_argument("--metric",
                       help="Metric to run (default: compilation_check). See --list-metrics.")

    parser.add_argument("--tlc-timeout", type=int,
                       help="Timeout for TLC model checking in seconds")
    parser.add_argument("--inv-translator-type", choices=["direct", "agent"], default="agent",
                       help="Invariant translator: 'agent' (Claude Code agent, default) or 'direct' (single LLM call)")

    # Transition validation parameters (--metric transition_validation).
    parser.add_argument("--tv-agent", help="Coding-agent CLI for transition validation (e.g. claude-code, codex)")
    parser.add_argument("--tv-model", help="Model override passed to the coding-agent CLI for transition validation")
    parser.add_argument("--tv-budget", type=float, help="Max API budget (USD) for transition validation (default: 5)")
    parser.add_argument("--tv-timeout", type=int, help="Timeout (seconds) for transition validation (default: 1800)")
    parser.add_argument("--yes", action="store_true",
                       help="Skip the interactive cost confirmation for transition validation")

    parser.add_argument("--language", default="TLA+",
                       help="Specification language (default: TLA+). Available backends are auto-discovered.")
    parser.add_argument("--task", help="Task name")
    parser.add_argument("--method", help="Method name")
    parser.add_argument("--model", help="Model name")
    parser.add_argument("--source-file", help="Specific source file within task")
    parser.add_argument("--traces-folder", help="Specific traces folder within task")

    parser.add_argument("--spec-file", help="Path to existing TLA+ specification file (.tla)")
    parser.add_argument("--config-file", help="Path to existing TLC configuration file (.cfg)")

    parser.add_argument("--list-tasks", action="store_true", help="List available tasks")
    parser.add_argument("--list-methods", action="store_true", help="List available methods")
    parser.add_argument("--list-models", action="store_true", help="List available models")
    parser.add_argument("--list-metrics", action="store_true", help="List available metrics")

    args = parser.parse_args()

    if args.list_tasks:
        for task in get_task_loader().list_available_tasks():
            print(f"  - {task}")
        return

    if args.list_methods:
        from tla_eval.methods import list_available_methods
        for m in list_available_methods():
            print(f"  - {m}")
        return

    if args.list_models:
        for m in get_config_manager().list_available_models():
            print(f"  - {m}")
        return

    if args.list_metrics:
        for metric in get_metric_registry().list_metrics():
            print(f"  - {metric.name}: {metric.description}")
        return

    if not (args.task and args.method and args.model):
        parser.error("Must specify --task, --method, and --model")

    if args.spec_file:
        if not Path(args.spec_file).exists():
            print(f"Error: Specified spec file does not exist: {args.spec_file}")
            sys.exit(1)
        args.spec_file = str(Path(args.spec_file).resolve())
    if args.config_file:
        if not Path(args.config_file).exists():
            print(f"Error: Specified config file does not exist: {args.config_file}")
            sys.exit(1)
        args.config_file = str(Path(args.config_file).resolve())

    if not validate_prerequisites():
        sys.exit(1)

    # Backend availability check — fail fast before generation/evaluator setup
    # so missing helper jars / missing mono / etc. surface as tooling issues
    # rather than getting wrapped as syntax errors inside the correction loop.
    try:
        from tla_eval.languages import get as _get_backend
        _backend = _get_backend(args.language)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)
    _backend_err = _backend.check_available()
    if _backend_err:
        print(
            f"Error: backend '{_backend.name}' is not ready: {_backend_err}\n"
            f"Fix the tooling above and rerun. (Use --language to pick a different backend.)"
        )
        sys.exit(1)

    metric_params = {}
    if args.tlc_timeout is not None:
        metric_params['tlc_timeout'] = args.tlc_timeout
    if args.metric == "invariant_verification":
        metric_params['translator_type'] = args.inv_translator_type
    if args.tv_agent:
        metric_params['tv_agent'] = args.tv_agent
    if args.tv_model:
        metric_params['tv_model'] = args.tv_model
    if args.tv_budget is not None:
        metric_params['tv_budget'] = args.tv_budget
    if args.tv_timeout is not None:
        metric_params['tv_timeout'] = args.tv_timeout

    if args.metric == "transition_validation" and not args.yes:
        print(
            "\n[!] transition_validation runs an external coding agent against the live system harness.\n"
            "    Expect 30 min to several hours per spec and roughly $1-4 in agent API spend.\n"
            "    Pass --yes to skip this confirmation.\n"
        )
        try:
            answer = input("Proceed? [y/N] ").strip().lower()
        except EOFError:
            answer = ""
        if answer not in ("y", "yes"):
            print("Aborted.")
            sys.exit(0)

    result = run_single_benchmark(
        args.task,
        args.method,
        args.model,
        metric=args.metric,
        source_file=args.source_file,
        traces_folder=args.traces_folder,
        spec_file=args.spec_file,
        config_file=args.config_file,
        language=args.language,
        **metric_params,
    )

    if result["success"]:
        _display_evaluation_results(result["evaluation_result"])
    else:
        print(f"Benchmark failed: {result['error']}")
        sys.exit(1)


if __name__ == "__main__":
    main()
