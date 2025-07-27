# Usage Guide

This document provides detailed instructions for using the LLM-Generated TLA+ Benchmark Framework.

## Command Line Interface

The main entry point for running benchmarks is the `run_benchmark.py` script located in the `scripts/` directory.

### Basic Command Structure

```bash
python3 scripts/run_benchmark.py --task <task> --method <method> --model <model> [--metric <metric>] [options]
```

### Required Parameters

Three parameters are required for single benchmark runs:

- `--task`: Target system to evaluate (e.g., `etcd`)
- `--method`: Generation method (e.g., `direct_call`)
- `--model`: LLM model to use (e.g., `gpt-4`, `my_yunwu`)

### Optional Evaluation Parameters

- `--metric`: Specific metric to run (e.g., `compilation_check`, `trace_validation`)
- `--evaluation-type`: Evaluation dimension (`syntax`, `semantics`, or `consistency`) - uses default metric for the dimension
- `--k`: Number of attempts for pass@k metrics
- `--level`: Granularity level for progressive metrics

### Environment Setup

Before running benchmarks, ensure you have the necessary API keys set as environment variables:

#### For OpenAI Models
```bash
export OPENAI_API_KEY="your-openai-api-key"
```

#### For Yunwu Models
```bash
export YUNWU_API_KEY="your-yunwu-api-key"
```

#### For Claude Models
```bash
export ANTHROPIC_API_KEY="your-anthropic-api-key"
```

## Single Benchmark Examples

### Syntax Evaluation: Compilation Check
Test whether the generated TLA+ specification compiles successfully.

```bash
# Using default metric for syntax dimension
export OPENAI_API_KEY="your-key"
python3 scripts/run_benchmark.py --task etcd --method direct_call --model gpt-4 --evaluation-type syntax

# Or specify the metric explicitly
python3 scripts/run_benchmark.py --task etcd --method direct_call --model gpt-4 --metric compilation_check
```

### Semantics Evaluation: Invariant Verification
Test whether TLC can validate the specification's invariants (requires syntax evaluation to pass first).

```bash
# Using default metric for semantics dimension
export OPENAI_API_KEY="your-key"
python3 scripts/run_benchmark.py --task etcd --method direct_call --model gpt-4 --evaluation-type semantics

# Or specify the metric explicitly
python3 scripts/run_benchmark.py --task etcd --method direct_call --model gpt-4 --metric invariant_verification
```

### Consistency Evaluation: Trace Validation
Test whether TLC can validate real system traces against the specification (most comprehensive).

```bash
# Using default metric for consistency dimension
export OPENAI_API_KEY="your-key"
python3 scripts/run_benchmark.py --task etcd --method direct_call --model my_yunwu --evaluation-type consistency

# Or specify the metric explicitly
python3 scripts/run_benchmark.py --task etcd --method direct_call --model my_yunwu --metric trace_validation
```

## Batch Benchmark Examples

Run multiple combinations of tasks, methods, and models:

### Multiple Models
```bash
export OPENAI_API_KEY="your-openai-key"
export ANTHROPIC_API_KEY="your-anthropic-key"

python3 scripts/run_benchmark.py \
  --tasks etcd \
  --methods direct_call \
  --models gpt-4 my_claude \
  --evaluation-type syntax \
  --output results/comparison
```

### Multiple Evaluation Types
```bash
# Run all evaluation types for a single configuration
for eval_type in syntax semantics consistency; do
  python3 scripts/run_benchmark.py \
    --task etcd \
    --method direct_call \
    --model gpt-4 \
    --evaluation-type $eval_type \
    --output results/evaluation_progression
done
```



## Information Commands

### List Available Options

```bash
# List all available tasks
python3 scripts/run_benchmark.py --list-tasks

# List all available methods
python3 scripts/run_benchmark.py --list-methods

# List all configured models
python3 scripts/run_benchmark.py --list-models

# List all available metrics
python3 scripts/run_benchmark.py --list-metrics

# List metrics for specific evaluation type
python3 scripts/run_benchmark.py --list-metrics-for syntax
python3 scripts/run_benchmark.py --list-metrics-for semantics
python3 scripts/run_benchmark.py --list-metrics-for consistency
```

## Configuration Files

### Model Configuration
Models can be configured in YAML files or through environment variables. See the main README for configuration examples.

### Task Configuration
Each task has its configuration in `tla_eval/tasks/<task_name>/task.yaml` with system-specific settings.

## Advanced Usage

### Custom Generation Configuration
```python
# For programmatic usage
from tla_eval.models.base import GenerationConfig

config = GenerationConfig(
    max_tokens=4096,
    temperature=0.1,
    top_p=0.9
)
```

### Direct Consistency Evaluation
```python
# For direct consistency evaluation usage
from tla_eval.evaluation.consistency.trace_validation import TraceValidationEvaluator

evaluator = TraceValidationEvaluator()
result = evaluator.evaluate('etcd', {
    'node_count': 3,
    'duration_seconds': 60,
    'client_qps': 10.0
})
```