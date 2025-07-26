# Usage Guide

This document provides detailed instructions for using the LLM-Generated TLA+ Benchmark Framework.

## Command Line Interface

The main entry point for running benchmarks is the `run_benchmark.py` script located in the `scripts/` directory.

### Basic Command Structure

```bash
python3 scripts/run_benchmark.py --task <task> --method <method> --model <model> --phase <phase> [options]
```

### Required Parameters

All four parameters are required for single benchmark runs:

- `--task`: Target system to evaluate (e.g., `etcd`)
- `--method`: Generation method (e.g., `direct_call`)
- `--model`: LLM model to use (e.g., `gpt-4`, `my_yunwu`)
- `--phase`: Evaluation phase (1, 2, or 3)

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

### Phase 1: Compilation Check
Test whether the generated TLA+ specification compiles successfully.

```bash
# Using OpenAI GPT-4
export OPENAI_API_KEY="your-key"
python3 scripts/run_benchmark.py --task etcd --method direct_call --model gpt-4 --phase 1
```

### Phase 2: Invariant Verification
Test whether TLC can validate the specification's invariants (requires Phase 1 to pass first).

```bash
export OPENAI_API_KEY="your-key"
python3 scripts/run_benchmark.py --task etcd --method direct_call --model gpt-4 --phase 2
```

### Phase 3: Trace Validation
Test whether TLC can validate real system traces against the specification (most comprehensive).

```bash
export OPENAI_API_KEY="your-key"
python3 scripts/run_benchmark.py --task etcd --method direct_call --model my_yunwu --phase 3
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
  --phase 1 \
  --output results/comparison
```

### Multiple Phases
```bash
# Run all phases for a single configuration
for phase in 1 2 3; do
  python3 scripts/run_benchmark.py \
    --task etcd \
    --method direct_call \
    --model gpt-4 \
    --phase $phase \
    --output results/phase_progression
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

### Direct Phase 3 Evaluation
```python
# For direct Phase 3 usage
from tla_eval.evaluation.phases.phase3 import Phase3Evaluator

evaluator = Phase3Evaluator()
result = evaluator.evaluate('etcd', {
    'node_count': 3,
    'duration_seconds': 60,
    'client_qps': 10.0
})
```