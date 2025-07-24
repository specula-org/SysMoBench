# LLM-Generated TLA+ Benchmark Framework

A framework for evaluating Large Language Models' capabilities in generating code-level TLA+ specifications from real-world distributed and concurrent systems.

## Repository Structure

```
LLM_Gen_TLA_benchmark_framework/
  ├── README.md
  ├── LICENSE
  ├── requirements.txt            # Dependencies
  ├── pyproject.toml             # Modern Python project config
  ├── Makefile                   # Common commands
  ├── tla_eval/                  # Main package (like lm_eval, bigcode_eval)
  │   ├── __init__.py
  │   ├── models/                # Model adapters
  │   ├── tasks/                 # Benchmark tasks (like evals/registry)
  │   ├── evaluation/            # Evaluation logic
  │   └── utils/                 # Utilities
  ├── scripts/                   # Executable scripts
  │   ├── run_benchmark.py      # Main entry point
  │   └── evaluate_spec.py      # Single spec evaluation
  ├── config/                    # Configuration files
  │   ├── models.yaml
  │   └── tasks.yaml
  ├── data/                      # Data storage
  │   ├── specifications/       # Generated TLA+ specs
  │   ├── results/              # Evaluation results
  │   └── traces/               # Execution traces
  ├── tests/                     # Test suite
  │   ├── test_models/
  │   ├── test_evaluation/
  │   └── test_tasks/
  ├── docs/                      # Documentation
  └── examples/                  # Usage examples
      ├── basic_usage.py
      └── custom_model.py
```

## Getting Started

### Prerequisites

TODO

### Installation

```bash
git clone https://github.com/Qian-Cheng-nju/LLM_Gen_TLA_benchmark_framework.git
cd LLM_Gen_TLA_benchmark_framework
TODO
```

### Quick Start

```bash
TODO
```

## Evaluation Metrics

1. **Compilation Success**: Does the generated TLA+ specification compile without errors?
2. **Runtime Checking**: Can the specification run model checking?
3. **Consistency Verification**: Does the specification keep consistent with the system?
