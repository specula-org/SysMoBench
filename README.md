# LLM-Generated TLA+ Benchmark Framework

A comprehensive benchmark framework for evaluating Large Language Models' ability to generate and validate TLA+ specifications for distributed systems.

## Overview

This benchmark framework evaluates LLMs across multiple dimensions to assess their capability in formal specification generation and validation. The framework is designed with extensibility in mind, allowing for systematic evaluation of different models, target systems, and interaction methods.

### Framework Dimensions

The benchmark operates across three key dimensions:

#### 1. **Models**
- Support for multiple LLM providers (OpenAI, Claude, etc.)
- Configurable model parameters and API integration
- Extensible adapter pattern for adding new models

#### 2. **Tasks (Target Systems)**
- **etcd raft**
- **Future systems...**

#### 3. **Invocation Methods**
- **Direct API calls**: Immediate LLM responses to prompts
- **Agent-based interaction**: Multi-step reasoning agents (TODO)

### Evaluation Metrics

The framework employs three evaluation metrics:

1. **Compilation Check**: Whether the generated TLA+ specification compiles successfully
2. **Invariant Verification**: Whether TLC model checker validates the specification's invariants
3. **Trace Validation**: Whether TLC successfully validates real system execution traces against the specification

## Architecture

The framework implements a three-phase evaluation pipeline:

- **Phase 1**: Basic TLA+ specification generation and compilation validation
- **Phase 2**: Invariant generation and Invariant-based model checking with TLC
- **Phase 3**: Real system trace generation and validation against TLA+ specifications

## Quick Start

### Prerequisites

- **Python 3.8+**
- **Java 11+** (for TLC model checker)
- **Go 1.19+** (for etcd trace generation)
- **Git** (for repository management)

### Installation
1. Install Python dependencies:
```bash
pip install -r requirements.txt
```

2. Set up environment variables:
```bash
# For OpenAI models
export OPENAI_API_KEY="your-api-key"
```

3. Initialize repositories and TLA+ tools:
```bash
# The framework will automatically set up required repositories and tools
python -m tla_eval.setup
```

### Running Your First Benchmark

```bash
# Basic usage: python scripts/run_benchmark.py --task <task> --method <method> --model <model> --phase <phase>
python3 scripts/run_benchmark.py --task etcd --method direct_call --model gpt-4 --phase 3
```

For detailed usage instructions and advanced configuration options, see [Usage.md](./docs/Usage.md).

## File Structure

```
LLM_Gen_TLA_benchmark_framework/
├── README.md                         # This file
├── requirements.txt                  # Python dependencies
├── lib/                              # External tools
├── data/                             # Data and specifications
│   ├── spec/                         # TLA+ specifications
│   │   ├── common/                   # Common TLA+ modules
│   │   └── etcd/                     # etcd-specific specifications
│   ├── sys_traces/                   # System-generated traces
│   ├── traces/                       # Converted traces for validation
│   └── repositories/                 # External system repositories
├── tla_eval/                         # Main framework code
│   ├── core/                         # Core functionality
│   │   ├── trace_generation/         # System trace generation
│   │   ├── spec_processing/          # TLA+ specification processing
│   │   └── verification/             # TLC verification
│   ├── evaluation/                   # Evaluation phases
│   │   └── phases/                   # Phase implementations
│   ├── models/                       # LLM model adapters
│   ├── tasks/                        # Task-specific configurations
├── config/                           # Configuration management
└── tests/                            # Test suites
```

## Configuration

### Model Configuration

Configure LLM models in your environment or configuration files:

```python
# Example OpenAI model configuration
{
    "model_name": "gpt-4",
    "api_key": "your-openai-api-key",
    "base_url": "https://api.openai.com/v1",
    "temperature": 0.1,
    "max_tokens": 4000
}
```

### Task Configuration

Each task has its own configuration in `tla_eval/tasks/<task_name>/`:

```yaml
# Example etcd task configuration
system_name: "etcd"
specification_files:
  - "etcdraft.tla"
  - "etcdraft.cfg"
trace_generation:
  node_count: 3
  duration_seconds: 60
  client_qps: 10.0
patch_files:
  - "src": "patches/etcd_raft_trace.patch"
    "target": "raft"
```
