# SysSpecBench

A comprehensive benchmark framework for evaluating AI agents' capability in formally specifying real-world concurrent and distributed systems using TLA+ specifications.

## Overview

This benchmark framework implements a rigorous four-phase evaluation methodology to assess whether AI-generated TLA+ specifications meet the demands of practical software development. The framework targets industrial-grade systems and provides progressive quality assessment from basic syntactic correctness to behavioral conformance with real-world systems.

## Four-Phase Evaluation Methodology

### Phase 1: Syntax Validation
**Goal**: Ensure the generated TLA+ specification is syntactically correct and well-formed.
- **Method**: Use SANY parser to check grammar, operators, and module structure, including individual action compilation
- **Metrics**: `compilation_check`, `action_decomposition`

### Phase 2: Semantic Execution
**Goal**: Verify that the specification is executable and semantically meaningful.
- **Method**: Run TLC model checker to test executability and action coverage
- **Metrics**: `runtime_check`, `coverage`

### Phase 3: Correctness Verification
**Goal**: Validate safety and liveness properties using expert-designed invariants.
- **Method**: Test against domain-specific invariants (TypeOK, Safety, Liveness)
- **Metrics**: `invariant_verification`

### Phase 4: Behavioral Conformance
**Goal**: Ensure the specification matches real system behavior.
- **Method**: Compare formal model execution against actual system traces
- **Metrics**: `trace_validation`

## Quick Start

### Prerequisites

- **Python 3.8+**
- **Java 11+** (for TLC model checker)
- **Git** (for repository management)

### Installation

1. Install Python dependencies:
```bash
pip install -r requirements.txt
```

2. Set up environment variables for LLM models:
```bash
# For OpenAI models
export OPENAI_API_KEY="your-api-key"

# For Anthropic models
export ANTHROPIC_API_KEY="your-api-key"

# For Google GenAI models
export GENAI_API_KEY="your-api-key"
```

3. Initialize TLA+ tools (automatic on first run):
```bash
python scripts/run_benchmark.py --list-metrics
```

## Usage

### Basic Usage Pattern

```bash
python scripts/run_benchmark.py --task <task> --method <method> --model <model> --metric <metric>
```

### Available Tasks
- `etcd`: etcd Raft consensus protocol
- `spin`: Spin lock implementation
- More systems can be added following the framework extension process

### Available Methods
- `direct_call`: Single-shot LLM prompting
- `agent_based`: Multi-step reasoning with automatic error correction (in development)

### Available Models

The framework supports models from major providers:
- **OpenAI**: 
- **Anthropic**: 
- **Google GenAI**: 
- **DeepSeek**:

**Special Model:**
- `with_exist_spec`: Use existing TLA+ specification files

### Available Metrics

**Phase 1 (Syntax):**
- `compilation_check`: Basic TLA+ compilation validation using SANY parser
- `action_decomposition`: Individual action syntax validation with granular error reporting

**Phase 2 (Semantics):**
- `runtime_check`: Model checking with specification's own invariants
- `coverage`: TLA+ specification coverage analysis using TLC statistics

**Phase 3 (Invariant Checking):**
- `invariant_verification`: Expert-designed invariant verification

**Phase 4 (Consistency):**
- `trace_validation`: Trace generation and test

**Integrated:**
- `composite`: Multi-phase evaluation combining action decomposition, compilation, runtime, and invariant verification with agent correcting loop

## Usage Examples

### Phase 1-2: Universal Evaluation (Works with Any TLA+ Spec)

These metrics are fully automated and can test any TLA+ specification:

```bash
# Test syntax validation
python scripts/run_benchmark.py --task spin --method direct_call --model gpt5 --metric compilation_check
python scripts/run_benchmark.py --task spin --method direct_call --model gpt5 --metric action_decomposition

# Test semantic execution  
python scripts/run_benchmark.py --task spin --method direct_call --model gpt5 --metric runtime_check
python scripts/run_benchmark.py --task spin --method direct_call --model gpt5 --metric coverage
```

### Phase 3-4: System-Specific Evaluation

These metrics require task-specific configurations:

```bash
# Correctness verification (requires task-specific invariants)
python scripts/run_benchmark.py --task etcd --method direct_call --model openai_gpt4 --metric invariant_verification

# Behavioral conformance (requires manual instrumentation alignment)
python scripts/run_benchmark.py --task etcd --method direct_call --model openai_gpt4 --metric trace_validation
```

### Using Existing Specifications

```bash
# Test your own TLA+ specification with universal metrics
python scripts/run_benchmark.py --task spin --method direct_call --model with_exist_spec --metric runtime_check --spec-file your_spec.tla

# Use existing spec with existing config
python scripts/run_benchmark.py --task spin --method direct_call --model with_exist_spec --metric runtime_check --spec-file your_spec.tla --config-file your_config.cfg
```

### List Available Options

```bash
python scripts/run_benchmark.py --list-metrics  # All metrics
python scripts/run_benchmark.py --list-models   # Supported models  
python scripts/run_benchmark.py --list-tasks    # Available tasks
```

## Framework Architecture

```
LLM_Gen_TLA_benchmark_framework/
├── scripts/
│   └── run_benchmark.py              # Main benchmark runner
├── tla_eval/                         # Framework core
│   ├── evaluation/                   # Evaluation phases
│   │   ├── syntax/                   # Phase 1: Compilation
│   │   ├── semantics/                # Phase 2 & 3: Execution & Invariants
│   │   ├── consistency/              # Phase 4: Conformance Testing
│   │   └── composite/                # Integrated evaluation
│   ├── models/                       # LLM model adapters
│   ├── tasks/                        # Task-specific configurations
│   ├── core/                         # Core functionality
│   │   ├── trace_generation/         # System trace generation
│   │   ├── spec_processing/          # TLA+ specification processing
│   │   └── verification/             # TLC verification
│   └── utils/                        # Utilities
├── data/                             # Evaluation data
│   ├── spec/                         # TLA+ specifications
│   ├── sys_traces/                   # System traces
│   ├── traces/                       # Converted traces
│   └── invariant_templates/          # Expert invariant templates
└── output/                           # Evaluation results
```

## Extending the Framework

### Adding New Systems

The framework supports systematic addition of new concurrent and distributed systems through three steps:

1. **Task Formulation**: Create specification prompts describing modules requiring formal modeling
2. **Invariant Template Design**: Develop invariant templates capturing domain-specific correctness requirements
3. **System Instrumentation**: Implement execution harnesses and instrumentation aligned with modeling granularity

See the existing `etcd` and `spin` tasks as examples.

## Output and Results

Results are saved in structured directories under `output/`:
```
output/
├── <metric>/
│   └── <task>/
│       └── <method>/
│           └── <model>/
│               └── <timestamp>/
│                   ├── specification.tla      # Generated/used specification
│                   ├── configuration.cfg      # TLC configuration
│                   ├── results.json          # Evaluation results
│                   └── metadata.json         # Execution metadata
```

