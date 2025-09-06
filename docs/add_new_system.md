# Adding a New System to the Framework

This document explains how to add a new system to the framework using etcd as an example. The process consists of three main steps:

## 1. Define Task Configuration

Create task definition in `tla_eval/tasks/<system_name>` directory:

### task.yaml
Create a system configuration file containing:
- Basic system information (name, etc.)
- Repository information (URL, branch, version)
- Source file paths
- TLA+ specification module name

### prompts directory
Create prompt files based on etcd naming convention:
- `agent_based.txt` - Agent-based generation prompts
- `direct_call.txt` - Direct call prompts
- `phase2_config.txt` - Phase 2 configuration
- `phase2_invariants.txt` - Phase 2 invariants
- `phase3_invariant_implementation.txt` - Phase 3 invariant implementation
- `trace_config_generation.txt` - Trace configuration generation

## 2. Create Invariant Templates

Create invariant definitions in `data/invariant_templates/<system_name>` directory:

### invariants.yaml
Define core system invariants, each containing:
- `name` - Invariant name
- `type` - Safety or liveness
- `natural_language` - Natural language description
- `formal_description` - Formal description
- `tla_example` - TLA+ code example

These templates are used for Phase 3 invariant verification.

## 3. Implement Trace Validation

Implement `module.py` in `tla_eval/core/trace_generation/<system_name>` directory:

### Required Interface Classes
- `TraceGenerator` - Implements trace generation logic
- `TraceConverter` - Implements trace format conversion
- `SystemModule` - System module entry point

