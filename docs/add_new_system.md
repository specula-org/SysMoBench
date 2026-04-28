# Adding a New System to SysMoBench

Adding a new system involves four steps: declaring the task, writing prompts, defining invariant templates, and bootstrapping a Phase-3 trace harness.

## 1. Task configuration

Create `tla_eval/tasks/<system_name>/task.yaml`:

```yaml
name: "<system_name>"
description: "<one-line summary>"
system_type: "concurrent" | "distributed" | ...
language: "rust" | "go" | "java" | ...

repository:
  url: "https://github.com/.../<repo>.git"
  branch: "main"

source_files:
  - path: "path/in/repo/to/file.go"

default_source_file: "path/in/repo/to/file.go"
specModule: "<TLA+ module name>"
traces_folder: "data/sys_traces/<system_name>"

# Phase 3 (window verification) configuration
wv:
  repo_path: "artifacts/<system_name>"
  target_actions: ["<ActionA>", "<ActionB>"]
```

The `wv:` block is consumed by `scripts/launch_wv_eval.sh`.

## 2. Prompts

Create `tla_eval/tasks/<system_name>/prompts/` with:

- `direct_call.txt` — full TLA+ spec generation prompt
- `phase2_config.txt` — TLC `.cfg` generation prompt (Phase 2)
- `phase3_invariant_implementation.txt` — invariant translation prompt (Phase 4)

Use `tla_eval/tasks/etcd/prompts/` as the reference layout.

## 3. Invariant templates

Create `data/invariant_templates/<system_name>/invariants.yaml` listing each core invariant:

```yaml
- name: "<InvariantName>"
  type: "safety" | "liveness"
  natural_language: "<plain-English statement>"
  formal_description: "<math / pseudocode>"
  tla_example: "<TLA+ snippet>"
```

Phase 4 (`invariant_verification`) translates each entry against the generated spec and runs TLC.

## 4. Phase-3 trace harness

Use the `harness-gen` skill to bootstrap `artifacts/<system_name>/` with an instrumented build that emits NDJSON traces at the granularity declared in `task.yaml`. The harness is one-time per system and is then reused by every spec evaluation through the `wv-eval` skill (driven by `scripts/launch_wv_eval.sh`).
