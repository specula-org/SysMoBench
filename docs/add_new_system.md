# Adding a new system

Four files. Once they exist the rest of the pipeline picks the system up automatically.

## 1. `tla_eval/tasks/<name>/task.yaml`

```yaml
name: "<name>"
description: "<one-line summary>"
system_type: "concurrent" | "distributed"
language: "rust" | "go" | "java" | ...

repository:
  url: "https://github.com/.../<repo>.git"
  branch: "main"

source_files:
  - path: "path/in/repo/to/file.go"

default_source_file: "path/in/repo/to/file.go"
specModule: "<TLA+ module name>"

tv:
  repo_path: "artifacts/<name>"
  target_actions: ["<ActionA>", "<ActionB>"]
```

The `tv:` block is consumed by `scripts/launch_tv_eval.sh`. See `tla_eval/tasks/spin/task.yaml` (docker harness) and `tla_eval/tasks/zookeeper/task.yaml` (native harness with action map) for full examples.

## 2. `tla_eval/tasks/<name>/prompts/`

Three prompt files:

- `direct_call.txt` — full TLA+ generation prompt
- `phase2_config.txt` — TLC `.cfg` generation prompt
- `phase3_invariant_implementation.txt` — Phase-4 invariant translation prompt

Reference layout: `tla_eval/tasks/etcd/prompts/`.

## 3. `data/invariant_templates/<name>/invariants.yaml`

```yaml
- name: "<InvariantName>"
  type: "safety" | "liveness"
  natural_language: "<plain-English statement>"
  formal_description: "<math / pseudocode>"
  tla_example: "<TLA+ snippet>"
```

## 4. Trace harness

Use the `harness-gen` skill to bootstrap `artifacts/<name>/` with an instrumented build emitting NDJSON traces at the granularity the task requires. One-time per system; reused by every spec evaluation through the `tv-eval` skill.
