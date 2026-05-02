# Usage Guide

SysMoBench evaluates AI-generated TLA+ models across **4 phases**:

1. **Phase 1 — Compilation** (syntax, via SANY)
2. **Phase 2 — Runtime** (bounded model checking, via TLC)
3. **Phase 3 — Conformance** (transition validation against system traces, agent-driven)
4. **Phase 4 — Invariant** (TLC with system-specific invariants, agent-translated)

Three ways to use the benchmark:

- **One metric on one (model, system)** → [Single-cell evaluation](#single-cell-evaluation)
- **Full 4-phase pipeline on a batch of models × systems** → [Batch evaluation](#batch-evaluation)
- **Repair broken specs then rescore** → [Spec repair](#spec-repair)

The scored data lives under [`docs/leaderboard/`](leaderboard/). See the [Leaderboard](#leaderboard) section for how to rebuild it.

---

## Tasks

11 system artifacts:

| System | Type | Phase-3 harness |
|---|---|---|
| `spin` | concurrent sync | docker + ktest |
| `mutex` | concurrent sync | docker + ktest |
| `rwmutex` | concurrent sync | docker + ktest |
| `dqueue` | distributed queue | PGo |
| `ringbuffer` | concurrent queue | host-side patch |
| `locksvc` | distributed lock | PGo |
| `curp` | distributed replication | madsim + patch |
| `raftkvs` | distributed consensus (Raft) | PGo |
| `redisraft` | distributed consensus (Raft) | wrapped Specula case |
| `zookeeper` | distributed coordination | patch-drift (blocked) |
| `etcd` | distributed consensus (Raft) | trace replay |

List from CLI: `sysmobench --list-tasks`.

---

## Single-cell evaluation

```bash
sysmobench --task <task> --method <method> --model <model> --metric <metric> [options]
```

### Required

- `--task` — one of the 11 task names above
- `--method` — `direct_call`
- `--model` — entry in `config/models.yaml`
- `--metric` — see [Metrics](#metrics)

### Common

- `--spec-file <path>` / `--config-file <path>` — use an existing spec (skip generation)
- `--output <dir>` — default `results/`

### List options

```bash
sysmobench --list-tasks
sysmobench --list-methods
sysmobench --list-models
sysmobench --list-metrics
```

### Metrics

**Phase 1 — Syntax correctness**

| Metric | Description |
|---|---|
| `compilation_check` | Full-model compilation with SANY |
| `action_decomposition` | Per-action validation with recovery |

**Phase 2 — Runtime correctness**

| Metric | Description | Parameters |
|---|---|---|
| `runtime_check` | Model checking without invariants | `--tlc-timeout <seconds>` |
| `coverage` | Action coverage via TLC statistics | `--tlc-timeout <seconds>` |
| `runtime_coverage` | Simulation-based coverage | `--tlc-timeout <seconds>` |

**Phase 3 — Conformance to implementation**

| Metric | Description | Parameters |
|---|---|---|
| `transition_validation` | Per-action conformance against captured system traces (agent-driven; see [§ Phase 3](#phase-3--transition-validation)) | `--tv-agent <name>`, `--tv-model <id>`, `--tv-budget <USD>`, `--tv-timeout <seconds>` |

**Phase 4 — Invariant correctness**

| Metric | Description | Parameters |
|---|---|---|
| `invariant_verification` | TLC with agent-translated system invariants | `--tlc-timeout <seconds>`, `--inv-translator-type <type>` |

---

## Batch evaluation

Run the full pipeline (P1 → P2 → optional TV → optional P4) across multiple
models × systems.

```bash
python3 scripts/run_batch_experiment.py [options]
```

Outputs land under `experiments/batch_<timestamp>/<system>/run_*.json`. Key flags:

- `--all` / `--systems <name> ...` — pick all 11 or a subset
- `--model <id>` — generation model (entry in `config/models.yaml`)
- `--runs <N>` — runs per (model, system); default 5
- `--threads <N>` — parallelism; default 5
- `--skip-tv` — skip Phase 3 transition validation (TV runs by default; this is the cost opt-out)
- `--tv-agent <name>` / `--tv-model <id>` — TV agent adapter and model
- `--tv-budget <USD>` / `--tv-timeout <s>` — per-TV cost cap (default 5) and timeout (default 1800s)
- `--inv-model <id>` — Phase-4 invariant-translator model

`python3 scripts/run_batch_experiment.py --list-systems` and `--list-agents` enumerate the choices.

End-to-end example (one model across all 11 systems, all four stages):

```bash
python3 scripts/run_batch_experiment.py \
    --all \
    --model claude \
    --runs 3 --threads 8 \
    --tv-agent claude-code --tv-model sonnet \
    --inv-model sonnet
```

TV adds roughly **\$1–4 per (model, system)** spec via the TV agent; budget on a 5-model × 11-system sweep is in the low-hundreds of USD. Phase 4 invariant translation runs through the agent's own credentials, not your API key — see [API usage policy](../README.md#) and CLAUDE.md.

---

## Phase 3 — Transition validation

Transition validation cuts every captured trace into per-action windows of the
form (pre-state, post-state) and asks TLC whether the spec's action allows the
transition. The pipeline is driven by the `tv-eval` skill, which the launcher
hands to a coding agent (`claude-code` or `codex`).

Two ways to invoke it:

```bash
# As a single-cell metric, on a spec already on disk.
sysmobench --task <name> --method direct_call --model <id> \
  --metric transition_validation --spec-file <path-to-.tla>

# Directly via the launcher (used internally by the batch pipeline).
bash scripts/launch_tv_eval.sh \
  --task=<name> \
  --spec=<dir-containing-.tla-and-.cfg> \
  [--agent=claude-code|codex] \
  [--model=<id>] \
  [--workspace-root=./tv-workspaces]
```

Each launch creates `tv-workspaces/<timestamp>_<task>/` with the agent's
`reports/final_report.md` (per-action pass rates, audit summary) and the
machine-readable `reports/tv_results.json` consumed by the metric registry.

> **TODO:** scoring rule (zero-tolerance per action), audit step, cost range.

---

## Phase 4 — Invariant verification

Each system has a set of expert-written invariant templates under
`data/invariant_templates/<task>/invariants.yaml`. A translator turns each
template into a TLA+ predicate against the generated spec's variable names,
then TLC verifies them.

```bash
sysmobench --task etcd --method direct_call --model claude \
  --metric invariant_verification \
  --inv-translator-type {direct|agent} \
  --tlc-timeout 600
```

- `--inv-translator-type direct` — single LLM call (cheap, fast, default)
- `--inv-translator-type agent` — Claude Code agent (more thorough; uses agent's
  own credentials per the API usage policy in CLAUDE.md)
- `--tlc-timeout` — per-invariant TLC timeout, seconds

Expected runtime varies by system: simple synchronization specs finish in seconds,
distributed consensus systems can take minutes per invariant.

---

## Spec repair

When a spec fails P1 or P2, the `spec-repair` skill applies bounded syntactic
edits so P3/P4 can still be measured on a comparable spec. Use the batch
orchestrator to repair every cell in `docs/leaderboard/specs/`:

```bash
python3 scripts/batch_repair_and_tv.py [--phase repair|tv|all] [--dry-run]
```

Repaired specs go to `docs/leaderboard/specs_repaired/<model>/<system>/` with a
`repair_manifest.json` + `repair_report.md`.

> **TODO:** allow-list (A1-A17) vs forbid-list (F1-F7), quota discipline.

---

## Leaderboard

Scripts that populate `docs/leaderboard/`:

| Script | Produces |
|---|---|
| `scripts/build_leaderboard.py` | Baseline `detail.csv`, `aggregate.csv`, `paper_summary.csv`, `data.json` |
| `scripts/build_leaderboard_repaired.py` | Rescored `*_repaired.csv` using repaired specs + fresh P3/P4 |
| `scripts/reweight_leaderboard.py` | Re-rank `detail_repaired.csv` with arbitrary phase weights |
| `scripts/build_spec_showcase.py` | Per-cell spec archive under `docs/leaderboard/specs/` |

Canonical phase weights: **P1=0.15, P2=0.15, P3=0.35, P4=0.35**.

CSV/JSON schemas, model canonicalization rules, and the abandoned-runs policy
are documented in [`docs/leaderboard/schema.md`](leaderboard/schema.md).

To reproduce the public leaderboard from a clean clone:

```bash
python3 scripts/run_batch_experiment.py --all --model <model>                # populate experiments/
python3 scripts/build_leaderboard.py                                         # baseline CSVs
python3 scripts/batch_repair_and_tv.py --phase all                           # repair + rescore
python3 scripts/build_leaderboard_repaired.py                                # *_repaired.csv
```

---

## Skills

Three agent-driven skills under `tla_eval/skills/`:

| Skill | When to use |
|---|---|
| `harness-gen` | Bootstrap a trace harness for a new system |
| `tv-eval` | Run Phase 3 transition validation + audit on one spec |
| `spec-repair` | Repair a P1- or P2-failing spec without changing its modeling intent |

Each skill has its own `SKILL.md` + `guide.md`.

These skills are intended to be invoked from a Claude Code session inside the
repo — the CLI agent loads `SKILL.md` and follows it. Typical triggers:

- **`harness-gen`** — "bootstrap a trace harness for system `<name>`". The agent
  clones the system into `artifacts/<name>/`, instruments it to emit NDJSON
  traces at the granularity declared in `tla_eval/tasks/<name>/task.yaml`,
  writes `run.sh`, and produces `INSTRUMENTATION.md`.
- **`tv-eval`** — typically launched non-interactively via
  `scripts/launch_tv_eval.sh --task=<name> --spec=<dir>`; the wrapper invokes
  the configured agent (Claude Code or Codex) with the skill's guide.
- **`spec-repair`** — "repair the spec at `<path>` so it passes P1/P2 without
  changing modeling intent". Outputs a `repair_manifest.json` listing the edits
  applied (allow-list) and any forbidden changes that were rejected.

---

## Model configuration

### File location

`config/models.yaml`

### Install

```
pip install -e .
```

### Configuration format

```yaml
models:
  <model_name>:
    provider: "litellm" | "openai" | "anthropic" | "genai" | "deepseek" | "yunwu"
    model_name: "<litellm-model-name>"
    api_key_env: "<ENV_VAR_NAME>"
    temperature: <float>
    max_tokens: <int>
    timeout: <int>        # seconds, optional
    top_p: <float>        # optional
    url: "<endpoint>"     # optional, for OpenAI-compatible endpoints
    litellm_params:       # optional, provider-specific passthrough params
      <key>: <value>
```

### Example

```yaml
models:
  claude:
    provider: "litellm"
    model_name: "anthropic/claude-sonnet-4-20250514"
    api_key_env: "ANTHROPIC_API_KEY"
    temperature: 0.1
    max_tokens: 64000
    top_p: 0.9
```

All hosted providers (`openai`, `anthropic`, `genai`, `deepseek`, `yunwu`, etc.) route through the unified LiteLLM adapter.

### Using a model

1. Add the entry to `config/models.yaml`
2. Export the API key: `export YOUR_API_KEY="..."`
3. Reference by name: `sysmobench --task spin --method direct_call --model custom --metric compilation_check`

---

## Output structure

Single-cell runs land under `output/<metric>/<task>/<method>_<model>/<timestamp>/`:

```
output/compilation_check/spin/direct_call_claude/20260101_120000/
├── spin.tla
├── spin.cfg
├── result.json
├── metadata.json
└── generation_usage.json
```

Batch runs land under `experiments/batch_<timestamp>/<system>/run_*.json`.

TV workspaces land under `tv-workspaces/<timestamp>_<system>/` (gitignored).

---

## Task configuration

Each task is defined by `tla_eval/tasks/<task>/task.yaml`:

```yaml
name: "spin"
description: "Asterinas OS spinlock"
system_type: "concurrent"
language: "rust"

repository:
  url: "https://github.com/asterinas/asterinas.git"
  branch: "main"

source_files:
  - path: "ostd/src/sync/spin.rs"

default_source_file: "ostd/src/sync/spin.rs"
specModule: "spin"
traces_folder: "data/sys_traces/spin"

tv:
  repo_path: "artifacts/spin"
  target_actions: ["AcquireLock", "ReleaseLock"]
```

The `tv:` block is consumed by `scripts/launch_tv_eval.sh` to set up Phase 3.

### `tv:` block fields

| Field | Required | Meaning |
|---|---|---|
| `repo_path` | yes | Where the TV agent should keep its instrumented system clone (e.g., `artifacts/<task>/`) |
| `target_actions` | yes | Spec-level action names to score (the agent maps these onto the spec's own action names) |
| `harness.type` | yes | `docker` or `native` |
| `harness.docker_image` | docker | Docker image with the test target installed |
| `harness.test_target` | docker | Test that emits NDJSON traces |
| `harness.reference_patch` | optional | Existing instrumentation patch to apply / re-base |
| `harness.instrumentation_file` | native | Patch file driving the instrumentation |
| `harness.test_file` | native | Source file under test |
| `harness.run_command` | native | Shell command to run the harness |
| `harness.traces_output_env` | native | Env var pointing at the trace output directory |
| `harness.trace_action_map` | optional | Maps raw trace event names → spec action names when they differ |
| `harness.coverage_gaps` | optional | Documented gaps where some target actions are under-sampled |

See `tla_eval/tasks/spin/task.yaml` (docker) and `tla_eval/tasks/zookeeper/task.yaml` (native + map) for full examples.

### Invariant templates

Phase 4 reads `data/invariant_templates/<task>/invariants.yaml`. Each entry has
`name`, `type` (safety/liveness), `natural_language`, `formal_description`, and
`tla_example`. The translator concretizes each template against the generated
spec's variable names; TLC then checks the resulting predicate.
