# Usage Guide

SysMoBench evaluates AI-generated TLA+ models across **4 phases**:

1. **Phase 1 — Compilation** (syntax, via SANY)
2. **Phase 2 — Runtime** (bounded model checking, via TLC)
3. **Phase 3 — Conformance** (window verification against system traces, agent-driven)
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

The canonical Phase-3 flow is **Window Verification (WV)**, an agent-driven
evaluation launched via `scripts/launch_wv_eval.sh`. See [Phase 3 — WV](#phase-3--window-verification-wv).

Low-level trace-validation metrics (pre-WV):

| Metric | Applies to | Parameters |
|---|---|---|
| `trace_validation` | `spin`, `mutex`, `rwmutex`, `etcd`, `redisraft`, `curp` | `--with-exist-traces <N>`, `--with-exist-specTrace`, `--create-mapping` |
| `pgo_trace_validation` | `dqueue`, `locksvc`, `raftkvs` | `--with-exist-traces <N>` |

**Phase 4 — Invariant correctness**

| Metric | Description | Parameters |
|---|---|---|
| `invariant_verification` | TLC with agent-translated system invariants | `--tlc-timeout <seconds>`, `--inv-translator-type <type>` |

> **Note:** the old `composite` metric has been removed. A new 4-phase
> composite is on the roadmap.

---

## Batch evaluation

Run the full pipeline (P1 → P2 → optional WV → optional P4) across multiple
models × systems.

```bash
python3 scripts/run_batch_experiment.py [options]
```

Outputs land under `experiments/batch_<timestamp>/`. Key flags:

- `--enable-wv` — also run Phase 3 WV for cells that pass P2
- `--wv-agent <name>` / `--wv-model <id>` — WV agent adapter + model override
- `--inv-model <id>` — invariant-translator model (Phase 4)

> **TODO:** concrete end-to-end example with `--models`, `--systems`, cost expectations.

---

## Phase 3 — Window verification (WV)

WV validates each action's transitions against windows cut from system traces,
running TLC on every (pre-state, post-state) pair. Driven by the `wv-eval` skill.

```bash
bash scripts/launch_wv_eval.sh \
  --task=<name> \
  --spec=<dir-containing-.tla-and-.cfg> \
  [--agent=claude-code|codex] \
  [--model=<id>] \
  [--workspace-root=./wv-workspaces]
```

Each launch creates `wv-workspaces/<timestamp>_<task>/` containing the agent's
`reports/final_report.md` with per-action pass rates and an audit summary.

> **TODO:** scoring rule (zero-tolerance per action), audit step, cost range.

---

## Phase 4 — Invariant verification

Agent translates the system's invariant templates (under
`data/invariant_templates/<task>/`) against the generated spec, then TLC
verifies them.

> **TODO:** flag summary, translator models, expected runtime.

---

## Spec repair

When a spec fails P1 or P2, the `spec-repair` skill applies bounded syntactic
edits so P3/P4 can still be measured on a comparable spec. Use the batch
orchestrator to repair every cell in `docs/leaderboard/specs/`:

```bash
python3 scripts/batch_repair_and_wv.py [--phase repair|wv|all] [--dry-run]
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

> **TODO:** schema of each CSV, how the website consumes them, reproduce steps.

---

## Skills

Three agent-driven skills under `tla_eval/skills/`:

| Skill | When to use |
|---|---|
| `harness-gen` | Bootstrap a trace harness for a new system |
| `wv-eval` | Run Phase 3 window verification + audit on one spec |
| `spec-repair` | Repair a P1- or P2-failing spec without changing its modeling intent |

Each skill has its own `SKILL.md` + `guide.md`.

> **TODO:** one-paragraph invocation example per skill.

---

## Model configuration

### File location

`config/models.yaml`

### Install modes

- Recommended: `pip install -e .`
- With legacy native SDK adapters: `pip install -e ".[legacy-providers]"`

### Configuration format

```yaml
models:
  <model_name>:
    provider: "litellm" | "openai" | "anthropic" | "genai" | "deepseek" | "yunwu" | "legacy_openai" | "legacy_anthropic" | "legacy_genai"
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

Compatibility notes:

- Existing `provider: "openai" | "anthropic" | "genai" | "deepseek" | "yunwu"` entries still work and route through the LiteLLM adapter by default.
- Use `legacy_openai`, `legacy_anthropic`, or `legacy_genai` to force the old native SDK adapters.

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

WV workspaces land under `wv-workspaces/<timestamp>_<system>/` (gitignored).

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

wv:
  repo_path: "artifacts/spin"
  target_actions: ["AcquireLock", "ReleaseLock"]
```

The `wv:` block is consumed by `scripts/launch_wv_eval.sh` to set up Phase 3.

> **TODO:** field reference for `wv:` block and invariant template locations.
