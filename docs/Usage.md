# Usage

## Single-cell evaluation

Run one (task, model, metric) combination:

```bash
sysmobench --task <task> --method direct_call --model <model> --metric <metric> [options]
```

Required: `--task`, `--method`, `--model`, `--metric`. Use `--spec-file <path>` (and optionally `--config-file <path>`) to evaluate an existing spec instead of generating one. `sysmobench --list-{tasks,methods,models,metrics}` enumerates the choices.

### Metrics

| Stage | Metric | Parameters |
|---|---|---|
| Syntax | `compilation_check`, `action_decomposition` | — |
| Runtime | `runtime_check`, `coverage`, `runtime_coverage` | `--tlc-timeout <s>` |
| Transition validation | `transition_validation` | `--tv-agent`, `--tv-model`, `--tv-budget`, `--tv-timeout` |
| Invariant verification | `invariant_verification` | `--tlc-timeout <s>`, `--inv-translator-type {direct,agent}` |

## Batch evaluation

Run the full pipeline (Syntax → Runtime → Transition validation → Invariant verification) over multiple systems:

```bash
python3 scripts/run_batch_experiment.py --all --model claude
```

Outputs land under `experiments/batch_<timestamp>/<system>/run_*.json`.

| Flag | Purpose |
|---|---|
| `--all` / `--systems <name>...` | All 11 systems or a subset |
| `--model <id>` | Generation model (entry in `config/models.yaml`) |
| `--runs <N>` | Runs per (model, system); default 5 |
| `--threads <N>` | Parallelism; default 5 |
| `--skip-tv` | Skip transition validation (cost opt-out) |
| `--tv-agent`, `--tv-model`, `--tv-budget`, `--tv-timeout` | Transition-validation knobs |
| `--inv-model <id>` | Phase-4 translator model |

Transition validation costs roughly **\$1–4 per (model, system) cell** through the coding-agent CLI; a five-model sweep across all 11 systems is in the low hundreds of USD.

## Transition validation

Each captured trace is cut into per-action windows of the form (pre-state, post-state). For every window TLC is asked whether the spec's action permits the transition. The flow is driven by the `tv-eval` skill, which the launcher hands to a coding agent (`claude-code` or `codex`).

Two ways to invoke it:

```bash
# As a single-cell metric, on a spec already on disk.
sysmobench --task <name> --method direct_call --model <id> \
  --metric transition_validation --spec-file <path-to-.tla>

# Directly via the launcher (used internally by the batch pipeline).
bash scripts/launch_tv_eval.sh --task=<name> --spec=<dir-with-.tla-and-.cfg>
```

Each launch creates `tv-workspaces/<timestamp>_<task>/` containing `reports/final_report.md` (per-action pass rates and audit summary) and `reports/tv_results.json` (machine-readable, consumed by the metric registry).

## Invariant verification

Each system has expert-written templates at `data/invariant_templates/<task>/invariants.yaml`. A translator concretizes each template against the generated spec's variable names; TLC then checks the resulting predicate.

```bash
sysmobench --task <name> --method direct_call --model <id> \
  --metric invariant_verification --inv-translator-type {direct|agent} --tlc-timeout 600
```

`direct` uses a single LLM call (cheap, fast, default); `agent` uses a Claude Code agent.

## Spec repair

When a spec fails the syntax or runtime check, the `spec-repair` skill applies bounded edits so the later stages can still be measured on a comparable spec. The batch orchestrator repairs every cell under `docs/leaderboard/specs/`:

```bash
python3 scripts/batch_repair_and_tv.py [--phase repair|tv|all] [--dry-run]
```

Repaired specs land at `docs/leaderboard/specs_repaired/<model>/<system>/` with a `repair_manifest.json` and `repair_report.md`.

## Leaderboard

Scripts that populate `docs/leaderboard/`:

| Script | Produces |
|---|---|
| `scripts/build_leaderboard.py` | Baseline `detail.csv`, `aggregate.csv`, `paper_summary.csv`, `data.json` |
| `scripts/build_leaderboard_repaired.py` | Rescored `*_repaired.csv` using repaired specs |
| `scripts/reweight_leaderboard.py` | Re-rank `detail_repaired.csv` with custom phase weights |
| `scripts/build_spec_showcase.py` | Per-cell spec archive under `docs/leaderboard/specs/` |

Canonical phase weights: P1=0.15, P2=0.15, P3=0.35, P4=0.35. Schema, model canonicalization, and abandoned-runs policy: [`docs/leaderboard/schema.md`](leaderboard/schema.md).

To rebuild the public leaderboard from a clean clone:

```bash
python3 scripts/run_batch_experiment.py --all --model <model>
python3 scripts/build_leaderboard.py
python3 scripts/batch_repair_and_tv.py --phase all
python3 scripts/build_leaderboard_repaired.py
```

## Skills

Three agent-driven skills under `tla_eval/skills/`:

| Skill | When |
|---|---|
| `harness-gen` | Bootstrap a trace harness when adding a new system |
| `tv-eval` | Run transition validation on one spec |
| `spec-repair` | Repair a syntax- or runtime-failing spec without changing modeling intent |

Each skill ships its own `SKILL.md` and `guide.md`. They are invoked from a Claude Code session inside the repo.

## Model configuration

`config/models.yaml`:

```yaml
models:
  <model_name>:
    provider: "litellm" | "openai" | "anthropic" | "genai" | "deepseek" | "yunwu"
    model_name: "<litellm-model-name>"
    api_key_env: "<ENV_VAR_NAME>"
    temperature: <float>
    max_tokens: <int>
    timeout: <int>           # optional, seconds
    top_p: <float>           # optional
    url: "<endpoint>"        # optional, OpenAI-compatible endpoints
    litellm_params: { ... }  # optional, provider-specific passthrough
```

All hosted providers route through the unified LiteLLM adapter. Reference a model by name on the CLI: `--model <model_name>`.

## Output layout

```
output/<metric>/<task>/<method>_<model>/<timestamp>/   single-cell results
experiments/batch_<timestamp>/<system>/run_*.json       batch runs
tv-workspaces/<timestamp>_<system>/                     transition validation (gitignored)
```
