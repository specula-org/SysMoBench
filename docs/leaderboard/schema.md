# Leaderboard data schema

The files in this directory are the canonical source for the SysMoBench leaderboard. The website at [sysmobench.com](https://sysmobench.com) reads them directly. Do not hand-edit; regenerate via `scripts/build_leaderboard.py` (idempotent — scans `experiments/batch_*/` and `tv-workspaces/*/`, picks best-of-N per (model, system)).

## Files

| File | Format | Use |
|---|---|---|
| `data.json` | JSON | Full structured data; preferred by the website. |
| `detail.csv` | CSV | One row per (model, system); spreadsheet-friendly. |
| `aggregate.csv` | CSV | One row per model; averages across evaluated systems. |

## `data.json`

```json
{
  "generated_at": "2026-04-18T11:15:00+00:00",
  "project": "SysMoBench",
  "schema_version": 1,
  "rows": [
    {
      "model": "kimi_k25_ds",
      "system": "curp",
      "phase1_score": 1.0,
      "phase2_score": 1.0,
      "phase2_coverage": 1.0,
      "phase2_runtime_check_passed": true,
      "phase3b_score": 0.9231,
      "phase_a_total": 1.0,
      "phase3_tv_rate": 1.0,
      "phase3_audit_run": true,
      "phase3_audit_bugs": [
        {"action": "Commit", "line": "Commit | 1.0 | yes | wrong (NullLeader, ...) | 0"}
      ],
      "phase3_final_score": 0.833,
      "overall_score": 0.954,
      "best_run_spec_path": "output/compilation_check/.../curp.tla",
      "best_run_json_path": "experiments/batch_.../curp/run_4.json",
      "tv_workspace_path": "tv-workspaces/.../reports/",
      "gen_tokens_in": 12345,
      "gen_tokens_out": 6789,
      "tv_agent_cost_usd": 4.18,
      "tv_agent_duration_s": 1230.0,
      "tv_agent_turns": 52,
      "notes": []
    }
  ]
}
```

## Fields

| Group | Field | Meaning |
|---|---|---|
| Phase A | `phase1_score` | SANY compilation pass (1.0 = pass) |
| | `phase2_score` | Runtime coverage (0–1, TLC state-space) |
| | `phase2_coverage` | Raw coverage fraction |
| | `phase2_runtime_check_passed` | TLC completed without violation |
| | `phase3b_score` | Invariant-check pass rate (translated invariants) |
| | `phase_a_total` | Mean over phases the batch runner actually ran |
| Phase 3 | `phase3_tv_rate` | Mean of per-action TV pass rates |
| | `phase3_audit_run` | Whether Step 9 audit was performed |
| | `phase3_audit_bugs` | `{action, line}` entries downgraded to 0 by audit |
| | `phase3_final_score` | TV score after audit downgrade |
| Composite | `overall_score` | Headline metric: mean over phase1 / phase2 / phase3_final / phase3b |
| Provenance | `best_run_spec_path`, `best_run_json_path`, `tv_workspace_path` | Drill-down paths for the winning run |
| Cost | `gen_tokens_in`, `gen_tokens_out` | Phase-A LLM usage (provider-billed) |
| | `tv_agent_cost_usd`, `tv_agent_duration_s`, `tv_agent_turns` | Coding-agent usage during TV |

A missing field means the phase did not run (e.g., compile failed, downstream skipped).

`overall_score` is the ranking metric. `audit_bugs_total > 0` (in `aggregate.csv`) flags a model whose specs hide real bugs behind high TV pass rates. `phase3_final_score < phase3_tv_rate` means audit downgraded one or more actions.

## Model canonicalization

`build_leaderboard.py` collapses config-level model names into canonical leaderboard names (e.g. `claude_sonnet_proxy` + `claude_sonnet_direct` → `claude_sonnet`; best-of-N picks the higher of the two routes).

Models that never completed a full benchmark are listed in `data.json` under `abandoned_rows` and excluded from `aggregate.csv` / `detail.csv`. Promote a model back to primary by removing it from `ABANDONED_MODELS` in `scripts/build_leaderboard.py` and re-running.
