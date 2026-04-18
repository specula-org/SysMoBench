# SysMoBench Leaderboard — Data Schema

The files in this directory are the **canonical source** for the SysMoBench
leaderboard. A website agent reads them to populate the public scoreboard.

## Files

| File | Format | Use |
|---|---|---|
| `data.json` | JSON | Full structured data. Website agent should prefer this. |
| `detail.csv` | CSV | One row per (model, system). For spreadsheets / ad-hoc analysis. |
| `aggregate.csv` | CSV | One row per model, averages across evaluated systems. |

## Regeneration

```
python3 scripts/build_leaderboard.py
```

Scans `experiments/batch_*/` (Phase A) and `wv-workspaces/*/` (Phase 3 WV + audit),
merges best-of-N per (model, system), writes the three files above. **Idempotent**.

Run this whenever new experiments complete. The script is the single source of
truth; don't hand-edit the CSV/JSON.

## `data.json` structure

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
      "phase3_wv_rate": 1.0,
      "phase3_audit_run": true,
      "phase3_audit_bugs": [
        {"action": "Commit", "line": "Commit | 1.0 | yes | wrong (NullLeader, ...) | 0"}
      ],
      "phase3_final_score": 0.833,
      "overall_score": 0.954,
      "best_run_spec_path": "output/compilation_check/.../curp.tla",
      "best_run_json_path": "experiments/batch_.../curp/run_4.json",
      "wv_workspace_path": "wv-workspaces/.../reports/",
      "gen_tokens_in": 12345,
      "gen_tokens_out": 6789,
      "wv_agent_cost_usd": 4.18,
      "wv_agent_duration_s": 1230.0,
      "wv_agent_turns": 52,
      "notes": []
    }
  ]
}
```

## Field semantics

### Phase A (generation + compile + runtime + invariant)
- `phase1_score`: SANY compilation pass (1.0 = pass, fractional = partial)
- `phase2_score`: runtime coverage (0.0–1.0) — TLC state-space coverage
- `phase2_coverage`: raw coverage fraction
- `phase2_runtime_check_passed`: boolean — did TLC complete without violation?
- `phase3b_score`: invariant-check pass rate (agent-translated invariants)
- `phase_a_total`: current total_score from batch runner (mean over ran phases)

### Phase 3 (Window Validation + Audit)
- `phase3_wv_rate`: mean of per-action WV pass rates (TLC window check)
- `phase3_audit_run`: was Step 9 audit performed?
- `phase3_audit_bugs`: list of `{action, line}` — actions downgraded to 0 by audit
  because audit found a TLC-verified impossibility the spec accepts
- `phase3_final_score`: WV score AFTER audit downgrade

### Composite
- `overall_score`: mean over {phase1, phase2, phase3_final (or _wv_rate fallback), phase3b}

### Provenance (for auditability)
- `best_run_spec_path`: the .tla file that won best-of-N
- `best_run_json_path`: raw run JSON in experiments/
- `wv_workspace_path`: where WV + audit artifacts live (for drill-down)

### Cost
- `gen_tokens_in` / `gen_tokens_out`: Phase A LLM usage (provider-billed)
- `wv_agent_cost_usd` / `duration_s` / `turns`: Claude Code agent (subscription,
  not billed per call — the USD value is the equivalent API cost)

## Notes on score interpretation

- **`overall_score` is the headline ranking metric** — higher = better spec quality
- **`audit_bugs_total > 0`** (in aggregate.csv) flags a model whose specs hide
  real bugs behind high WV pass rates. Example from pilot: kimi curp has 1 bug
  (NullLeader), deepseek mutex has 1 bug (DoubleQueue)
- **`phase3_final_score < phase3_wv_rate`** means audit downgraded something
- Missing values mean the phase didn't run (e.g., spec failed to compile
  → downstream phases skipped)

## Update discipline

- **Don't hand-edit these files.** Re-run `scripts/build_leaderboard.py` instead.
- **Don't modify run_*.json in `experiments/`** after the fact — those are
  evidence. If a run needs correcting, re-run the experiment.
- **WV workspaces under `wv-workspaces/`** are per-evaluation. Historical ones
  stay even when new ones are added; the script picks the latest matching one.

## Known data quirks (2026-04-18)

- `claude_sonnet_proxy` etcd: 5/5 gen failures (Cloudflare 520). Use
  `claude_sonnet_direct` rows for etcd.
- `gemini31_proxy`: mutex/rwmutex 5/5 failures (504). Only spin usable.
- `qwen36_plus_ds` etcd: uses 1h timeout variant (`qwen36_plus_ds_1h`) under
  the hood. Leaderboard merges back to `qwen36_plus_ds` for display.
- `glm51_ds`: 0/55 successful runs (DashScope backend broken).
- Older batches from 2026-04-17 include exploratory runs (minimax_m27,
  glm51, grok4_proxy, deepseek_r1_proxy, gpt54_proxy) that were abandoned
  mid-way. Treat those model rows as partial / experimental.
