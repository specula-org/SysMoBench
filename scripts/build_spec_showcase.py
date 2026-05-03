#!/usr/bin/env python3
"""
Build docs/leaderboard/specs/ — the "show me the spec behind each score" folder.

For every (model, system) cell in docs/leaderboard/paper_summary.csv, copy:
  - the leaderboard-picked .tla + .cfg
  - a scores.json with all four phase scores + their status (ran/skipped/etc.)
  - the TV final_report.md if Phase 3 TV ran

Pick rule: max total_score across all batches for that (canonical_model, system).
Same rule the leaderboard uses (see scripts/build_leaderboard.py).

Layout:
    docs/leaderboard/specs/
      README.md
      INDEX.csv                      # flat overview
      <model>/
        <system>/
          <module>.tla
          <module>.cfg
          scores.json
          tv_report.md               # only present if Phase 3 TV ran

Idempotent — re-run any time.
"""
from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_JSON = PROJECT_ROOT / "docs" / "leaderboard" / "data.json"
PAPER_CSV = PROJECT_ROOT / "docs" / "leaderboard" / "paper_summary.csv"
OUT_ROOT = PROJECT_ROOT / "docs" / "leaderboard" / "specs"


def phase_status(phase_score: float | None, phase_ran_flag=None) -> str:
    """Human-readable status for a phase."""
    if phase_score is None:
        return "skipped"  # cascade-skipped or not evaluated
    if phase_score == 0.0:
        return "ran_failed"
    if phase_score == 1.0:
        return "ran_passed"
    return "ran_partial"


def build_scores(row: dict) -> dict:
    """Turn a detail-row into a compact, human-readable phase report."""
    p1 = row.get("phase1_score")
    p2 = row.get("phase2_score")
    p2_rc = row.get("phase2_runtime_check_passed")
    p2_cov = row.get("phase2_coverage")
    p3_tv_rate = row.get("phase3_tv_rate")
    p3_final = row.get("phase3_final_score")
    p4 = row.get("phase3b_score")
    return {
        "overall_score": row.get("overall_score"),
        "formula": "mean(P1, P2, P3_final, P4); missing phases count as 0",
        "phase1_compilation": {
            "score": p1,
            "status": phase_status(p1),
        },
        "phase2_runtime": {
            "score": p2,
            "status": phase_status(p2),
            "coverage": p2_cov,
            "runtime_check_passed": p2_rc,
        },
        "phase3_conformance": {
            "final_score": p3_final,
            "tv_rate": p3_tv_rate,
            "audit_run": row.get("phase3_audit_run"),
            "audit_bugs": row.get("phase3_audit_bugs") or [],
            "status": ("ran_passed" if p3_final == 1.0
                      else "ran_failed" if p3_final == 0.0
                      else "ran_partial" if p3_final is not None
                      else "skipped"),
        },
        "phase4_invariant": {
            "score": p4,
            "status": phase_status(p4),
        },
        "source": {
            "best_run_json": row.get("best_run_json_path"),
            "best_run_spec": row.get("best_run_spec_path"),
            "tv_workspace": row.get("tv_workspace_path"),
            "notes": row.get("notes") or [],
        },
    }


def copy_cell(row: dict, model_dir: Path) -> dict:
    system = row["system"]
    cell_dir = model_dir / system
    cell_dir.mkdir(parents=True, exist_ok=True)

    # Copy the spec .tla and .cfg from the picked run's source dir.
    spec_path = row.get("best_run_spec_path")
    copied_spec = None
    if spec_path:
        sp = Path(spec_path)
        if sp.is_absolute() and sp.exists():
            src_dir = sp.parent
        else:
            src_dir = PROJECT_ROOT / sp.parent
        if src_dir.exists():
            for f in src_dir.iterdir():
                if f.is_file() and (f.suffix in {".tla", ".cfg"}):
                    shutil.copy2(f, cell_dir / f.name)
                    if f.suffix == ".tla":
                        copied_spec = f.name

    # scores.json
    (cell_dir / "scores.json").write_text(
        json.dumps(build_scores(row), indent=2, ensure_ascii=False)
    )

    # TV report if any
    tv = row.get("tv_workspace_path")
    if tv:
        report = PROJECT_ROOT / tv / "reports" / "final_report.md"
        if report.exists():
            shutil.copy2(report, cell_dir / "tv_report.md")

    return {
        "model": row["model"],
        "system": system,
        "overall_score": row.get("overall_score"),
        "phase1": row.get("phase1_score"),
        "phase2": row.get("phase2_score"),
        "phase3_conformance": row.get("phase3_final_score"),
        "phase4": row.get("phase3b_score"),
        "spec_file": copied_spec,
        "has_tv_report": (cell_dir / "tv_report.md").exists(),
    }


def main():
    if OUT_ROOT.exists():
        shutil.rmtree(OUT_ROOT)
    OUT_ROOT.mkdir(parents=True)

    data = json.loads(DATA_JSON.read_text())
    paper_models = [ln.split(",")[0]
                    for ln in PAPER_CSV.read_text().strip().splitlines()[1:]]

    rows_by_model: dict[str, list[dict]] = {}
    for r in data["primary_rows"]:
        if r["model"] in paper_models:
            rows_by_model.setdefault(r["model"], []).append(r)

    index_rows = []
    for model in paper_models:
        model_dir = OUT_ROOT / model
        model_dir.mkdir()
        for r in sorted(rows_by_model.get(model, []), key=lambda x: x["system"]):
            index_rows.append(copy_cell(r, model_dir))

    with (OUT_ROOT / "INDEX.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "model", "system", "overall_score",
            "phase1", "phase2", "phase3_conformance", "phase4",
            "spec_file", "has_tv_report",
        ])
        w.writeheader()
        w.writerows(index_rows)

    (OUT_ROOT / "README.md").write_text(README_MD.strip() + "\n")

    print(f"Wrote {len(index_rows)} cells ({len(paper_models)} models × 11 systems) "
          f"to {OUT_ROOT.relative_to(PROJECT_ROOT)}")
    tv_count = sum(1 for r in index_rows if r["has_tv_report"])
    spec_count = sum(1 for r in index_rows if r["spec_file"])
    print(f"  specs copied: {spec_count}/{len(index_rows)}")
    print(f"  tv_report.md: {tv_count}/{len(index_rows)}")


README_MD = """
# Leaderboard Specs

One folder per (model, system) cell, containing the exact spec that produced
that row's score in `paper_summary.csv`.

## Pick rule

For each (canonical_model, system), we scan every `experiments/batch_*/<sys>/run_*.json`
and keep the run with the **highest `total_score`** (= mean of the 4 phases,
missing phases count as 0). That run's `.tla` and `.cfg` are copied here; its
phase-level details go into `scores.json`.

## Layout

    specs/
      INDEX.csv                    flat list of all 121 cells
      <model>/
        <system>/
          <module>.tla             the exact spec that got scored
          <module>.cfg             TLC config used for P1/P2
          scores.json              P1/P2/P3/P4 with status + provenance
          tv_report.md             present only if Phase 3 TV ran

## scores.json schema

    overall_score        : float — mean of the 4 phases
    formula              : "mean(P1, P2, P3_final, P4); missing count as 0"
    phase1_compilation   : { score, status }
    phase2_runtime       : { score, status, coverage, runtime_check_passed }
    phase3_conformance   : { final_score, tv_rate, audit_run, audit_bugs, status }
    phase4_invariant     : { score, status }
    source               : { best_run_json, best_run_spec, tv_workspace, notes }

`status` values: `ran_passed` (1.0), `ran_failed` (0.0), `ran_partial`
(0 < score < 1), `skipped` (phase didn't run — cascade from an upstream failure,
or TV gate not met).

## Regenerating

    python3 scripts/build_spec_showcase.py

Re-runs safely (wipes and rebuilds `specs/`).
"""


if __name__ == "__main__":
    main()
