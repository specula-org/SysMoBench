#!/usr/bin/env python3
"""
Recompute total scores across all experiment runs using a pluggable formula.

Per-phase scores are stored raw in each run_1.json. This tool lets you apply
any weighting scheme after the fact, without re-running experiments.

Usage:
    python3 scripts/compute_scores.py                     # default: equal-weighted
    python3 scripts/compute_scores.py --formula tv_only   # use custom formula
    python3 scripts/compute_scores.py --systems spin etcd # filter systems
    python3 scripts/compute_scores.py --csv > results.csv

Add new formulas in the FORMULAS dict below. A formula is just
    fn(phase_scores: dict[str, float | None]) -> float | None
where the dict keys are: "compilation", "runtime", "tv", "invariant".
Return None to indicate "not scoreable under this formula".
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Callable, Dict, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"

PhaseScores = Dict[str, Optional[float]]
Formula = Callable[[PhaseScores], Optional[float]]


def equal_weighted(s: PhaseScores) -> Optional[float]:
    """Unweighted mean of all non-None phase scores."""
    vals = [v for v in s.values() if v is not None]
    return sum(vals) / len(vals) if vals else None


def tv_only(s: PhaseScores) -> Optional[float]:
    """Only the transition validation score."""
    return s.get("tv")


def strict_all_or_nothing(s: PhaseScores) -> Optional[float]:
    """1.0 only if every phase passed perfectly."""
    vals = [v for v in s.values() if v is not None]
    if not vals:
        return None
    return 1.0 if all(v >= 1.0 for v in vals) else 0.0


def semantic_heavy(s: PhaseScores) -> Optional[float]:
    """Weights favor P3 TV (the trace-faithfulness signal): 0.1/0.2/0.5/0.2."""
    weights = {"compilation": 0.1, "runtime": 0.2, "tv": 0.5, "invariant": 0.2}
    present = {k: v for k, v in s.items() if v is not None}
    if not present:
        return None
    total = sum(weights[k] * present[k] for k in present)
    norm = sum(weights[k] for k in present)
    return total / norm


FORMULAS: Dict[str, Formula] = {
    "equal_weighted": equal_weighted,
    "tv_only": tv_only,
    "strict": strict_all_or_nothing,
    "semantic_heavy": semantic_heavy,
}


def extract_scores(run: dict) -> PhaseScores:
    return {
        "compilation": (run.get("phase1_compilation") or {}).get("score"),
        "runtime": (run.get("phase2_runtime") or {}).get("score"),
        "tv": (run.get("phase3_tv") or {}).get("score"),
        "invariant": (run.get("phase3_invariant") or {}).get("score"),
    }


def collect_runs(systems_filter: Optional[list] = None):
    """Yield (batch_dir_name, system, run_path, run_data)."""
    for run_path in sorted(EXPERIMENTS_DIR.glob("batch_*/*/run_*.json")):
        system = run_path.parent.name
        if systems_filter and system not in systems_filter:
            continue
        try:
            with open(run_path) as f:
                data = json.load(f)
        except Exception as e:
            print(f"skip {run_path}: {e}", file=sys.stderr)
            continue
        yield run_path.parent.parent.name, system, run_path, data


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--formula", default="equal_weighted", choices=list(FORMULAS),
                    help="scoring formula (default: equal_weighted)")
    ap.add_argument("--systems", nargs="+", help="restrict to these systems")
    ap.add_argument("--csv", action="store_true", help="CSV output instead of table")
    args = ap.parse_args()

    formula = FORMULAS[args.formula]

    rows = []
    for batch, system, path, data in collect_runs(args.systems):
        scores = extract_scores(data)
        total = formula(scores)
        model = None
        try:
            from pathlib import Path as _P
            exp_log = _P(path).parent.parent / "experiment.log"
            if exp_log.exists():
                for line in exp_log.read_text().splitlines():
                    if "] Model:" in line:
                        model = line.split("] Model:")[-1].strip()
                        break
        except Exception:
            pass
        gen = (data.get("phase0_usage") or {}).get("usage") or {}
        tv_usage = data.get("phase3_tv_usage") or {}
        rows.append({
            "batch": batch,
            "system": system,
            "model": model or "?",
            "p1": scores["compilation"],
            "p2": scores["runtime"],
            "p3_tv": scores["tv"],
            "p3b": scores["invariant"],
            "total": total,
            "gen_prompt_tok": gen.get("prompt_tokens"),
            "gen_completion_tok": gen.get("completion_tokens"),
            "gen_reasoning_tok": (gen.get("completion_tokens_details") or {}).get("reasoning_tokens"),
            "tv_cost_usd": tv_usage.get("cost_usd"),
            "tv_min": (tv_usage.get("duration_ms") or 0) / 1000 / 60 if tv_usage.get("duration_ms") else None,
        })

    if args.csv:
        w = csv.DictWriter(sys.stdout, fieldnames=list(rows[0].keys()) if rows else [])
        w.writeheader()
        for r in rows:
            w.writerow(r)
        return

    print(f"Formula: {args.formula}")
    print(f"{'Model':25} {'System':10} {'P1':>5} {'P2':>5} {'TV':>5} {'P3b':>5} {'Total':>7} {'TV $':>7} {'TV min':>7}")
    print("-" * 95)
    for r in rows:
        fmt = lambda x: f"{x:.2f}" if isinstance(x, float) else str(x or "-")
        print(f"{r['model']:25} {r['system']:10} {fmt(r['p1']):>5} {fmt(r['p2']):>5} "
              f"{fmt(r['p3_tv']):>5} {fmt(r['p3b']):>5} {fmt(r['total']):>7} "
              f"{('$' + fmt(r['tv_cost_usd'])) if r['tv_cost_usd'] else '-':>7} "
              f"{fmt(r['tv_min']):>7}")


if __name__ == "__main__":
    main()
