#!/usr/bin/env python3
"""Reweight the repaired leaderboard with arbitrary phase weights.

Reads docs/leaderboard/detail_repaired.csv and prints a model-level
ranking under user-specified (w1, w2, w3, w4). Missing phases count as 0.

Usage:
  python3 scripts/reweight_leaderboard.py [--weights w1,w2,w3,w4]
                                          [--compare]
                                          [--top N]
                                          [--csv path]

Examples:
  # default equal weights (0.25 each) = same as current leaderboard
  python3 scripts/reweight_leaderboard.py

  # put most weight on the two discriminating phases (P3 / P4)
  python3 scripts/reweight_leaderboard.py --weights 0.05,0.05,0.45,0.45

  # P3-only ranking (pure behavioral conformance)
  python3 scripts/reweight_leaderboard.py --weights 0,0,1,0

  # side-by-side comparison against the equal-weight baseline
  python3 scripts/reweight_leaderboard.py --weights 0,0,0.5,0.5 --compare

Weights don't need to sum to 1 — they're auto-normalized.
"""

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DETAIL_CSV = PROJECT_ROOT / "docs" / "leaderboard" / "detail_repaired.csv"


def load_rows():
    rows = list(csv.DictReader(open(DETAIL_CSV)))
    for r in rows:
        for k in ("p1", "p2", "p3", "p4"):
            r[k] = float(r[k]) if r[k] not in ("", None) else None
    return rows


def aggregate(rows, weights):
    w = [float(x) for x in weights]
    total = sum(w)
    if total <= 0:
        raise SystemExit("weights sum to 0")
    w = [x / total for x in w]

    by_model = defaultdict(list)
    for r in rows:
        by_model[r["model"]].append(r)

    out = []
    for model, items in by_model.items():
        def mean_of(k):
            vals = [(it[k] or 0.0) for it in items]
            return sum(vals) / len(vals) if vals else 0.0
        p = [mean_of(f"p{i}") for i in (1, 2, 3, 4)]
        overall = sum(wi * pi for wi, pi in zip(w, p))
        out.append({
            "model": model, "overall": overall,
            "p1": p[0], "p2": p[1], "p3": p[2], "p4": p[3],
            "n": len(items),
        })
    out.sort(key=lambda x: x["overall"], reverse=True)
    return out, w


def print_table(agg, weights, title=""):
    if title:
        print(f"\n=== {title} ===")
    print(f"weights: P1={weights[0]:.3f}  P2={weights[1]:.3f}  "
          f"P3={weights[2]:.3f}  P4={weights[3]:.3f}")
    print(f"{'#':>2} {'model':<26} {'overall':>8}   "
          f"{'P1':>5} {'P2':>5} {'P3':>5} {'P4':>5}")
    print("-" * 68)
    for i, r in enumerate(agg, 1):
        print(f"{i:>2} {r['model']:<26} {r['overall']:>8.4f}   "
              f"{r['p1']:>5.3f} {r['p2']:>5.3f} "
              f"{r['p3']:>5.3f} {r['p4']:>5.3f}")


def print_compare(agg_new, agg_base):
    pos_base = {r["model"]: i for i, r in enumerate(agg_base, 1)}
    print(f"\n{'#new':>4} {'model':<26} {'new':>8} {'base':>8}   {'Δrank':>7}")
    print("-" * 64)
    for i, r in enumerate(agg_new, 1):
        base_score = next(b["overall"] for b in agg_base if b["model"] == r["model"])
        drank = pos_base[r["model"]] - i
        arrow = f"{drank:+d}" if drank else "—"
        print(f"{i:>4} {r['model']:<26} {r['overall']:>8.4f} "
              f"{base_score:>8.4f}   {arrow:>7}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", default="0.25,0.25,0.25,0.25",
                    help="Comma-separated w1,w2,w3,w4")
    ap.add_argument("--compare", action="store_true",
                    help="Also show baseline (0.25 each) side-by-side")
    ap.add_argument("--top", type=int, default=None)
    ap.add_argument("--csv", type=str, default=None)
    args = ap.parse_args()

    try:
        weights = [float(x) for x in args.weights.split(",")]
    except ValueError:
        raise SystemExit("--weights must be 4 numbers separated by commas")
    if len(weights) != 4:
        raise SystemExit(f"--weights needs 4 values, got {len(weights)}")

    rows = load_rows()
    agg, wnorm = aggregate(rows, weights)
    if args.top:
        agg = agg[:args.top]
    print_table(agg, wnorm, title="Reweighted Leaderboard")

    if args.compare:
        agg_base, _ = aggregate(rows, [0.25] * 4)
        if args.top:
            agg_base_print = agg_base[:args.top]
        else:
            agg_base_print = agg_base
        print_table(agg_base_print, [0.25] * 4, title="Baseline (0.25 each)")
        print_compare(agg, agg_base)

    if args.csv:
        out_path = Path(args.csv)
        with out_path.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["rank", "model", "overall",
                        "p1_mean", "p2_mean", "p3_mean", "p4_mean",
                        "w1", "w2", "w3", "w4"])
            for i, r in enumerate(agg, 1):
                w.writerow([i, r["model"], round(r["overall"], 4),
                            round(r["p1"], 4), round(r["p2"], 4),
                            round(r["p3"], 4), round(r["p4"], 4),
                            *wnorm])
        print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
