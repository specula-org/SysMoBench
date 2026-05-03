#!/usr/bin/env python3
"""Rebuild the leaderboard using the repaired specs + fresh P3/P4 runs.

Sources per cell:
  P1, P2    — docs/leaderboard/specs/<model>/<system>/scores.json
              (ORIGINAL, pre-repair values — P1/P2 measure the model's own
              syntax/runtime fluency. Using the post-repair values would
              make every cell 1.0 and erase the signal.)
  P3 (TV)   — latest TV workspace whose spec symlink points at
              docs/leaderboard/specs_repaired/<model>/<system>/, else fall back
              to the old workspace used for the original scores.json.
              Parsed via the shared helper in build_leaderboard.py
              (zero-tolerance per-action: rate==1.0 AND not 'wrong' → 1.0).
  P4 (inv)  — batch_logs/p4_results/<model>__<system>.json marker + the agent
              log it points at (parse 'Passed invariants: N' / 'Total ... K').
              Fallback: scores.json phase4_invariant.score.

Overall score weighting (canonical): P1=0.15  P2=0.15  P3=0.35  P4=0.35.
P1/P2 are partially informative signals (all post-repair cells are 1.0 so we
read the ORIGINAL pre-repair values); P3/P4 discriminate more, hence larger
weight. To try a different weighting without rewriting the CSVs, use
scripts/reweight_leaderboard.py.

Outputs:
  docs/leaderboard/detail_repaired.csv
  docs/leaderboard/aggregate_repaired.csv
  docs/leaderboard/paper_summary_repaired.csv
  Prints the paper_summary table to stdout.
"""

from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path

# Canonical phase weights
W_P1, W_P2, W_P3, W_P4 = 0.15, 0.15, 0.35, 0.35

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from build_leaderboard import parse_tv_final_report  # type: ignore

SPECS_DIR = PROJECT_ROOT / "docs" / "leaderboard" / "specs"
REPAIRED_DIR = PROJECT_ROOT / "docs" / "leaderboard" / "specs_repaired"
TV_ROOT = PROJECT_ROOT / "tv-workspaces"
P4_RESULTS = PROJECT_ROOT / "batch_logs" / "p4_results"
OUT_DIR = PROJECT_ROOT / "docs" / "leaderboard"

SYSTEMS = ["curp", "dqueue", "etcd", "locksvc", "mutex", "raftkvs",
           "redisraft", "ringbuffer", "rwmutex", "spin", "zookeeper"]


# ── P1/P2 from repair manifest ──

def read_repair_manifest(model: str, system: str) -> dict | None:
    p = REPAIRED_DIR / model / system / "repair_manifest.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def read_original_scores(model: str, system: str) -> dict | None:
    p = SPECS_DIR / model / system / "scores.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


# ── P3 (TV) ──

def find_latest_tv_for(model: str, system: str) -> Path | None:
    """Prefer workspaces whose spec symlink points at specs_repaired/<cell>.
    Fall back to the one referenced by original scores.json if no new TV.
    """
    target = (REPAIRED_DIR / model / system).resolve()
    candidates = []
    if TV_ROOT.exists():
        for wdir in TV_ROOT.iterdir():
            if not wdir.is_dir():
                continue
            spec_link = wdir / "spec"
            if not spec_link.exists():
                continue
            try:
                resolved = spec_link.resolve()
            except Exception:
                continue
            if resolved == target and (wdir / "reports" / "final_report.md").exists():
                candidates.append(wdir)
    if candidates:
        candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return candidates[0]
    # Fallback: use the one recorded in original scores.json
    orig = read_original_scores(model, system)
    if orig:
        ws = (orig.get("source") or {}).get("tv_workspace")
        if ws:
            p = PROJECT_ROOT / ws
            if (p / "reports" / "final_report.md").exists():
                return p
    return None


# ── P4 (manual invariant verification) ──

P4_FINAL_RE = re.compile(
    r"Manual invariant testing:\s*(\d+)\s*/\s*(\d+)\s+invariants passed",
    re.IGNORECASE,
)
P4_PASSED_RE = re.compile(r"Passed invariants:\s*(\d+)")
P4_TOTAL_RE = re.compile(r"Total invariants tested:\s*(\d+)")


def read_p4_from_new_batch(model: str, system: str) -> tuple[float, int, int] | None:
    marker = P4_RESULTS / f"{model}__{system}.json"
    if not marker.exists():
        return None
    try:
        d = json.loads(marker.read_text())
    except Exception:
        return None
    if not d.get("completed"):
        return None
    log_path = d.get("log_path")
    if not log_path or not Path(log_path).exists():
        return None
    try:
        text = Path(log_path).read_text(errors="replace")
    except Exception:
        return None
    m = P4_FINAL_RE.search(text)
    if m:
        p, t = int(m.group(1)), int(m.group(2))
        return (p / t if t else 0.0, p, t)
    p = P4_PASSED_RE.search(text)
    t = P4_TOTAL_RE.search(text)
    if p and t:
        pi, ti = int(p.group(1)), int(t.group(1))
        return (pi / ti if ti else 0.0, pi, ti)
    return None


def read_p4_from_original(model: str, system: str) -> float | None:
    d = read_original_scores(model, system)
    if not d:
        return None
    v = (d.get("phase4_invariant") or {}).get("score")
    try:
        return float(v) if v is not None else None
    except Exception:
        return None


# ── Build ──

def build_one(model: str, system: str) -> dict:
    row = {
        "model": model, "system": system,
        "p1": None, "p2": None, "p3": None, "p4": None,
        "p3_source": "", "p4_source": "", "repair_status": "",
        "tv_workspace": "", "edits": 0,
    }
    man = read_repair_manifest(model, system)
    orig = read_original_scores(model, system)

    # P1/P2 always from the ORIGINAL (pre-repair) scores — they measure
    # the model's own syntax/runtime fluency. Post-repair values would
    # be 1.0 for everyone and erase the signal.
    if orig is not None:
        row["p1"] = (orig.get("phase1_compilation") or {}).get("score")
        row["p2"] = (orig.get("phase2_runtime") or {}).get("score")
    if man is not None:
        row["repair_status"] = man.get("status", "")
        row["edits"] = man.get("edit_count", 0) or 0

    ws = find_latest_tv_for(model, system)
    if ws is not None:
        row["tv_workspace"] = str(ws.relative_to(PROJECT_ROOT))
        tv = parse_tv_final_report(ws)
        row["p3"] = tv.get("phase3_final_score")
        # Mark where P3 came from
        target = (REPAIRED_DIR / model / system).resolve()
        try:
            row["p3_source"] = ("repaired"
                                if (ws / "spec").resolve() == target
                                else "original")
        except Exception:
            row["p3_source"] = "original"
    # If no TV workspace found, P3 stays None

    new_p4 = read_p4_from_new_batch(model, system)
    if new_p4 is not None:
        row["p4"] = new_p4[0]
        row["p4_source"] = f"repaired ({new_p4[1]}/{new_p4[2]})"
    else:
        v = read_p4_from_original(model, system)
        if v is not None:
            row["p4"] = v
            row["p4_source"] = "original"

    # Overall = weighted sum of the 4, treating missing as 0.
    def _v(x):
        return x if x is not None else 0.0
    row["overall"] = (W_P1 * _v(row["p1"]) + W_P2 * _v(row["p2"])
                      + W_P3 * _v(row["p3"]) + W_P4 * _v(row["p4"]))
    return row


def build_all() -> list[dict]:
    rows = []
    if not REPAIRED_DIR.exists():
        return rows
    for model_dir in sorted(REPAIRED_DIR.iterdir()):
        if not model_dir.is_dir():
            continue
        model = model_dir.name
        for system in SYSTEMS:
            if (model_dir / system).is_dir():
                rows.append(build_one(model, system))
    return rows


# ── Outputs ──

def write_detail(rows, path: Path):
    cols = ["model", "system", "overall", "p1", "p2", "p3", "p4",
            "p3_source", "p4_source", "repair_status", "edits",
            "tv_workspace"]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in cols})


def write_aggregate(rows, path: Path):
    by = {}
    for r in rows:
        by.setdefault(r["model"], []).append(r)
    out = []
    for model, items in sorted(by.items()):
        def m(field, default=0.0):
            vals = [(x.get(field) or default) for x in items]
            return sum(vals) / len(vals) if vals else 0.0
        out.append({
            "model": model,
            "n_systems": len(items),
            "overall_mean": round(m("overall"), 4),
            "p1_mean": round(m("p1"), 4),
            "p2_mean": round(m("p2"), 4),
            "p3_mean": round(m("p3"), 4),
            "p4_mean": round(m("p4"), 4),
            "n_p1_fail": sum(1 for x in items if (x.get("p1") or 0) < 1.0),
            "n_p2_fail": sum(1 for x in items if (x.get("p2") or 0) < 1.0),
            "n_no_p3": sum(1 for x in items if x.get("p3") is None),
            "n_no_p4": sum(1 for x in items if x.get("p4") is None),
            "repairs": sum(1 for x in items
                           if x.get("repair_status") not in ("none", "", None)),
        })
    out.sort(key=lambda x: x["overall_mean"], reverse=True)
    cols = ["model", "n_systems", "overall_mean", "p1_mean", "p2_mean",
            "p3_mean", "p4_mean", "n_p1_fail", "n_p2_fail", "n_no_p3",
            "n_no_p4", "repairs"]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in out:
            w.writerow(r)
    return out


def write_paper_summary(agg, rows, path: Path):
    """Website-sync-friendly CSV (matches the layout of the old
    paper_summary.csv so the website agent can diff easily):

        model, n_systems, overall_score,
        phase1_compilation, phase2_runtime, phase3_conformance, phase4_invariant,
        sys_spin, sys_mutex, sys_rwmutex, sys_dqueue, sys_ringbuffer,
        sys_locksvc, sys_curp, sys_raftkvs, sys_redisraft, sys_zookeeper, sys_etcd

    Per-model phase columns are means across the model's 11 systems.
    Per-system columns hold the cell's overall score (mean of the 4 phases,
    missing counts as 0), matching the old file's semantics.
    """
    # Keep the old non-alphabetical column order (simple -> complex)
    system_order = ["spin", "mutex", "rwmutex", "dqueue", "ringbuffer",
                    "locksvc", "curp", "raftkvs", "redisraft", "zookeeper",
                    "etcd"]
    by_cell: dict[tuple[str, str], float] = {}
    for r in rows:
        by_cell[(r["model"], r["system"])] = r["overall"]

    cols = (["model", "n_systems", "overall_score",
             "phase1_compilation", "phase2_runtime",
             "phase3_conformance", "phase4_invariant"]
            + [f"sys_{s}" for s in system_order])

    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in agg:
            row = [
                r["model"], r["n_systems"],
                round(r["overall_mean"], 3),
                round(r["p1_mean"], 3), round(r["p2_mean"], 3),
                round(r["p3_mean"], 3), round(r["p4_mean"], 3),
            ]
            for s in system_order:
                v = by_cell.get((r["model"], s))
                row.append(round(v, 3) if v is not None else "")
            w.writerow(row)


def print_table(agg):
    print()
    print(f"{'model':<25} {'n':>3} {'overall':>8} {'P1':>6} {'P2':>6} {'P3(TV)':>8} {'P4(Inv)':>8}  {'gaps (p3/p4 miss)':>18}")
    print("-" * 92)
    for r in agg:
        gap = f"{r['n_no_p3']}/{r['n_no_p4']}"
        print(f"{r['model']:<25} {r['n_systems']:>3} "
              f"{r['overall_mean']:>8.4f} "
              f"{r['p1_mean']:>6.3f} {r['p2_mean']:>6.3f} "
              f"{r['p3_mean']:>8.4f} {r['p4_mean']:>8.4f}  "
              f"{gap:>18}")


def main():
    rows = build_all()
    if not rows:
        print("no cells found", file=sys.stderr)
        return 1

    detail_path = OUT_DIR / "detail_repaired.csv"
    aggregate_path = OUT_DIR / "aggregate_repaired.csv"
    paper_path = OUT_DIR / "paper_summary_repaired.csv"

    write_detail(rows, detail_path)
    agg = write_aggregate(rows, aggregate_path)
    write_paper_summary(agg, rows, paper_path)

    print_table(agg)
    print()
    print(f"wrote {detail_path}")
    print(f"wrote {aggregate_path}")
    print(f"wrote {paper_path}")
    print(f"\n{len(rows)} cells scored")
    return 0


if __name__ == "__main__":
    sys.exit(main())
