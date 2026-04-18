#!/usr/bin/env python3
"""
Re-interpret existing run_1.json files under experiments/ with the new
status semantics (ran / skipped / pending / not_evaluated) and recompute
total_score using the equal-weighted mean over `status=ran` phases only.

Also rescues spin WV scores: if the workspace has a final_report.md and the
stored score is 0 / missing due to the old parser bug, re-parse with the
new regex.

Idempotent. Modifies files in place.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXP_ROOT = PROJECT_ROOT / "experiments"


def infer_status(phase: dict) -> str:
    """Classify an existing PhaseResult dict into a status."""
    if phase is None:
        return None
    # explicit from current code paths
    if phase.get("status"):
        return phase["status"]
    details = phase.get("details") or {}
    if details.get("reason") == "generation_failed":
        return "not_evaluated"
    if "skipped" in details:
        return "skipped"
    pending_reasons = {
        "cannot_evaluate",
        "no_workspace_created",
        "no_results_in_workspace",
        "wv_timeout",
        "spec_broken",  # legacy label for WV "Cannot evaluate"
    }
    if details.get("reason") in pending_reasons:
        return "pending"
    err = (phase.get("error") or "").lower()
    if any(s in err for s in ("no results", "no workspace", "timeout")):
        return "pending"
    # an evaluation that completed with an agent-reported failure is still "ran"
    return "ran"


def parse_wv_report(report_path: Path) -> tuple[int, int] | None:
    """Re-parse a WV final_report.md for per-action pass rates."""
    if not report_path.exists():
        return None
    text = report_path.read_text()
    if "Cannot evaluate" in text or "CANNOT EVALUATE" in text:
        return None
    pair_re = re.compile(r"(\d+)\s*/\s*(\d+)")
    matches = []
    for line in text.splitlines():
        if not line.lstrip().startswith("|"):
            continue
        for m in pair_re.finditer(line):
            x, y = int(m.group(1)), int(m.group(2))
            if 0 <= x <= y and y > 0:
                matches.append((x, y))
                break
    if not matches:
        return None
    tp = sum(m[0] for m in matches)
    tw = sum(m[1] for m in matches)
    return tp, tw


def find_wv_workspace(system: str, timestamp: str) -> Path | None:
    """Find the wv-workspace that matches this run by timestamp + system."""
    ws_root = PROJECT_ROOT / "wv-workspaces"
    if not ws_root.exists():
        return None
    # workspace naming: <yyyymmdd_HHMMSS>_<system>/
    # timestamp from run_1.json is <yyyymmdd_HHMMSS>
    # match system name and pick the closest workspace within ±30 min
    from datetime import datetime, timedelta
    try:
        t0 = datetime.strptime(timestamp, "%Y%m%d_%H%M%S")
    except Exception:
        return None
    candidates = []
    for d in ws_root.iterdir():
        if not d.is_dir():
            continue
        if not d.name.endswith(f"_{system}"):
            continue
        try:
            ts = d.name.split("_")[0] + "_" + d.name.split("_")[1]
            wt = datetime.strptime(ts, "%Y%m%d_%H%M%S")
            if abs((wt - t0).total_seconds()) < 3600:  # within 1h
                candidates.append((abs((wt - t0).total_seconds()), d))
        except Exception:
            continue
    return min(candidates)[1] if candidates else None


def rescue_wv(phase: dict, system: str = None, timestamp: str = None) -> dict:
    """If spin-style WV was parsed as 0 due to old parser, recompute from report."""
    if phase is None:
        return phase
    details = phase.get("details") or {}
    workspace = details.get("workspace")
    # Fallback: scan wv-workspaces/ for a matching ws by system + timestamp
    if not workspace and system and timestamp:
        ws = find_wv_workspace(system, timestamp)
        if ws:
            workspace = str(ws)
            print(f"    found WV workspace by scan: {ws.name}")
    if not workspace:
        return phase
    report = Path(workspace) / "reports" / "final_report.md"
    parsed = parse_wv_report(report)
    if parsed is None:
        return phase  # agent really couldn't evaluate
    tp, tw = parsed
    old_score = phase.get("score")
    new_score = tp / tw if tw > 0 else 0.0
    # Only rescue if old score was 0 or None and new score > 0 (avoid overwriting
    # genuine 0 results).
    if (old_score in (0, 0.0, None)) and new_score > 0:
        phase["score"] = new_score
        phase["passed"] = new_score > 0
        phase["status"] = "ran"
        phase["error"] = None
        phase["details"] = {
            **details,
            "total_passed": tp,
            "total_windows": tw,
            "rescued_from_report": True,
        }
        print(f"    rescued WV: {tp}/{tw} = {new_score:.2%}")
    return phase


def apply_phase2_cascade(run: dict) -> None:
    """
    Enforce the same cascade as the runner: if Phase 2 didn't actually
    exercise the spec (runtime_check failed, OR coverage == 0 which means
    TLC explored zero states and the "pass" is vacuous), downstream phases
    are disqualified. Convert any scored Phase 3 / Phase 3b result into
    status=skipped / score=None so they do NOT average into total_score.

    Observed 2026-04-17 on kimi_k25_ds/etcd: P2 coverage=0 but P3b
    invariant (agent-translator, TLC-independent) scored 0.92 and
    inflated total to 0.64.
    """
    p2 = run.get("phase2_runtime") or {}
    details = p2.get("details") or {}
    runtime_check_passed = details.get("runtime_check_passed", True)
    coverage = details.get("coverage", None)
    zero_coverage = (coverage is not None and coverage <= 0.0)
    if runtime_check_passed and not zero_coverage:
        return
    reason_tag = (
        "phase2_runtime_check_failed" if not runtime_check_passed
        else "phase2_zero_coverage"
    )
    for key in ("phase3_wv", "phase3_invariant"):
        ph = run.get(key)
        if ph is None:
            continue
        if ph.get("status") == "ran" and ph.get("score") is not None:
            ph["_original_score"] = ph.get("score")
        ph["status"] = "skipped"
        ph["score"] = None
        ph["passed"] = False
        ph["details"] = {**(ph.get("details") or {}), "skipped": reason_tag}


def recompute_total(run: dict) -> float:
    """Mean over {ran at score} ∪ {skipped as 0}.

    Skipped phases count as 0 — a cascade-skip means the spec could not
    demonstrate anything in that phase. not_evaluated / pending / None
    remain excluded (those mean the phase was never attempted).
    """
    scored = []
    for key in ("phase1_compilation", "phase2_runtime", "phase3_wv", "phase3_invariant"):
        p = run.get(key)
        if not p:
            continue
        st = p.get("status")
        if st == "ran" and p.get("score") is not None:
            scored.append(p["score"])
        elif st == "skipped":
            scored.append(0.0)
    return sum(scored) / len(scored) if scored else 0.0


def fixup_phase(phase: dict, phase_name: str, run: dict = None) -> dict:
    """Apply status + clean up score=None semantics in one pass."""
    if phase is None:
        # Legacy runs left phase fields as literal None when generation failed
        # (before we filled in not_evaluated explicitly). Synthesize a
        # not_evaluated marker so downstream consumers see clear status.
        if run and not run.get("generation_success"):
            return {
                "phase_name": phase_name.replace("phase1_compilation", "compilation")
                                         .replace("phase2_runtime", "runtime")
                                         .replace("phase3_wv", "wv")
                                         .replace("phase3_invariant", "invariant"),
                "status": "not_evaluated",
                "score": None,
                "passed": False,
                "details": {"reason": "generation_failed"},
                "error": None,
                "passed_items": [],
                "failed_items": [],
            }
        return None
    status = infer_status(phase)
    phase["status"] = status
    if status != "ran":
        # Non-ran statuses should report score=None so downstream aggregators
        # treat them as absent, not as real 0.
        phase["score"] = None
    return phase


def backfill_one(run_path: Path) -> dict:
    with open(run_path) as f:
        run = json.load(f)
    before_total = run.get("total_score")

    # Try WV rescue FIRST (while old score=0 might still be there)
    if run.get("phase3_wv"):
        run["phase3_wv"] = rescue_wv(
            run["phase3_wv"],
            system=run.get("system"),
            timestamp=run.get("timestamp"),
        )

    for key in ("phase1_compilation", "phase2_runtime", "phase3_wv", "phase3_invariant"):
        run[key] = fixup_phase(run.get(key), key, run)

    # Retroactive cascade: disqualify P3/P3b if P2 didn't truly exercise the spec.
    apply_phase2_cascade(run)

    run["total_score"] = recompute_total(run)
    run["is_perfect"] = all(
        run.get(k) and run[k].get("status") == "ran" and (run[k].get("score") or 0) >= 1.0
        for k in ("phase1_compilation", "phase2_runtime", "phase3_wv", "phase3_invariant")
    )

    with open(run_path, "w") as f:
        json.dump(run, f, indent=2, ensure_ascii=False)

    return {
        "system": run.get("system"),
        "before": before_total,
        "after": run["total_score"],
        "phases": {
            k: (run.get(k) or {}).get("status", "?")
            for k in ("phase1_compilation", "phase2_runtime",
                      "phase3_wv", "phase3_invariant")
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batches", nargs="*", help="Batch dir names (default: all)")
    args = ap.parse_args()

    targets = []
    for batch in EXP_ROOT.glob("batch_*"):
        if args.batches and batch.name not in args.batches:
            continue
        for run in batch.glob("*/run_*.json"):
            targets.append(run)

    print(f"Processing {len(targets)} run files\n")
    for run_path in sorted(targets):
        rel = run_path.relative_to(PROJECT_ROOT)
        print(f"{rel}")
        try:
            r = backfill_one(run_path)
            b = r["before"]
            a = r["after"]
            changed = "" if b == a else f"  (was {b:.3f})"
            phases = " ".join(f"{k[:3]}:{v[:4]}" for k, v in r["phases"].items())
            print(f"  {r['system']:12} total={a:.3f}{changed}  [{phases}]")
        except Exception as e:
            print(f"  ERROR: {e}")


if __name__ == "__main__":
    main()
