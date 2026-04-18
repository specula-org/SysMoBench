#!/usr/bin/env python3
"""
Build SysMoBench leaderboard by scanning:
  - experiments/batch_*/    → Phase A (gen + P1 + P2 + P3b) results per model
  - wv-workspaces/*/        → Phase 3 WV + Audit results per model/system

Outputs (all under docs/leaderboard/):
  - detail.csv       one row per (model, system)
  - aggregate.csv    one row per model (averaged across systems)
  - data.json        full structured dataset (for website agent consumption)
  - schema.md        schema documentation (read this to understand the data)

Idempotent: re-run at any time to refresh.
"""

from __future__ import annotations

import csv
import json
import re
import sys
from dataclasses import dataclass, asdict, field
from pathlib import Path
from datetime import datetime, timezone

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXP_ROOT = PROJECT_ROOT / "experiments"
WV_ROOT = PROJECT_ROOT / "wv-workspaces"
OUT_ROOT = PROJECT_ROOT / "docs" / "leaderboard"

SYSTEMS = ["spin", "etcd", "curp", "dqueue", "locksvc", "mutex",
           "raftkvs", "redisraft", "ringbuffer", "rwmutex", "zookeeper"]


@dataclass
class SystemResult:
    model: str
    system: str
    # Phase A
    phase1_score: float | None = None
    phase2_score: float | None = None
    phase2_coverage: float | None = None
    phase2_runtime_check_passed: bool | None = None
    phase3b_score: float | None = None
    phase_a_total: float | None = None
    # Phase 3 WV+Audit
    phase3_wv_rate: float | None = None  # mean of per-action WV pass rates
    phase3_audit_run: bool = False
    phase3_audit_bugs: list = field(default_factory=list)
    phase3_final_score: float | None = None
    # Composite
    overall_score: float | None = None
    # Provenance
    best_run_spec_path: str | None = None
    best_run_json_path: str | None = None
    wv_workspace_path: str | None = None
    # Cost / usage
    gen_tokens_in: int | None = None
    gen_tokens_out: int | None = None
    wv_agent_cost_usd: float | None = None
    wv_agent_duration_s: float | None = None
    wv_agent_turns: int | None = None
    # Status flags
    notes: list = field(default_factory=list)


def scan_phase_a_batches():
    """Group experiments/batch_*/ runs by (model, system), keep best by total_score."""
    by_ms: dict[tuple[str, str], tuple[float, Path, dict]] = {}
    for batch_dir in sorted(EXP_ROOT.glob("batch_*")):
        exp_log = batch_dir / "experiment.log"
        if not exp_log.exists():
            continue
        m = re.search(r"Model:\s*(\S+)", exp_log.read_text())
        if not m:
            continue
        model = m.group(1)
        # Normalise qwen etcd 1h variant back to base model for leaderboard display
        model_display = model.replace("_1h", "")
        for sys_dir in batch_dir.iterdir():
            if not sys_dir.is_dir() or sys_dir.name not in SYSTEMS:
                continue
            for run_file in sorted(sys_dir.glob("run_*.json")):
                try:
                    d = json.loads(run_file.read_text())
                except Exception:
                    continue
                ts = d.get("total_score")
                if ts is None:
                    continue
                key = (model_display, sys_dir.name)
                if key not in by_ms or ts > by_ms[key][0]:
                    by_ms[key] = (ts, run_file, d)
    return by_ms


def find_wv_workspace_for(model: str, system: str) -> Path | None:
    """Find the latest WV workspace whose spec symlink points to this model's output."""
    if not WV_ROOT.exists():
        return None
    candidates = []
    for ws in sorted(WV_ROOT.glob(f"*_{system}")):
        spec_link = ws / "spec" / f"{system}.tla"
        if not spec_link.exists():
            continue
        target = spec_link.resolve()
        target_str = str(target)
        # Spec path pattern:
        # output/compilation_check/tla/<sys>/direct_call_<model>/<ts>/<sys>.tla
        if f"direct_call_{model}/" in target_str:
            candidates.append(ws)
    # Return newest (lexicographic on timestamped ws name works here)
    return candidates[-1] if candidates else None


def parse_wv_final_report(ws: Path) -> dict:
    """Extract Phase 3 final score + per-action verdicts from the report."""
    rep = ws / "reports" / "final_report.md"
    out = {
        "phase3_final_score": None,
        "phase3_wv_rate": None,
        "audit_run": False,
        "audit_bugs": [],  # list of {action, reason}
    }
    if not rep.exists():
        return out
    text = rep.read_text()
    # final score: "Phase3_score = (...) = 0.833" or "Phase 3 score = 0.833"
    m = re.search(r"Phase\s*3[_ ]?score\s*=[^=]*=\s*([0-9.]+)", text)
    if m:
        out["phase3_final_score"] = float(m.group(1))
    # audit run flag: look for "## Phase 3" + "Audited" column header
    if re.search(r"(audited|Audit Results|Step 9)", text, re.IGNORECASE):
        out["audit_run"] = True
    # per-action wrong verdicts
    # match a markdown table row with "wrong" in the Verdict column
    for line in text.splitlines():
        if "|" not in line:
            continue
        if re.search(r"\bwrong\b", line, re.IGNORECASE):
            cells = [c.strip() for c in line.split("|")]
            # cells[0] is usually empty, cells[1] likely action name
            if len(cells) >= 3:
                action = cells[1]
                # trim markdown emphasis
                action = re.sub(r"[*`]", "", action).strip()
                if action and action.lower() not in ("action", "---"):
                    # reason = the line itself minus leading pipes
                    reason = line.strip("| ").strip()
                    out["audit_bugs"].append({"action": action, "line": reason[:200]})
    # Mean WV pass rate (from WV-only section)
    rates = []
    for m in re.finditer(r"\|\s*(\d+)\s*/\s*(\d+)\s*\|", text):
        num, den = int(m.group(1)), int(m.group(2))
        if den > 0 and num <= den:
            rates.append(num / den)
    if rates:
        out["phase3_wv_rate"] = sum(rates) / len(rates)
    return out


def parse_wv_cost(ws: Path) -> dict:
    u = ws / ".run.usage.json"
    if not u.exists():
        return {}
    try:
        d = json.loads(u.read_text())
    except Exception:
        return {}
    return {
        "wv_agent_cost_usd": d.get("total_cost_usd"),
        "wv_agent_duration_s": (d.get("duration_ms") or 0) / 1000 or None,
        "wv_agent_turns": d.get("num_turns"),
    }


def build_rows():
    rows: list[SystemResult] = []
    by_ms = scan_phase_a_batches()
    for (model, system), (ts, run_file, d) in sorted(by_ms.items()):
        r = SystemResult(model=model, system=system)
        r.best_run_json_path = str(run_file.relative_to(PROJECT_ROOT))
        r.best_run_spec_path = d.get("spec_path")
        r.phase_a_total = ts
        p1 = d.get("phase1_compilation") or {}
        p2 = d.get("phase2_runtime") or {}
        p2d = p2.get("details") or {}
        p3b = d.get("phase3_invariant") or {}
        if p1.get("status") == "ran":
            r.phase1_score = p1.get("score")
        if p2.get("status") == "ran":
            r.phase2_score = p2.get("score")
        r.phase2_coverage = p2d.get("coverage")
        r.phase2_runtime_check_passed = p2d.get("runtime_check_passed")
        if p3b.get("status") == "ran":
            r.phase3b_score = p3b.get("score")
        # gen usage
        usage = ((d.get("phase0_usage") or {}).get("usage") or {})
        r.gen_tokens_in = usage.get("prompt_tokens")
        r.gen_tokens_out = usage.get("completion_tokens")
        # WV workspace lookup
        ws = find_wv_workspace_for(model, system)
        if ws:
            r.wv_workspace_path = str(ws.relative_to(PROJECT_ROOT))
            wv_info = parse_wv_final_report(ws)
            r.phase3_wv_rate = wv_info["phase3_wv_rate"]
            r.phase3_audit_run = wv_info["audit_run"]
            r.phase3_audit_bugs = wv_info["audit_bugs"]
            r.phase3_final_score = wv_info["phase3_final_score"]
            r.__dict__.update(parse_wv_cost(ws))
        # overall = mean over ran phases including phase3_final_score (if available)
        scored = [x for x in [r.phase1_score, r.phase2_score,
                              r.phase3_final_score if r.phase3_final_score is not None
                              else r.phase3_wv_rate,
                              r.phase3b_score] if x is not None]
        r.overall_score = sum(scored) / len(scored) if scored else None
        rows.append(r)
    return rows


def write_detail_csv(rows, path: Path):
    fields = [
        "model", "system",
        "phase1_score", "phase2_score", "phase2_coverage",
        "phase2_runtime_check_passed", "phase3b_score",
        "phase3_wv_rate", "phase3_audit_run", "phase3_final_score",
        "overall_score",
        "gen_tokens_in", "gen_tokens_out",
        "wv_agent_cost_usd", "wv_agent_duration_s",
        "best_run_spec_path", "wv_workspace_path",
    ]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(asdict(r))


def write_aggregate_csv(rows, path: Path):
    by_model: dict[str, list[SystemResult]] = {}
    for r in rows:
        by_model.setdefault(r.model, []).append(r)
    fields = [
        "model", "systems_evaluated", "overall_score_mean",
        "phase1_mean", "phase2_mean", "phase3b_mean", "phase3_final_mean",
        "audit_bugs_total", "total_gen_tokens_in", "total_gen_tokens_out",
        "total_wv_cost_usd",
    ]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for model, items in sorted(by_model.items()):
            def mean(key):
                vals = [getattr(r, key) for r in items if getattr(r, key) is not None]
                return round(sum(vals) / len(vals), 4) if vals else None
            def total(key):
                vals = [getattr(r, key) for r in items if getattr(r, key) is not None]
                return sum(vals) if vals else None
            w.writerow({
                "model": model,
                "systems_evaluated": len(items),
                "overall_score_mean": mean("overall_score"),
                "phase1_mean": mean("phase1_score"),
                "phase2_mean": mean("phase2_score"),
                "phase3b_mean": mean("phase3b_score"),
                "phase3_final_mean": mean("phase3_final_score"),
                "audit_bugs_total": sum(len(r.phase3_audit_bugs) for r in items),
                "total_gen_tokens_in": total("gen_tokens_in"),
                "total_gen_tokens_out": total("gen_tokens_out"),
                "total_wv_cost_usd": round(total("wv_agent_cost_usd") or 0, 2) or None,
            })


def write_json(rows, path: Path):
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "project": "SysMoBench",
        "schema_version": 1,
        "rows": [asdict(r) for r in rows],
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))


def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    rows = build_rows()
    write_detail_csv(rows, OUT_ROOT / "detail.csv")
    write_aggregate_csv(rows, OUT_ROOT / "aggregate.csv")
    write_json(rows, OUT_ROOT / "data.json")
    print(f"Wrote {len(rows)} detail rows to {OUT_ROOT}")
    models = sorted({r.model for r in rows})
    print(f"Models: {models}")
    print(f"  detail.csv     — per (model, system) row")
    print(f"  aggregate.csv  — per model averages")
    print(f"  data.json      — full structured data (website agent reads this)")


if __name__ == "__main__":
    main()
