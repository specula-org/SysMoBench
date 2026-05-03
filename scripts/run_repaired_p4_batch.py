#!/usr/bin/env python3
"""
Run P4 (invariant_verification) on repaired specs, sequentially by default.

Targets only cells with a real repair manifest (`applied: true`) because
trivial-copy cells keep the original spec and would not change P4.

Each cell writes:
  - batch_logs/p4/<timestamp>_<model>__<system>.log
  - batch_logs/p4_results/<model>__<system>.json

The script is resumable: completed markers are skipped unless --force.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import datetime as dt
import json
import re
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import scripts.batch_repair_and_tv as btv

REPAIRED_DIR = PROJECT_ROOT / "docs" / "leaderboard" / "specs_repaired"
BATCH_LOG_DIR = PROJECT_ROOT / "batch_logs"
P4_LOG_DIR = BATCH_LOG_DIR / "p4"
P4_RESULT_DIR = BATCH_LOG_DIR / "p4_results"
RUN_BENCHMARK = PROJECT_ROOT / "scripts" / "run_benchmark.py"


def log(msg: str) -> None:
    print(f"[{dt.datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def discover_repaired_cells() -> list[tuple[str, str, str]]:
    return [
        (model, system, module)
        for (model, system, module) in btv.discover_cells()
        if btv.repair_was_real(model, system)
    ]


def parse_cells(cells_arg: str | None, all_cells: list[tuple[str, str, str]]) -> list[tuple[str, str, str]]:
    if not cells_arg:
        return all_cells
    wanted = {x.strip() for x in cells_arg.split(",") if x.strip()}
    idx = {f"{m}/{s}": (m, s, mod) for (m, s, mod) in all_cells}
    picked = [idx[w] for w in wanted if w in idx]
    missing = [w for w in wanted if w not in idx]
    if missing:
        log(f"WARN: unknown cells: {missing}")
    return picked


def result_marker(model: str, system: str) -> Path:
    return P4_RESULT_DIR / f"{model}__{system}.json"


def p4_already_done(model: str, system: str) -> bool:
    marker = result_marker(model, system)
    if not marker.exists():
        return False
    try:
        data = json.loads(marker.read_text())
    except Exception:
        return False
    return bool(data.get("completed"))


def extract_output_dir(text: str) -> str | None:
    for pattern in (
        r"Created experiment directory:\s*(.+)",
        r"Results saved to:\s*(.+)",
    ):
        m = re.search(pattern, text)
        if m:
            return m.group(1).strip()
    return None


def extract_overall_success(text: str) -> bool | None:
    m = re.search(r"Overall success:\s*(True|False)", text)
    if m:
        return m.group(1) == "True"
    if "Semantics Evaluation Results: ✓ PASS" in text:
        return True
    if "Semantics Evaluation Results: ✗ FAIL" in text:
        return False
    return None


def run_one(model: str, system: str, module: str,
            model_name: str, force: bool, dry: bool) -> dict:
    if not force and p4_already_done(model, system):
        return {"cell": f"{model}/{system}", "skipped": "p4 already done"}

    spec_dir = REPAIRED_DIR / model / system
    spec_file = spec_dir / f"{module}.tla"
    cfg_file = spec_dir / f"{module}.cfg"
    if not spec_file.exists() or not cfg_file.exists():
        return {
            "cell": f"{model}/{system}",
            "error": f"missing spec or cfg under {spec_dir}",
        }

    if dry:
        return {
            "cell": f"{model}/{system}",
            "would": f"p4:{model_name}",
        }

    P4_LOG_DIR.mkdir(parents=True, exist_ok=True)
    P4_RESULT_DIR.mkdir(parents=True, exist_ok=True)

    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = P4_LOG_DIR / f"{ts}_{model}__{system}.log"
    cmd = [
        "python3", str(RUN_BENCHMARK),
        "--evaluation-type", "semantics",
        "--task", system,
        "--method", "direct_call",
        "--model", model_name,
        "--metric", "invariant_verification",
        "--spec-file", str(spec_file),
        "--config-file", str(cfg_file),
        "--inv-translator-type", "agent",
    ]

    try:
        r = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=PROJECT_ROOT,
        )
        output = (r.stdout or "") + (r.stderr or "")
        log_path.write_text(output)
        result = {
            "cell": f"{model}/{system}",
            "model": model,
            "system": system,
            "spec_module": module,
            "agent_model": model_name,
            "completed": True,
            "rc": r.returncode,
            "overall_success": extract_overall_success(output),
            "output_dir": extract_output_dir(output),
            "log_path": str(log_path),
            "finished_at": dt.datetime.now().isoformat(),
        }
        result_marker(model, system).write_text(json.dumps(result, indent=2) + "\n")
        return result
    except Exception as e:
        result = {
            "cell": f"{model}/{system}",
            "model": model,
            "system": system,
            "spec_module": module,
            "agent_model": model_name,
            "completed": False,
            "error": str(e),
            "log_path": str(log_path),
            "finished_at": dt.datetime.now().isoformat(),
        }
        result_marker(model, system).write_text(json.dumps(result, indent=2) + "\n")
        return result


def run_batch(cells: list[tuple[str, str, str]], concurrency: int,
              model_name: str, force: bool, dry: bool) -> list[dict]:
    results: list[dict] = []
    if concurrency <= 1:
        for (m, s, mod) in cells:
            r = run_one(m, s, mod, model_name, force, dry)
            results.append(r)
            log(f"[p4 done] {r.get('cell')}: " + ", ".join(f"{k}={v}" for k, v in r.items() if k != "cell"))
    else:
        with cf.ThreadPoolExecutor(max_workers=concurrency) as ex:
            futs = {
                ex.submit(run_one, m, s, mod, model_name, force, dry): (m, s)
                for (m, s, mod) in cells
            }
            for fut in cf.as_completed(futs):
                r = fut.result()
                results.append(r)
                log(f"[p4 done] {r.get('cell')}: " + ", ".join(f"{k}={v}" for k, v in r.items() if k != "cell"))
    return results


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cells", default=None,
                    help="Comma-separated model/system pairs to process")
    ap.add_argument("--concurrency", type=int, default=1,
                    help="Cells in flight simultaneously (default: 1)")
    ap.add_argument("--model", default="gpt-5",
                    help="Codex/OpenAI model id passed to run_benchmark.py (default: gpt-5)")
    ap.add_argument("--force", action="store_true",
                    help="Re-run even if a result marker already exists")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    all_cells = discover_repaired_cells()
    cells = parse_cells(args.cells, all_cells)
    log(f"real-repair cells={len(all_cells)}; operating on {len(cells)}")
    pending = [c for c in cells if args.force or not p4_already_done(c[0], c[1])]
    log(f"[p4 plan] pending={len(pending)}")

    if args.dry_run:
        log("--dry-run: exiting without launching")
        return 0

    results = run_batch(pending, args.concurrency, args.model, args.force, dry=False)
    summary_path = BATCH_LOG_DIR / f"p4_summary_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    summary_path.write_text(json.dumps(results, indent=2) + "\n")
    log(f"[p4] summary -> {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
