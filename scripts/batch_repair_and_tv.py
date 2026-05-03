#!/usr/bin/env python3
"""
Batch orchestrator: run spec-repair then tv-eval on every leaderboard cell.

Pipeline:
  Phase REPAIR:
    For every cell in docs/leaderboard/specs/<model>/<system>/:
      - If the cell already passes P1+P2 (per scores.json), verbatim-copy
        the .tla/.cfg into docs/leaderboard/specs_repaired/<model>/<system>/
        and write a trivial manifest. No agent invoked.
      - Else, spawn a claude-code agent to run the /spec-repair skill,
        writing the repaired spec + manifest + logs into specs_repaired/.

  Phase TV:
    For every cell whose repair ended with p1_passed && p2_passed,
    spawn a claude-code agent to run the /tv-eval skill on the repaired
    spec via scripts/launch_tv_eval.sh.

Quota discipline (matches Specula's wait_for_quota pattern):
  Before every agent launch, check 5h and 7d utilization via the Anthropic
  OAuth usage endpoint. If 5h >= QUOTA_5H (default 90) or 7d >= QUOTA_7D
  (default 95), sleep until resets_at + 120s, repeat. Abort after
  QUOTA_MAX_WAITS (default 6) consecutive blocks.

Usage:
  python3 scripts/batch_repair_and_tv.py [options]

Options:
  --phase {repair,tv,all}   Run only one phase or both (default: all)
  --concurrency N           Cells in flight simultaneously (default: 3)
  --cells <csv>             Comma-separated model/system pairs to process
                            (default: every cell under docs/leaderboard/specs/)
  --dry-run                 Print what would run; do not launch any agent
  --claude-alias NAME       Claude CLI profile (default: claude)
  --force-repair            Re-run repair even if out_dir already has a
                            successful manifest
  --force-tv                Re-run TV even if a workspace with a report exists
  --repair-model MODEL      Model for repair agent (default: sonnet)
  --tv-model MODEL          Model for TV agent (default: agent-specific;
                            claude-code=sonnet, codex=CLI default)

Resumable: skips cells whose repair_manifest.json indicates success,
           or whose tv-workspaces/ has a final_report.md pointing at
           the repaired spec.

Env:
  QUOTA_5H (default 90)
  QUOTA_7D (default 95)
  QUOTA_MAX_WAITS (default 6)
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import datetime as dt
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SPECS_DIR = PROJECT_ROOT / "docs" / "leaderboard" / "specs"
REPAIRED_DIR = PROJECT_ROOT / "docs" / "leaderboard" / "specs_repaired"
TV_ROOT = PROJECT_ROOT / "tv-workspaces"
REPAIR_SKILL = PROJECT_ROOT / "tla_eval" / "skills" / "spec-repair"
ADAPTER = PROJECT_ROOT / "scripts" / "launch" / "adapters" / "claude-code.sh"
LAUNCH_TV = PROJECT_ROOT / "scripts" / "launch_tv_eval.sh"
TLA_TOOLS = PROJECT_ROOT / "lib" / "tla2tools.jar"
BATCH_LOG_DIR = PROJECT_ROOT / "batch_logs"

QUOTA_5H = float(os.environ.get("QUOTA_5H", "90"))
QUOTA_7D = float(os.environ.get("QUOTA_7D", "100"))
QUOTA_MAX_WAITS = int(os.environ.get("QUOTA_MAX_WAITS", "6"))


# ──────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────

def log(msg: str) -> None:
    print(f"[{dt.datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


# ──────────────────────────────────────────────────────────
# Quota check (matches Specula's usage.sh + wait_for_quota pattern)
# ──────────────────────────────────────────────────────────

def _read_access_token(claude_alias: str) -> str | None:
    creds_path = Path.home() / (f".{claude_alias}" if claude_alias != "claude" else ".claude") / ".credentials.json"
    if not creds_path.exists():
        log(f"WARN: credentials not at {creds_path}")
        return None
    try:
        d = json.loads(creds_path.read_text())
    except Exception as e:
        log(f"WARN: failed to parse credentials: {e}")
        return None
    inner = d.get("claudeAiOauth", d)
    return inner.get("accessToken") or None


def fetch_usage(claude_alias: str) -> dict | None:
    token = _read_access_token(claude_alias)
    if not token:
        return None
    req = urllib.request.Request(
        "https://api.anthropic.com/api/oauth/usage",
        headers={
            "Authorization": f"Bearer {token}",
            "anthropic-beta": "oauth-2025-04-20",
            "Content-Type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            return json.loads(r.read().decode())
    except Exception as e:
        log(f"WARN: usage fetch failed: {e}")
        return None


def wait_for_quota(claude_alias: str) -> bool:
    """Return True once quota is under threshold; False on exhausted wait budget."""
    waits = 0
    while True:
        usage = fetch_usage(claude_alias)
        if usage is None:
            log("WARN: usage unavailable, proceeding without check")
            return True

        five = (usage.get("five_hour") or {})
        seven = (usage.get("seven_day") or {})
        u5 = float(five.get("utilization") or 0)
        u7 = float(seven.get("utilization") or 0)

        over = None
        reset_at = None
        if u5 >= QUOTA_5H:
            over = f"5h={u5:.0f}% (limit {QUOTA_5H:.0f}%)"
            reset_at = five.get("resets_at")
        elif u7 >= QUOTA_7D:
            over = f"7d={u7:.0f}% (limit {QUOTA_7D:.0f}%)"
            reset_at = seven.get("resets_at")

        if over is None:
            if waits == 0:
                log(f"quota ok: 5h={u5:.0f}%  7d={u7:.0f}%")
            return True

        waits += 1
        if waits > QUOTA_MAX_WAITS:
            log(f"ERROR: quota still over after {QUOTA_MAX_WAITS} waits, aborting")
            return False

        if reset_at:
            try:
                reset_dt = dt.datetime.fromisoformat(reset_at.replace("Z", "+00:00"))
                now = dt.datetime.now(dt.timezone.utc)
                sleep_secs = max(60, int((reset_dt - now).total_seconds()) + 120)
            except Exception:
                sleep_secs = 600
        else:
            sleep_secs = 600

        log(f"quota: {over} — sleeping {sleep_secs}s (wait {waits}/{QUOTA_MAX_WAITS})")
        time.sleep(sleep_secs)


# ──────────────────────────────────────────────────────────
# Cell discovery & status
# ──────────────────────────────────────────────────────────

def discover_cells() -> list[tuple[str, str, str]]:
    """Return [(model, system, module), ...] from all existing cells."""
    out: list[tuple[str, str, str]] = []
    for model_dir in sorted(SPECS_DIR.iterdir()):
        if not model_dir.is_dir():
            continue
        model = model_dir.name
        for sys_dir in sorted(model_dir.iterdir()):
            if not sys_dir.is_dir():
                continue
            tlas = list(sys_dir.glob("*.tla"))
            if not tlas:
                continue
            module = tlas[0].stem
            out.append((model, sys_dir.name, module))
    return out


def cell_passes_baseline(model: str, system: str) -> bool:
    sj = SPECS_DIR / model / system / "scores.json"
    if not sj.exists():
        return False
    try:
        d = json.loads(sj.read_text())
    except Exception:
        return False
    p1 = d.get("phase1_compilation", {}).get("status") == "ran_passed"
    p2 = bool(d.get("phase2_runtime", {}).get("runtime_check_passed"))
    return p1 and p2


def repair_already_done(model: str, system: str) -> bool:
    m = REPAIRED_DIR / model / system / "repair_manifest.json"
    if not m.exists():
        return False
    try:
        d = json.loads(m.read_text())
    except Exception:
        return False
    return bool(d.get("p1_passed")) and bool(d.get("p2_passed"))


def tv_already_done(model: str, system: str) -> bool:
    # launch_tv_eval.sh names workspaces <TIMESTAMP>_<system>, losing the
    # model. Identify the cell via the `spec/` symlink target, which points
    # back at docs/leaderboard/specs_repaired/<model>/<system>/.
    target = (REPAIRED_DIR / model / system).resolve()
    if not TV_ROOT.exists():
        return False
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
            return True
    return False


def repair_was_real(model: str, system: str) -> bool:
    """True iff repair applied non-trivial edits (not just a verbatim copy).

    Used to select cells for TV: 'just-repaired' means actually repaired,
    not cells that baseline-passed and were only mirrored.
    """
    m = REPAIRED_DIR / model / system / "repair_manifest.json"
    if not m.exists():
        return False
    try:
        d = json.loads(m.read_text())
    except Exception:
        return False
    return bool(d.get("applied")) and bool(d.get("p1_passed")) and bool(d.get("p2_passed"))


# ──────────────────────────────────────────────────────────
# Trivial copy (cells already passing P1+P2)
# ──────────────────────────────────────────────────────────

def trivial_copy(model: str, system: str, module: str) -> None:
    src = SPECS_DIR / model / system
    dst = REPAIRED_DIR / model / system
    (dst / "repair_logs").mkdir(parents=True, exist_ok=True)
    for ext in ("tla", "cfg"):
        s = src / f"{module}.{ext}"
        if s.exists():
            shutil.copy2(s, dst / f"{module}.{ext}")
    (dst / "repair_manifest.json").write_text(json.dumps({
        "applied": False,
        "status": "none",
        "p1_passed": True,
        "p2_passed": True,
        "edit_count": 0,
        "ops": [],
        "note": "cell already passed P1+P2 at source; copied verbatim",
    }, indent=2))
    (dst / "repair_report.md").write_text(
        f"# Repair Report — {model}/{system}\n\nCell already passed P1+P2 at source. "
        "The spec was copied verbatim into the repaired mirror; no edits applied.\n"
    )


# ──────────────────────────────────────────────────────────
# Prompt builders
# ──────────────────────────────────────────────────────────

def repair_prompt(model: str, system: str, module: str) -> str:
    in_dir = SPECS_DIR / model / system
    out_dir = REPAIRED_DIR / model / system
    task_yaml = PROJECT_ROOT / "tla_eval" / "tasks" / system / "task.yaml"
    task_prompts = PROJECT_ROOT / "tla_eval" / "tasks" / system / "prompts"

    return f"""You are running the SysMoBench `/spec-repair` skill on ONE cell. Batch-repair mode, fully autonomous. No confirmation.

STEP 1 — Read the skill in full and follow it exactly:
  {REPAIR_SKILL / 'guide.md'}
  {REPAIR_SKILL / 'SKILL.md'}

The allow-list (A1-A17) is intentionally liberal. The forbid-list (F1-F7) covers only edits that compromise P3 (TV) fairness: adding/removing actions, weakening guards, changing an action's write-set, adding invariants, changing VARIABLES, weakening model-written invariants. Everything else is fair game. Default to fixing; halt with `unrepairable` ONLY if a fix truly requires F1-F7.

STEP 2 — Cell context:
- Cell: {model} / {system}
- in_dir:  {in_dir}/
- out_dir: {out_dir}/
- Module filename stem: `{module}` (so `{module}.tla`, `{module}.cfg`)
- Task contract: {task_yaml}
- Task prompt dir: {task_prompts}/
- tla2tools.jar:  {TLA_TOOLS}

Commands:
- P1: `java -cp {TLA_TOOLS} tla2sany.SANY -error-codes {module}.tla`
- P2: `cd <dir> && timeout 60 java -cp {TLA_TOOLS} tlc2.TLC -config {module}.cfg -deadlock {module}.tla`
  Pass = TLC runs with no violation/deadlock/error; a clean 60s timeout with no errors counts as pass.

STEP 3 — Produce outputs at {out_dir}/ per the skill:
  {module}.tla, {module}.cfg, repair_report.md, repair_manifest.json,
  repair_logs/{{sany,tlc}}_{{before,after}}.log

STEP 4 — End with a short status (<150 words): final repair_status, edit_count,
rules applied (e.g. "A11 x2, A13 x1"), P1/P2 final pass/fail, one-sentence root
cause, one-sentence repair summary. Do NOT paste the full spec or full logs.
"""


# ──────────────────────────────────────────────────────────
# Agent invocation via the claude-code adapter
# ──────────────────────────────────────────────────────────

def run_adapter(prompt_text: str, log_path: Path, claude_alias: str,
                max_budget: str | None = None) -> tuple[int, str]:
    BATCH_LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    prompt_file = log_path.with_suffix(".prompt.md")
    prompt_file.write_text(prompt_text)

    cmd = [str(ADAPTER),
           f"--prompt-file={prompt_file}",
           f"--log={log_path}"]
    if max_budget:
        cmd.append(f"--max-budget={max_budget}")

    # SysMoBench's adapter doesn't support --claude-alias yet; pass via env.
    env = os.environ.copy()
    if claude_alias and claude_alias != "claude":
        env["CLAUDE_CONFIG_DIR"] = str(Path.home() / f".{claude_alias}")

    try:
        r = subprocess.run(cmd, capture_output=True, text=True, cwd=PROJECT_ROOT, env=env)
        output_tail = (r.stdout or "") + (r.stderr or "")
        return r.returncode, output_tail[-4000:]
    except Exception as e:
        return 1, f"adapter launch failed: {e}"


def run_tv_launcher(model: str, system: str, log_path: Path,
                    agent: str, agent_model: str | None) -> tuple[int, str]:
    BATCH_LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    spec_path = REPAIRED_DIR / model / system
    cmd = [
        "bash", str(LAUNCH_TV),
        f"--spec={spec_path}",
        f"--task={system}",
        f"--workspace-root={TV_ROOT}",
        f"--agent={agent}",
    ]
    if agent_model:
        cmd.append(f"--model={agent_model}")

    try:
        r = subprocess.run(cmd, capture_output=True, text=True, cwd=PROJECT_ROOT)
        output = (r.stdout or "") + (r.stderr or "")
        log_path.write_text(output)
        return r.returncode, output[-4000:]
    except Exception as e:
        msg = f"tv launch failed: {e}"
        log_path.write_text(msg)
        return 1, msg


# ──────────────────────────────────────────────────────────
# Phase drivers
# ──────────────────────────────────────────────────────────

def do_repair(model: str, system: str, module: str,
              claude_alias: str, force: bool, dry: bool,
              tv_agent: str = "claude-code",
              tv_model: str | None = None) -> dict:
    if not force and repair_already_done(model, system):
        return {"cell": f"{model}/{system}", "skipped": "already repaired"}

    if cell_passes_baseline(model, system):
        if dry:
            return {"cell": f"{model}/{system}", "would": "trivial-copy"}
        trivial_copy(model, system, module)
        return {"cell": f"{model}/{system}", "result": "trivial-copy"}

    if dry:
        return {"cell": f"{model}/{system}", "would": "repair-agent"}

    if not wait_for_quota(claude_alias):
        return {"cell": f"{model}/{system}", "error": "quota abort"}

    log(f"[repair] {model}/{system} — launching")
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = BATCH_LOG_DIR / "repair" / f"{ts}_{model}__{system}.log"
    rc, tail = run_adapter(repair_prompt(model, system, module), log_path, claude_alias)
    manifest = REPAIRED_DIR / model / system / "repair_manifest.json"
    if manifest.exists():
        try:
            m = json.loads(manifest.read_text())
            ok = bool(m.get("p1_passed")) and bool(m.get("p2_passed"))
            return {"cell": f"{model}/{system}",
                    "result": m.get("status"),
                    "p1": m.get("p1_passed"), "p2": m.get("p2_passed"),
                    "edits": m.get("edit_count"), "rc": rc, "ok": ok}
        except Exception as e:
            return {"cell": f"{model}/{system}", "error": f"manifest parse: {e}", "rc": rc}
    return {"cell": f"{model}/{system}", "error": "no manifest written", "rc": rc, "tail": tail}


def do_tv(model: str, system: str, module: str,
          claude_alias: str, force: bool, dry: bool,
          tv_agent: str, tv_model: str | None) -> dict:
    if not force and tv_already_done(model, system):
        return {"cell": f"{model}/{system}", "skipped": "tv already done"}

    # Run TV only on cells we actually repaired. Trivial-copied cells
    # (applied=False) already had TV reports from the original pipeline.
    if not repair_was_real(model, system):
        return {"cell": f"{model}/{system}", "skipped": "no real repair; trivial copy or still broken"}

    if dry:
        return {"cell": f"{model}/{system}", "would": f"tv-agent:{tv_agent}/{tv_model or 'default'}"}

    if tv_agent == "claude-code" and not wait_for_quota(claude_alias):
        return {"cell": f"{model}/{system}", "error": "quota abort"}

    log(f"[tv] {model}/{system} — launching via {tv_agent} ({tv_model or 'default'})")
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = BATCH_LOG_DIR / "tv" / f"{ts}_{model}__{system}.log"
    rc, tail = run_tv_launcher(model, system, log_path, tv_agent, tv_model)
    return {"cell": f"{model}/{system}", "rc": rc}


# ──────────────────────────────────────────────────────────
# Orchestration
# ──────────────────────────────────────────────────────────

def run_phase(phase: str, cells: list[tuple[str, str, str]], concurrency: int,
              claude_alias: str, force: bool, dry: bool,
              tv_agent: str, tv_model: str | None) -> list[dict]:
    fn = do_repair if phase == "repair" else do_tv

    results: list[dict] = []
    if concurrency <= 1:
        for (m, s, mod) in cells:
            results.append(fn(m, s, mod, claude_alias, force, dry, tv_agent, tv_model))
    else:
        # Quota check happens inside fn for each cell; concurrency is I/O-bound
        with cf.ThreadPoolExecutor(max_workers=concurrency) as ex:
            futs = {ex.submit(fn, m, s, mod, claude_alias, force, dry, tv_agent, tv_model): (m, s)
                    for (m, s, mod) in cells}
            for fut in cf.as_completed(futs):
                r = fut.result()
                results.append(r)
                log(f"[{phase} done] {r.get('cell')}: "
                    + ", ".join(f"{k}={v}" for k, v in r.items() if k != 'cell'))
    return results


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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--phase", choices=["repair", "tv", "all"], default="all")
    ap.add_argument("--concurrency", type=int, default=3)
    ap.add_argument("--cells", default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--claude-alias", default="claude")
    ap.add_argument("--force-repair", action="store_true")
    ap.add_argument("--force-tv", action="store_true")
    ap.add_argument("--tv-agent", default="claude-code")
    ap.add_argument("--tv-model", default=None,
                    help="TV agent model override. Defaults to sonnet for claude-code and CLI default for codex.")
    args = ap.parse_args()

    if not ADAPTER.exists():
        log(f"ERROR: adapter not found: {ADAPTER}")
        return 1
    if args.phase in ("tv", "all") and not LAUNCH_TV.exists():
        log(f"ERROR: TV launcher not found: {LAUNCH_TV}")
        return 1

    all_cells = discover_cells()
    cells = parse_cells(args.cells, all_cells)
    log(f"discovered {len(all_cells)} cells; operating on {len(cells)}")

    # Phase summary
    if args.phase in ("repair", "all"):
        need_agent = [(m, s, mod) for (m, s, mod) in cells
                      if not cell_passes_baseline(m, s)
                      and (args.force_repair or not repair_already_done(m, s))]
        trivial = [(m, s, mod) for (m, s, mod) in cells
                   if cell_passes_baseline(m, s)
                   and (args.force_repair or not repair_already_done(m, s))]
        already = len(cells) - len(need_agent) - len(trivial)
        log(f"[repair plan] agent={len(need_agent)} trivial-copy={len(trivial)} already-done={already}")
    if args.phase in ("tv", "all"):
        pending_tv = [(m, s, mod) for (m, s, mod) in cells
                      if repair_was_real(m, s)
                      and (args.force_tv or not tv_already_done(m, s))]
        log(f"[tv plan] pending={len(pending_tv)}  (real-repair cells only)")

    if args.dry_run:
        log("--dry-run: exiting without launching agents")
        return 0

    # ── Execute ──
    if args.phase in ("repair", "all"):
        log("═" * 60)
        log(f"PHASE REPAIR — concurrency={args.concurrency}")
        log("═" * 60)
        rr = run_phase("repair", cells, args.concurrency,
                       args.claude_alias, args.force_repair, dry=False,
                       tv_agent=args.tv_agent, tv_model=args.tv_model)
        summary_path = BATCH_LOG_DIR / f"repair_summary_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(rr, indent=2))
        log(f"[repair] summary -> {summary_path}")

    if args.phase in ("tv", "all"):
        log("═" * 60)
        log(f"PHASE TV — concurrency={args.concurrency}")
        log("═" * 60)
        # Re-discover which cells are TV-ready (repaired successfully, real edits)
        tv_cells = [(m, s, mod) for (m, s, mod) in cells
                    if repair_was_real(m, s)
                    and (args.force_tv or not tv_already_done(m, s))]
        rr = run_phase("tv", tv_cells, args.concurrency,
                       args.claude_alias, args.force_tv, dry=False,
                       tv_agent=args.tv_agent, tv_model=args.tv_model)
        summary_path = BATCH_LOG_DIR / f"tv_summary_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        summary_path.write_text(json.dumps(rr, indent=2))
        log(f"[tv] summary -> {summary_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
