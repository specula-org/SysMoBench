#!/usr/bin/env python3
"""
Build SysMoBench leaderboard by scanning:
  - experiments/batch_*/    → Phase A (gen + P1 + P2 + P3b) results per model
  - tv-workspaces/*/        → Phase 3 TV + Audit results per model/system

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
TV_ROOT = PROJECT_ROOT / "tv-workspaces"
OUT_ROOT = PROJECT_ROOT / "docs" / "leaderboard"

SYSTEMS = ["spin", "etcd", "curp", "dqueue", "locksvc", "mutex",
           "raftkvs", "redisraft", "ringbuffer", "rwmutex", "zookeeper"]

# Model canonicalization: collapse config-level model names that are really the
# same underlying model accessed via different routes (proxy vs direct API) or
# with different timeout variants. Best-of-N picks the highest run across all
# aliased configs.
MODEL_ALIASES = {
    # claude-sonnet-4-6 via gptsapi proxy OR direct Anthropic API
    "claude_sonnet_proxy": "claude_sonnet",
    "claude_sonnet_direct": "claude_sonnet",
    # qwen3.6-plus variants — all merge into one leaderboard row. OpenRouter
    # entry added 2026-04-19 to backfill the 1 missing Phase A system
    # (redisraft) after DashScope kept timing out.
    "qwen36_plus_ds_1h": "qwen36_plus_ds",
    "qwen36_plus_openrouter": "qwen36_plus_ds",
    # gpt-5.2 via gptsapi proxy OR OpenAI direct API. Proxy failed with
    # Cloudflare 504 on raftkvs/redisraft/etcd (~45K input); those three
    # systems were refilled via the official OpenAI entry.
    "gpt52_proxy": "gpt52",
    "gpt52_openai": "gpt52",
    # MiniMax-M2.7: earlier direct-API attempts (config "minimax_m27") had
    # completion issues; DashScope route (config "minimax_m27_ds") finished
    # the full 11-system run. Specs from the original direct attempts are
    # still usable (same underlying model), so merge into one leaderboard row.
    "minimax_m27": "minimax_m27_ds",
}

# Abandoned / exploratory runs from earlier provider trials that never reached
# full-batch completion. Excluded from the primary leaderboard (the website
# shows only PRIMARY_MODELS by default; all rows stay in data.json under
# "all_rows" for completeness).
ABANDONED_MODELS = {
    "deepseek_r1_proxy",  # only 1 system ever ran
    "gpt54_proxy",        # replaced by gpt54_azure (free Azure tier)
    "gemini31_proxy",     # proxy 504 on mutex-size prompts; to be re-run direct
    "glm51",              # DashScope glm-5.1 routing broken
    "glm51_ds",           # same as above
    "grok4_proxy",        # proxy retry storm, cost control issue
}

# (model, system) pairs where TV was launched despite the cascade gate saying
# it shouldn't have been. These scores are degenerate (TV on a spec that
# failed Phase 2 is meaningless — TLC can't run it meaningfully) and would
# give the model unfair credit for partial traces. Treat as if TV never ran.
# Observed 2026-04-18 from a TV batch that bypassed P2 eligibility gating.
CASCADE_VIOLATIONS = {
    ("gpt52", "raftkvs"),     # P1 PARTIAL 0.47, shouldn't have reached P2
    ("gpt52", "ringbuffer"),  # P2 runtime_check FAILED (cov=0.30)
    ("gpt52", "zookeeper"),   # P2 runtime_check FAILED (cov=0.00)
}


def canonical(model: str) -> str:
    """Map config-level model name to canonical leaderboard name."""
    return MODEL_ALIASES.get(model, model)


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
    # Phase 3 TV+Audit
    phase3_tv_rate: float | None = None  # mean of per-action TV pass rates
    phase3_audit_run: bool = False
    phase3_audit_bugs: list = field(default_factory=list)
    phase3_final_score: float | None = None
    # Composite
    overall_score: float | None = None
    # Provenance
    best_run_spec_path: str | None = None
    best_run_json_path: str | None = None
    tv_workspace_path: str | None = None
    # Cost / usage
    gen_tokens_in: int | None = None
    gen_tokens_out: int | None = None
    tv_agent_cost_usd: float | None = None
    tv_agent_duration_s: float | None = None
    tv_agent_turns: int | None = None
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
        model_display = canonical(model)
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


def find_tv_workspace_for(model: str, system: str) -> Path | None:
    """Find the latest TV workspace whose spec symlink points to this canonical
    model's output.

    Workspace directory names use the TLA+ MODULE name, which can differ
    from the task/system name (e.g. task "etcd" → module "etcdraft").
    So we search for both "*_<system>/" and "*_<module>/" if task.yaml
    declares a specModule.
    """
    if not TV_ROOT.exists():
        return None
    aliases = {k for k, v in MODEL_ALIASES.items() if v == model}
    aliases.add(model)
    # Module name from task.yaml (may differ from system name)
    module_name = system
    task_yaml = PROJECT_ROOT / "tla_eval" / "tasks" / system / "task.yaml"
    if task_yaml.exists():
        try:
            import yaml
            td = yaml.safe_load(task_yaml.read_text()) or {}
            module_name = td.get("specModule") or td.get("spec_module") or system
        except Exception:
            pass
    candidates = []
    search_names = {system, module_name}  # set handles the spin==spin case
    for name in search_names:
        for ws in sorted(TV_ROOT.glob(f"*_{name}")):
            spec_link = ws / "spec" / f"{name}.tla"
            if not spec_link.exists():
                continue
            target_str = str(spec_link.resolve())
            # Direct attribution: spec points at output/.../direct_call_<model>/
            matched = any(f"direct_call_{a}/" in target_str for a in aliases)
            # Indirect attribution: spec was fed from a batch's best_specs/
            # copy (regular file, not symlink). The batch's experiment.log
            # "Model:" header is NOT authoritative — concurrent batches can
            # collide into a shared dir, so best_specs/<sys>.tla may have
            # come from a different model's run. Check the per-run JSONs
            # for that system in that batch: the highest-total-score run's
            # phase0_usage.model is the authoritative attribution.
            if not matched and "/best_specs/" in target_str:
                batch_dir = Path(target_str).parent.parent
                sys_dir = batch_dir / system
                best_model = None
                best_score = -1.0
                for rj in sys_dir.glob("run_*.json"):
                    try:
                        rd = json.loads(rj.read_text())
                    except Exception:
                        continue
                    ts = rd.get("total_score")
                    if ts is None:
                        continue
                    if ts > best_score:
                        best_score = ts
                        p0 = rd.get("phase0_usage") or {}
                        best_model = p0.get("model")
                if best_model:
                    # phase0_usage.model is the LiteLLM routing string
                    # (e.g. "openai/deepseek/deepseek-r1-0528"). Map it to
                    # the canonical leaderboard model by probing each alias.
                    for a in aliases:
                        marker = a.split("_")[0]  # crude fallback
                        if a.lower() in best_model.lower() or marker.lower() in best_model.lower():
                            matched = True
                            break
                    # Handle common provider→config mappings where the alias
                    # name doesn't appear verbatim in the LiteLLM string.
                    if not matched:
                        # e.g. minimax_m27_ds ↔ "openai/MiniMax/MiniMax-M2.7"
                        #      deepseek_r1_openrouter ↔ "openai/deepseek/deepseek-r1-0528"
                        #      qwen36_plus_ds ↔ "openai/qwen/qwen3.6-plus"
                        provider_hints = {
                            "minimax_m27_ds": ["minimax-m2.7", "minimax/m2.7"],
                            "minimax_m27":    ["minimax-m2.7", "minimax/m2.7"],
                            "deepseek_r1_openrouter": ["deepseek-r1"],
                            "qwen36_plus_ds": ["qwen3.6-plus"],
                            "qwen36_plus_openrouter": ["qwen3.6-plus"],
                            "qwen36_plus_ds_1h": ["qwen3.6-plus"],
                        }
                        for a in aliases:
                            for hint in provider_hints.get(a, []):
                                if hint in best_model.lower():
                                    matched = True
                                    break
                            if matched:
                                break
            if matched:
                candidates.append(ws)
    # dedupe while preserving order-by-mtime (glob is name-sorted; resort by
    # mtime so the newest workspace wins even across module/system name mix).
    candidates = sorted(set(candidates), key=lambda p: p.stat().st_mtime)
    return candidates[-1] if candidates else None


def parse_tv_final_report(ws: Path) -> dict:
    """Extract Phase 3 final score + per-action verdicts from the report.

    Phase 3 uses ZERO-TOLERANCE scoring at the action level:
      final(A) = 1.0  iff TV rate == 1.0 AND audit verdict is not 'wrong'
      final(A) = 0    otherwise (any TV failure OR audit-verified bug)
      phase3_final_score = mean(final(A_i))

    The per-action rate (x/y) is still parsed and reported for diagnostic
    transparency, but the score is binary per action — a spec that mismodels
    even one real transition is wrong, and the average should reflect that.
    Historical reports (some written with the old fractional rule) are
    rescored here from their per-action table rows so all models are
    comparable without regenerating workspaces.
    """
    rep = ws / "reports" / "final_report.md"
    out = {
        "phase3_final_score": None,
        "phase3_tv_rate": None,
        "audit_run": False,
        "audit_bugs": [],  # list of {action, reason}
    }
    if not rep.exists():
        return out
    text = rep.read_text()

    # audit run flag: look for "## Phase 3" + "Audited" column header
    if re.search(r"(audited|Audit Results|Step 9)", text, re.IGNORECASE):
        out["audit_run"] = True

    # Locate the "Final Score" table. Row shapes we accept:
    #   | Action | 10/11 = 0.909 | yes/no | correct/wrong(...) | 0.909 |
    #   | Action | 1.0          | yes    | correct             | 1.0   |
    #   | Action | 123 / 123 = 1.000 | yes | correct            | 1.0   |
    # We ignore the original `Final` column (old rule was fractional) and
    # recompute under zero-tolerance: 1.0 iff rate==1.0 AND not 'wrong'.
    per_action = []  # (action, rate, is_wrong)

    def _extract_rate(cell: str) -> float | None:
        # Rate cells must be unambiguously rates. Bare integers like "0" /
        # "1" / "22" are rejected — in most tables those are window counts,
        # not rates, and accepting them causes PASS/FAIL columns to be
        # mis-parsed as per-action pass rates.
        # fraction form: "10/11", "123 / 123"
        m = re.search(r"(\d+)\s*/\s*(\d+)", cell)
        if m:
            num, den = int(m.group(1)), int(m.group(2))
            if den > 0 and num <= den:
                return num / den
        # percentage form: "100%", "**95.5%**"
        m = re.search(r"(\d+(?:\.\d+)?)\s*%", cell)
        if m:
            try:
                v = float(m.group(1)) / 100.0
                if 0.0 <= v <= 1.0:
                    return v
            except ValueError:
                pass
        # float form: must have a decimal point — "0.909", "1.0", "0.00"
        m = re.search(r"\b(0|1)\.\d+\b", cell)
        if m:
            try:
                v = float(m.group(0))
                if 0.0 <= v <= 1.0:
                    return v
            except ValueError:
                pass
        return None

    for line in text.splitlines():
        if "|" not in line:
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        # Per-action rows need ≥3 columns — modern reports use 5-column
        # final tables (Action | TV | Audited | Verdict | Final) but older
        # ones use 3-col Pass-Rate tables (Action | Pass/Total | Rate).
        # 2-column rows are typically step-status tables (e.g. "| 0 Contract
        # | PASS ... 420/420 compliant |") — rejecting those plus the
        # "starts with a letter" filter below is enough to keep step-status
        # rows out while accepting both table shapes.
        if len(cells) < 3:
            continue
        action = re.sub(r"[*`]", "", cells[0]).strip()
        if not action or action.lower() in (
            "action", "---", "", "total", "spec action", "window", "symptom",
        ):
            continue
        if re.fullmatch(r"[-\s|:]+", action):
            continue
        # Action names are TLA+ identifiers — start with a letter. Rejects
        # status-table rows like "0 Contract", "9 Audit", "Step 1", etc.
        if not re.match(r"[A-Za-z_]", action):
            continue
        # Rate can live in any column (different report formats put it at
        # different positions: col 1 in the Final table, col 4 in some older
        # Pass-Rate tables). Scan from left and take the first valid rate we
        # find — but skip pure counts ("82", "3") which _extract_rate
        # naturally ignores because they have no "/" / "%" / decimal point.
        rate = None
        for c in cells[1:]:
            rate = _extract_rate(c)
            if rate is not None:
                break
        if rate is None:
            continue
        is_wrong = bool(re.search(r"\bwrong\b", line, re.IGNORECASE))
        per_action.append((action, rate, is_wrong))
        if is_wrong:
            out["audit_bugs"].append({
                "action": action,
                "line": line.strip("| ").strip()[:200],
            })

    if per_action:
        # Deduplicate (action names can appear in multiple tables within the
        # report — TV-only table + Final-Score table). Keep the entry whose
        # rate matches the action's most-recent mention, which is the Final
        # table since it's written last. A dict overwrite by action name
        # achieves this given Python's insertion-order semantics.
        dedup = {}
        for action, rate, is_wrong in per_action:
            dedup[action] = (rate, is_wrong)
        rates = [r for (r, _) in dedup.values()]
        out["phase3_tv_rate"] = sum(rates) / len(rates) if rates else None
        # Zero-tolerance final score.
        finals = [
            1.0 if (r == 1.0 and not wrong) else 0.0
            for (r, wrong) in dedup.values()
        ]
        out["phase3_final_score"] = sum(finals) / len(finals)
    return out


def parse_tv_cost(ws: Path) -> dict:
    u = ws / ".run.usage.json"
    if not u.exists():
        return {}
    try:
        d = json.loads(u.read_text())
    except Exception:
        return {}
    return {
        "tv_agent_cost_usd": d.get("total_cost_usd"),
        "tv_agent_duration_s": (d.get("duration_ms") or 0) / 1000 or None,
        "tv_agent_turns": d.get("num_turns"),
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
        # TV workspace lookup (skip if this pair is on the cascade-violation
        # blocklist — the TV ran but shouldn't have, score is invalid).
        if (model, system) in CASCADE_VIOLATIONS:
            r.notes.append("cascade_violation_tv_excluded")
        else:
            ws = find_tv_workspace_for(model, system)
            if ws:
                r.tv_workspace_path = str(ws.relative_to(PROJECT_ROOT))
                tv_info = parse_tv_final_report(ws)
                r.phase3_tv_rate = tv_info["phase3_tv_rate"]
                r.phase3_audit_run = tv_info["audit_run"]
                r.phase3_audit_bugs = tv_info["audit_bugs"]
                r.phase3_final_score = tv_info["phase3_final_score"]
                r.__dict__.update(parse_tv_cost(ws))
        # overall_score per (model, system) = fixed mean of 4 phases.
        # Missing / skipped / not_evaluated phases ALL count as 0.
        # No selective denominator — the formula is literally (P1+P2+P3+P4)/4.
        p1v = r.phase1_score or 0.0
        p2v = r.phase2_score or 0.0
        p3v = (r.phase3_final_score if r.phase3_final_score is not None
               else r.phase3_tv_rate) or 0.0
        p4v = r.phase3b_score or 0.0
        r.overall_score = (p1v + p2v + p3v + p4v) / 4.0
        rows.append(r)
    return rows


def write_detail_csv_all(rows, path: Path):
    # detail.csv contains PRIMARY models only (abandoned excluded to keep the
    # main leaderboard focused). Full data lives in data.json.
    write_detail_csv([r for r in rows if r.model not in ABANDONED_MODELS], path)


def write_detail_csv(rows, path: Path):
    fields = [
        "model", "system",
        "phase1_score", "phase2_score", "phase2_coverage",
        "phase2_runtime_check_passed", "phase3b_score",
        "phase3_tv_rate", "phase3_audit_run", "phase3_final_score",
        "overall_score",
        "gen_tokens_in", "gen_tokens_out",
        "tv_agent_cost_usd", "tv_agent_duration_s",
        "best_run_spec_path", "tv_workspace_path",
    ]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(asdict(r))


def write_aggregate_csv(rows, path: Path):
    by_model: dict[str, list[SystemResult]] = {}
    for r in rows:
        if r.model in ABANDONED_MODELS:
            continue
        by_model.setdefault(r.model, []).append(r)
    fields = [
        "model", "systems_evaluated", "overall_score_mean",
        "phase1_mean", "phase2_mean", "phase3b_mean", "phase3_final_mean",
        "audit_bugs_total", "total_gen_tokens_in", "total_gen_tokens_out",
        "total_tv_cost_usd",
    ]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for model, items in sorted(by_model.items()):
            def total(key):
                vals = [getattr(r, key) for r in items if getattr(r, key) is not None]
                return sum(vals) if vals else None
            # ALL phase means use a FIXED denominator = number of systems for
            # this model. Missing / not-run phases count as 0, not excluded.
            # User rule: "选择性分母毫无意义，分母就是 11".
            # Rationale: a model that fails P1 on 7 systems shouldn't score
            # P2_mean=0.8 just because its 4 P1-passing systems happened to
            # have good P2 — P2 never ran on the failed 7 and that's a model
            # deficiency, counted as 0.
            def fixed_mean(key):
                n = len(items)
                if n == 0:
                    return None
                vals = [(getattr(r, key) or 0.0) for r in items]
                return round(sum(vals) / n, 4)
            w.writerow({
                "model": model,
                "systems_evaluated": len(items),
                "overall_score_mean": fixed_mean("overall_score"),
                "phase1_mean": fixed_mean("phase1_score"),
                "phase2_mean": fixed_mean("phase2_score"),
                "phase3b_mean": fixed_mean("phase3b_score"),
                "phase3_final_mean": fixed_mean("phase3_final_score"),
                "audit_bugs_total": sum(len(r.phase3_audit_bugs) for r in items),
                "total_gen_tokens_in": total("gen_tokens_in"),
                "total_gen_tokens_out": total("gen_tokens_out"),
                "total_tv_cost_usd": round(total("tv_agent_cost_usd") or 0, 2) or None,
            })


def write_json(rows, path: Path):
    primary = [r for r in rows if r.model not in ABANDONED_MODELS]
    abandoned = [r for r in rows if r.model in ABANDONED_MODELS]
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "project": "SysMoBench",
        "schema_version": 2,
        "primary_rows": [asdict(r) for r in primary],
        "abandoned_rows": [asdict(r) for r in abandoned],
        "abandoned_models": sorted(ABANDONED_MODELS),
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))


BENCHMARK_SYSTEMS = [
    "spin", "mutex", "rwmutex", "dqueue", "ringbuffer", "locksvc",
    "curp", "raftkvs", "redisraft", "zookeeper", "etcd",
]


def write_paper_summary_csv(rows, path: Path):
    """Paper-ready summary table.

    Includes ONLY models that have FINISHED the full pipeline:

    1. Phase A ran on every one of the 11 benchmark systems
       (phase1_score is not None for every system).

    2. For every system whose P2 runtime_check passed (rc=True),
       Phase 3 TV has completed — i.e. a final_report.md exists and
       produced a parseable phase3_final_score, OR the agent wrote
       an explicit "Cannot evaluate" marker (captured upstream as
       status=pending with a reason).

    If a system is TV-eligible but the TV agent is still running (no
    report yet), the ENTIRE model is treated as in-progress and
    excluded from this file. Paper readers should not see half-done
    data masquerading as a score.

    Paper-standard phase names:
        Phase 1 = compilation
        Phase 2 = runtime
        Phase 3 = conformance (TV, zero-tolerance)
        Phase 4 = invariant correctness (agent-translated invariants)
    Means use fixed denominator = 11 (rule: "没跑的算零分").
    Cost excluded — inconsistent across workspaces.
    """
    by_model = {}
    for r in rows:
        if r.model in ABANDONED_MODELS:
            continue
        by_model.setdefault(r.model, []).append(r)

    def is_complete(items):
        # Phase A on all 11 systems?
        have = {r.system for r in items if r.phase1_score is not None}
        if not set(BENCHMARK_SYSTEMS).issubset(have):
            return False
        # TV-eligibility rule (user 2026-04-19):
        #   - P2 rc=False          → skip TV (runtime error)
        #   - P2 score=0 / cov=0   → skip TV (TLC explored 0 states; TV would
        #                            be degenerate — no traces to score)
        #   - P2 rc=True, score>0  → TV must have produced a final_report
        # An in-progress TV (workspace exists, report missing) keeps the model
        # marked in-progress.
        by_sys = {r.system: r for r in items}
        for s in BENCHMARK_SYSTEMS:
            r = by_sys.get(s)
            if r is None:
                return False
            if r.phase2_runtime_check_passed is not True:
                continue
            # P2 ran with no runtime error, but coverage may still be 0.
            if (r.phase2_score or 0) == 0 or (r.phase2_coverage or 0) == 0:
                continue  # legitimate "no TV" — TLC explored nothing
            # Otherwise TV is required.
            if r.phase3_final_score is None:
                ws = (PROJECT_ROOT / r.tv_workspace_path) if r.tv_workspace_path else None
                report = ws / "reports" / "final_report.md" if ws else None
                if not (report and report.exists()):
                    return False
        return True

    fields = [
        "model", "n_systems", "overall_score",
        "phase1_compilation", "phase2_runtime",
        "phase3_conformance", "phase4_invariant",
    ] + [f"sys_{s}" for s in BENCHMARK_SYSTEMS]
    out_rows = []
    for model, items in sorted(by_model.items()):
        if not is_complete(items):
            continue
        n = len(BENCHMARK_SYSTEMS)

        def mean(key):
            vals = [(getattr(r, key) or 0.0) for r in items]
            return round(sum(vals) / n, 3)

        # Per-system overall score (same formula as detail.csv's
        # overall_score column: mean over phases that ran at their score,
        # plus phases that cascaded-skipped as 0). Missing system (shouldn't
        # happen for completed models, but defensive) → 0.
        by_sys = {r.system: r for r in items}
        per_sys = {
            f"sys_{s}": round(
                (getattr(by_sys.get(s), "overall_score", None) or 0.0), 3
            )
            for s in BENCHMARK_SYSTEMS
        }

        out_rows.append({
            "model": model,
            "n_systems": n,
            "overall_score": mean("overall_score"),
            "phase1_compilation": mean("phase1_score"),
            "phase2_runtime": mean("phase2_score"),
            "phase3_conformance": mean("phase3_final_score"),
            "phase4_invariant": mean("phase3b_score"),
            **per_sys,
        })

    # Sort by overall_score descending.
    out_rows.sort(key=lambda x: -x["overall_score"])
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(out_rows)
    return out_rows


def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    rows = build_rows()
    write_detail_csv_all(rows, OUT_ROOT / "detail.csv")
    write_aggregate_csv(rows, OUT_ROOT / "aggregate.csv")
    summary = write_paper_summary_csv(rows, OUT_ROOT / "paper_summary.csv")
    write_json(rows, OUT_ROOT / "data.json")
    primary = {r.model for r in rows if r.model not in ABANDONED_MODELS}
    abandoned = {r.model for r in rows if r.model in ABANDONED_MODELS}
    print(f"Wrote {len(rows)} total detail rows ({len(primary)} primary models, "
          f"{len(abandoned)} abandoned)")
    print(f"Primary models:    {sorted(primary)}")
    print(f"Abandoned models:  {sorted(abandoned)}")
    print(f"  detail.csv         — primary rows, per (model, system)")
    print(f"  aggregate.csv      — primary rows, per model averages (all models)")
    print(f"  paper_summary.csv  — paper-ready summary, FULLY-COMPLETED models only")
    print(f"  data.json          — full data (primary_rows + abandoned_rows)")
    print(f"Completed models for paper: {[x['model'] for x in summary]}")


if __name__ == "__main__":
    main()
