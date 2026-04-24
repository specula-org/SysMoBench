#!/usr/bin/env python3
"""
Collect 5 most-recent specs per (canonical_model, system) into a clean archive
under specs_collection/. For internal re-scoring — not the leaderboard pipeline.

Layout:
    specs_collection/
      <canonical_model>/
        MANIFEST.json               model-level index (all 55 runs)
        <system>/
          run_1/                    newest
            <module>.tla
            <module>.cfg
            scores.json             phase scores + provenance
          run_2/
          ...
          run_5/                    oldest of the selected 5

Selection rule: for each (canonical_model, system), gather every timestamped
run directory across ALL config-level aliases that canonicalize to that model
(e.g. minimax_m27 + minimax_m27_ds → minimax_m27_ds), sort by directory name
(lexicographic == chronological for `YYYYMMDDHHMMSS`), keep the newest 5.

Re-runnable: overwrites the canonical-model subtree each time. Models not yet
on the leaderboard paper_summary.csv are skipped.
"""
from __future__ import annotations

import hashlib
import json
import re
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from build_leaderboard import MODEL_ALIASES  # noqa: E402  authoritative source

OUT_ROOT = PROJECT_ROOT / "specs_collection"
# Pipeline writes the same spec into separate timestamp dirs under each phase
# root as it progresses (comp_check → runtime_check → runtime_coverage). All
# three are valid sources of ts-dir + .tla/.cfg; we scan all of them so cells
# like gpt52/zookeeper can reach 5 runs (only 2 distinct generations but the
# pipeline duplicated them across phases — user rule: duplicates count).
PHASE_ROOTS = [
    PROJECT_ROOT / "output" / "compilation_check" / "tla",
    PROJECT_ROOT / "output" / "runtime_check" / "tla",
    PROJECT_ROOT / "output" / "runtime_coverage" / "tla",
]
EXP_ROOT = PROJECT_ROOT / "experiments"
PAPER_CSV = PROJECT_ROOT / "docs" / "leaderboard" / "paper_summary.csv"

SYSTEMS = ["spin", "mutex", "rwmutex", "dqueue", "ringbuffer", "locksvc",
           "curp", "raftkvs", "redisraft", "zookeeper", "etcd"]

RUNS_PER_CELL = 5


def canonical(alias: str) -> str:
    return MODEL_ALIASES.get(alias, alias)


def load_paper_models() -> list[str]:
    """Read the paper_summary.csv to know which models are 'complete'."""
    if not PAPER_CSV.exists():
        print(f"ERROR: {PAPER_CSV} not found — run scripts/build_leaderboard.py first")
        sys.exit(1)
    lines = PAPER_CSV.read_text().strip().splitlines()
    return [ln.split(",")[0] for ln in lines[1:]]  # skip header


def index_run_jsons() -> dict[str, tuple[Path, dict]]:
    """Map every run_*.json's spec_path → (json_path, parsed_data).

    Scans experiments/batch_*/<sys>/run_*.json once; later lookups are O(1).
    """
    idx: dict[str, tuple[Path, dict]] = {}
    for rj in EXP_ROOT.glob("batch_*/*/run_*.json"):
        try:
            d = json.loads(rj.read_text())
        except Exception:
            continue
        sp = d.get("spec_path")
        if sp:
            idx[sp] = (rj, d)
    return idx


def _spec_md5(ts_dir: Path) -> str | None:
    tlas = list(ts_dir.glob("*.tla"))
    if not tlas:
        return None
    return hashlib.md5(tlas[0].read_bytes()).hexdigest()


def _dir_quality(ts_dir: Path) -> tuple[int, int]:
    """Higher is better. Prefer dirs that have a .cfg (→ scorable) and more files."""
    has_cfg = 1 if any(ts_dir.glob("*.cfg")) else 0
    file_count = sum(1 for _ in ts_dir.iterdir() if _.is_file())
    return (has_cfg, file_count)


def find_run_dirs(model: str, system: str) -> tuple[list[Path], dict[str, Path], dict[str, str]]:
    """All timestamp dirs under output/.../<sys>/direct_call_<alias>/ where
    alias canonicalizes to `model`, sorted newest-first.

    Returns (hits, md5_to_cfgdir, ts_dir_md5).
    md5_to_cfgdir: md5 → sibling dir with .cfg, for borrowing cfg.
    ts_dir_md5: str(ts_dir) → md5, for sibling-score lookup by md5.
    Identical generations count as separate runs — we only borrow cfg and
    batch-score attribution from an md5-matching sibling.
    """
    candidates = []
    for phase_root in PHASE_ROOTS:
        sys_dir = phase_root / system
        if not sys_dir.exists():
            continue
        for alias_dir in sys_dir.iterdir():
            if not alias_dir.is_dir() or not alias_dir.name.startswith("direct_call_"):
                continue
            alias = alias_dir.name[len("direct_call_"):]
            if canonical(alias) != model:
                continue
            for ts_dir in alias_dir.iterdir():
                if not ts_dir.is_dir():
                    continue
                md5 = _spec_md5(ts_dir)
                if md5 is None:
                    continue
                candidates.append((ts_dir, md5))

    # md5 → best dir with that spec (prefer one with .cfg) for sibling lookup.
    md5_to_cfgdir: dict[str, Path] = {}
    for td, md5 in candidates:
        if not any(td.glob("*.cfg")):
            continue
        existing = md5_to_cfgdir.get(md5)
        if existing is None or td.name > existing.name:
            md5_to_cfgdir[md5] = td

    ts_dir_md5 = {str(td): md5 for td, md5 in candidates}
    hits = [td for td, _ in candidates]
    hits.sort(key=lambda p: p.name, reverse=True)  # newest first
    return hits, md5_to_cfgdir, ts_dir_md5


def extract_scores(run_json: dict) -> dict:
    """Pull the compact score summary for scores.json."""
    p1 = run_json.get("phase1_compilation") or {}
    p2 = run_json.get("phase2_runtime") or {}
    p2d = p2.get("details") or {}
    p3 = run_json.get("phase3_invariant") or {}
    usage = ((run_json.get("phase0_usage") or {}).get("usage") or {})
    return {
        "total_score": run_json.get("total_score"),
        "phase1_compilation": {
            "score": p1.get("score"),
            "passed": p1.get("passed"),
            "details": p1.get("details"),
        },
        "phase2_runtime": {
            "score": p2.get("score"),
            "passed": p2.get("passed"),
            "status": p2.get("status"),
            "coverage": p2d.get("coverage"),
            "runtime_check_passed": p2d.get("runtime_check_passed"),
        },
        "phase3_invariant": {
            "score": p3.get("score"),
            "passed": p3.get("passed"),
            "status": p3.get("status"),
        },
        "generation": {
            "success": run_json.get("generation_success"),
            "duration_s": run_json.get("generation_time"),
            "tokens_in": usage.get("prompt_tokens"),
            "tokens_out": usage.get("completion_tokens"),
            "error": run_json.get("error"),
        },
    }


def copy_run(ts_dir: Path, dest: Path, run_index: dict,
             md5_to_cfgdir: dict[str, Path],
             md5_to_scored_specpath: dict[str, str]) -> dict:
    """Mirror ts_dir's files into dest and layer scores.json on top.

    Source dirs contain generator artifacts (.tla/.cfg/metadata.json/
    result.json/generation_usage.json). We copy everything so the archive
    is self-contained. Then if a batch scored this run, its phase-level
    scores go into scores.json; if not (orphan generation), scores.json
    still carries provenance + the P1 outcome parsed from result.json so
    re-scoring can start from something.

    If the source dir has no .cfg, borrow one from a sibling ts_dir whose
    .tla has the same md5 — identical spec → identical cfg semantics. The
    borrow is recorded in provenance.borrowed_cfg_from.
    """
    dest.mkdir(parents=True, exist_ok=True)
    tla_files = list(ts_dir.glob("*.tla"))
    module = tla_files[0].stem if tla_files else None

    for src in ts_dir.iterdir():
        if src.is_file():
            shutil.copy2(src, dest / src.name)

    # Borrow .cfg from md5-sibling when missing.
    borrowed_cfg_from = None
    if module and not (dest / f"{module}.cfg").exists():
        md5 = _spec_md5(ts_dir)
        sibling = md5_to_cfgdir.get(md5) if md5 else None
        if sibling and sibling != ts_dir:
            for src_cfg in sibling.glob("*.cfg"):
                shutil.copy2(src_cfg, dest / src_cfg.name)
            borrowed_cfg_from = str(sibling.relative_to(PROJECT_ROOT))

    # Direct attribution: run_index is keyed by the run's own spec_path.
    spec_path_key = str(tla_files[0]) if tla_files else None
    run_json_info = run_index.get(spec_path_key)
    attribution = "direct" if run_json_info else None

    # Fallback: if this ts_dir has no batch score of its own, look for an
    # md5-matching sibling ts_dir that was batch-scored (same physical spec
    # written under a different phase root or timestamp — e.g. comp_check
    # vs runtime_check copy). Pull the score from there.
    if run_json_info is None:
        this_md5 = _spec_md5(ts_dir)
        sibling_spec_path = md5_to_scored_specpath.get(this_md5) if this_md5 else None
        if sibling_spec_path:
            run_json_info = run_index.get(sibling_spec_path)
            if run_json_info:
                attribution = "md5_sibling"

    provenance = {
        "source_timestamp_dir": str(ts_dir.relative_to(PROJECT_ROOT)),
        "source_alias": ts_dir.parent.name[len("direct_call_"):],
        "module": module,
        "has_batch_scoring": run_json_info is not None,
        "batch_score_attribution": attribution,
    }
    if borrowed_cfg_from:
        provenance["borrowed_cfg_from"] = borrowed_cfg_from
    if run_json_info:
        scores = extract_scores(run_json_info[1])
        provenance["source_run_json"] = str(run_json_info[0].relative_to(PROJECT_ROOT))
        provenance["source_batch"] = run_json_info[0].parent.parent.name
        provenance["run_id_in_batch"] = run_json_info[1].get("run_id")
    else:
        # Orphan generation — no batch scored it. Salvage P1 from result.json.
        scores = {}
        result_json = ts_dir / "result.json"
        if result_json.exists():
            try:
                rd = json.loads(result_json.read_text())
                scores = {"phase1_compilation_from_result_json": {
                    "compilation_successful": rd.get("compilation_successful"),
                    "generation_successful": rd.get("generation_successful"),
                    "syntax_errors": rd.get("syntax_errors") or [],
                    "semantic_errors": rd.get("semantic_errors") or [],
                }}
            except Exception:
                pass

    (dest / "scores.json").write_text(
        json.dumps({"provenance": provenance, "scores": scores}, indent=2)
    )

    return {
        "dir": dest.relative_to(OUT_ROOT).as_posix(),
        "module": module,
        "timestamp": ts_dir.name,
        "alias": provenance["source_alias"],
        "batch": provenance.get("source_batch"),
        "has_batch_scoring": provenance["has_batch_scoring"],
        "total_score": (scores or {}).get("total_score"),
    }


def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    models = load_paper_models()
    print(f"Collecting specs for {len(models)} paper models "
          f"({RUNS_PER_CELL} newest per system):\n")
    print("  " + ", ".join(models) + "\n")

    run_index = index_run_jsons()
    print(f"Indexed {len(run_index)} run_*.json files\n")

    overall = {}
    for model in models:
        model_root = OUT_ROOT / model
        if model_root.exists():
            shutil.rmtree(model_root)
        model_root.mkdir(parents=True)

        manifest = {"model": model, "systems": {}}
        total_runs = 0
        batch_scored = 0
        missing = []
        for system in SYSTEMS:
            ts_dirs, md5_to_cfgdir, ts_dir_md5 = find_run_dirs(model, system)
            # Build md5 → scored spec_path index (for sibling attribution):
            # for each md5, find any ts_dir whose spec_path is in run_index.
            md5_to_scored_specpath: dict[str, str] = {}
            for td in ts_dirs:
                md5 = ts_dir_md5.get(str(td))
                if md5 is None or md5 in md5_to_scored_specpath:
                    continue
                tlas = list(td.glob("*.tla"))
                if tlas and str(tlas[0]) in run_index:
                    md5_to_scored_specpath[md5] = str(tlas[0])
            picks = ts_dirs[:RUNS_PER_CELL]
            entries = []
            for i, td in enumerate(picks, start=1):
                dest = model_root / system / f"run_{i}"
                entries.append(copy_run(td, dest, run_index, md5_to_cfgdir,
                                        md5_to_scored_specpath))
            manifest["systems"][system] = {
                "runs_selected": len(entries),
                "runs_available": len(ts_dirs),
                "entries": entries,
            }
            total_runs += len(entries)
            batch_scored += sum(1 for e in entries if e["has_batch_scoring"])
            if len(entries) < RUNS_PER_CELL:
                missing.append(f"{system}={len(entries)}")

        (model_root / "MANIFEST.json").write_text(json.dumps(manifest, indent=2))
        overall[model] = {
            "total_runs": total_runs,
            "batch_scored": batch_scored,
            "per_system": {s: manifest["systems"][s]["runs_selected"] for s in SYSTEMS},
        }
        status = f"{total_runs}/{11*RUNS_PER_CELL} ({batch_scored} with batch scores)"
        short = f"  short: {', '.join(missing)}" if missing else ""
        print(f"  {model:28} {status}{short}")

    (OUT_ROOT / "SUMMARY.json").write_text(json.dumps({
        "generated_from": "scripts/collect_specs.py",
        "selection_rule": f"newest {RUNS_PER_CELL} runs (by timestamp) per "
                          f"(canonical_model, system). Identical-md5 duplicates "
                          f"are kept as separate runs; cfg-less dirs borrow .cfg "
                          f"from an md5-matching sibling for re-scoring.",
        "systems": SYSTEMS,
        "runs_per_cell_target": RUNS_PER_CELL,
        "models": overall,
    }, indent=2))

    print(f"\nArchive: {OUT_ROOT}")


if __name__ == "__main__":
    main()
