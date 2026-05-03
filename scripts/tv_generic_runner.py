#!/usr/bin/env python3
"""
Generic TV validation runner. Discovers (TV_<action>.tla, TV_<action>.cfg,
windows_<action>.ndjson) tuples in a tv/ directory, calls run_tv_batch
per action, prints a standardized summary the mutation tester can parse.

Usage: python3 scripts/tv_generic_runner.py <workspace-tv-dir>
"""

import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from tla_eval.tv_tools.runner import run_tv_batch, summarize


def main():
    tv_dir = Path(sys.argv[1])
    tla_jar = str(PROJECT_ROOT / "lib" / "tla2tools.jar")
    cm_jar = str(PROJECT_ROOT / "lib" / "CommunityModules-deps.jar")

    actions = []
    for tla_file in sorted(tv_dir.glob("TV_*.tla")):
        name = tla_file.stem.replace("TV_", "")
        if "TTrace" in name:
            continue
        cfg_file = tv_dir / f"TV_{name}.cfg"
        windows_file = tv_dir / f"windows_{name}.ndjson"
        if not (cfg_file.exists() and windows_file.exists()):
            continue
        n = sum(1 for _ in windows_file.open())
        actions.append((name, tla_file.name, cfg_file.name, n))

    results_by_action = {}
    for name, tla, cfg, n in actions:
        print(f"\n=== {name} ({n} windows) ===")
        results = run_tv_batch(
            num_windows=n,
            tv_tla=tla,
            tv_cfg=cfg,
            work_dir=str(tv_dir),
            workers=8,
            timeout=60,
            tla_jar=tla_jar,
            community_jar=cm_jar,
        )
        stats = summarize(results)
        results_by_action[name] = stats
        print(f"  pass={stats['passed']} fail={stats['failed']} "
              f"error={stats['errored']} timeout={stats['timeout']} "
              f"rate={stats['pass_rate']:.1%}")

    print("\n=== Summary ===")
    for name, stats in results_by_action.items():
        print(f"  {name:30s}: {stats['pass_rate']:.1%}  ({stats['passed']}/{stats['total']})")


if __name__ == "__main__":
    main()
