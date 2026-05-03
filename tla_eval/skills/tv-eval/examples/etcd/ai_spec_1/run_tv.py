#!/usr/bin/env python3
"""
Run TV for a given action against etcd AI spec.
Usage: run_tv.py <action_name>  (default: ClientProposal)
"""

import os
import sys
import subprocess
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

SPEC_DIR = Path(__file__).parent
ACTION = sys.argv[1] if len(sys.argv) > 1 else "ClientProposal"
WINDOWS = SPEC_DIR / f"windows_{ACTION}.ndjson"
TV_TLA = f"TV_{ACTION}.tla"
TV_CFG = f"TV_{ACTION}.cfg"
PROJECT_ROOT = SPEC_DIR.resolve().parents[5]
TLA_JAR = str(PROJECT_ROOT / "lib" / "tla2tools.jar")
COMM_JAR = str(PROJECT_ROOT / "lib" / "CommunityModules-deps.jar")


def check_one(idx):
    env = {**os.environ, "WINDOW_INDEX": str(idx)}
    r = subprocess.run(
        ["java", "-cp", f"{TLA_JAR}:{COMM_JAR}", "tlc2.TLC",
         "-dfid", "2", "-deadlock", "-nowarning",
         "-metadir", f"/tmp/tv_etcd_{idx}",
         "-config", TV_CFG, TV_TLA],
        capture_output=True, text=True, timeout=60,
        cwd=SPEC_DIR, env=env,
    )
    return idx, r.returncode == 12, r.returncode


def main():
    if not WINDOWS.exists():
        print(f"ERROR: {WINDOWS} not found. Run make_windows.py first.")
        sys.exit(1)
    with open(WINDOWS) as f:
        n = sum(1 for _ in f)
    print(f"Running {n} windows against AI spec's {ACTION}...")

    start = time.time()
    results = {}
    with ProcessPoolExecutor(max_workers=8) as pool:
        for idx, ok, code in pool.map(check_one, range(1, n + 1)):
            results[idx] = (ok, code)

    passed = sum(1 for ok, _ in results.values() if ok)
    elapsed = time.time() - start
    os.system("rm -rf /tmp/tv_etcd_*")
    print(f"\nResult: {passed}/{n} PASSED ({passed/n*100:.1f}%)  time={elapsed:.1f}s")
    failed = [i for i, (ok, _) in results.items() if not ok]
    if failed:
        print(f"Failed: {failed[:10]}{' ...' if len(failed) > 10 else ''}")


if __name__ == "__main__":
    main()
