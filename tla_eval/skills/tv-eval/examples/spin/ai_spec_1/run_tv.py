#!/usr/bin/env python3
"""
Run TV for AcquireLock on all windows.
Data-code separation: windows.ndjson is pure data, TV_*.tla is pure code.
"""

import os
import json
import subprocess
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

SPEC_DIR = Path(__file__).parent
WINDOWS = SPEC_DIR / "windows_AcquireLock.ndjson"
TV_TLA = "TV_AcquireLock.tla"
TV_CFG = "TV_AcquireLock.cfg"
PROJECT_ROOT = SPEC_DIR.resolve().parents[5]
TLA_JAR = str(PROJECT_ROOT / "lib" / "tla2tools.jar")
COMM_JAR = str(PROJECT_ROOT / "lib" / "CommunityModules-deps.jar")


def check_one(idx):
    env = {**os.environ, "WINDOW_INDEX": str(idx)}
    r = subprocess.run(
        ["java", "-cp", f"{TLA_JAR}:{COMM_JAR}", "tlc2.TLC",
         "-dfid", "2", "-deadlock", "-nowarning",
         "-metadir", f"/tmp/tv_spin_{idx}",
         "-config", TV_CFG, TV_TLA],
        capture_output=True, text=True, timeout=30,
        cwd=SPEC_DIR, env=env,
    )
    return idx, r.returncode == 12, r.returncode


def main():
    with open(WINDOWS) as f:
        n = sum(1 for _ in f)

    print(f"Running {n} windows against AI spec's AcquireLock...")
    start = time.time()
    results = {}
    with ProcessPoolExecutor(max_workers=8) as pool:
        for idx, ok, code in pool.map(check_one, range(1, n + 1)):
            results[idx] = (ok, code)

    passed = sum(1 for ok, _ in results.values() if ok)
    elapsed = time.time() - start

    # Cleanup metadirs
    os.system("rm -rf /tmp/tv_spin_*")

    print(f"\nResult: {passed}/{n} PASSED ({passed/n*100:.1f}%)  time={elapsed:.1f}s")

    failed = [i for i, (ok, _) in results.items() if not ok]
    if failed:
        print(f"Failed: {failed[:10]}{' ...' if len(failed) > 10 else ''}")


if __name__ == "__main__":
    main()
