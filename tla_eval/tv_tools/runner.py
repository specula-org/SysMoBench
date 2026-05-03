"""
Generic parallel TLC runner for transition validation.

Each window is checked by one TLC invocation. WINDOW_INDEX env var selects
which window from the JSON file to check.

TLC exit code convention:
    12 = invariant violated = post-state reachable = PASS
    0  = no violation       = post-state not reachable = FAIL
    -1 (internal) = TIMEOUT
"""

from __future__ import annotations

import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

from tla_eval.utils.setup_utils import (
    get_tla_tools_path,
    get_community_modules_path,
)

DEFAULT_TLA_JAR = str(get_tla_tools_path())
DEFAULT_COMMUNITY_JAR = str(get_community_modules_path())


@dataclass
class WindowResult:
    window_id: int
    status: str  # "PASS" | "FAIL" | "TIMEOUT"
    exit_code: int


def _tlc_cmd(tla_jar: str, community_jar: str) -> list[str]:
    cp = f"{tla_jar}:{community_jar}" if community_jar else tla_jar
    return ["java", "-cp", cp, "tlc2.TLC",
            "-dfid", "2", "-deadlock", "-nowarning"]


def check_one_window(
    window_index: int,
    tv_tla: str,
    tv_cfg: str,
    work_dir: str,
    tla_jar: str = DEFAULT_TLA_JAR,
    community_jar: str = DEFAULT_COMMUNITY_JAR,
    timeout: int = 60,
) -> WindowResult:
    """Run TLC for a single window. Uses a unique metadir per window so
    parallel runs don't collide."""
    env = {**os.environ, "WINDOW_INDEX": str(window_index)}
    metadir = f"/tmp/tv_{os.getpid()}_{window_index}"
    cmd = _tlc_cmd(tla_jar, community_jar) + [
        "-metadir", metadir,
        "-config", tv_cfg,
        tv_tla,
    ]
    try:
        r = subprocess.run(
            cmd, capture_output=True, text=True,
            timeout=timeout, cwd=work_dir, env=env,
        )
        if r.returncode == 12:
            status = "PASS"
        elif r.returncode == 0:
            status = "FAIL"
        else:
            status = "ERROR"
        return WindowResult(window_index, status, r.returncode)
    except subprocess.TimeoutExpired:
        return WindowResult(window_index, "TIMEOUT", -1)
    finally:
        subprocess.run(["rm", "-rf", metadir], check=False)


def _worker(args):
    return check_one_window(*args)


def run_tv_batch(
    num_windows: int,
    tv_tla: str,
    tv_cfg: str,
    work_dir: str,
    workers: int = 8,
    timeout: int = 60,
    tla_jar: str = DEFAULT_TLA_JAR,
    community_jar: str = DEFAULT_COMMUNITY_JAR,
    progress_every: int = 50,
) -> dict[int, WindowResult]:
    """Run TV on windows 1..num_windows in parallel.

    Returns {window_index: WindowResult}. Window indices are 1-based.
    """
    tasks = [
        (i, tv_tla, tv_cfg, work_dir, tla_jar, community_jar, timeout)
        for i in range(1, num_windows + 1)
    ]
    results: dict[int, WindowResult] = {}
    start = time.time()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for res in pool.map(_worker, tasks):
            results[res.window_id] = res
            done = len(results)
            if done % progress_every == 0 or done == num_windows:
                elapsed = time.time() - start
                rate = done / elapsed if elapsed > 0 else 0
                print(f"  [{done}/{num_windows}]  {rate:.1f} windows/s")
    return results


def summarize(results: dict[int, WindowResult]) -> dict:
    total = len(results)
    passed = sum(1 for r in results.values() if r.status == "PASS")
    failed = sum(1 for r in results.values() if r.status == "FAIL")
    errored = sum(1 for r in results.values() if r.status == "ERROR")
    timeout = sum(1 for r in results.values() if r.status == "TIMEOUT")
    return {
        "total": total,
        "passed": passed,
        "failed": failed,
        "errored": errored,
        "timeout": timeout,
        "pass_rate": passed / total if total > 0 else 0.0,
    }
