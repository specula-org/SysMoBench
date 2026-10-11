"""Run the spin examples without modifying their checked-in specifications/data."""

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT))

from tla_eval.tv_tools.runner import run_tv_batch, summarize


def main(spec_dir, module, default_windows):
    windows_name = default_windows.name
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--windows", type=Path, default=default_windows)
    parser.add_argument(
        "--tla-jar", type=Path, default=PROJECT_ROOT / "lib/tla2tools.jar"
    )
    parser.add_argument(
        "--community-jar",
        type=Path,
        default=PROJECT_ROOT / "lib/CommunityModules-deps.jar",
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--report", type=Path, help="Write per-window results as JSON")
    args = parser.parse_args()
    if args.workers < 1 or args.timeout < 1:
        parser.error("--workers and --timeout must be positive")
    if not shutil.which("java"):
        parser.error("Java is required; install Java 11 or newer")
    for jar in (args.tla_jar, args.community_jar):
        if not jar.is_file() or jar.stat().st_size == 0:
            parser.error(
                f"Missing tool: {jar}. Run sysmobench-setup or specify both jar paths."
            )
    try:
        windows = [
            json.loads(line)
            for line in args.windows.read_text().splitlines()
            if line.strip()
        ]
        if not windows or not all(isinstance(w, dict) for w in windows):
            raise ValueError("expected a nonempty NDJSON window file")
    except (OSError, ValueError) as exc:
        parser.error(str(exc))

    # TLC may write generated files beside the module; keep the checkout clean.
    with tempfile.TemporaryDirectory(prefix="spin-tv-") as tmp:
        work_dir = Path(tmp)
        for pattern in ("*.tla", "*.cfg"):
            for source in spec_dir.glob(pattern):
                shutil.copy2(source, work_dir / source.name)
        (work_dir / windows_name).write_text(
            "".join(json.dumps(w) + "\n" for w in windows)
        )
        results = run_tv_batch(
            len(windows),
            f"{module}.tla",
            f"{module}.cfg",
            str(work_dir),
            workers=args.workers,
            timeout=args.timeout,
            tla_jar=str(args.tla_jar.resolve()),
            community_jar=str(args.community_jar.resolve()),
        )
    stats = summarize(results)
    print(json.dumps(stats, indent=2))
    if args.report:
        args.report.write_text(
            json.dumps(
                {
                    "summary": stats,
                    "windows": [vars(r) for _, r in sorted(results.items())],
                },
                indent=2,
            )
            + "\n"
        )
    if stats["errored"] or stats["timeout"]:
        print(
            "TLC errors/timeouts are infrastructure failures, not model mismatches.",
            file=sys.stderr,
        )
        return 2
    return 1 if stats["failed"] else 0
