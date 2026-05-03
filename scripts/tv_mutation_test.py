#!/usr/bin/env python3
"""
Guard / body mutation testing for TV score discrimination.

For a given TV workspace (already populated with windows_*.ndjson and
TV_*.tla window validators), apply a series of spec mutations and re-run
the validation. Report per-action pass rate for each mutant.

Usage:
  python3 scripts/tv_mutation_test.py <tv-workspace-dir> --system <name> \\
      [--mutations <json>]

Mutation definitions live in `scripts/tv_mutations.json` keyed by system
name (e.g. `spin`, `mutex`, `locksvc`). Pass `--mutations` to point at an
alternate catalog; otherwise the default path is used.
"""

import argparse
import json
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

DEFAULT_MUTATIONS = Path(__file__).resolve().parent / "tv_mutations.json"


@dataclass
class Mutation:
    name: str
    description: str
    pattern: str
    replacement: str


def load_mutations_from_json(path: Path, system: str) -> list:
    data = json.loads(path.read_text())
    if system not in data:
        raise SystemExit(f"System '{system}' not in {path}. Available: {list(data.keys())}")
    return [Mutation(**m) for m in data[system]]


def apply_mutation(text: str, mut: Mutation) -> str:
    new, n = re.subn(mut.pattern, mut.replacement, text, count=1, flags=re.DOTALL)
    if n != 1:
        raise RuntimeError(f"Mutation {mut.name} matched {n} times (expected 1)")
    return new


def parse_rates(stdout: str) -> dict:
    rates = {}
    for m in re.finditer(
        r"(\w+)\s*:\s*([\d.]+)%\s+\((\d+)/(\d+)\)", stdout
    ):
        action = m.group(1)
        passed = int(m.group(3))
        total = int(m.group(4))
        rates[action] = (passed, total, passed / total if total else 0.0)
    return rates


def run_validation(runner) -> dict:
    if isinstance(runner, (list, tuple)):
        cmd = ["python3"] + [str(a) for a in runner]
    else:
        cmd = ["python3", str(runner)]
    r = subprocess.run(
        cmd,
        capture_output=True, text=True, timeout=900,
    )
    rates = parse_rates(r.stdout)
    if not rates:
        print("--- STDOUT ---")
        print(r.stdout[-2000:])
        print("--- STDERR ---")
        print(r.stderr[-1000:])
        raise RuntimeError("Failed to parse any rates")
    return rates


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ws_dir", help="TV workspace directory")
    ap.add_argument("--system", required=True,
                    help="System key in mutations JSON (e.g. spin, mutex, locksvc)")
    ap.add_argument("--mutations", default=str(DEFAULT_MUTATIONS),
                    help=f"Path to mutations JSON (default: {DEFAULT_MUTATIONS.name})")
    args = ap.parse_args()

    ws_dir = Path(args.ws_dir)
    mutations = load_mutations_from_json(Path(args.mutations), args.system)
    tlas = list((ws_dir / "spec").glob("*.tla"))
    if len(tlas) == 1:
        spec = tlas[0]
    else:
        raise SystemExit(f"Expected exactly 1 .tla in {ws_dir}/spec/, got {len(tlas)}")

    # The TV validators do `INSTANCE <system>` which resolves to tv/<system>.tla
    # (same directory). When that file is a symlink chained back to spec/<sys>.tla,
    # mutating spec/ is enough. But when tv/<sys>.tla is a standalone copy
    # (curp / rwmutex / spin), mutating spec/ alone does NOT propagate — TV keeps
    # reading the untouched copy and all mutations appear as "no-change". Write
    # the mutant to BOTH locations to be safe.
    tv_spec = ws_dir / "tv" / spec.name
    targets = [spec]
    if tv_spec.exists() and not tv_spec.is_symlink():
        targets.append(tv_spec)
    elif tv_spec.is_symlink():
        # Already chained back to spec/ — leave it, write_text on spec will
        # propagate through the symlink chain.
        pass
    print(f"Mutation targets: {[str(t) for t in targets]}")
    # Use the generic runner so each per-system workspace doesn't need its
    # own matching output format. The generic runner discovers TV_*.tla and
    # windows_*.ndjson files in the tv/ dir and prints a standard summary.
    runner_args = [Path(__file__).parent / "tv_generic_runner.py", str(ws_dir / "tv")]
    runner = runner_args

    # Backup each target; content is identical across them at this point.
    backups = {t: t.with_suffix(".tla.orig") for t in targets}
    for t, b in backups.items():
        shutil.copy(t, b)
    original = backups[targets[0]].read_text()

    def write_mutant(text: str):
        for t in targets:
            t.write_text(text)

    def restore_all():
        for t, b in backups.items():
            t.write_text(b.read_text())
            b.unlink()

    print(f"Spec:   {spec}")
    print(f"Runner: {runner}\n")

    print("=== BASELINE (original spec) ===")
    baseline = run_validation(runner)
    for a, (p, t, r) in baseline.items():
        print(f"  {a:20s} {r:>6.1%}  ({p}/{t})")

    print()
    results = {}
    try:
        for mut in mutations:
            print(f"=== {mut.name}: {mut.description} ===")
            try:
                mutant = apply_mutation(original, mut)
            except Exception as e:
                print(f"  SKIP (mutation failed to apply: {e})\n")
                continue
            write_mutant(mutant)
            try:
                rates = run_validation(runner)
                results[mut.name] = rates
                for a, (p, t, r) in rates.items():
                    b_rate = baseline.get(a, (0, 0, 0))[2]
                    diff = r - b_rate
                    arrow = "↓" if diff < -1e-6 else ("↑" if diff > 1e-6 else " ")
                    print(f"  {a:20s} {r:>6.1%}  ({p}/{t})  {arrow} {diff:+.1%}")
            except Exception as e:
                print(f"  RUN FAILED: {e}")
            print()
    finally:
        restore_all()
        print(f"Restored {len(targets)} target file(s) from backup.")

    print("\n=== Summary ===")
    print(f"{'Mutation':<32}{'Baseline → Mutant (delta per action)'}")
    for name, rates in results.items():
        parts = []
        for a, (p, t, r) in rates.items():
            b = baseline.get(a, (0, 0, 0))[2]
            parts.append(f"{a}={b:.0%}→{r:.0%}")
        print(f"  {name:<32}{' | '.join(parts)}")


if __name__ == "__main__":
    main()
