#!/usr/bin/env bash
# Capture the focused spin TV contract test from an isolated Asterinas checkout.
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "$SCRIPT_DIR/../../.." && pwd)
SOURCE_COMMIT=d73f1016a1d0cfb7b1f2ea72d2e9dd67ced0e0fa
IMAGE=asterinas/asterinas:0.16.0-20250822@sha256:10d52ce84f72116ca2e41d6170f8882893592ccfc166d6b7d2262ccfeedacab2
OUT=${1:-"$PROJECT_ROOT/tla_eval/skills/tv-eval/examples/spin/traces"}
mkdir -p "$PROJECT_ROOT/artifacts" "$OUT"
OUT=$(cd "$OUT" && pwd)
WORK=$(mktemp -d "$PROJECT_ROOT/artifacts/spin-tv.XXXXXX")
PATCH="$PROJECT_ROOT/data/patches/asterinas_tla_trace.patch"
cleanup() {
    if [[ -s "$WORK/container-id" ]]; then
        docker rm -f "$(cat "$WORK/container-id")" >/dev/null 2>&1 || true
    fi
}
trap cleanup EXIT
printf 'Capture workspace: %s\n' "$WORK"
git init -q "$WORK/source"
git -C "$WORK/source" fetch --depth 1 https://github.com/asterinas/asterinas.git "$SOURCE_COMMIT"
git -C "$WORK/source" checkout -q --detach FETCH_HEAD
git -C "$WORK/source" apply --whitespace=nowarn "$PATCH"
# No host checkout, cargo installation, or other container is modified.
# --locked preserves the OSDK tool's dependency versions; make initramfs also
# prepares the disk images referenced by the pinned OSDK configuration.
timeout 1200 docker run --rm --cidfile "$WORK/container-id" --cpus 8 \
    -e CARGO_BUILD_JOBS=8 -v "$WORK/source:/build" "$IMAGE" bash -c '
        set -e
        cd /build
        OSDK_LOCAL_DEV=1 cargo install cargo-osdk --path osdk --locked
        make initramfs
        cd ostd
        timeout 900 cargo osdk test --features tla-trace --target-arch x86_64 \
            --qemu-args="-accel tcg" test_spin_tv_contract
    ' > "$WORK/kernel.log" 2>&1 || {
    tail -40 "$WORK/kernel.log" >&2
    exit 1
}
python3 "$SCRIPT_DIR/parse_trace.py" "$WORK/kernel.log" "$OUT/trace_01.jsonl"
python3 - "$OUT" "$SOURCE_COMMIT" "$IMAGE" "$PATCH" "$WORK/kernel.log" <<'PY'
import hashlib, json, sys
from pathlib import Path
out, commit, image, patch, log = sys.argv[1:]
out = Path(out)
text = Path(log).read_text()
# Keep the test output and result, without the large dependency/build log.
start = text.index("=== SPIN_TV_BEGIN ===")
(out / "kernel-test.txt").write_text(text[start:])
manifest = {
    "source_repository": "https://github.com/asterinas/asterinas",
    "source_commit": commit,
    "docker_image": image,
    "patch": "data/patches/asterinas_tla_trace.patch",
    "patch_sha256": hashlib.sha256(Path(patch).read_bytes()).hexdigest(),
    "trace_sha256": hashlib.sha256((out / "trace_01.jsonl").read_bytes()).hexdigest(),
    "test": "test_spin_tv_contract",
    "capture_command": "bash scripts/harness/spin_tv/run.sh",
    "scope": "Executed SpinTrace operations with three logical actors on one CPU; no synthetic events or blocking contention."
}
(out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
PY
