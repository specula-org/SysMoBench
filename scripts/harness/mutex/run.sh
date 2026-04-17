#!/usr/bin/env bash
# scripts/harness/mutex/run.sh — collect mutex execution traces from
# Asterinas kernel test_mutex_trace ktest, run inside the asterinas docker image.
#
# Usage (from project root):
#   bash scripts/harness/mutex/run.sh
#
# Output: trace_NN.jsonl files (one per scenario, NDJSON lines + leading
# '#' comment header) under $TRACES_DIR (default: artifacts/mutex/traces).
#
# Requires: docker, the asterinas/asterinas:0.16.0-20250822 image, and the
# shared Asterinas clone at artifacts/spin (reused across mutex/spin/ringbuffer/rwmutex).

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "$SCRIPT_DIR/../../.." && pwd)
SOURCE_DIR="${ASTERINAS_SOURCE_DIR:-$PROJECT_ROOT/artifacts/spin}"
TRACES_DIR="${TRACES_DIR:-$PROJECT_ROOT/artifacts/mutex/traces}"
DOCKER_IMAGE="${ASTERINAS_DOCKER_IMAGE:-asterinas/asterinas:0.16.0-20250822}"
TEST_TARGET="${TEST_TARGET:-test_mutex_trace}"
BUILD_TIMEOUT="${BUILD_TIMEOUT:-600}"

if [[ ! -d "$SOURCE_DIR/ostd/src/sync" ]]; then
  echo "ERROR: Asterinas source not found at $SOURCE_DIR" >&2
  echo "Set ASTERINAS_SOURCE_DIR or ensure artifacts/spin is populated." >&2
  exit 1
fi

mkdir -p "$TRACES_DIR"
LOG_FILE="$TRACES_DIR/docker_run.log"

echo "[run.sh] source:    $SOURCE_DIR" >&2
echo "[run.sh] test:      $TEST_TARGET" >&2
echo "[run.sh] traces to: $TRACES_DIR" >&2
echo "[run.sh] log:       $LOG_FILE" >&2

# --locked is required because osdk/Cargo.lock pins libflate 2.1.0, but
# `cargo install --path osdk` (the Makefile's install_osdk target) ignores
# the lockfile and resolves libflate 2.3.0, which needs a rustc newer than
# the one in this docker image.
docker run --rm --privileged --network host \
  -v "$SOURCE_DIR:/workspace" "$DOCKER_IMAGE" \
  /bin/bash -c "
    set -e
    cd /workspace
    export PATH=/nix/store/4zpvbvn0cvmmn9k05b1qgr5xh7i6r9ka-nix-2.31.1/bin:\$PATH
    echo 'connect-timeout = 60000' >> /etc/nix/nix.conf
    OSDK_LOCAL_DEV=1 cargo install cargo-osdk --path osdk --locked
    make initramfs
    cd ostd
    timeout $BUILD_TIMEOUT cargo osdk test --features tla-trace \
      --target-arch x86_64 --qemu-args='-accel tcg' $TEST_TARGET 2>&1
  " > "$LOG_FILE" 2>&1 || {
    echo "[run.sh] docker run exited non-zero; see $LOG_FILE" >&2
    tail -40 "$LOG_FILE" >&2
    exit 1
  }

python3 "$SCRIPT_DIR/parse_traces.py" "$LOG_FILE" "$TRACES_DIR"
ls "$TRACES_DIR"/trace_*.jsonl 2>/dev/null | wc -l | xargs -I{} echo "[run.sh] done, {} trace files"
