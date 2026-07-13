#!/usr/bin/env bash
# scripts/harness/spin/run.sh — capture Asterinas spinlock execution traces for
# the JS-SAM Phase-3 (transition validation) direct path.
#
# Produces a 2-thread (pre, post) NDJSON trace set at
#   data/sys_traces/spin/spin_2thread.ndjson
# from a real run of the instrumented Asterinas spinlock under QEMU.
#
# Pipeline:
#   1. Clone Asterinas at v0.16.0 (matches the docker image toolchain; the
#      reference patch does not apply to current main) with LF line endings.
#   2. Apply data/patches/asterinas_tla_trace.patch (reference TLA+ trace
#      instrumentation) + data/patches/spin_2thread_ktest.patch (adds the
#      2-actor `test_spin_2thread` ktest — the reference tests are single-actor).
#   3. Build + run the ktest under QEMU-TCG inside the asterinas image, on an
#      ext4 docker volume (the source may be on a host FS without fallocate).
#   4. Fold the kernel serial JSON into JS-SAM windows via parse_traces.py.
#
# Requires: docker, the asterinas/asterinas:0.16.0-20250822 image, git, python3.
# Windows note: run from Git Bash with Docker Desktop; the CRLF and fallocate
# handling below is what makes the Linux-oriented flow work on a Windows host.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "$SCRIPT_DIR/../../.." && pwd)

SRC="${ASTERINAS_SOURCE_DIR:-$PROJECT_ROOT/artifacts/spin}"
IMAGE="${ASTERINAS_DOCKER_IMAGE:-asterinas/asterinas:0.16.0-20250822}"
REF_PATCH="$PROJECT_ROOT/data/patches/asterinas_tla_trace.patch"
ADD_PATCH="$PROJECT_ROOT/data/patches/spin_2thread_ktest.patch"
TRACES_OUT="${TRACES_OUT:-$PROJECT_ROOT/data/sys_traces/spin/spin_2thread.ndjson}"
LOG="${LOG:-$PROJECT_ROOT/artifacts/spin_capture.log}"

# 1-2. Clone + patch (idempotent: skip if already instrumented). The guard is
# a sentinel written only after BOTH patches apply — checking for a file the
# first patch creates would leave a half-patched checkout unrecoverable if the
# second apply fails (set -e aborts, the next run would skip the whole block).
SENTINEL="$SRC/.sysmobench_instrumented"
if [ ! -e "$SENTINEL" ]; then
  echo "[run.sh] cloning asterinas v0.16.0 (LF) into $SRC" >&2
  rm -rf "$SRC"
  git -c core.autocrlf=false clone --depth 1 --branch v0.16.0 \
    https://github.com/asterinas/asterinas.git "$SRC"
  git -C "$SRC" config core.autocrlf false
  # Patches (and the sysmobench checkout) may carry CRLF on Windows; normalize
  # to LF before applying so the emitted Rust/shell files build in Linux.
  echo "[run.sh] applying reference + 2-thread patches" >&2
  tr -d '\r' < "$REF_PATCH" | git -C "$SRC" apply --whitespace=nowarn
  tr -d '\r' < "$ADD_PATCH" | git -C "$SRC" apply --whitespace=nowarn
  touch "$SENTINEL"
fi

# 3. Build + run under QEMU, capturing serial output. Source is read-only; the
# build happens on the ext4 volume `spin-work`; cargo cache on `spin-cargo`.
echo "[run.sh] building + running test_spin_2thread under QEMU (see $LOG)" >&2
docker run --rm --privileged \
  -v "$SRC:/src:ro" \
  -v "$SCRIPT_DIR:/harness:ro" \
  -v "spin-work:/build" \
  -v "spin-cargo:/root/.cargo" \
  "$IMAGE" bash /harness/build_and_test.sh > "$LOG" 2>&1 || {
    echo "[run.sh] docker run failed; tail of $LOG:" >&2
    tail -40 "$LOG" >&2
    exit 1
  }

# 4. Parse serial JSON -> NDJSON windows.
python3 "$SCRIPT_DIR/parse_traces.py" "$LOG" "$TRACES_OUT"
echo "[run.sh] traces written to $TRACES_OUT" >&2
