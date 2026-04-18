#!/usr/bin/env bash
# scripts/harness/ringbuffer/run.sh — collect RingBuffer execution traces
# from Asterinas kernel test_rb_trace_randomized ktest.
#
# Unlike mutex/rwmutex (which run fully inside the asterinas docker image),
# ringbuffer is executed directly on the HOST using:
#   - rustup toolchain nightly-2025-02-01 (from rust-toolchain.toml)
#   - cargo-osdk rebuilt from the patched artifacts/spin/osdk/ sources
#   - host-side qemu-system-x86_64 + OVMF + grub-mkrescue + xorriso
#   - a materialized initramfs.cpio.gz (not the dangling nix-store symlink)
#
# Why host-side? `cargo osdk test` regenerates the kernel test-base as its
# own workspace, which re-resolves dependencies from crates.io. The
# docker image's rustc 1.86-nightly is too old for libflate 2.3.0 / time
# 0.3.47 / fixed 1.31.0, and core2 0.4.0 is now yanked. Our host has the
# same rustc but with user-space tooling recent enough to drive the build,
# and we pin the supply-chain-affected crates via a modified osdk template
# (see artifacts/spin/osdk/src/base_crate/Cargo.toml.template).
#
# Usage (from project root):
#   bash scripts/harness/ringbuffer/run.sh

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "$SCRIPT_DIR/../../.." && pwd)

SOURCE_DIR="${ASTERINAS_SOURCE_DIR:-$PROJECT_ROOT/artifacts/spin}"
TRACES_DIR="${TRACES_DIR:-$PROJECT_ROOT/artifacts/ringbuffer/traces}"
OVMF_PATH="${OVMF_PATH:-$PROJECT_ROOT/artifacts/ringbuffer/ovmf}"
TEST_TARGET="${TEST_TARGET:-test_rb_trace_randomized}"

if [[ ! -d "$SOURCE_DIR/kernel/src/util" ]]; then
  echo "ERROR: Asterinas kernel source not found at $SOURCE_DIR/kernel/src/util" >&2
  exit 1
fi
if [[ ! -x "$SOURCE_DIR/osdk/target/release/cargo-osdk" ]]; then
  echo "ERROR: patched cargo-osdk not found at $SOURCE_DIR/osdk/target/release/cargo-osdk" >&2
  echo "Rebuild it inside the asterinas docker image once:" >&2
  echo "  docker run --rm -v \"\$(pwd)/artifacts/spin:/workspace\" asterinas/asterinas:0.16.0-20250822 \\" >&2
  echo "    /bin/bash -c 'cd /workspace/osdk && OSDK_LOCAL_DEV=1 cargo build --release'" >&2
  exit 1
fi
if [[ ! -s "$SOURCE_DIR/test/build/initramfs.cpio.gz" ]] || [[ -L "$SOURCE_DIR/test/build/initramfs.cpio.gz" && ! -e "$SOURCE_DIR/test/build/initramfs.cpio.gz" ]]; then
  echo "ERROR: initramfs.cpio.gz missing or dangling at $SOURCE_DIR/test/build/" >&2
  echo "Materialize it once via docker:" >&2
  echo "  docker run --rm -v \"\$(pwd)/artifacts/spin:/workspace\" asterinas/asterinas:0.16.0-20250822 \\" >&2
  echo "    /bin/bash -c 'cd /workspace && make initramfs && cp -L test/build/initramfs.cpio.gz test/build/initramfs.cpio.gz.real && mv test/build/initramfs.cpio.gz.real test/build/initramfs.cpio.gz'" >&2
  exit 1
fi

# Stage host-side OVMF files (copy once — OVMF_VARS.fd needs to be writable)
mkdir -p "$OVMF_PATH"
if [[ ! -f "$OVMF_PATH/OVMF_CODE.fd" ]]; then
  cp /usr/share/OVMF/OVMF_CODE_4M.fd "$OVMF_PATH/OVMF_CODE.fd"
fi
if [[ ! -f "$OVMF_PATH/OVMF_VARS.fd" ]]; then
  cp /usr/share/OVMF/OVMF_VARS_4M.fd "$OVMF_PATH/OVMF_VARS.fd"
  chmod u+w "$OVMF_PATH/OVMF_VARS.fd"
fi

for bin in qemu-system-x86_64 grub-mkrescue xorriso rustup; do
  if ! command -v "$bin" >/dev/null 2>&1; then
    echo "ERROR: required tool '$bin' not on PATH" >&2
    echo "Install via: sudo apt-get install -y qemu-system-x86 ovmf grub-common grub-pc-bin xorriso mtools" >&2
    exit 1
  fi
done

mkdir -p "$TRACES_DIR"
LOG="$TRACES_DIR/host_run.log"

echo "[run.sh] source:    $SOURCE_DIR" >&2
echo "[run.sh] test:      $TEST_TARGET" >&2
echo "[run.sh] traces to: $TRACES_DIR" >&2

# Host-side osdk test. Env vars:
#   CARGO_RESOLVER_INCOMPATIBLE_RUST_VERSIONS=fallback — let cargo pick older
#     versions when newer ones require a rustc we don't have.
#   OVMF=off — asterinas' default QEMU args bind OVMF to /root/ovmf/release
#     (a docker-internal path). With OVMF=off the boot flow falls back to a
#     direct grub-iso + BIOS chain that works on host.
#   OVMF_PATH=... — patched qemu_args.sh honors this for the OVMF=on path.
export PATH="$SOURCE_DIR/osdk/target/release:$PATH"
export CARGO_RESOLVER_INCOMPATIBLE_RUST_VERSIONS=fallback
export OVMF=off

cd "$SOURCE_DIR/kernel"
cargo-osdk osdk test --features tla-trace --target-arch x86_64 \
  --qemu-args="-accel tcg" "$TEST_TARGET" 2>&1 | tee "$LOG" >&2

python3 "$SCRIPT_DIR/parse_traces.py" "$LOG" "$TRACES_DIR"

echo "[run.sh] done — $(ls "$TRACES_DIR"/trace_*.jsonl 2>/dev/null | wc -l) trace files"
