#!/bin/bash
# In-container build + run of the Asterinas 2-thread spin trace ktest.
#
# Runs inside asterinas/asterinas:0.16.0-20250822. The patched source is
# bind-mounted read-only at /src (host FS — may not support fallocate), so we
# build on an ext4 docker volume at /build where mke2fs/fallocate work. /build
# persists across runs, so re-runs after editing the ktest are incremental.
#
# Emits kernel serial JSON on stdout: lines like
#   {"seq":N,"thread":T,"lock":0,"state":"locked|unlocked","action":A,"actor":T}
# which scripts/harness/spin/parse_traces.py folds into NDJSON windows.
set -e
export PATH=/nix/store/4zpvbvn0cvmmn9k05b1qgr5xh7i6r9ka-nix-2.31.1/bin:$PATH
echo 'connect-timeout = 60000' >> /etc/nix/nix.conf

if [ ! -f /build/Makefile ]; then
  echo "===== populating /build from /src (first run) ====="
  cp -a /src/. /build/
fi
# Refresh instrumentation + tests so edits take effect on incremental re-runs.
cp -f /src/ostd/src/sync/spin_trace.rs       /build/ostd/src/sync/spin_trace.rs
cp -f /src/ostd/src/sync/spin_trace_tests.rs /build/ostd/src/sync/spin_trace_tests.rs

cd /build
echo "===== installing cargo-osdk ====="
OSDK_LOCAL_DEV=1 cargo install cargo-osdk --path osdk --locked
echo "===== make initramfs ====="
make initramfs
echo "===== cargo osdk test (test_spin_2thread) ====="
cd ostd
timeout 1200 cargo osdk test --features tla-trace --target-arch x86_64 \
  --qemu-args='-accel tcg' test_spin_2thread 2>&1
echo "===== DONE rc=$? ====="
