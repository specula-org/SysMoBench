# RingBuffer Harness — Instrumentation & Run Notes

Collects execution traces from the Asterinas kernel's lock-free SPSC
`RingBuffer` (`kernel/src/util/ring_buffer.rs`) for TV action-window
validation.

## Where the code lives

- **Shared Asterinas clone**: `artifacts/spin/` — the same tree the
  mutex/rwmutex/spin tasks use, but augmented with the following
  ringbuffer-specific additions:
  - `ostd/src/trace_support.rs` — public helpers `in_bootstrap_context()`
    / `send_byte(u8)` so kernel-level crates can emit bytes via OSTD's
    serial path without reaching through `pub(crate)` internals.
    (`ostd/src/lib.rs` re-exports `pub mod trace_support;`.)
  - `kernel/src/util/ring_buffer_trace.rs` — instrumented RingBuffer
    plus `#[ktest] test_rb_trace_randomized` (5 randomized scenarios).
  - `vendored/core2-0.4.0/`, `vendored/libflate-2.2.1/`,
    `vendored/libflate_lz77-2.2.0/` — local crate extracts to bypass
    crates.io supply-chain rot (see below).
  - `osdk/src/base_crate/Cargo.toml.template` — SysMoBench patch that
    adds `[patch.crates-io]` redirects in every generated test-base:
    - `core2` / `libflate` / `libflate_lz77` → vendored paths
    - `ostd` / `osdk-test-kernel` / `osdk-frame-allocator` /
      `osdk-heap-allocator` → workspace paths (otherwise test-base pulls
      a second copy from crates.io and we get duplicate
      `#[global_allocator]` errors).
  - `tools/qemu_args.sh` — small `OVMF_PATH=${OVMF_PATH:-/root/ovmf/release}`
    indirection so host-side runs can point at `/usr/share/OVMF/`.
  - `test/build/initramfs.cpio.gz` — a *materialized* (not nix-store
    symlinked) rootfs so host cargo-osdk can open it.
- **Harness orchestration**:
  - `scripts/harness/ringbuffer/run.sh` — host-side driver (see below)
  - `scripts/harness/ringbuffer/parse_traces.py` — splits the kernel
    serial output on `=== TRACE_RANDOM_<n> ===` banners into
    `trace_01.jsonl` … `trace_NN.jsonl`.

## Why host-side (not docker)

mutex / rwmutex run the whole build inside `asterinas/asterinas:0.16.0-20250822`
because their kernel deps (in `ostd/`) are closed. RingBuffer's test
lives in `kernel/`, which means `cargo osdk test` generates a separate
test-base workspace that re-resolves dependencies fresh from crates.io.

That fresh resolution now picks packages that don't compile against the
Asterinas-pinned `nightly-2025-02-01` rustc:

| Pkg                | Picked fresh | Needs            | Our pin         |
|--------------------|--------------|------------------|-----------------|
| `core2`            | 0.4.0        | (yanked on crates.io) | vendored 0.4.0 (bypass yank) |
| `libflate`         | 2.3.0        | Rust 1.88 (`let_chains`) | vendored 2.2.1 |
| `libflate_lz77`    | 2.3.0        | API != libflate 2.2.1 | vendored 2.2.0 |
| `fixed`            | 1.31.0       | Rust 1.93        | cargo fallback → 1.29.0 |
| `time`             | 0.3.47       | Rust 1.88        | cargo fallback → 0.3.44 |

Docker's rustc is also 1.86-nightly — we'd hit the same wall inside it,
plus docker's older glibc can't run cargo-osdk binaries built against
newer user-space libraries. Host, by contrast, has the *same* rustc via
`rustup` (pinned by `rust-toolchain.toml`) plus modern enough user-space
tooling to drive the build.

## Emit events

`ring_buffer_trace.rs` writes one NDJSON line per RingBuffer method call:

```json
{"seq":N,"action":"Push","actor":"producer","rb":R,"head":H,"tail":T,"capacity":C,"success":true}
```

| Field      | Type                           | Meaning                           |
|------------|--------------------------------|-----------------------------------|
| `seq`      | int (monotonic, NO per-scenario reset) | Global sequence number     |
| `action`   | string                         | `Create` / `Split` / `Push` / `PushSlice` / `Pop` / `PopSlice` |
| `actor`    | `"system"` / `"producer"` / `"consumer"` | Which side triggered it |
| `rb`       | int                            | RingBuffer instance ID (0-indexed per scenario) |
| `head`     | int                            | Consumer pointer                  |
| `tail`     | int                            | Producer pointer                  |
| `capacity` | int                            | Ring capacity                     |
| `success`  | bool                           | False on empty-Pop / full-Push    |

### Raw → spec-action mapping

The ringbuffer task prompt requires just `Push` and `Pop`. The TV agent
maps success-only versions of the raw events:

| Trace `action` (with `success=true`) | Spec action |
|--------------------------------------|-------------|
| `Push` / `PushSlice`                 | `Push`      |
| `Pop`  / `PopSlice`                  | `Pop`       |
| `Create` / `Split`                   | — (setup)   |
| any action with `success=false`      | — (failure path; spec only models successful enqueue/dequeue) |

## How to run

```bash
# from project root — does not need docker
bash scripts/harness/ringbuffer/run.sh
```

Typical output: 5 scenarios / ~75-90 events total. First run takes
~30-60 s (cold cargo build of the kernel test binary); subsequent runs
~10-20 s.

### Environment overrides

| Var                    | Default                                  | Purpose                                  |
|------------------------|------------------------------------------|------------------------------------------|
| `TRACES_DIR`           | `artifacts/ringbuffer/traces`            | Where parsed NDJSON files land           |
| `ASTERINAS_SOURCE_DIR` | `artifacts/spin`                         | Asterinas clone with SysMoBench patches  |
| `OVMF_PATH`            | `artifacts/ringbuffer/ovmf`              | Auto-populated from `/usr/share/OVMF/` on first run |
| `TEST_TARGET`          | `test_rb_trace_randomized`               | ktest name                               |

## First-run prerequisites (host)

The harness requires these installed on host:

```bash
sudo apt-get install -y qemu-system-x86 ovmf grub-common grub-pc-bin xorriso mtools
# plus rustup with nightly-2025-02-01 (asterinas' rust-toolchain.toml)
```

Two one-time setup steps (run.sh self-checks both):

1. **Build the patched cargo-osdk** (inside docker, once):
   ```bash
   cd artifacts/spin
   docker run --rm --network host -v "$(pwd):$(pwd)" -w "$(pwd)/osdk" \
     asterinas/asterinas:0.16.0-20250822 /bin/bash -c \
     'export PATH=/nix/store/4zpvbvn0cvmmn9k05b1qgr5xh7i6r9ka-nix-2.31.1/bin:$PATH && \
      OSDK_LOCAL_DEV=1 cargo build --release'
   ```
   The matching `-v path:path` mount keeps absolute workspace paths
   consistent between docker and host so the generated test-base's path
   deps resolve on either side.

2. **Materialize initramfs.cpio.gz** (asterinas' Makefile builds it via
   Nix; dereference the symlink so host can open it):
   ```bash
   docker run --rm -v "$(pwd)/artifacts/spin:/workspace" \
     asterinas/asterinas:0.16.0-20250822 /bin/bash -c '
       cd /workspace && make initramfs && \
       cp -L test/build/initramfs.cpio.gz test/build/initramfs.cpio.gz.real && \
       mv test/build/initramfs.cpio.gz.real test/build/initramfs.cpio.gz'
   ```

## Known coverage gaps

- **Single-threaded simulation**. Although spec-level SPSC is two-process,
  the ktest runs producer + consumer sequentially in one kernel thread.
  `success=false` paths (empty Pop, full Push) are exercised, but genuine
  producer/consumer concurrency isn't.
- **No `PushSlice`/`PopSlice`-distinct scoring**. We map both to
  `Push`/`Pop`; the spec doesn't model batch-vs-single. Windows for
  slice operations have `tail' - tail > 1` — harmless if the spec
  accepts multi-step increments, a 0-score if it insists on atomic single.
- **Single RingBuffer per scenario** (rb IDs 0-4 across 5 scenarios);
  cross-ring invariants can't be tested.
