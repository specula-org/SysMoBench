# Mutex Harness — Instrumentation & Run Notes

This harness collects execution traces from the Asterinas OS `Mutex`
synchronization primitive (ostd/src/sync/mutex.rs) for TV (action-window
validation) of generated TLA+ specs.

## Where the code lives

- **Shared Asterinas clone**: `artifacts/spin/` (also used by `spin`,
  `ringbuffer`, `rwmutex`). Do NOT re-clone here; reuse to avoid 7+GB
  duplication.
- **Instrumented mutex**: `artifacts/spin/ostd/src/sync/mutex_trace.rs`
  — a copy of `mutex.rs` where each state-changing call emits a JSON
  event on the kernel serial port.
- **Ktest driver**: the `test_mutex_trace` test at the bottom of
  `mutex_trace.rs`. It runs 20 randomized scenarios with 3 competing
  threads sharing one `MutexTrace<u32>`. Each scenario resets
  `TRACE_SEQUENCE` to 0 — that's how the parser splits scenarios.
- **Module wiring**: `ostd/src/sync/mod.rs` declares `mod mutex_trace;`
  and re-exports `MutexTrace`, `MutexTraceGuard`, `ArcMutexTraceGuard`.

## Emitted events

Each mutex operation emits one NDJSON line on the serial port:

```json
{"seq":N,"thread":T,"mutex":0,"state":"locked|unlocked","action":A,"actor":T}
```

| Field    | Type                  | Meaning                                 |
|----------|-----------------------|-----------------------------------------|
| `seq`    | int (0..99, wraps)    | Monotonic counter, reset per scenario   |
| `thread` | int (0..2)            | Thread ID (3-thread contention)         |
| `mutex`  | int (always 0)        | Fixed — one mutex per scenario          |
| `state`  | `"locked"/"unlocked"` | Post-action mutex state                 |
| `action` | string                | `Lock` / `TryLock` / `Unlock`           |
| `actor`  | int                   | Alias of `thread` for schema uniformity |

### Action → spec-action mapping

The mutex prompt (`tla_eval/tasks/mutex/prompts/direct_call.txt`) requires
these action names in the generated TLA+ spec. The TV agent maps raw
trace events → spec actions when building windows:

| Trace `action` | Spec action       | Semantic                            |
|----------------|-------------------|-------------------------------------|
| `Lock`         | `AcquireLock`     | Blocking acquire (waited if needed) |
| `TryLock`      | `TryAcquireLock`  | Non-blocking successful attempt     |
| `Unlock`       | `ReleaseLock`     | Guard drop / explicit unlock        |

Failed `try_lock` does not emit an event (line 175 of `mutex_trace.rs`)
— the spec's `TryAcquireLock` action only covers success. A future
improvement would be to emit a `TryLockFail` event for coverage.

## What we changed vs. upstream Asterinas

These are the only additions to the upstream Asterinas kernel tree at
`artifacts/spin/`:

1. **New file** `ostd/src/sync/mutex_trace.rs` — copy of `mutex.rs` with
   `trace_event()` calls at each state-changing point. Tracing is gated
   by `crate::IN_BOOTSTRAP_CONTEXT` so it only fires after the kernel
   finishes booting, avoiding noise during init.
2. **Edit** `ostd/src/sync/mod.rs` — added `mod mutex_trace;` and
   re-exports.
3. **Edit** `ostd/Cargo.toml` — added `tla-trace = []` to `[features]`
   (currently a no-op gate; present so `--features tla-trace` succeeds).
4. **Edit** `ostd/src/lib.rs` — added `pub(crate) static
   IN_BOOTSTRAP_CONTEXT: AtomicBool` plus its `store(false)` at the end
   of bootstrap.

(1)–(3) are shared with the `spin`, `ringbuffer`, and `rwmutex` tasks.

## How to run the harness

```bash
# from project root
bash scripts/harness/mutex/run.sh
```

Writes `trace_01.jsonl` .. `trace_20.jsonl` into
`artifacts/mutex/traces/` (or `$TRACES_DIR` if overridden). Each file
is a `#`-commented header followed by one JSON event per line. Total
runtime on a warm build is ~5 min (cold build ≈ 10 min).

Files:
- `scripts/harness/mutex/run.sh` — docker orchestration (tracked in git)
- `scripts/harness/mutex/parse_traces.py` — splits kernel serial output
  into per-scenario trace files (tracked in git)
- `artifacts/mutex/traces/` — output directory (gitignored)
- `artifacts/spin/ostd/src/sync/mutex_trace.rs` — instrumented mutex
  (in the gitignored clone; reproducible via `data/patches/asterinas_tla_trace.patch`)

### Environment overrides

| Var                     | Default                                | Purpose                         |
|-------------------------|----------------------------------------|---------------------------------|
| `TRACES_DIR`            | `artifacts/mutex/traces`               | Where parsed NDJSON files land  |
| `ASTERINAS_SOURCE_DIR`  | `artifacts/spin`                       | Location of the kernel clone    |
| `ASTERINAS_DOCKER_IMAGE`| `asterinas/asterinas:0.16.0-20250822`  | Build/test container            |
| `TEST_TARGET`           | `test_mutex_trace`                     | ktest name                      |

## Rebuilding from a clean checkout

```bash
# 1. clone upstream
git clone https://github.com/asterinas/asterinas.git artifacts/spin

# 2. apply the trace instrumentation patch (spin/mutex/rwmutex families)
cd artifacts/spin
git apply ../../data/patches/asterinas_tla_trace.patch

# 3. run
cd ../.. && bash scripts/harness/mutex/run.sh
```

The patch may need minor adaptation if `main` has moved — search for
`sync/mod.rs` and verify the `mod mutex_trace;` insertion still
applies cleanly.

## Known issues

- `cargo install --path osdk` inside the docker image must be invoked
  with `--locked`, otherwise it pulls libflate 2.3.0 which needs a
  newer rustc than the image provides. `run.sh` handles this.
- Failed `try_lock` attempts are not traced. The spec's
  `TryAcquireLock` action can therefore only be scored on successful
  paths.
- Running `run.sh` writes build artifacts into `artifacts/spin/` —
  the source tree is not kept pristine between runs.
