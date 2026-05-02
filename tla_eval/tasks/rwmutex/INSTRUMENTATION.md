# RwMutex Harness — Instrumentation & Run Notes

This harness collects execution traces from the Asterinas OS `RwMutex`
read-write lock (ostd/src/sync/rwmutex.rs) for TV (action-window
validation) of generated TLA+ specs.

## Where the code lives

- **Shared Asterinas clone**: `artifacts/spin/` (also used by `mutex`,
  `spin`, `ringbuffer`). Do NOT re-clone.
- **Instrumented rwmutex**: `artifacts/spin/ostd/src/sync/rwmutex_trace.rs`
  — a copy of `rwmutex.rs` where each state-changing call emits a JSON
  event on the kernel serial port.
- **Ktest driver**: `test_rwmutex_trace` (inside `rwmutex_trace.rs`).
  Runs 100 randomized scenarios with 3 competing threads against one
  `RwMutexTrace<u32>`. Each scenario resets `TRACE_SEQUENCE` to 0 — the
  parser uses that to split scenarios.
- **Module wiring**: `ostd/src/sync/mod.rs` declares `mod rwmutex_trace;`
  and re-exports `RwMutexTrace`, `RwMutexTraceReadGuard`,
  `RwMutexTraceWriteGuard`, `RwMutexTraceUpgradeableGuard`, plus Arc
  variants.

## Emitted events

Each rwmutex operation emits one NDJSON line on the serial port:

```json
{"seq":N,"thread":T,"rwmutex":0,"state":S,"lock_type":L,"action":A,"actor":T}
```

| Field       | Type                                          | Meaning                                          |
|-------------|-----------------------------------------------|--------------------------------------------------|
| `seq`       | int (resets to 0 per scenario)                | Monotonic counter within a scenario              |
| `thread`    | int (0..2)                                    | Thread ID                                        |
| `rwmutex`   | int (always 0)                                | Fixed — one rwmutex per scenario                 |
| `state`     | string (`free`, `1_reader`, `multi_readers`, `writer_locked`, `upread_locked`, …) | Post-action lock state |
| `lock_type` | `"read"`/`"write"`/`"upread"`                 | Which role the actor is playing                  |
| `action`    | string                                        | One of the raw event names (see table)           |
| `actor`     | int                                           | Alias of `thread`                                |

### Raw event → spec-action mapping

The rwmutex prompt (`tla_eval/tasks/rwmutex/prompts/direct_call.txt`)
requires these 4 spec actions. The TV agent maps raw trace events →
spec actions when building windows:

| Trace `action`                 | Spec action         | Semantic                            |
|--------------------------------|---------------------|-------------------------------------|
| `ReadLock` / `TryReadLock`     | `AcquireReadLock`   | Successful shared (reader) acquire  |
| `WriteLock` / `TryWriteLock`   | `AcquireWriteLock`  | Successful exclusive acquire        |
| `UpreadLock` / `TryUpreadLock` | `AcquireUpReadLock` | Successful upgradeable-reader acquire |
| `ReadUnlock` / `WriteUnlock` / `UpreadUnlock` | `ReleaseLock` | Guard drop / explicit release |
| `UpgradeLock` / `TryUpgradeLock` | *(out of scope)*  | Upgrade-from-upread — not in target_actions |

Failed `try_*` attempts emit console log lines (`thread0: try_write
failed`) but NOT trace events — only successful paths are scored.

## Known coverage gaps

- **`AcquireUpReadLock` has 0 windows**. `test_rwmutex_trace` doesn't
  exercise the upread / upgrade paths — TV will report the action as
  untested. Future work: extend the ktest to include upread scenarios.
- **No failed-try paths**. Same gap as mutex — spec's handling of
  contention-failure paths (`try_lock` returns `None`) can only be
  scored if we add those events.

## What we changed vs. upstream Asterinas

1. **New file** `ostd/src/sync/rwmutex_trace.rs` — copy of `rwmutex.rs`
   with `trace_event()` calls at each state-changing point. Gated by
   `crate::IN_BOOTSTRAP_CONTEXT` so it only emits after bootstrap.
2. **Edit** `ostd/src/sync/mod.rs` — added `mod rwmutex_trace;` and
   re-exports.
3. Shared with mutex/spin/ringbuffer: `ostd/Cargo.toml` has
   `tla-trace = []` feature; `ostd/src/lib.rs` defines
   `IN_BOOTSTRAP_CONTEXT`.

## How to run the harness

```bash
# from project root
bash scripts/harness/rwmutex/run.sh
```

Writes `trace_01.jsonl` .. `trace_NN.jsonl` into
`artifacts/rwmutex/traces/` (or `$TRACES_DIR` if overridden). Warm
build ≈ 5 min, cold ≈ 10 min.

Typical smoke yield: ~100 scenarios, ~2000-2500 events (ReadLock,
TryReadLock, ReadUnlock, WriteLock, TryWriteLock, WriteUnlock).

### Environment overrides

| Var                     | Default                                | Purpose                         |
|-------------------------|----------------------------------------|---------------------------------|
| `TRACES_DIR`            | `artifacts/rwmutex/traces`             | Where parsed NDJSON files land  |
| `ASTERINAS_SOURCE_DIR`  | `artifacts/spin`                       | Location of the kernel clone    |
| `ASTERINAS_DOCKER_IMAGE`| `asterinas/asterinas:0.16.0-20250822`  | Build/test container            |
| `TEST_TARGET`           | `test_rwmutex_trace`                   | ktest name                      |

## Rebuilding from a clean checkout

```bash
# 1. clone upstream
git clone https://github.com/asterinas/asterinas.git artifacts/spin

# 2. apply the trace instrumentation patch (spin/mutex/rwmutex families)
cd artifacts/spin
git apply ../../data/patches/asterinas_tla_trace.patch

# 3. run
cd ../.. && bash scripts/harness/rwmutex/run.sh
```
