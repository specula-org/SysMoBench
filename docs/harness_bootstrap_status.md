# Harness Bootstrap Status

Per-task completion of the 9-task harness bootstrap, per
`tla_eval/skills/harness-gen/BOOTSTRAP_9_SYSTEMS.md`.

## mutex
- Category: B (concurrent kernel primitive; uses ktest+serial, not rdtsc timebox)
- `artifacts/spin/`: shared Asterinas clone (upstream github.com/asterinas/asterinas, commit in `artifacts/spin/.git`)
- Instrumentation already present (from prior work): `ostd/src/sync/mutex_trace.rs`, wired in `mod.rs`, `tla-trace` feature in Cargo.toml
- Harness orchestration added this pass:
  - `scripts/harness/mutex/run.sh` — docker runner
  - `scripts/harness/mutex/parse_traces.py` — splits 20 scenarios from kernel serial output
  - `tla_eval/tasks/mutex/INSTRUMENTATION.md` — event schema, action→spec mapping, rebuild recipe
  - `tla_eval/tasks/mutex/task.yaml` — `wv.repo_path`, `wv.harness.*` fields populated
- Events emitted: `Lock`, `TryLock`, `Unlock` (mapped to spec `AcquireLock`, `TryAcquireLock`, `ReleaseLock`)
- Smoke: 20 trace files, 260 events total, all 3 action types present, ktest `1 passed; 0 failed`
- WV smoke: **PASS** (workspace `wv-workspaces/20260417_102949_mutex/`). Agent ran harness fresh from copied repo, produced 20 traces + 260 windows, Step 0 contract check 100% compliant, per-action window counts: `AcquireLock=82`, `TryAcquireLock=48`, `ReleaseLock=130`. TLC couldn't run because the sample spec (`data/spec/mutex/mutex.tla`) has two unrelated `Spec`-operator bugs (`vars` vs `Vars`; `\A t` scope collision); agent's manual check against action bodies concluded ~100% pass on all three. Harness is verified end-to-end; spec-side fix is out of scope for this task.
- Open issues:
  - Failed `try_lock` attempts are NOT traced (only successes). Spec's `TryAcquireLock` scoring will only cover successes.
  - Docker run leaves root-owned build artifacts in `artifacts/spin/` and in any `workspace/repo/target/` the WV agent creates; requires sudo or chmod to clean up. `launch_wv_eval.sh` now handles the workspace-side cleanup gracefully.
  - `cargo install --path osdk` needs `--locked` (run.sh handles it); upstream Makefile target omits the flag.

## ringbuffer
- Not started

## rwmutex
- Not started (instrumentation file `rwmutex_trace.rs` already drafted in `artifacts/spin/`, similar to mutex; follows same pattern)

## redisraft
- Not started

## dqueue / locksvc / raftkvs (shared PGo clone)
- Not started

## curp
- Not started

## zookeeper
- Not started
