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
- **Blocked** on Asterinas upstream supply-chain issue: `core2 0.4.0` is yanked, and `kernel/Cargo.toml` pins `core2 = "^0.4"` (no other 0.4.x exists on crates.io). `cargo osdk test` generates a fresh `target/osdk/aster-nix-test-base` project whose resolver ignores our `cargo update --precise 0.4.0` in the kernel workspace. User has a prior working solution and will package it — pause until then.
- Instrumentation already present: `kernel/src/util/ring_buffer_trace.rs` (541 lines, 4 ktests, emits Create/Split/Push/Pop/PushSlice/PopSlice events with head/tail/capacity/success fields).

## rwmutex
- Category: B (concurrent kernel primitive; uses ktest+serial, single-threaded kernel context)
- `artifacts/spin/`: shared Asterinas clone
- Instrumentation already present: `ostd/src/sync/rwmutex_trace.rs` (wired in `mod.rs`, same pattern as mutex)
- Harness orchestration added this pass:
  - `scripts/harness/rwmutex/run.sh`
  - `scripts/harness/rwmutex/parse_traces.py` (same seq==0 split strategy as mutex)
  - `tla_eval/tasks/rwmutex/INSTRUMENTATION.md`
  - `tla_eval/tasks/rwmutex/task.yaml` wv section populated
- Events: `ReadLock`, `TryReadLock`, `WriteLock`, `TryWriteLock`, `ReadUnlock`, `WriteUnlock` (+ UpreadLock/TryUpreadLock/UpreadUnlock/UpgradeLock/TryUpgradeLock available but not exercised by current ktest)
- Action mapping: Read/TryRead → `AcquireReadLock`, Write/TryWrite → `AcquireWriteLock`, Upread/TryUpread → `AcquireUpReadLock` (unused), all `*Unlock` → `ReleaseLock`
- Smoke: 100 trace files, 2230 events total, ktest `1 passed; 0 failed`
- WV smoke: in progress (see latest `wv-workspaces/*_rwmutex/`)
- Known coverage gaps:
  - `test_rwmutex_trace` doesn't exercise upread/upgrade — `AcquireUpReadLock` will have 0 scoring windows
  - Failed `try_*` attempts not traced; only successful paths scored

## redisraft
- Category: A (distributed Raft, NDJSON single-writer trace)
- Reuses Specula case dir at `/home/ubuntu/Specula/case-studies/redisraft/` (symlink-free). Canonical instrumented raft library + harness already present from Specula.
- Harness orchestration added this pass:
  - `scripts/harness/redisraft/run.sh` — thin wrapper that invokes the Specula `harness/run.sh` and mirrors `.ndjson` outputs to `artifacts/redisraft/traces/`
  - `tla_eval/tasks/redisraft/INSTRUMENTATION.md` — event schema, action mapping, coverage tables
  - `tla_eval/tasks/redisraft/task.yaml` — `wv.harness.*` populated, including full `trace_action_map` for 5 events
- Events emitted (via Specula `tla_trace.c` patch into `deps/raft/src/raft_server.c`): `Timeout`, `BecomeLeader`, `HandleRequestVoteRequest`, `HandleAppendEntriesRequest`, `ClientRequest` plus out-of-scope: response events, `AdvanceCommitIndex`, `TakeSnapshot`, `ProposeAddServer`, `ProposeRemoveServer`, install/end-snapshot
- Action mapping: `Timeout`→`ElectionTimeout`+`BecomeCandidate` (same event, two spec actions); `HandleRequestVoteRequest`→`RecvRequestVote`; `HandleAppendEntriesRequest`→`RecvAppendEntries`; `ClientRequest`→`LogAppend`
- Smoke: 3 scenarios (basic_consensus/leader_failover/snapshot_basic), 74-78 events total, all tests `ok 1`
- WV smoke: **PASS** (workspace `wv-workspaces/20260417_132542_redisraft/`). Agent ran harness fresh from workspace copy, produced 43 windows across all 6 actions. Sample spec at `data/spec/redisraft/redisraft.tla` has a parse error (`_` as identifier in `SortedSeq`), blocking TLC. Agent's manual semantic check: `ElectionTimeout=4/4`, `BecomeCandidate=4/4`, `RecvRequestVote=8/8` would pass; `BecomeLeader=0/4` (spec's `UNCHANGED log` doesn't model the leader-elected noop append), `RecvAppendEntries`/`LogAppend` 0 (spec's `Entry={}` model gap). All blockers are spec-side, not harness.
- Open issues:
  - Coverage thin (~78 events, 3 scenarios). Future work: split-vote / network-partition / config-change scenarios to strengthen scoring.
  - `RecvRequestVote` only scored on granted votes — no reject path traced.
  - Many harness events (AdvanceCommitIndex, snapshot, response pairs) are emitted but out-of-scope under current `target_actions`.

## dqueue (PGo)
- Category: A (distributed message-passing, MPCal archetypes)
- `data/repositories/pgo/` — shared clone (also for locksvc/raftkvs)
- Go 1.23.5 installed at `/usr/local/go/` (Go 1.24.0 has a swissmap linkname bug)
- Harness orchestration:
  - `scripts/harness/dqueue/dqueue_trace_test.go` — duplicates `TestProducerConsumer`, wires `distsys.SetTraceRecorder(MakeLocalFileRecorder(f))` so every MPCal block emits JSON; `run.sh` copies it into the clone, removes on exit
  - `scripts/harness/dqueue/parse_traces.py` — converts PGo's native `csElements`+`.pc` format to `{"tag":"trace","event":{...,"action":...,"label":...}}`
  - `scripts/harness/dqueue/run.sh` — orchestration
  - `tla_eval/tasks/dqueue/INSTRUMENTATION.md`, `task.yaml` (wv section filled)
- Label→action mapping: `AConsumer.c1`→`Request`, `AProducer.p2`→`Produce`; other labels emitted with `action=null` for the agent's awareness
- Smoke: 1 scenario (1p/1c, 3 values), 19 events total, 3 `Request` + 3 `Produce` + 13 out-of-scope — `go test` passes
- WV smoke: **PASS** (workspace `wv-workspaces/20260417_142657_dqueue/`). Agent ran harness in workspace copy, 19 events, Step 0 contract compliance OK, target actions 3+3 windows present. Sample spec at `data/spec/dqueue/dqueue.tla` has `RemoveAt` symbol collision with `SequencesExt` (spec-side bug), blocking TLC — harness is verified end-to-end.

## locksvc (PGo)
- Not started

## raftkvs (PGo)
- Not started

## curp
- Not started

## zookeeper
- Not started
