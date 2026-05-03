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
  - `tla_eval/tasks/mutex/task.yaml` — `tv.repo_path`, `tv.harness.*` fields populated
- Events emitted: `Lock`, `TryLock`, `Unlock` (mapped to spec `AcquireLock`, `TryAcquireLock`, `ReleaseLock`)
- Smoke: 20 trace files, 260 events total, all 3 action types present, ktest `1 passed; 0 failed`
- TV smoke: **PASS** (workspace `tv-workspaces/20260417_102949_mutex/`). Agent ran harness fresh from copied repo, produced 20 traces + 260 windows, Step 0 contract check 100% compliant, per-action window counts: `AcquireLock=82`, `TryAcquireLock=48`, `ReleaseLock=130`. TLC couldn't run because the sample spec (`data/spec/mutex/mutex.tla`) has two unrelated `Spec`-operator bugs (`vars` vs `Vars`; `\A t` scope collision); agent's manual check against action bodies concluded ~100% pass on all three. Harness is verified end-to-end; spec-side fix is out of scope for this task.
- Open issues:
  - Failed `try_lock` attempts are NOT traced (only successes). Spec's `TryAcquireLock` scoring will only cover successes.
  - Docker run leaves root-owned build artifacts in `artifacts/spin/` and in any `workspace/repo/target/` the TV agent creates; requires sudo or chmod to clean up. `launch_tv_eval.sh` now handles the workspace-side cleanup gracefully.
  - `cargo install --path osdk` needs `--locked` (run.sh handles it); upstream Makefile target omits the flag.

## ringbuffer
- Category: B in spirit (lock-free SPSC), executed via Asterinas ktest (single-threaded simulation) rather than true timebox rdtsc.
- Uses shared `artifacts/spin/` clone with **ringbuffer-specific overlays**:
  - `ostd/src/trace_support.rs` + `ostd/src/lib.rs` pub-mod export: lets kernel-level code emit serial bytes via public OSTD helpers (avoids `pub(crate)` access to `IN_BOOTSTRAP_CONTEXT` / `arch::serial`).
  - `kernel/src/util/ring_buffer_trace.rs` (641 lines, replaces the 541-line stub) with `test_rb_trace_randomized` ktest.
  - `vendored/core2-0.4.0` + `vendored/libflate-2.2.1` + `vendored/libflate_lz77-2.2.0` + patched `osdk/src/base_crate/Cargo.toml.template` injecting `[patch.crates-io]` into every generated test-base (redirecting the three supply-chain-rotten crates plus `ostd`/`osdk-test-kernel`/`osdk-frame-allocator`/`osdk-heap-allocator` to in-workspace paths to avoid duplicate `#[global_allocator]` errors).
  - Patched `tools/qemu_args.sh` honoring `OVMF_PATH` so host OVMF files work.
  - Materialized `test/build/initramfs.cpio.gz` (not the original dangling nix-store symlink).
- Harness is **host-side** (not docker):
  - `scripts/harness/ringbuffer/run.sh` — invokes the patched cargo-osdk with `CARGO_RESOLVER_INCOMPATIBLE_RUST_VERSIONS=fallback` + `OVMF=off`; rustup's `nightly-2025-02-01` toolchain drives the build.
  - `scripts/harness/ringbuffer/parse_traces.py` — splits on `=== TRACE_RANDOM_<n> ===` banners.
  - `tla_eval/tasks/ringbuffer/INSTRUMENTATION.md` documents the one-time setup (install qemu/ovmf/grub tools, rebuild cargo-osdk inside docker with matched `-v $(pwd):$(pwd)` mount, materialize initramfs).
- Why host-side: `cargo osdk test` generates a fresh test-base workspace that re-resolves from crates.io. Docker's rustc 1.86-nightly can't build libflate 2.3.0 / time 0.3.47 / fixed 1.31.0; core2 0.4.0 is yanked. Our host has the same rustc via rustup but with modern tooling + the patched osdk template pinning the supply-chain-affected crates.
- Smoke: 5 scenarios, ~80-95 events (Push/PushSlice/Pop/PopSlice/Create/Split), `test result: ok`
- TV smoke: **PASS** (workspace `tv-workspaces/20260418_080338_ringbuffer/`). Agent ran the harness from the workspace copy, got 93 events, Step 0 contract compliance confirmed (correctly excluded `success=false` and `Create`/`Split` as Type B out-of-scope). Sample spec at `data/spec/ringbuffer/ringbuffer.tla` has an undefined `vars` identifier in its `Fairness` definition (4 SANY errors) — spec-side bug, harness verified.
- Open issues:
  - Host-side only, which diverges from the docker-based mutex/rwmutex flow. Future cleanup: unify when Asterinas upstream either unyaks core2 or pins to a non-yanked version.
  - Single-threaded simulation (no genuine producer/consumer race); success=false paths exercised but may need spec extension to score.
  - One-time setup: rebuild cargo-osdk in docker (~10s) + materialize initramfs (~5 min cold nix build, cached afterwards).

## rwmutex
- Category: B (concurrent kernel primitive; uses ktest+serial, single-threaded kernel context)
- `artifacts/spin/`: shared Asterinas clone
- Instrumentation already present: `ostd/src/sync/rwmutex_trace.rs` (wired in `mod.rs`, same pattern as mutex)
- Harness orchestration added this pass:
  - `scripts/harness/rwmutex/run.sh`
  - `scripts/harness/rwmutex/parse_traces.py` (same seq==0 split strategy as mutex)
  - `tla_eval/tasks/rwmutex/INSTRUMENTATION.md`
  - `tla_eval/tasks/rwmutex/task.yaml` tv section populated
- Events: `ReadLock`, `TryReadLock`, `WriteLock`, `TryWriteLock`, `ReadUnlock`, `WriteUnlock` (+ UpreadLock/TryUpreadLock/UpreadUnlock/UpgradeLock/TryUpgradeLock available but not exercised by current ktest)
- Action mapping: Read/TryRead → `AcquireReadLock`, Write/TryWrite → `AcquireWriteLock`, Upread/TryUpread → `AcquireUpReadLock` (unused), all `*Unlock` → `ReleaseLock`
- Smoke: 100 trace files, 2230 events total, ktest `1 passed; 0 failed`
- TV smoke: in progress (see latest `tv-workspaces/*_rwmutex/`)
- Known coverage gaps:
  - `test_rwmutex_trace` doesn't exercise upread/upgrade — `AcquireUpReadLock` will have 0 scoring windows
  - Failed `try_*` attempts not traced; only successful paths scored

## redisraft
- Category: A (distributed Raft, NDJSON single-writer trace)
- Reuses Specula case dir at `/home/ubuntu/Specula/case-studies/redisraft/` (symlink-free). Canonical instrumented raft library + harness already present from Specula.
- Harness orchestration added this pass:
  - `scripts/harness/redisraft/run.sh` — thin wrapper that invokes the Specula `harness/run.sh` and mirrors `.ndjson` outputs to `artifacts/redisraft/traces/`
  - `tla_eval/tasks/redisraft/INSTRUMENTATION.md` — event schema, action mapping, coverage tables
  - `tla_eval/tasks/redisraft/task.yaml` — `tv.harness.*` populated, including full `trace_action_map` for 5 events
- Events emitted (via Specula `tla_trace.c` patch into `deps/raft/src/raft_server.c`): `Timeout`, `BecomeLeader`, `HandleRequestVoteRequest`, `HandleAppendEntriesRequest`, `ClientRequest` plus out-of-scope: response events, `AdvanceCommitIndex`, `TakeSnapshot`, `ProposeAddServer`, `ProposeRemoveServer`, install/end-snapshot
- Action mapping: `Timeout`→`ElectionTimeout`+`BecomeCandidate` (same event, two spec actions); `HandleRequestVoteRequest`→`RecvRequestVote`; `HandleAppendEntriesRequest`→`RecvAppendEntries`; `ClientRequest`→`LogAppend`
- Smoke: 3 scenarios (basic_consensus/leader_failover/snapshot_basic), 74-78 events total, all tests `ok 1`
- TV smoke: **PASS** (workspace `tv-workspaces/20260417_132542_redisraft/`). Agent ran harness fresh from workspace copy, produced 43 windows across all 6 actions. Sample spec at `data/spec/redisraft/redisraft.tla` has a parse error (`_` as identifier in `SortedSeq`), blocking TLC. Agent's manual semantic check: `ElectionTimeout=4/4`, `BecomeCandidate=4/4`, `RecvRequestVote=8/8` would pass; `BecomeLeader=0/4` (spec's `UNCHANGED log` doesn't model the leader-elected noop append), `RecvAppendEntries`/`LogAppend` 0 (spec's `Entry={}` model gap). All blockers are spec-side, not harness.
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
  - `tla_eval/tasks/dqueue/INSTRUMENTATION.md`, `task.yaml` (tv section filled)
- Label→action mapping: `AConsumer.c1`→`Request`, `AProducer.p2`→`Produce`; other labels emitted with `action=null` for the agent's awareness
- Smoke: 1 scenario (1p/1c, 3 values), 19 events total, 3 `Request` + 3 `Produce` + 13 out-of-scope — `go test` passes
- TV smoke: **PASS** (workspace `tv-workspaces/20260417_142657_dqueue/`). Agent ran harness in workspace copy, 19 events, Step 0 contract compliance OK, target actions 3+3 windows present. Sample spec at `data/spec/dqueue/dqueue.tla` has `RemoveAt` symbol collision with `SequencesExt` (spec-side bug), blocking TLC — harness is verified end-to-end.

## locksvc (PGo)
- Category: A (centralized lock service, MPCal 1 server + N clients)
- Same PGo clone at `data/repositories/pgo/`
- Harness orchestration:
  - `scripts/harness/locksvc/locksvc_trace_test.go` — shadow of `testNClients(3)` adding `SetTraceRecorder` on server + client contexts, reusing `matcherResource`/`whileHoldingLock`/`addressFn` from the in-tree external test package
  - `scripts/harness/locksvc/parse_traces.py` — same shape as dqueue's; disambiguates `AServer.serverRespond` by whether it wrote `GrantMsg (=3)` to `network[x]`
  - `scripts/harness/locksvc/run.sh` — copies test into clone, runs go test, parses, reverts
  - `tla_eval/tasks/locksvc/INSTRUMENTATION.md`, `task.yaml` (tv section filled with 4-way trace_action_map)
- Label→action mapping: `AClient.acquireLock`→`ClientLockRequest`, `AClient.criticalSection`→`ClientCriticalSection`, `AClient.unlock`→`ClientUnlockRequest`, `AServer.serverRespond` (with `network[*]=3` write)→`ServerGrantLock`
- Smoke: 3-client scenario, ~22-27 events, all 4 target actions represented
- TV smoke: **PASS** (workspace `tv-workspaces/20260417_143727_locksvc/`). 25 events, 3 of each action. Sample spec has two defects (line 28 mixes record/function syntax; `ServerGrantLock` uses double-primed `network''`) — spec-side bugs, harness verified.

## raftkvs (PGo)
- Category: A (monolithic Raft-based KV store, 5 server archetypes + 1 client archetype)
- Same PGo clone at `data/repositories/pgo/`
- Harness orchestration (heaviest PGo case — requires patching bootstrap):
  - `scripts/harness/raftkvs/trace_hook.go` — new `bootstrap/` file with `var TraceRecorder trace.Recorder`
  - `run.sh` Python-patches `bootstrap/server.go` (adds `SetTraceRecorder` append inside `genResources`) and `bootstrap/client.go` (adds option to `AClient`'s `NewMPCalContext`); trap-restores on exit
  - `scripts/harness/raftkvs/raftkvs_trace_test.go` — sets `bootstrap.TraceRecorder` then spins up 3 servers + 1 client via `configs/test-3-1.yaml`
  - `scripts/harness/raftkvs/parse_traces.py` — disambiguates `AServer.handleMsg` by parsing `mtype` from PGo's TLA-record reads (`("mtype") :> ("rvq")` etc.); drops 30K+ tight-polling events (`applyLoop`, `serverAdvanceCommitIndexLoop`) at parse time
  - `tla_eval/tasks/raftkvs/INSTRUMENTATION.md`, `task.yaml` (tv section filled with 5-way trace_action_map)
- Label+mtype→action mapping: `AClient.sndReq`→`ClientRequest`; `AServerRequestVote.serverRequestVoteLoop`→`ElectionTimeout`; `AServer.handleMsg` {rvq→`HandleRequestVoteRequest`, apq→`HandleAppendEntriesRequest`, app→`HandleAppendEntriesResponse`}
- Smoke: 3-server/1-client/3-request-pair scenario, ~1039 events (post-filter): 2 ElectionTimeout, 2 HandleRequestVoteRequest, 144 HandleAppendEntriesRequest, 144 HandleAppendEntriesResponse, 7 ClientRequest
- TV smoke: **PARTIAL** (workspace `tv-workspaces/20260417_145041_raftkvs/`). Agent short-circuited at Step 1 (spec compile) before running the harness — sample spec has parse error at line 198 (`HandleRequestVoteRequest`), plus double-primed vars and a broken `∀` assignment. Harness itself is verified via direct run (`bash scripts/harness/raftkvs/run.sh` — 1039 events, 5 target actions present).

## curp
- Category: A (distributed consensus, Madsim-simulated 3-node CURP cluster)
- `data/repositories/Xline/` — commit `71479a4fabdd12fe67eec0bee95e652976937541` + 2 submodules (curp-proto, xline-proto) + `data/patches/xline-instrumented.patch` auto-applied by run.sh
- The patch ships a complete tracing stack:
  - new `crates/trace/` crate with a global `ClusterState` tracker and `trace_event(name, args)` function
  - new `crates/simulation/src/bin/trace_generator.rs` — CLI that spins up a 3-node Madsim cluster and drives N ops
  - instrumentation calls into `raw_curp/mod.rs`, `curp_node.rs`, `client/unary/propose_impl.rs`, `cmd_worker/mod.rs`, `conflict/uncommitted_pool.rs`
  - `.cargo/config.toml` + Cargo.toml additions to enable `--cfg madsim`
- Harness orchestration (thin wrapper only):
  - `scripts/harness/curp/run.sh` — auto-applies patch if `crates/trace/` missing; runs `cargo run --release --bin trace_generator --nodes $N --op-count $M --trace-file …`
  - `tla_eval/tasks/curp/INSTRUMENTATION.md`, `task.yaml` (tv section filled; trace_action_map is identity since event names already match spec)
- Events emit with canonical names: `Propose`, `ProcessProposeLeader`, `ProcessProposeNonLeader`, `Commit`, `ProcessCommitMsg`, `LeaderChange`. No converter needed.
- Smoke: 3-node / 30-op / 10% packet-loss default — ~180 events, all 6 actions present
- TV smoke: **PASS** (workspace `data/repositories/Xline/tv-workspaces/20260417_145848_curp/`). Agent ran harness fresh from workspace copy, 179 events, all 6 actions have windows (26 Propose + 23 ProcessProposeLeader + 37 ProcessProposeNonLeader + 23 Commit + 69 ProcessCommitMsg + 1 LeaderChange). Sample spec has `Keys(cmd)` where it should be `Keys[cmd]` — spec-side bug, harness verified. Agent also flagged a genuine instrumentation gap: `uncommitted_pool` is not cleared during `ProcessCommitMsg` in the trace, causing stale pre-state for ProcessProposeLeader windows 2–23. Real finding — future improvement.
- Open issues:
  - `uncommitted_pool` staleness during `ProcessCommitMsg` (see above). Trace patch needs a fix: emit removal when leader commits.
  - Leader churn is minimal (1 `LeaderChange` in default seed). Longer scenarios / seed variation would improve coverage.
  - First-time build is ~8–15 min (xline + rocksdb + madsim cold compile); incremental ≈ 30 s.

## zookeeper
- Category: A (distributed, ZooKeeper FastLeaderElection scope)
- `data/repositories/Remix/` (upstream `github.com/Lingzhi-Ouyang/Remix`, HEAD `81869f1`)
- Original upstream patches (`remix_ndjson_output.patch`, `remix_ndjson_output_complete.patch`) targeted a pre-rewrite version of Remix and no longer apply — they left NDJSON writer fields but no emit points.
- **New SysMoBench patch**: `data/patches/remix_ndjson_output_v2.patch`. 3 emit points mapped directly to the 3 target spec actions:
  - `Notification` — `ReplayService.offerElectionMessage` (after `sendingSubnode.setState(SENDING)`)
  - `HandleNotification` — `ElectionMessageExecutor.releaseMessage` (after receiving subnode enters PROCESSING — new receive-side hook that upstream's patch didn't have)
  - `BecomeLeader` — `ReplayService.updateLeaderElectionState` when `state == LEADING`
  - Plus `LocalEvent` inherited from upstream hunk 8 (out of scope; filtered by tv-eval)
- Harness orchestration:
  - `scripts/harness/zookeeper/run.sh` — auto-applies v2 patch if `writeNdjsonEvent` helper absent; incrementally builds via `scripts/build.sh`; runs `replay.sh demo` with `NDJSON_OUTPUT` env; waits for nohup java via `pgrep -f '^java .*zookeeper-ensemble-jar'` (anchored so it doesn't self-match)
  - `tla_eval/tasks/zookeeper/INSTRUMENTATION.md`, `task.yaml` (tv section filled)
- Smoke: 3 demo scenarios, 77-78 events — 32 Notification + 19-20 HandleNotification + 2 BecomeLeader + ~24 out-of-scope LocalEvent
- TV smoke: **PASS** (workspace `tv-workspaces/20260418_061627_zookeeper/`). Agent ran harness in workspace copy (78 events, all 3 target actions). Sample spec at `data/spec/zookeeper/zookeeper.tla` has a symbol collision — `Notification` is defined both as a 0-arg record type (line 24) and a 2-arg action (line 70). SANY rejects with 3 semantic errors; TLC cannot load. Harness itself verified end-to-end.
- Open issues:
  - Demo replay has no vote-reject scenarios; `HandleNotification` only scored on grant path.
  - Only 2 `BecomeLeader` windows per run (one per successful election); richer scenarios would need upgrades to `generator/generate_traces.sh`.
