# RedisRaft Harness — Instrumentation & Run Notes

Collects execution traces from RedisRaft's embedded Raft library
(`deps/raft/src/raft_server.c`) for TV (transition validation) of
generated TLA+ specs.

## Where the code lives

- **Specula case dir** (canonical instrumented source, shared with
  Specula's own harness-gen pipeline):
  `/home/ubuntu/Specula/case-studies/redisraft/`
  Structure:
  - `harness/run.sh` — full build+trace driver (used as our inner harness)
  - `harness/apply.sh` — patch applicator
  - `harness/patches/instrumentation.patch` — trace call-sites against `raft_server.c`
  - `harness/src/tla_trace.{h,c}` — NDJSON emitter
  - `harness/src/test_trace.c` — three CuTest scenarios
  - `artifact/redisraft/` — cloned github.com/RedisLabs/redisraft
  - `traces/` — inner harness output
- **SysMoBench wrapper**: `scripts/harness/redisraft/run.sh` — thin shell that
  calls the Specula harness and mirrors `.ndjson` files into
  `artifacts/redisraft/traces/`.

## Emitted events

Lines tagged `"tag":"trace"` carry per-action state snapshots; lines
tagged `"tag":"config"` carry one-time server-ID mapping. Event schema:

```json
{"tag":"trace","ts":"<ns>","event":{"name":"<Event>","nid":"s1","state":{...},"msg":{...}}}
```

### Instrumented events

| Event name (raw)              | Where                                    | Capture level      |
|-------------------------------|------------------------------------------|--------------------|
| `Timeout`                     | `raft_become_candidate()` after term++   | Full               |
| `BecomeLeader`                | `raft_become_leader()` after noop append | Full               |
| `HandleRequestVoteRequest`    | `raft_recv_requestvote()` after response | Full + `msg`       |
| `HandleRequestVoteResponse`   | `raft_recv_requestvote_response()`       | Weak + `msg`       |
| `ClientRequest`               | `raft_recv_entry()` NORMAL path          | Full               |
| `ProposeAddServer` / `ProposeRemoveServer` | `raft_recv_entry()` config paths | Full + target     |
| `HandleAppendEntriesRequest`  | `raft_recv_appendentries()` after log+commit | Full + `msg`  |
| `HandleAppendEntriesResponse` | `raft_recv_appendentries_response()`     | Weak + `msg`       |
| `AdvanceCommitIndex`          | after `raft_set_commit_idx()`            | Commit             |
| `TakeSnapshot`                | `raft_end_snapshot()`                    | Full + snapshot    |
| `HandleInstallSnapshotRequest`| `raft_begin_load_snapshot()`             | Weak + `msg`       |
| `EndLoadSnapshot`             | `raft_end_load_snapshot()`               | Full + snapshot    |

Full details: `/home/ubuntu/Specula/case-studies/redisraft/harness/INSTRUMENTATION.md`.

### Raw event → spec-action mapping

The redisraft prompt requires 6 spec actions. TV windowing translates:

| Trace event                  | Spec action(s)                          |
|------------------------------|-----------------------------------------|
| `Timeout`                    | `ElectionTimeout`, `BecomeCandidate`    |
| `BecomeLeader`               | `BecomeLeader`                          |
| `HandleRequestVoteRequest`   | `RecvRequestVote`                       |
| `HandleAppendEntriesRequest` | `RecvAppendEntries`                     |
| `ClientRequest`              | `LogAppend`                             |
| (others above)               | out of scope — listed as Future Work    |

`Timeout` captures both the election-timeout firing AND the
become-candidate transition in one event (RedisRaft does both
atomically). Agent must decide per-action windowing: a single
`Timeout` can produce windows for either action from its pre-state.

## Running the harness

```bash
# from SysMoBench project root
bash scripts/harness/redisraft/run.sh
```

Writes `basic_consensus.ndjson`, `leader_failover.ndjson`,
`snapshot_basic.ndjson` (~74 events total) into
`artifacts/redisraft/traces/`.

Build+run is fast — ~5 seconds on a warm tree, ~15 seconds cold.

### Environment overrides

| Var          | Default                                         | Purpose                      |
|--------------|-------------------------------------------------|------------------------------|
| `TRACES_DIR` | `artifacts/redisraft/traces`                    | Where NDJSON files land      |
| `REPO_PATH`  | `/home/ubuntu/Specula/case-studies/redisraft`   | Specula case dir (or copy)   |

When the tv-eval agent runs in a workspace copy, pass
`REPO_PATH=<workspace>/repo` so it uses its own sandboxed tree.

## Coverage

Three scenarios baked into `test_trace.c`:

| Scenario          | Lines | Covers                                                |
|-------------------|-------|--------------------------------------------------------|
| `basic_consensus` | 17    | 3-node cluster: election + first log replication       |
| `leader_failover` | 27    | Leader dies, followers elect new leader, resume        |
| `snapshot_basic`  | 30    | Snapshot taken, restart with snapshot load             |

Per-action coverage from the 3 traces:

| Action             | Events in traces                     |
|--------------------|--------------------------------------|
| `ElectionTimeout`  | 4 (`Timeout`)                        |
| `BecomeCandidate`  | 4 (`Timeout` — same events)          |
| `BecomeLeader`     | 4 (`BecomeLeader`)                   |
| `RecvRequestVote`  | 8 (`HandleRequestVoteRequest`)       |
| `RecvAppendEntries`| 18 (`HandleAppendEntriesRequest`)    |
| `LogAppend`        | 5 (`ClientRequest`)                  |

## Known gaps

- Only 3 scenarios, ~74 events total. Mutex/rwmutex yield hundreds/thousands.
  More scenarios (split-vote, network partition, config change) would
  strengthen scoring confidence but are out of scope for bootstrap.
- `RecvRequestVote` only scored on granted votes — no reject path in traces.
- Config-change events (`ProposeAddServer` / `ProposeRemoveServer`) are
  emitted by the harness but not in the current spec action scope.
