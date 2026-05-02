# raftkvs Harness — Instrumentation & Run Notes

Collects traces from PGo's `raftkvs` (monolithic Raft-based key-value
store, 3 servers × 5 archetypes each + client) for TV action-window
validation.

## Where the code lives

- **PGo clone**: `data/repositories/pgo/` (shared with `dqueue`, `locksvc`)
- **System source**: `data/repositories/pgo/systems/raftkvs/` — in
  particular `raftkvs.tla` (MPCal), `bootstrap/server.go`,
  `bootstrap/client.go`
- **Our additions (copied + patched by `run.sh`, reverted on exit)**:
  - `scripts/harness/raftkvs/trace_hook.go` → `bootstrap/trace_hook.go`
    Declares `var TraceRecorder trace.Recorder` at the package level so
    the tracing opt-in can cross file boundaries without wrapper structs.
  - Patch to `bootstrap/server.go` inside `genResources`:
    ```go
    if TraceRecorder != nil {
        resourcesConfig = append(resourcesConfig, distsys.SetTraceRecorder(TraceRecorder))
    }
    ```
    (inserted directly above `return resourcesConfig`) — threads the
    recorder into all 5 server archetypes (`AServer`,
    `AServerRequestVote`, `AServerAppendEntries`,
    `AServerAdvanceCommitIndex`, `AServerBecomeLeader`).
  - Patch to `bootstrap/client.go`: appends `distsys.SetTraceRecorder(TraceRecorder)`
    to the sole `NewMPCalContext(...)` for `AClient`.
  - `scripts/harness/raftkvs/raftkvs_trace_test.go` →
    `systems/raftkvs/raftkvs_trace_test.go` — new test function
    `TestSafety_ThreeServers_WithTrace` that sets
    `bootstrap.TraceRecorder = trace.MakeLocalFileRecorder(f)` before
    spinning up servers + client (config `test-3-1.yaml`).
  - `scripts/harness/raftkvs/parse_traces.py` converts PGo-native NDJSON
    to SysMoBench-canonical `{"tag":"trace","event":{...}}`.
  - `scripts/harness/raftkvs/run.sh` — orchestrator.

## Label → spec action mapping

raftkvs spans 5 server archetypes + 1 client archetype. The label alone
is not enough: `AServer.handleMsg` dispatches on the received message's
`mtype`. Our parser reads the message record inside the event's reads and
disambiguates:

| MPCal label / condition                                     | Spec action                       |
|-------------------------------------------------------------|-----------------------------------|
| `AClient.sndReq`                                            | `ClientRequest`                   |
| `AServerRequestVote.serverRequestVoteLoop`                  | `ElectionTimeout`                 |
| `AServer.handleMsg` + `mtype=="rvq"`                        | `HandleRequestVoteRequest`        |
| `AServer.handleMsg` + `mtype=="apq"`                        | `HandleAppendEntriesRequest`      |
| `AServer.handleMsg` + `mtype=="app"`                        | `HandleAppendEntriesResponse`     |
| `AServer.handleMsg` + `mtype=="rvp"`                        | — (out of scope)                  |
| `AServer.handleMsg` + `mtype=="cpq"/"cgq"/"cpp"/"cgp"`      | — (client request/response, not the vote/replication RPCs) |
| `AServerBecomeLeader.serverBecomeLeaderLoop`                | — (candidate→leader promotion; out of scope) |
| `AServerAppendEntries.appendEntriesLoop`                    | — (leader-side send loop)         |
| all other labels                                            | — (internal)                      |

mtype values are declared in `raftkvs.tla` (`RequestVoteRequest == "rvq"`,
etc.). PGo serializes TLA records as `("mtype") :> ("rvq") @@ ...` in
the event's reads/writes — the parser's regex handles that form.

## Noise filter

Two labels fire tens of thousands of times per run with no spec-relevant
state change (they poll-wait on local `newCommitIndex`):

- `AServerAdvanceCommitIndex.applyLoop`
- `AServerAdvanceCommitIndex.serverAdvanceCommitIndexLoop`

Both are dropped at parse time. If a future TV upgrade cares about apply
loop ordering, remove them from `NOISE_LABELS` in `parse_traces.py`.

## How to run

```bash
# from project root
bash scripts/harness/raftkvs/run.sh
```

Writes `artifacts/raftkvs/traces/trace_01.ndjson`. The file contains
roughly:

| Action                            | Events (typical 3-server run) |
|-----------------------------------|-------------------------------|
| `ClientRequest`                   | 6                             |
| `ElectionTimeout`                 | 1-3                           |
| `HandleRequestVoteRequest`        | 2                             |
| `HandleAppendEntriesRequest`      | 140-170                       |
| `HandleAppendEntriesResponse`     | 140-170                       |
| in-scope subtotal                 | ~300                          |
| internal labels (kept for context)| ~700                          |
| **total lines**                   | **~1000**                     |

Runtime ≈ 7-10 s. Exit cleanly; `trap` in `run.sh` restores
`bootstrap/server.go` and `bootstrap/client.go` and removes
`trace_hook.go` and `raftkvs_trace_test.go`.

### Environment overrides

| Var          | Default                            | Purpose                      |
|--------------|------------------------------------|------------------------------|
| `TRACES_DIR` | `artifacts/raftkvs/traces`         | Where the NDJSON lands       |
| `REPO_PATH`  | `data/repositories/pgo`            | PGo clone root               |
| `GOROOT`     | `/usr/local/go`                    | Go toolchain (≥ 1.23)        |

## Known gaps

- **Single scenario**. Only 3-server / 1-client / 3 request-pairs. Richer
  scenarios (5-server, partition, leader crash) would produce more
  ElectionTimeout windows — currently 1-3 per run, scoring is sparse.
- **No reject-vote coverage**. The scenario doesn't trigger vote rejects;
  `HandleRequestVoteRequest` only scored on grant paths.
- **Config-change actions**. The spec's full contract includes
  `AddServer`/`RemoveServer`; these are not in the `target_actions` scope
  for this task.
- **Leader-election serial**. In a 3-server run with a fixed seed,
  election often elects server 1 immediately; occasional second elections
  happen during leader-timeout drift but are scenario-dependent.
