# locksvc Harness — Instrumentation & Run Notes

Collects traces from PGo's `locksvc` (centralized lock service with one
server and N clients) for TV transition validation.

## Where the code lives

- **PGo clone**: `data/repositories/pgo/` (shared with `dqueue` and `raftkvs`)
- **System source**: `data/repositories/pgo/systems/locksvc/locksvc.{go,tla}`
- **Trace-enabled test**: `scripts/harness/locksvc/locksvc_trace_test.go`
  — duplicates the shape of `testNClients(3)` but adds
  `distsys.SetTraceRecorder(...)` on both server and client contexts.
  Reuses `matcherResource`, `whileHoldingLock`, `addressFn` from the
  existing `locksvc_test.go` (same external test package).
- **Harness wrapper**: `scripts/harness/locksvc/run.sh`
- **Parser**: `scripts/harness/locksvc/parse_traces.py`

## Emitted events → spec action mapping

| MPCal label                          | Condition                           | Spec action |
|--------------------------------------|-------------------------------------|-------------|
| `AClient.acquireLock`                | always                              | **`ClientLockRequest`** |
| `AClient.criticalSection`            | always                              | **`ClientCriticalSection`** |
| `AClient.unlock`                     | always                              | **`ClientUnlockRequest`** |
| `AServer.serverRespond`              | wrote `GrantMsg (=3)` to `network[c]` | **`ServerGrantLock`** |
| `AServer.serverRespond`              | Lock queued behind holder (no grant write) | — (internal) |
| `AServer.serverLoop`, `serverReceive` | always                             | — (internal) |

`serverRespond` is the tricky case. It fires for BOTH Lock and Unlock
messages:
- Lock + empty queue → sends `GrantMsg` to requester, appends to q → `ServerGrantLock`
- Lock + non-empty queue → just appends to q, no network write → internal
- Unlock + non-empty remaining q → sends `GrantMsg` to next waiter → `ServerGrantLock`
- Unlock + empty remaining q → no network write → internal

Our parser disambiguates by scanning `writes` for any `network[.*]=3`.

## Output

One NDJSON line per non-aborted MPCal step:

```json
{"tag":"trace","event":{
  "name":"ServerGrantLock",
  "action":"ServerGrantLock",
  "label":"AServer.serverRespond",
  "pid":"0",
  "archetype":"AServer",
  "reads":{"AServer.msg":"...","AServer.q":"<<>>"},
  "writes":{"AServer.network[3]":3,"AServer.q":"<<3>>"},
  "clock":[...],
  "start":"...","end":"..."
}}
```

## How to run

```bash
# from project root
bash scripts/harness/locksvc/run.sh
```

Writes `artifacts/locksvc/traces/trace_01.ndjson` (~22-27 events, of
which 3 `ClientLockRequest` + 3 `ServerGrantLock` + 2-3 `ClientCriticalSection`
+ 2-3 `ClientUnlockRequest`). Runtime ≈ 5 seconds (includes a 3-second
"wait for prior tests to settle" sleep inherited from locksvc's test
pattern).

### Environment overrides

| Var          | Default                             | Purpose                      |
|--------------|-------------------------------------|------------------------------|
| `TRACES_DIR` | `artifacts/locksvc/traces`          | Where the NDJSON lands       |
| `REPO_PATH`  | `data/repositories/pgo`             | PGo clone root               |
| `GOROOT`     | `/usr/local/go`                     | Go toolchain (≥ 1.23)        |

## Known coverage gaps

- **Small scenario**. 3 clients mean at most 3 of each action. Re-scoring
  across multiple runs / multiple scenarios (e.g. Test5ClientsWithTrace,
  Test20ClientsWithTrace) would strengthen statistics.
- **CS/Unlock count variability**. Goroutine scheduling sometimes leaves
  the last client's unlock un-observed (the test returns once the counter
  hits N, before all unlocks finalize). Count varies 2-3 per run.
- **serverRespond grant misclassification risk**. Our parser keys on the
  GrantMsg integer constant (3). If upstream locksvc.tla renumbers
  `GrantMsg`, update `GRANT_MSG` in `parse_traces.py`.
- **No failure/retry paths traced**. `isAbort=true` events are dropped;
  contention outcomes are only the successful paths.
