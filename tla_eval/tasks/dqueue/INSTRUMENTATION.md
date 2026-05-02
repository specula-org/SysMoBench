# dqueue Harness — Instrumentation & Run Notes

Collects execution traces from the PGo-compiled `dqueue` system
(distributed producer-consumer queue, MPCal archetype-based) for TV
transition validation.

## Where the code lives

- **PGo clone**: `data/repositories/pgo/` (shared with `locksvc` and
  `raftkvs`). Upstream `github.com/DistCompiler/pgo`.
- **System source**: `data/repositories/pgo/systems/dqueue/dqueue.{go,tla}`
- **Our trace-enabled test**: `scripts/harness/dqueue/dqueue_trace_test.go`
  — copied into the clone by `run.sh`, removed on exit. A duplicate of
  `TestProducerConsumer` that wires a `trace.Recorder` via
  `distsys.SetTraceRecorder(...)` so every MPCal block emits a JSON line.
- **Harness wrapper**: `scripts/harness/dqueue/run.sh`
- **Parser**: `scripts/harness/dqueue/parse_traces.py` — converts PGo's
  native NDJSON to the canonical `{"tag":"trace","event":{...}}` schema.

## What PGo emits per MPCal block

PGo's `trace.Recorder` writes one JSON line per archetype step:

```json
{"archetypeName":"AConsumer","self":"1","isAbort":false,
 "startTime":"...","endTime":"...",
 "csElements":[
   {"tag":"read","name":{"name":".pc","self":"1"},"value":"\"AConsumer.c1\""},
   {"tag":"write","indices":["0"],"name":{"prefix":"AConsumer","name":"net","self":"1"},
    "value":"1"},
   {"tag":"write","name":{"name":".pc","self":"1"},
    "oldValue":"\"AConsumer.c1\"","value":"\"AConsumer.c2\""}
 ],
 "clock":[[...vector clock...]]}
```

Key fields our parser uses:
- `isAbort=true` lines are dropped (failed PlusCal step attempts).
- The `.pc` write's `oldValue` is the **MPCal label that just executed**
  (e.g. `"AConsumer.c1"`); `value` is the next label.
- Non-`.pc` reads/writes describe state touched during the block.

## Label → spec action mapping (dqueue)

From `dqueue.tla` (the canonical PlusCal translation):

| MPCal label        | Semantic                                | Spec action |
|--------------------|------------------------------------------|-------------|
| `AConsumer.c`      | `while(TRUE)` check                      | — (internal) |
| `AConsumer.c1`     | `net[PRODUCER] := self` (ask for data)   | **`Request`** |
| `AConsumer.c2`     | `proc := net[self]` (receive data)       | — (internal) |
| `AProducer.p`      | `while(TRUE)` check                      | — (internal) |
| `AProducer.p1`     | `requester := net[self]` (receive req)   | — (internal) |
| `AProducer.p2`     | `net[requester] := stream++` (send data) | **`Produce`** |

Events with no mapped action carry `"action": null` in the output — the
tv-eval windowing step is free to drop them.

## Output schema (after `parse_traces.py`)

One NDJSON line per non-aborted MPCal step:

```json
{"tag":"trace","event":{
  "name":"Request",             // or the raw label if unmapped
  "action":"Request",           // null when out of scope
  "label":"AConsumer.c1",
  "next_label":"AConsumer.c2",
  "pid":"1",                    // MPCal self
  "archetype":"AConsumer",
  "clock":[[["AConsumer","1"],2]],
  "start":"2026-04-17T14:24:08.289630451Z",
  "end":"2026-04-17T14:24:08.290694458Z",
  "reads":{"net[0]":1},
  "writes":{"AConsumer.net[0]":1,".pc→":"AConsumer.c2"}
}}
```

## How to run

```bash
# from project root
bash scripts/harness/dqueue/run.sh
```

Writes `artifacts/dqueue/traces/trace_01.ndjson` (~19 events: 3 `Request`
+ 3 `Produce` + 13 internal labels). Runtime ≈ 2 seconds.

### Environment overrides

| Var          | Default                          | Purpose                      |
|--------------|----------------------------------|------------------------------|
| `TRACES_DIR` | `artifacts/dqueue/traces`        | Where the NDJSON lands       |
| `REPO_PATH`  | `data/repositories/pgo`          | PGo clone root (or workspace copy) |
| `GOROOT`     | `/usr/local/go`                  | Go toolchain (≥ 1.23)        |

## Known coverage gaps

- **Single scenario**, 3 produced values → only 3 Request + 3 Produce
  windows. A richer scenario suite (multiple consumers, buffer-full
  backpressure, producer-crash) would strengthen scoring. Out of scope
  for bootstrap.
- **Partial state in each event**. PGo emits the fields the block
  touched, not the full global snapshot. The tv-eval agent must
  accumulate `writes` across events to reconstruct running state.
- **No retry/failure scoring**. `isAbort=true` steps are dropped; a
  failure-aware spec would have nothing to be scored against.
- **Vector clocks are present but unused** by the current tv-eval
  skill (it uses start/end monotonic order like the distributed
  NDJSON cases). A future upgrade could use `clock` for stronger
  happens-before reasoning.
