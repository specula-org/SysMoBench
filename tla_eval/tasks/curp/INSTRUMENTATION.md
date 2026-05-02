# curp Harness — Instrumentation & Run Notes

Collects traces from Xline's CURP (Consistent Unordered Replication
Protocol) implementation via the `madsim` deterministic simulation
framework. Unlike the other tasks, curp ships with a purpose-built
trace generator — we only wrap it.

## Where the code lives

- **Xline clone**: `data/repositories/Xline/` (pinned to commit
  `71479a4fabdd12fe67eec0bee95e652976937541`)
- **Instrumentation patch**: `data/patches/xline-instrumented.patch`
  — applied by `run.sh` if the target tree is pristine. The patch:
  - adds `crates/trace/` (a new crate exposing `trace_event(name, args)`
    and a global `ClusterState` tracker)
  - instruments `raw_curp/mod.rs`, `curp_node.rs`,
    `client/unary/propose_impl.rs`, `cmd_worker/mod.rs`, and
    `conflict/uncommitted_pool.rs` to call `trace_event("<Action>", …)`
    at the 6 spec-relevant code paths
  - adds `crates/simulation/src/bin/trace_generator.rs` — a CLI that
    spins up a 3-node Madsim cluster, issues N client Puts+Gets, and
    writes NDJSON to the file named by `--trace-file`
  - adds `crates/curp/tla+/curp.tla` — the Gemini-generated spec this
    trace module mirrors
  - adds `.cargo/config.toml` with `rustflags = ["--cfg", "madsim"]`
    (Madsim is build-tag-gated)
- **Harness wrapper**: `scripts/harness/curp/run.sh`

No extra parser needed — the trace generator emits NDJSON where every
line is `{"event": "<name>", …state deltas}`, matching the schema the
tv-eval skill already consumes.

## Emitted events (match spec 1:1)

| Event (raw == spec)          | When                                               |
|------------------------------|----------------------------------------------------|
| `Propose`                    | Client initiates propose via `propose_impl.rs`     |
| `ProcessProposeLeader`       | Leader appends proposed cmd to its spec pool       |
| `ProcessProposeNonLeader`    | Follower appends proposed cmd to its spec pool     |
| `Commit`                     | Leader calls `commit` on the append-entries path   |
| `ProcessCommitMsg`           | Replica applies commit, removes from uncommitted   |
| `LeaderChange`               | Role transition (follower→leader / leader→follower) |

Event fields carry **deltas** (json-patch-style) against a running
`ClusterState` = `{term, leader, role, spec_pool, uncommitted_pool}`.
The tv-eval agent replays deltas from the trace start to reconstruct
full state at each window boundary.

## How to run

```bash
# from project root
bash scripts/harness/curp/run.sh
```

Writes `artifacts/curp/traces/trace_01.ndjson`. Typical numbers on the
default 3-node / 30-operation / 10%-packet-loss scenario:

| Action                    | Events |
|---------------------------|--------|
| `Propose`                 | ~30    |
| `ProcessProposeLeader`    | ~20    |
| `ProcessProposeNonLeader` | ~40    |
| `Commit`                  | ~20    |
| `ProcessCommitMsg`        | ~60    |
| `LeaderChange`            | 1-3    |
| **total**                 | ~180   |

First-time runtime ≈ 8-15 min (cold build of xline + rocksdb + madsim).
Subsequent runs ≈ 30 s (incremental). Set `CURP_NODES=5` or
`CURP_OP_COUNT=100` to scale.

### Environment overrides

| Var              | Default                          | Purpose                     |
|------------------|----------------------------------|-----------------------------|
| `TRACES_DIR`     | `artifacts/curp/traces`          | Where the NDJSON lands      |
| `REPO_PATH`      | `data/repositories/Xline`        | Xline clone root            |
| `PATCH_FILE`     | `data/patches/xline-instrumented.patch` | Auto-applied if absent |
| `CURP_NODES`     | `3`                              | Cluster size                |
| `CURP_OP_COUNT`  | `30`                             | Total Put+Get operations    |

## Rebuild from a clean checkout

```bash
git clone --recursive https://github.com/xline-kv/Xline.git data/repositories/Xline
cd data/repositories/Xline
git checkout 71479a4fabdd12fe67eec0bee95e652976937541
git submodule update --init --recursive
cd ../../..
bash scripts/harness/curp/run.sh   # applies the patch on first run
```

## Known gaps

- **Leader churn**. The default Madsim seed/timing gives only 1
  `LeaderChange` event. Longer scenarios or different seeds produce more.
- **Client requests are uniform Put/Get**. No conflicting-key or
  linearizability-stress patterns. Extend `trace_generator.rs` if the
  spec needs those exercised.
- **State deltas, not snapshots**. Unlike the spin/etcd NDJSON format,
  curp trace lines are patches. The tv-eval agent replays them from
  `Init` to compute pre/post snapshots — standard enough, but worth
  noting if downstream consumers expect full snapshots.
