# zookeeper Harness — Instrumentation & Run Notes

Collects ZooKeeper FastLeaderElection traces via the Remix model-trace
replayer (a deterministic scheduler running a 3-node ZooKeeper ensemble
against model-level traces from `traces/demo/`).

## Where the code lives

- **Remix clone**: `data/repositories/Remix/` (upstream
  `github.com/Lingzhi-Ouyang/Remix`, HEAD `81869f1`)
- **Instrumentation patch**: `data/patches/remix_ndjson_output_v2.patch`
  — authored by SysMoBench. The upstream patches
  (`remix_ndjson_output.patch`, `remix_ndjson_output_complete.patch`)
  target an older Remix version and no longer apply cleanly; `v2` is the
  port of the NDJSON idea to the current file layout, with three emit
  points that map directly to the three target spec actions.
- **Harness wrapper**: `scripts/harness/zookeeper/run.sh`

## Emit points (all in `checker/server/`)

| Spec action           | File + location                                                                                              | Trigger                                                                               |
|-----------------------|--------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------|
| `Notification`        | `ReplayService.java::offerElectionMessage`, after `sendingSubnode.setState(SubnodeState.SENDING)`            | Any FLE message a node is offering for delivery (send side)                          |
| `HandleNotification`  | `executor/ElectionMessageExecutor.java::releaseMessage`, after `subnode.setState(SubnodeState.PROCESSING)`    | Recipient's `WORKER_RECEIVER` subnode transitions to PROCESSING — delivery moment     |
| `BecomeLeader`        | `ReplayService.java::updateLeaderElectionState`, inside `if (LeaderElectionState.LEADING.equals(state))`      | Any node updates its role to LEADING                                                  |

A `LocalEvent` emit is also inherited from upstream hunk 8 of the
complete patch (in `offerLocalEvent`); it stays out of
`tv.target_actions` — the tv-eval agent filters it as internal.

## Event schema

```json
{"event":"Notification","step":42,"node":"s1","subnode":3,
 "data":{"msgId":17,"from":1,"to":2,"electionEpoch":1,"leader":1,"payload":"..."}}
```

Every event has: `event`, monotonically-increasing `step`, `node` (`s<id>`),
`subnode` (integer; -1 when not applicable, e.g. BecomeLeader), and
`data` (event-specific fields).

## How to run

```bash
# from project root
bash scripts/harness/zookeeper/run.sh
```

Writes `artifacts/zookeeper/traces/trace_01.ndjson`. A default demo run
(traces/demo has 3 scenarios) yields ~75–78 events:

| Action              | Typical count |
|---------------------|---------------|
| `Notification`      | ~32           |
| `HandleNotification`| ~19–20        |
| `BecomeLeader`      | 2             |
| `LocalEvent`        | ~24 (out of scope) |

Runtime: ~55 s per demo trace × 3 traces ≈ 3 min (after a one-time
30-s Maven build).

### Environment overrides

| Var           | Default                              | Purpose                                          |
|---------------|--------------------------------------|--------------------------------------------------|
| `TRACES_DIR`  | `artifacts/zookeeper/traces`         | Where the NDJSON lands                           |
| `REPO_PATH`   | `data/repositories/Remix`            | Remix clone root                                 |
| `PATCH_FILE`  | `data/patches/remix_ndjson_output_v2.patch` | Auto-applied if `writeNdjsonEvent` is missing |
| `TRACE_SET`   | `demo`                               | Which sub-dir of `$REPO_PATH/traces/` to replay  |

## Re-deriving the v2 patch

If a future Remix upgrade breaks `v2`, regenerate:

```bash
cd data/repositories/Remix
# edit the 3 files manually at the anchors described above, then:
git diff checker/server/src/main/java/org/disalg/remix/server/ReplayService.java \
         checker/server/src/main/java/org/disalg/remix/server/executor/ElectionMessageExecutor.java \
  > ../../patches/remix_ndjson_output_v2.patch
```

## Known coverage gaps

- **Demo scenarios only** (3 traces, one BecomeLeader each modulo failed
  election rounds). Richer FLE scenarios — split vote, term bump races,
  follower timeouts — would tighten scoring but are a generator-side
  concern (see `generator/generate_traces.sh`).
- **No reject-vote asymmetry**. The demo traces lead directly to quorum
  formation; the spec's "vote denied" paths are never exercised.
- **Notification→HandleNotification pairing**. Because
  `ElectionMessageExecutor.releaseMessage` fires per-event (not per-thread),
  one Notification may be followed by multiple HandleNotifications if
  the scheduler broadcasts. TV windows should pair by `msgId` in the
  `data` field if strict pairing is needed.
