# zookeeper Harness — Instrumentation & Run Notes

**Status: BLOCKED — NDJSON output patch does not apply cleanly to the
current Remix clone.**

## Where the code lives

- **Remix clone**: `data/repositories/Remix/` — upstream
  `github.com/Lingzhi-Ouyang/Remix`, HEAD `81869f1 Update README.md`
- **Instrumentation patches**:
  - `data/patches/remix_ndjson_output.patch` (partial — 3 hunks)
  - `data/patches/remix_ndjson_output_complete.patch` (full — 8 hunks)
- **Harness wrapper**: `scripts/harness/zookeeper/run.sh` (placeholder;
  currently exits 2 — see comment block inside)

## The blocker

Both patches target `checker/server/src/main/java/org/disalg/remix/server/ReplayService.java`.
The current upstream version of that file has drifted substantially (+1400 lines)
from the version the patches were authored against. Result (tested with
both `git apply --recount --reject` and `patch -p1`):

| Hunk | Anchor                                          | Status    |
|------|--------------------------------------------------|-----------|
| 1    | Field declarations near `private FileWriter statisticsWriter` | ✅ applies |
| 2    | NDJSON init near `executionWriter = new FileWriter(...)`    | ✅ applies |
| 3    | `ndjsonWriter.close()` near `committedLogVerifier closed`   | ❌ rejected |
| 4    | `writeNdjsonEvent` helper method, after `return id;`        | ❌ rejected |
| 5    | Emit `ElectionMessage` in `offerElectionMessage`             | ❌ rejected |
| 6    | Emit `FollowerToLeaderMessage` in `offerFollowerToLeaderMessage` | ❌ rejected |
| 7    | Emit `LeaderToFollowerMessage` in `offerLeaderToFollowerMessage` | ❌ rejected |
| 8    | Final cleanup hunk                                            | ✅ applies (offset +1436) |

So the NDJSON file path resolves and the file is created, but no events are
ever written because the 4 emit sites didn't land. Demo replay completes
silently and the output file stays 0 bytes.

## What needs to happen

In the current `ReplayService.java`, the relevant methods are at:
- `offerElectionMessage` — line 2230
- `offerFollowerToLeaderMessage` — line 2342
- `offerLeaderToFollowerMessage` — line 2479
- `offerLocalEvent` — line 2697

Each needs a `writeNdjsonEvent(...)` call inserted at the right point in
the synchronized(controlMonitor) block (just before `addEvent(...)` and
state transition). The `writeNdjsonEvent` helper method + the
`ndjsonWriter.close()` on shutdown also need to be added manually.

## Raw event → spec action mapping (target)

Once instrumentation is repaired, the expected mapping is:

| Emitted event                | Spec action (`target_actions`)         |
|------------------------------|----------------------------------------|
| `ElectionMessage` (send)     | `Notification`                         |
| `ElectionMessage` (receive)  | `HandleNotification` *(need second emit point)* |
| `LocalEvent` with role change to Leader | `BecomeLeader` *(need to detect in metadata)* |
| `FollowerToLeaderMessage`, `LeaderToFollowerMessage` | — (out of scope; post-election Zab) |

Note that the current patch only emits on the SEND side. To cover
`HandleNotification` (receive), a second emit point in the dispatch
path is needed.

## Environment deps (already satisfied)

- Java 11+ ✓ (OpenJDK 21 present)
- Maven 3.5+ ✓ (Maven 3.8.7 present)
- Python 3+ ✓
- Submodules: n/a (Remix is self-contained)
- `bash scripts/build.sh` (from the Remix dir) builds cleanly in ~30s

## Out-of-scope fallback options for a future pass

1. **Manually port the 5 rejected hunks** to current line numbers (~1-2h).
2. **Use a different ZooKeeper instrumentation** — e.g. instrument
   `FastLeaderElection.java` directly with a JSON logger, bypass Remix
   entirely. Cleaner but more code to write and maintain.
3. **Add a second emit point for HandleNotification**: the current
   patch only covers the send side of election messages; the receive
   path (`FastLeaderElectionWrapper` or the QueuePeer state update) needs
   its own `writeNdjsonEvent("HandleNotification", …)` call.
