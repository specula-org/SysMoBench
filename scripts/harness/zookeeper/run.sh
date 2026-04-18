#!/usr/bin/env bash
# scripts/harness/zookeeper/run.sh — PLACEHOLDER.
#
# NOT YET FUNCTIONAL. The NDJSON emission path for the Remix replayer
# (checker/server/src/main/java/org/disalg/remix/server/ReplayService.java)
# depends on data/patches/remix_ndjson_output_complete.patch, which was
# authored against an older commit of github.com/Lingzhi-Ouyang/Remix
# than what `data/repositories/Remix/` now tracks (HEAD `81869f1 Update
# README.md`). Of the 8 hunks in the patch, only the first 2 apply cleanly
# — the 5 hunks that actually emit `ElectionMessage` /
# `FollowerToLeaderMessage` / `LeaderToFollowerMessage` / `LocalEvent`
# events fail (line numbers have shifted ~1400 lines, surrounding
# context has changed).
#
# Consequences:
# - Replay runs to completion but writes an empty NDJSON file
# - All 3 target actions (Notification, HandleNotification, BecomeLeader)
#   would score 0 windows
# - WV smoke must not proceed until this is repaired
#
# To repair (est. 1-2h):
#   1. Manually port the 5 rejected hunks to the current line numbers
#      (offerElectionMessage @2230, offerFollowerToLeaderMessage @2342,
#       offerLeaderToFollowerMessage @2479, offerLocalEvent @2697, plus
#       writeNdjsonEvent helper + ndjsonWriter.close in the cleanup path)
#   2. Re-run scripts/build.sh in the Remix clone
#   3. Add a translation step: ElectionMessage → Notification
#      (send side), HandleNotification (receive side — probably need
#      another emit point at offerElectionMessage receiver side);
#      LocalEvent + role="Leader" → BecomeLeader
#   4. Remove the early-exit below
#
# Alternatively: switch the harness to an implementation that already
# emits Notification / HandleNotification / BecomeLeader directly —
# e.g. a ZooKeeper fork with FastLeaderElection instrumentation.

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "$SCRIPT_DIR/../../.." && pwd)

REPO_PATH="${REPO_PATH:-$PROJECT_ROOT/data/repositories/Remix}"
TRACES_DIR="${TRACES_DIR:-$PROJECT_ROOT/artifacts/zookeeper/traces}"

cat >&2 <<'EOF'
[run.sh] zookeeper harness is NOT functional — see comment block at the
top of this file and tla_eval/tasks/zookeeper/INSTRUMENTATION.md. Exiting
without generating traces to make the failure explicit (per the "no fake
content" policy in the SysMoBench contract).
EOF

exit 2
