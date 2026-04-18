#!/usr/bin/env bash
# scripts/harness/zookeeper/run.sh — build the Remix checker with NDJSON
# instrumentation applied, replay the demo traces, and copy the resulting
# NDJSON into TRACES_DIR.
#
# Instrumentation is in data/patches/remix_ndjson_output_v2.patch (a
# SysMoBench-authored patch; the upstream-provided patches targeted an
# older Remix commit and no longer apply). The patch adds three emit
# points:
#   - Notification         — offerElectionMessage (send side of FLE msg)
#   - HandleNotification   — ElectionMessageExecutor.releaseMessage
#                            (receive side, after WORKER_RECEIVER enters PROCESSING)
#   - BecomeLeader         — updateLeaderElectionState when state==LEADING
# Plus a LocalEvent emit inherited from upstream hunk 8 (out of scope).
#
# Usage (from project root):
#   bash scripts/harness/zookeeper/run.sh

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "$SCRIPT_DIR/../../.." && pwd)

REPO_PATH="${REPO_PATH:-$PROJECT_ROOT/data/repositories/Remix}"
TRACES_DIR="${TRACES_DIR:-$PROJECT_ROOT/artifacts/zookeeper/traces}"
PATCH_FILE="${PATCH_FILE:-$PROJECT_ROOT/data/patches/remix_ndjson_output_v2.patch}"
TRACE_SET="${TRACE_SET:-demo}"

if [[ ! -d "$REPO_PATH/checker/server/src/main/java/org/disalg/remix/server" ]]; then
  echo "ERROR: Remix clone not found at $REPO_PATH" >&2
  exit 1
fi

# Apply instrumentation if writeNdjsonEvent helper is missing (sentinel)
if ! grep -q 'writeNdjsonEvent' \
      "$REPO_PATH/checker/server/src/main/java/org/disalg/remix/server/ReplayService.java"; then
  echo "[run.sh] applying $PATCH_FILE to $REPO_PATH..." >&2
  ( cd "$REPO_PATH" && git apply "$PATCH_FILE" ) || {
    echo "ERROR: patch failed to apply" >&2
    exit 1
  }
fi

# Build if the compiled jar is missing or older than our source
JAR="$REPO_PATH/checker/zookeeper-ensemble/target/zookeeper-ensemble-jar-with-dependencies.jar"
SRC="$REPO_PATH/checker/server/src/main/java/org/disalg/remix/server/ReplayService.java"
if [[ ! -f "$JAR" ]] || [[ "$SRC" -nt "$JAR" ]]; then
  echo "[run.sh] building Remix checker..." >&2
  ( cd "$REPO_PATH/scripts" && bash build.sh ) >&2
fi

mkdir -p "$TRACES_DIR"
OUT="$TRACES_DIR/trace_01.ndjson"
rm -f "$OUT"

echo "[run.sh] REPO_PATH:  $REPO_PATH" >&2
echo "[run.sh] TRACES_DIR: $TRACES_DIR" >&2
echo "[run.sh] replay:     $TRACE_SET" >&2

(
  cd "$REPO_PATH/scripts"
  NDJSON_OUTPUT="$OUT" bash replay.sh "$TRACE_SET"
) >&2

# replay.sh launches `nohup java ... &` then returns — wait for java to finish.
# `^java` anchor matches only processes whose argv[0] starts with "java",
# not our own bash eval strings that contain the jar name as a literal.
sleep 2
while pgrep -f '^java .*zookeeper-ensemble-jar' > /dev/null; do
  sleep 3
done

if [[ ! -s "$OUT" ]]; then
  echo "ERROR: no NDJSON output at $OUT" >&2
  exit 1
fi
echo "[run.sh] done — $(wc -l < "$OUT") events in $OUT"
