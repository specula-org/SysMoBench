#!/usr/bin/env bash
# scripts/harness/redisraft/run.sh — thin wrapper around the redisraft trace
# harness; expects an instrumented redisraft tree at $REPO_PATH (default
# artifacts/redisraft) following the harness-gen layout
# {REPO_PATH}/{harness/run.sh, artifact/redisraft}.
#
# The inner harness (harness/run.sh inside REPO_PATH) applies an
# instrumentation patch to deps/raft, builds libraft.a with
# -DREDISRAFT_ENABLE_TRACE, builds the test_trace CuTest binary, and runs
# three scenarios (basic_consensus, leader_failover, snapshot_basic).
# Traces land in REPO_PATH/traces/ by default; we copy them out to the
# SysMoBench-side TRACES_DIR.
#
# Usage (from project root):
#   bash scripts/harness/redisraft/run.sh
#
# Or from the tv-eval agent's workspace (passing its copy of the case dir):
#   REPO_PATH=/abs/path/to/workspace/repo bash scripts/harness/redisraft/run.sh

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "$SCRIPT_DIR/../../.." && pwd)

REPO_PATH="${REPO_PATH:-$PROJECT_ROOT/artifacts/redisraft}"
TRACES_DIR="${TRACES_DIR:-$PROJECT_ROOT/artifacts/redisraft/traces}"

if [[ ! -x "$REPO_PATH/harness/run.sh" ]]; then
  echo "ERROR: Specula-style redisraft harness not found at $REPO_PATH/harness/run.sh" >&2
  echo "Expected structure: \$REPO_PATH/{harness,artifact/redisraft}" >&2
  exit 1
fi

mkdir -p "$TRACES_DIR"

echo "[run.sh] REPO_PATH:   $REPO_PATH" >&2
echo "[run.sh] TRACES_DIR:  $TRACES_DIR" >&2

# Inner harness writes to $REPO_PATH/traces/. Run it, then mirror outputs.
bash "$REPO_PATH/harness/run.sh"

cp -f "$REPO_PATH"/traces/*.ndjson "$TRACES_DIR/"

count=$(ls "$TRACES_DIR"/*.ndjson 2>/dev/null | wc -l)
echo "[run.sh] done — $count trace files in $TRACES_DIR"
