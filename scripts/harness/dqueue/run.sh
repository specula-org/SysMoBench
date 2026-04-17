#!/usr/bin/env bash
# scripts/harness/dqueue/run.sh — run dqueue PGo test with tracing enabled,
# then convert the PGo-native trace into SysMoBench-canonical NDJSON.
#
# Side effect: copies dqueue_trace_test.go into the PGo clone under
# systems/dqueue/ (removed on cleanup). Requires Go 1.23+.
#
# Usage (from project root):
#   bash scripts/harness/dqueue/run.sh
#
# Env:
#   REPO_PATH     — PGo clone root (default: data/repositories/pgo)
#   TRACES_DIR    — where the canonical NDJSON lands (default: artifacts/dqueue/traces)
#   GOROOT        — override Go install path (default: /usr/local/go)

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "$SCRIPT_DIR/../../.." && pwd)

REPO_PATH="${REPO_PATH:-$PROJECT_ROOT/data/repositories/pgo}"
TRACES_DIR="${TRACES_DIR:-$PROJECT_ROOT/artifacts/dqueue/traces}"
GOROOT="${GOROOT:-/usr/local/go}"

SYS_DIR="$REPO_PATH/systems/dqueue"
if [[ ! -f "$SYS_DIR/dqueue.go" ]]; then
  echo "ERROR: dqueue system not found at $SYS_DIR" >&2
  exit 1
fi
if [[ ! -x "$GOROOT/bin/go" ]]; then
  echo "ERROR: Go not found at $GOROOT/bin/go" >&2
  exit 1
fi

mkdir -p "$TRACES_DIR"
RAW_TRACE="$TRACES_DIR/pgo_raw.ndjson"
TEST_FILE="$SYS_DIR/dqueue_trace_test.go"

# Drop in our trace-enabled test variant
cp "$SCRIPT_DIR/dqueue_trace_test.go" "$TEST_FILE"
trap 'rm -f "$TEST_FILE" "$RAW_TRACE"' EXIT

export PATH="$GOROOT/bin:$PATH"
export DQUEUE_TRACE_FILE="$RAW_TRACE"

echo "[run.sh] REPO_PATH:   $REPO_PATH" >&2
echo "[run.sh] TRACES_DIR:  $TRACES_DIR" >&2
echo "[run.sh] running: go test -run TestProducerConsumerWithTrace" >&2

( cd "$SYS_DIR" && timeout 90 go test -run TestProducerConsumerWithTrace -count=1 ) >&2

python3 "$SCRIPT_DIR/parse_traces.py" "$RAW_TRACE" "$TRACES_DIR"

echo "[run.sh] done — $(grep -cv '^#' "$TRACES_DIR"/trace_01.ndjson) events (incl. out-of-scope labels)"
