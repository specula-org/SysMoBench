#!/usr/bin/env bash
# scripts/harness/locksvc/run.sh — run locksvc PGo test with tracing enabled,
# then convert the PGo-native trace into SysMoBench-canonical NDJSON.
#
# Usage (from project root):
#   bash scripts/harness/locksvc/run.sh

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "$SCRIPT_DIR/../../.." && pwd)

REPO_PATH="${REPO_PATH:-$PROJECT_ROOT/data/repositories/pgo}"
TRACES_DIR="${TRACES_DIR:-$PROJECT_ROOT/artifacts/locksvc/traces}"
GOROOT="${GOROOT:-/usr/local/go}"

SYS_DIR="$REPO_PATH/systems/locksvc"
if [[ ! -f "$SYS_DIR/locksvc.go" ]]; then
  echo "ERROR: locksvc system not found at $SYS_DIR" >&2
  exit 1
fi
if [[ ! -x "$GOROOT/bin/go" ]]; then
  echo "ERROR: Go not found at $GOROOT/bin/go" >&2
  exit 1
fi

mkdir -p "$TRACES_DIR"
RAW_TRACE="$TRACES_DIR/pgo_raw.ndjson"
TEST_FILE="$SYS_DIR/locksvc_trace_test.go"

cp "$SCRIPT_DIR/locksvc_trace_test.go" "$TEST_FILE"
trap 'rm -f "$TEST_FILE" "$RAW_TRACE"' EXIT

export PATH="$GOROOT/bin:$PATH"
export LOCKSVC_TRACE_FILE="$RAW_TRACE"

echo "[run.sh] REPO_PATH:   $REPO_PATH" >&2
echo "[run.sh] TRACES_DIR:  $TRACES_DIR" >&2
echo "[run.sh] running: go test -run Test3ClientsWithTrace" >&2

( cd "$SYS_DIR" && timeout 60 go test -run Test3ClientsWithTrace -count=1 ) >&2

python3 "$SCRIPT_DIR/parse_traces.py" "$RAW_TRACE" "$TRACES_DIR"

echo "[run.sh] done — $(grep -cv '^#' "$TRACES_DIR"/trace_01.ndjson) events"
