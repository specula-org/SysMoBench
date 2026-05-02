#!/usr/bin/env bash
# scripts/harness/curp/run.sh — run the Xline/CURP madsim trace generator,
# writing an NDJSON trace where each line is already {"event": ..., ...state}
# (no post-processing needed; the tv-eval skill consumes it directly).
#
# The harness depends on the xline-instrumented patch already being applied
# to the Xline clone. If this script sees a pristine tree it applies the
# patch automatically (idempotent check via a sentinel file).
#
# Usage (from project root):
#   bash scripts/harness/curp/run.sh

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "$SCRIPT_DIR/../../.." && pwd)

REPO_PATH="${REPO_PATH:-$PROJECT_ROOT/data/repositories/Xline}"
TRACES_DIR="${TRACES_DIR:-$PROJECT_ROOT/artifacts/curp/traces}"
PATCH_FILE="${PATCH_FILE:-$PROJECT_ROOT/data/patches/xline-instrumented.patch}"

NODES="${CURP_NODES:-3}"
OP_COUNT="${CURP_OP_COUNT:-30}"

if [[ ! -d "$REPO_PATH/crates/curp" ]]; then
  echo "ERROR: Xline clone not found at $REPO_PATH" >&2
  exit 1
fi

# Apply instrumentation patch if the trace crate is missing (sentinel)
if [[ ! -d "$REPO_PATH/crates/trace" ]]; then
  echo "[run.sh] applying $PATCH_FILE to $REPO_PATH..." >&2
  ( cd "$REPO_PATH" && git apply "$PATCH_FILE" ) || {
    echo "ERROR: patch failed to apply" >&2
    exit 1
  }
fi

mkdir -p "$TRACES_DIR"
OUT="$TRACES_DIR/trace_01.ndjson"

echo "[run.sh] REPO_PATH:  $REPO_PATH" >&2
echo "[run.sh] TRACES_DIR: $TRACES_DIR" >&2
echo "[run.sh] nodes=$NODES op-count=$OP_COUNT" >&2

(
  cd "$REPO_PATH"
  MADSIM_TEST_CONFIG="$REPO_PATH/config.toml" \
    cargo run --release --bin trace_generator -- \
      --nodes "$NODES" \
      --op-count "$OP_COUNT" \
      --trace-file "$OUT"
) >&2

if [[ ! -s "$OUT" ]]; then
  echo "ERROR: no trace output produced at $OUT" >&2
  exit 1
fi

echo "[run.sh] done — $(wc -l < "$OUT") events in $OUT"
