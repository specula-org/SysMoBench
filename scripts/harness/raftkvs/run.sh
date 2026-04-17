#!/usr/bin/env bash
# scripts/harness/raftkvs/run.sh — run raftkvs PGo test with tracing enabled,
# then convert the PGo-native trace into SysMoBench-canonical NDJSON.
#
# Side effects (reverted on exit via `trap`):
#   - copies trace_hook.go into REPO_PATH/systems/raftkvs/bootstrap/
#   - patches REPO_PATH/systems/raftkvs/bootstrap/server.go to append
#     distsys.SetTraceRecorder(TraceRecorder) to genResources' config list
#   - patches REPO_PATH/systems/raftkvs/bootstrap/client.go similarly
#   - copies raftkvs_trace_test.go into REPO_PATH/systems/raftkvs/
#
# Usage (from project root):
#   bash scripts/harness/raftkvs/run.sh

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "$SCRIPT_DIR/../../.." && pwd)

REPO_PATH="${REPO_PATH:-$PROJECT_ROOT/data/repositories/pgo}"
TRACES_DIR="${TRACES_DIR:-$PROJECT_ROOT/artifacts/raftkvs/traces}"
GOROOT="${GOROOT:-/usr/local/go}"

SYS_DIR="$REPO_PATH/systems/raftkvs"
BOOTSTRAP_DIR="$SYS_DIR/bootstrap"
SERVER_GO="$BOOTSTRAP_DIR/server.go"
CLIENT_GO="$BOOTSTRAP_DIR/client.go"
TRACE_HOOK="$BOOTSTRAP_DIR/trace_hook.go"
TRACE_TEST="$SYS_DIR/raftkvs_trace_test.go"

if [[ ! -f "$SERVER_GO" || ! -f "$CLIENT_GO" ]]; then
  echo "ERROR: raftkvs bootstrap not found at $BOOTSTRAP_DIR" >&2
  exit 1
fi
if [[ ! -x "$GOROOT/bin/go" ]]; then
  echo "ERROR: Go not found at $GOROOT/bin/go" >&2
  exit 1
fi

mkdir -p "$TRACES_DIR"
RAW_TRACE="$TRACES_DIR/pgo_raw.ndjson"

# Backup originals so we can revert on exit
BACKUP_DIR=$(mktemp -d)
cp "$SERVER_GO" "$BACKUP_DIR/server.go.orig"
cp "$CLIENT_GO" "$BACKUP_DIR/client.go.orig"

restore() {
  cp "$BACKUP_DIR/server.go.orig" "$SERVER_GO"
  cp "$BACKUP_DIR/client.go.orig" "$CLIENT_GO"
  rm -f "$TRACE_HOOK" "$TRACE_TEST" "$RAW_TRACE"
  rm -rf "$BACKUP_DIR"
}
trap restore EXIT

# Drop in our additions
cp "$SCRIPT_DIR/trace_hook.go" "$TRACE_HOOK"
cp "$SCRIPT_DIR/raftkvs_trace_test.go" "$TRACE_TEST"

# Patch server.go: append SetTraceRecorder to genResources' config list.
python3 - "$SERVER_GO" <<'PY'
import sys, pathlib
p = pathlib.Path(sys.argv[1])
src = p.read_text()
anchor = '\t\treturn resourcesConfig\n'
inject = '\t\tif TraceRecorder != nil {\n\t\t\tresourcesConfig = append(resourcesConfig, distsys.SetTraceRecorder(TraceRecorder))\n\t\t}\n' + anchor
if 'distsys.SetTraceRecorder(TraceRecorder)' in src:
    print('server.go already patched; skipping')
else:
    if anchor not in src:
        sys.exit(f'anchor not found in {p}: {anchor!r}')
    p.write_text(src.replace(anchor, inject, 1))
    print(f'patched {p}')
PY

# Patch client.go: inject SetTraceRecorder option into the sole NewMPCalContext.
python3 - "$CLIENT_GO" <<'PY'
import sys, pathlib
p = pathlib.Path(sys.argv[1])
src = p.read_text()
anchor = '\t\tdistsys.EnsureArchetypeRefParam("timeout", timeoutChRes),\n\t)\n'
inject = '\t\tdistsys.EnsureArchetypeRefParam("timeout", timeoutChRes),\n\t\tdistsys.SetTraceRecorder(TraceRecorder),\n\t)\n'
if 'distsys.SetTraceRecorder(TraceRecorder)' in src:
    print('client.go already patched; skipping')
else:
    if anchor not in src:
        sys.exit(f'anchor not found in {p}: {anchor!r}')
    p.write_text(src.replace(anchor, inject, 1))
    print(f'patched {p}')
PY

export PATH="$GOROOT/bin:$PATH"
export RAFTKVS_TRACE_FILE="$RAW_TRACE"

echo "[run.sh] REPO_PATH:   $REPO_PATH" >&2
echo "[run.sh] TRACES_DIR:  $TRACES_DIR" >&2
echo "[run.sh] running: go test -run TestSafety_ThreeServers_WithTrace" >&2

( cd "$SYS_DIR" && timeout 120 go test -run TestSafety_ThreeServers_WithTrace -count=1 -v ) >&2

python3 "$SCRIPT_DIR/parse_traces.py" "$RAW_TRACE" "$TRACES_DIR"

echo "[run.sh] done — $(grep -cv '^#' "$TRACES_DIR"/trace_01.ndjson) events"
