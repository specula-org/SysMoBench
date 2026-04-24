#!/usr/bin/env bash
# Adapter: codex
# Capabilities: auto-approve, json output, log capture
#
# Unified interface for invoking Codex CLI.
#
# Usage:
#   scripts/launch/adapters/codex.sh [options]
#
# Options:
#   --prompt "..."         Task prompt (mutually exclusive with --prompt-file)
#   --prompt-file file.md  Read prompt from file (mutually exclusive with --prompt)
#   --max-turns N          (DEPRECATED, ignored)
#   --max-budget N         Ignored (Codex CLI does not expose a budget flag)
#   --model=<id>           Optional Codex model override
#   --log output.log       Log file path (required)
#   --background           Parsed for compatibility, currently ignored
#   --help                 Show this help

set -euo pipefail

PROMPT=""
PROMPT_FILE=""
MAX_TURNS=""
MAX_BUDGET=""
MODEL=""
LOG_FILE=""
BACKGROUND=false

for arg in "$@"; do
  case "$arg" in
    --prompt=*)      PROMPT="${arg#*=}" ;;
    --prompt-file=*) PROMPT_FILE="${arg#*=}" ;;
    --max-turns=*)   MAX_TURNS="${arg#*=}" ;;
    --max-budget=*)  MAX_BUDGET="${arg#*=}" ;;
    --model=*)       MODEL="${arg#*=}" ;;
    --log=*)         LOG_FILE="${arg#*=}" ;;
    --background)    BACKGROUND=true ;;
    --help|-h)
      sed -n '2,/^$/{ s/^# //; s/^#//; p }' "$0"
      exit 0
      ;;
    *) echo "codex adapter: unknown option: $arg" >&2; exit 1 ;;
  esac
done

if [[ -n "$PROMPT" && -n "$PROMPT_FILE" ]]; then
  echo "codex adapter: --prompt and --prompt-file are mutually exclusive" >&2
  exit 1
fi

if [[ -z "$PROMPT" && -z "$PROMPT_FILE" ]]; then
  echo "codex adapter: one of --prompt or --prompt-file is required" >&2
  exit 1
fi

if [[ -z "$LOG_FILE" ]]; then
  echo "codex adapter: --log is required" >&2
  exit 1
fi

if [[ -n "$PROMPT_FILE" ]]; then
  if [[ ! -f "$PROMPT_FILE" ]]; then
    echo "codex adapter: prompt file not found: $PROMPT_FILE" >&2
    exit 1
  fi
  PROMPT_INPUT="$PROMPT_FILE"
else
  PROMPT_INPUT="$(mktemp)"
  echo "$PROMPT" > "$PROMPT_INPUT"
  trap "rm -f '$PROMPT_INPUT'" EXIT
fi

RAW_JSON="${LOG_FILE%.log}.raw.json"
USAGE_JSON="${LOG_FILE%.log}.usage.json"
START_MS="$(date +%s%3N)"

CMD=(
  codex exec
  --dangerously-bypass-approvals-and-sandbox
  --skip-git-repo-check
  -c 'model_reasoning_effort="high"'
  --json
  --output-last-message "$LOG_FILE"
)

if [[ -n "$MODEL" && "$MODEL" != "default" ]]; then
  CMD+=(--model "$MODEL")
fi

if "${CMD[@]}" < "$PROMPT_INPUT" > "$RAW_JSON" 2>&1; then
  CMD_RC=0
else
  CMD_RC=$?
fi

END_MS="$(date +%s%3N)"
DURATION_MS="$((END_MS - START_MS))"

if [[ ! -s "$LOG_FILE" ]]; then
  cp "$RAW_JSON" "$LOG_FILE" 2>/dev/null || : > "$LOG_FILE"
fi

python3 - "$RAW_JSON" "$USAGE_JSON" "$DURATION_MS" "$MODEL" "$CMD_RC" <<'PY'
import json
import sys
from pathlib import Path

raw_path = Path(sys.argv[1])
usage_path = Path(sys.argv[2])
duration_ms = int(sys.argv[3])
model = sys.argv[4]
exit_code = int(sys.argv[5])

usage = {
    "total_cost_usd": 0,
    "num_turns": 0,
    "duration_ms": duration_ms,
    "stop_reason": "",
    "usage": {},
    "model_usage": {},
    "exit_code": exit_code,
}
if model:
    usage["model_usage"] = {"model": model}

turns = 0
stop_reason = ""
for line in raw_path.read_text(errors="replace").splitlines():
    try:
        event = json.loads(line)
    except Exception:
        continue
    event_type = event.get("type", "")
    if event_type.startswith("response.") or event_type == "message":
        turns += 1
    if not stop_reason:
        stop_reason = (
            event.get("stop_reason")
            or event.get("response", {}).get("stop_reason", "")
            or stop_reason
        )

usage["num_turns"] = turns
usage["stop_reason"] = stop_reason
usage_path.write_text(json.dumps(usage, indent=2) + "\n")
PY

if grep -Eiq 'rate limit|too many requests|429|quota' "$RAW_JSON" 2>/dev/null; then
  exit 75
fi

exit "$CMD_RC"
