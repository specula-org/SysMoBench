#!/usr/bin/env bash
#
# Launch a WV (action-window validation) evaluation: spawn a Claude Code
# agent that walks the wv-eval skill to score a TLA+ spec against a real
# system's execution traces.
#
# Usage:
#   bash scripts/launch_wv_eval.sh --spec <path> --repo <path> [options]
#
# Required:
#   --spec=<file_or_dir>    TLA+ spec to evaluate (single .tla or dir with .tla+.cfg)
#   --repo=<dir>            Source code repo of the real system
#
# Optional:
#   --task=<name>           Task name (e.g. etcd, spin). Auto-detected if --spec
#                           lives under tla_eval/tasks/<name>/ or named accordingly.
#   --actions=<list>        Comma-separated list of actions to evaluate (locks scope).
#                           If unset, agent picks. Strongly recommended to set.
#                           Example: --actions=ElectionTimeout,HandleVoteRequest,ClientProposal
#   --workspace-root=<dir>  Where to create the per-eval workspace
#                           (default: ./wv-workspaces)
#   --agent=<name>          Agent adapter (default: claude-code)
#   --model=<id>            Model ID (default: claude-sonnet-4-5)
#   --max-budget=<usd>      Max API spend (default: unlimited)
#   --dry-run               Set up workspace and print prompt, don't launch
#   --keep-repo             Keep the repo copy in workspace (default: delete, save patch)
#   --help
#
# What this script does:
#   1. Creates a timestamped workspace under --workspace-root
#   2. Symlinks the spec (read-only reference)
#   3. COPIES the repo into workspace/repo/ so the agent can instrument it
#      without polluting the original
#   4. Generates a prompt pointing the agent at the wv-eval skill
#   5. Launches the agent via the configured adapter
#   6. On completion: git-diffs the repo copy, saves a patch to reports/,
#      deletes the copy (unless --keep-repo)
#
# Workspace layout:
#   wv-workspaces/<timestamp>_<spec_name>/
#     ├── spec/              symlink to input spec
#     ├── repo/              COPY of input repo (deleted after agent finishes)
#     ├── repo.patch         git patch of agent's changes (saved, repo deleted)
#     ├── traces/            agent writes
#     ├── windows/           agent writes (canonical format)
#     ├── wv/                agent writes WV_*.tla + make_windows.py
#     ├── reports/           final scoring report
#     ├── .prompt.md
#     └── .run.log

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SKILL_DIR="$PROJECT_ROOT/tla_eval/skills/wv-eval"

# Source .env if present (API keys, base URLs)
[[ -f "$PROJECT_ROOT/.env" ]] && source "$PROJECT_ROOT/.env"

SPEC_PATH=""
REPO_PATH=""
TASK_NAME=""
WORKSPACE_ROOT="$PWD/wv-workspaces"
AGENT="claude-code"
MODEL="sonnet"
MAX_BUDGET=""
ACTIONS=""
DRY_RUN=false
KEEP_REPO=false

for arg in "$@"; do
  case "$arg" in
    --spec=*)           SPEC_PATH="${arg#*=}" ;;
    --repo=*)           REPO_PATH="${arg#*=}" ;;
    --task=*)           TASK_NAME="${arg#*=}" ;;
    --workspace-root=*) WORKSPACE_ROOT="${arg#*=}" ;;
    --agent=*)          AGENT="${arg#*=}" ;;
    --model=*)          MODEL="${arg#*=}" ;;
    --max-budget=*)     MAX_BUDGET="${arg#*=}" ;;
    --actions=*)        ACTIONS="${arg#*=}" ;;
    --dry-run)          DRY_RUN=true ;;
    --keep-repo)        KEEP_REPO=true ;;
    --help|-h)
      sed -n '2,/^$/{ s/^# //; s/^#//; p }' "$0"
      exit 0
      ;;
    *) echo "Unknown option: $arg"; exit 1 ;;
  esac
done

# ── Load defaults from task.yaml if --task given ─────────

if [[ -n "$TASK_NAME" ]]; then
  TASK_YAML="$PROJECT_ROOT/tla_eval/tasks/$TASK_NAME/task.yaml"
  if [[ -f "$TASK_YAML" ]]; then
    # Fill in --repo and --actions from task.yaml's wv: block if not overridden
    if [[ -z "$REPO_PATH" ]]; then
      REPO_PATH=$(python3 -c "
import yaml, sys, os
with open('$TASK_YAML') as f:
    d = yaml.safe_load(f) or {}
r = (d.get('wv') or {}).get('repo_path')
if r and not os.path.isabs(r):
    r = os.path.join('$PROJECT_ROOT', r)
print(r if r else '')
" 2>/dev/null)
    fi
    if [[ -z "$ACTIONS" ]]; then
      ACTIONS=$(python3 -c "
import yaml
with open('$TASK_YAML') as f:
    d = yaml.safe_load(f) or {}
a = (d.get('wv') or {}).get('target_actions') or []
print(','.join(a))
" 2>/dev/null)
    fi
  fi
fi

# ── Validate inputs ──────────────────────────────────────

[[ -z "$SPEC_PATH" ]] && { echo "ERROR: --spec is required"; exit 1; }
[[ -z "$REPO_PATH" ]] && { echo "ERROR: --repo is required (or set wv.repo_path in task.yaml)"; exit 1; }
[[ ! -e "$SPEC_PATH" ]] && { echo "ERROR: spec not found: $SPEC_PATH"; exit 1; }
[[ ! -d "$REPO_PATH" ]] && { echo "ERROR: repo not found: $REPO_PATH"; exit 1; }

SPEC_PATH="$(cd "$(dirname "$SPEC_PATH")" && pwd)/$(basename "$SPEC_PATH")"
REPO_PATH="$(cd "$REPO_PATH" && pwd)"

# Auto-detect task name if not given
if [[ -z "$TASK_NAME" ]]; then
  if [[ -d "$SPEC_PATH" ]]; then
    TASK_NAME="$(basename "$SPEC_PATH")"
  else
    TASK_NAME="$(basename "$SPEC_PATH" .tla)"
  fi
fi

ADAPTER="$SCRIPT_DIR/launch/adapters/${AGENT}.sh"
[[ ! -f "$ADAPTER" ]] && { echo "ERROR: adapter not found: $ADAPTER"; exit 1; }

# ── Create workspace ─────────────────────────────────────

TIMESTAMP="$(date '+%Y%m%d_%H%M%S')"
SPEC_BASENAME="$(basename "$SPEC_PATH" .tla)"
WORKSPACE="$WORKSPACE_ROOT/${TIMESTAMP}_${SPEC_BASENAME}"

mkdir -p "$WORKSPACE"/{traces,windows,wv,reports}

# Spec: symlink (read-only reference)
if [[ -d "$SPEC_PATH" ]]; then
  ln -sf "$SPEC_PATH" "$WORKSPACE/spec"
else
  mkdir -p "$WORKSPACE/spec"
  ln -sf "$SPEC_PATH" "$WORKSPACE/spec/$(basename "$SPEC_PATH")"
  # also pick up a .cfg sibling if present
  CFG="${SPEC_PATH%.tla}.cfg"
  if [[ -f "$CFG" ]]; then
    ln -sf "$CFG" "$WORKSPACE/spec/$(basename "$CFG")"
  fi
fi

# Repo: COPY (agent may modify for instrumentation)
echo "Copying repo into workspace..."
cp -r "$REPO_PATH" "$WORKSPACE/repo"

# Snapshot original repo commit for later patch generation
(
  cd "$WORKSPACE/repo"
  if [[ -d .git ]]; then
    git rev-parse HEAD > "$WORKSPACE/.orig_commit" 2>/dev/null || echo "" > "$WORKSPACE/.orig_commit"
  else
    echo "" > "$WORKSPACE/.orig_commit"
  fi
)

# ── Generate prompt ──────────────────────────────────────

cat > "$WORKSPACE/.prompt.md" <<PROMPT_EOF
# Action-Window Validation Task

You are the evaluator (考官) for a TLA+ spec. Your job: score how faithfully the spec models the real system, producing per-action pass rates with defensible explanations.

## Inputs

- **Spec under evaluation**: $WORKSPACE/spec/
  (read-only reference — do not modify)
- **System source code**: $WORKSPACE/repo/
  (COPY — safe to modify for instrumentation; changes will be saved as a patch)
- **Task name**: $TASK_NAME
- **Task prompt**: $PROJECT_ROOT/tla_eval/tasks/$TASK_NAME/prompts/
  (contains the contract specs must follow)

## Workspace

All your work happens under: $WORKSPACE

Subdirectories:
- \`traces/\` — put NDJSON traces here
- \`windows/\` — put canonical-format window files here
- \`wv/\` — put WV_*.tla, WV_*.cfg, make_windows.py here
- \`reports/\` — final scoring report goes here

## Skill to follow

Read and follow the **wv-eval** skill:

  $SKILL_DIR/guide.md

Also consult as needed:
- $SKILL_DIR/references/canonical_window_format.md
- $SKILL_DIR/references/wv_module_template.md
- $SKILL_DIR/references/score_interpretation.md
- $SKILL_DIR/examples/ (worked examples for spin and etcd)

## Scope (HARD CONSTRAINT)

$(
if [[ -n "$ACTIONS" ]]; then
  echo "Evaluate EXACTLY these actions: $ACTIONS"
  echo ""
  echo "Do NOT expand scope to additional actions. If other actions look interesting,"
  echo "list them in 'Flagged Issues / Future Work' in the final report, but do not evaluate them."
else
  echo "Evaluate EXACTLY 3 core actions: pick them based on the task's most-emphasized behaviors."
  echo "Do NOT exceed 3. List any additional candidates in 'Future Work'."
fi
)

## Critical rules

1. Follow the skill. Don't invent your own methodology.
2. Step 0 (contract check) is a hard gate — if the trace violates task contract,
   STOP and report "benchmark data problem", do not proceed to scoring.
3. Every score you produce needs an explanation based on evidence (specific
   windows or patterns). No mystery numbers.
4. **Respect the scope above. No silent scope expansion.**
5. **Check existing examples first.** If spec/ matches an example under
   tla_eval/skills/wv-eval/examples/<task>/ai_spec_*/ (same file contents), you
   can REUSE existing WV_*.tla and make_windows.py by copying them. Don't rewrite
   from scratch if an identical spec is already worked out.
6. You can modify $WORKSPACE/repo/ for instrumentation. Original is preserved;
   your changes are saved as a patch automatically.

## Final output

Write to $WORKSPACE/reports/final_report.md with:
- Per-action pass rate (only for actions in scope)
- Explanation for each score
- Contract-compliance assessment
- Flagged issues / Future work (actions outside scope, abstraction limitations, etc.)
PROMPT_EOF

echo "================================================"
echo " Workspace: $WORKSPACE"
echo " Spec:      $SPEC_PATH"
echo " Repo:      $REPO_PATH → $WORKSPACE/repo (copy)"
echo " Task:      $TASK_NAME"
echo " Model:     $MODEL"
echo " Actions:   ${ACTIONS:-<agent picks ≤3>}"
echo " Skill:     $SKILL_DIR/guide.md"
echo "================================================"

if $DRY_RUN; then
  echo ""
  echo "[DRY RUN] Would launch: $ADAPTER --prompt-file=$WORKSPACE/.prompt.md ..."
  echo "Prompt preview:"
  echo "---"
  sed 's/^/  /' "$WORKSPACE/.prompt.md"
  echo "---"
  exit 0
fi

# ── Launch agent ─────────────────────────────────────────

LOG_FILE="$WORKSPACE/.run.log"
echo ""
echo "[$(date '+%H:%M:%S')] Launching $AGENT agent..."
echo "  Log: $LOG_FILE"

BUDGET_ARG=""
[[ -n "$MAX_BUDGET" ]] && BUDGET_ARG="--max-budget=$MAX_BUDGET"
MODEL_ARG=""
[[ -n "$MODEL" ]] && MODEL_ARG="--model=$MODEL"

"$ADAPTER" --prompt-file="$WORKSPACE/.prompt.md" --log="$LOG_FILE" $BUDGET_ARG $MODEL_ARG || AGENT_EXIT=$?

echo ""
echo "[$(date '+%H:%M:%S')] Agent finished."

# ── Post-process: save patch, clean repo copy ────────────

if [[ -d "$WORKSPACE/repo/.git" ]]; then
  (
    cd "$WORKSPACE/repo"
    git add -A 2>/dev/null || true
    git diff --cached > "$WORKSPACE/repo.patch" 2>/dev/null || echo "" > "$WORKSPACE/repo.patch"
  )
else
  # Non-git repo: use diff against original
  diff -rN "$REPO_PATH" "$WORKSPACE/repo" > "$WORKSPACE/repo.patch" 2>/dev/null || true
fi

if ! $KEEP_REPO; then
  rm -rf "$WORKSPACE/repo"
  echo "  Repo copy deleted. Agent's changes saved to: $WORKSPACE/repo.patch"
else
  echo "  Repo copy kept at: $WORKSPACE/repo (changes also in repo.patch)"
fi

# ── Summary ──────────────────────────────────────────────

echo ""
echo "================================================"
echo " Result"
echo "================================================"
REPORT="$WORKSPACE/reports/final_report.md"
if [[ -f "$REPORT" ]]; then
  echo "Report: $REPORT"
  echo ""
  head -30 "$REPORT"
else
  echo "No report generated. Check log: $LOG_FILE"
fi
