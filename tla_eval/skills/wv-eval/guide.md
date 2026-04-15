# WV-Eval Skill — Action-Window Validation

You are the examiner (考官). Given a TLA+ spec and a real system's codebase, score how faithfully the spec models the system. Produce per-action pass rates with evidence-based explanations.

## Core principle

**Agent does the semantic work. TLC does the mechanical scoring.**

You read task prompts, write instrumentation, write WV modules, interpret scores. TLC decides pass/fail per window. This keeps results reproducible (TLC is deterministic) while leveraging your language understanding (semantics are judgment calls).

## Inputs (set up by launcher)

| Item | Location | Notes |
|---|---|---|
| Spec under eval | `<workspace>/spec/` | Read-only reference |
| System source | `<workspace>/repo/` | COPY — modify freely for instrumentation |
| Task prompt | `tla_eval/tasks/<task>/prompts/` | The contract (what specs must model) |
| **Task config** | `tla_eval/tasks/<task>/task.yaml` | **Read the `wv:` block for evaluation scope + harness info** |
| Your workspace | `<workspace>/` | Where you write everything |

### The `wv:` block in task.yaml

This block contains only infrastructure and scope — not modeling answers:

- `repo_path`: local path to the instrumented system code.
- `target_actions`: canonical semantic names of actions in scope. Map them to whatever the spec under evaluation calls them (e.g., canonical `HandleVoteRequest` may be named `HandleVote` or `DeliverVote` in a given spec).
- `harness`: info for regenerating traces (instrumentation files, run command). Use this when you need to re-instrument or re-run the harness.

What's NOT in this block (derive from the spec yourself):
- Which variables are schema vs auxiliary — read the spec and decide based on what the task prompt requires to be modeled.
- Value conventions (e.g., how `None` is encoded) — read the spec's TypeOK and Init.
- Log entry shape, message field names, etc. — per-spec, you discover them.

## Outputs

- `<workspace>/windows/*.ndjson` — canonical-format window files
- `<workspace>/wv/WV_*.tla`, `wv/WV_*.cfg`, `wv/make_windows.py` — your WV modules
- `<workspace>/reports/final_report.md` — scoring + explanations

---

## Workflow

### Step 0 — Contract check (HARD GATE)

Before anything else, verify traces comply with the task's required granularity. **A non-compliant trace must not be scored against** — that would unfairly penalize the spec.

1. Read `tla_eval/tasks/<task>/prompts/` and `<workspace>/spec/` (mostly the task prompt).
2. Enumerate granularity requirements as explicit claims:
   - "state X must be modeled" (e.g., StatePreCandidate for etcd)
   - "transitions must be A → B → C" (e.g., idle → trying → locked for spin)
   - "event X must precede event Y for same actor"
3. Check (using your judgment, not mechanical rules):
   - Sample 1–3 traces. Read them.
   - For each claim, look for evidence or counter-evidence in the traces.
4. Classify the trace set:
   - **All compliant** → proceed to Step 1.
   - **Partial violation** (some windows comply, some don't) → exclude the non-compliant subset, record the exclusion in `reports/trace_compliance.md`, proceed with the compliant subset.
   - **Non-compliant (all or most traces)** → the trace is wrong, not the spec. Fix the trace:
     - If traces come from a configurable system (e.g., etcd): adjust config (e.g., enable PreVote) and re-instrument/re-run the harness.
     - If traces come from synthetic generation: fix the generator to emit the missing events.
     - Regenerate until traces satisfy the contract.
     - Only after traces are compliant do you start scoring.
     - If the trace cannot be fixed within this task (e.g., system doesn't support required behavior): STOP and report to user — this is a benchmark-setup problem, not a scoring problem.

See `references/score_interpretation.md` for how contract issues manifest in scores if you miss them here.

### Step 1 — Analyze the spec (and verify it compiles)

**First: verify the spec parses and compiles.** Broken specs can't be INSTANCEd, so WV can't score them.

```bash
cd <workspace>/spec
java -cp $TLA_JAR:$COMMUNITY_JAR tla2sany.SANY <spec>.tla
```

Look for:
- `Multiple declarations or definitions for symbol X` — spec defines an operator (like `Min`/`Max`) that collides with Community Modules (`FiniteSetsExt` etc.). The spec is broken.
- `Parse Error` — the AI produced malformed TLA+.

**If spec doesn't compile**: STOP scoring. Write `reports/spec_broken.md` with the error details and classify as:
- Severity: **cannot evaluate**
- Reason: compile error in AI-generated spec (quote the specific error)
- Recommendation: treat as 0 on all actions, OR mark evaluation as failed

This is itself a useful score signal: the AI produced invalid TLA+. Don't try to patch the spec yourself.

**If spec compiles**, produce the mental model:
- **Variables**: split into **schema variables** (what the task cares about, will appear in windows) and **auxiliary variables** (everything else).
- **Actions**: all disjuncts of `Next`. These are evaluation units.
- **Per-action preconditions and effects**: what each action reads, writes, and leaves UNCHANGED.
- **TypeOK**: the domain of each variable (used to choose aux defaults).

Write this to `<workspace>/reports/spec_analysis.md`.

**Known spec-quality issues to flag** (from prototype experience):
- Double-primed variables in a single action (e.g., `msgs' = RemoveAt(...) /\ msgs' = Append(...)`) — TLA+ semantic error, spec is broken.
- Missing TypeOK or Init — spec structurally incomplete.
- Action with empty `Next` disjunct — no actual behavior modeled.

### Step 2 — Design instrumentation

Decide what to log in the real system to produce traces at the task-required granularity.

For each schema variable:
- Find code locations where the system modifies that state.
- Design a log record that includes the updated value, the actor/node, and the event name.

Critical: follow the task contract's granularity. If task says `Follower → PreCandidate → Candidate` strict, instrument both transitions separately (don't let `PreCandidate → Candidate` be silent).

Write the plan to `<workspace>/reports/instrumentation_plan.md`.

### Step 3 — Apply instrumentation and run harness

Modify code under `<workspace>/repo/` to add the log calls. Then run the system's test harness to generate traces. Put NDJSON traces in `<workspace>/traces/`.

If the repo has an existing harness/test suite, use that. If not, you'll need to write one (out of scope for MVP — flag to user).

Re-run Step 0 contract check on the newly generated traces.

### Step 4 — Cut windows (canonical format)

Reconstruct cluster state from trace events and cut one window per target action event. Write per-system code at `<workspace>/wv/generate_windows.py` that outputs **canonical window format** (see `references/canonical_window_format.md`).

One output file per target action, e.g.:
- `<workspace>/windows/ElectionTimeout.ndjson`
- `<workspace>/windows/ClientProposal.ndjson`

### Step 5 — Write make_windows.py (per-spec value mapping)

Most specs use slightly different value names or types than the canonical windows. Write `<workspace>/wv/make_windows.py` that converts canonical windows into this spec's value space (e.g., `pc="acquiring"` → `pc="trying"`; `votedFor=0` → `"None"`).

Output goes to `<workspace>/wv/windows_<Action>.ndjson`.

### Step 6 — Write WV modules

For each target action, write `<workspace>/wv/WV_<Action>.tla` following the template in `references/wv_module_template.md`.

Key patterns:
- `EXTENDS ..., Json, IOUtils` — for runtime JSON loading
- `S == INSTANCE <spec_module>` — **never copy action bodies**
- Read window: `w == AllWindows[atoi(IOEnv.WINDOW_INDEX)]`
- Init: schema vars from `w.pre`, aux vars set to plausible defaults
- Next: `step=0 /\ \E params: S!Action(params) /\ step'=1`
- Invariant: `NeverPost == ~PostReached`, where `PostReached` checks `step=1 /\ <schema vars> = w.post.<field>`

Write the cfg: fixed, per-system constants only. One cfg per WV module. Never changes per window.

### Step 7 — Run validation

Use `tla_eval/wv_tools/runner.py` for parallel TLC execution:

```python
from tla_eval.wv_tools import run_wv_batch, summarize

results = run_wv_batch(
    num_windows=N,
    wv_tla="WV_AcquireLock.tla",
    wv_cfg="WV_AcquireLock.cfg",
    work_dir="<workspace>/wv",
    workers=8,
)
stats = summarize(results)
```

TLC exit codes:
- **12** = invariant violated = post reachable = **PASS**
- **0** = no violation = post unreachable = **FAIL**

### Step 8 — Interpret and report

For each action, write to `<workspace>/reports/final_report.md`:
- Pass rate
- Explanation: why is it this number? (see `references/score_interpretation.md`)
- Evidence: cite specific window IDs and their pre/post patterns

Rules:
- **Every score needs an explanation grounded in evidence.** No mystery numbers.
- **0% or 100% need special attention.** Check they aren't masking issues.
- **Middle scores** usually split along some pre-state pattern — identify it.

If any score lacks a clear explanation, iterate: dump TLC traces, look for patterns, adjust WV or re-check contract.

---

## What you do NOT do

- Don't modify `<workspace>/spec/` (the spec is what you're evaluating).
- Don't copy action bodies from spec into WV — use `INSTANCE`.
- Don't use CONSTANTS to pass window data — use JSON + IOEnv.
- Don't write per-system contract check code — use semantic judgment in Step 0.
- Don't quietly accept a score you can't explain — iterate or escalate.

---

## Common failure modes

| Symptom | Likely cause | Fix |
|---|---|---|
| 0% on an action | Trace granularity coarser than spec's | Re-check Step 0; trace is likely non-compliant |
| Middle score with clear pre-state pattern | Subset of traces non-compliant | Exclude non-compliant windows, re-score |
| Random-looking failures | Real spec quality issue | Dump TLC counterexample, examine actual vs expected |
| TLC timeouts | Aux variable defaults wrong / domain explosion | Tighten aux defaults per action precondition |
| Parse/compile errors | WV module syntax | Check INSTANCE usage, imports, cfg constants |

See `references/score_interpretation.md` for deeper diagnostic guide.
