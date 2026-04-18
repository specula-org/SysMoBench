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
   - **Any violations found** → do NOT silently filter. First diagnose WHY the windows violate the contract:

     **Type A — Instrumentation defect**: non-compliant windows share a consistent pattern (e.g., "all failures are missing event X for actor Y"), indicating the trace generator missed logging some transitions that DO happen in the real system. Examples: synthetic generator drops events on fast paths; production instrumentation enabled at wrong code location.
     - Fix: **re-instrument and regenerate traces until fully compliant**. Filtering is forbidden — it hides a trace-data bug.

     **Type B — Out-of-scope behavior**: non-compliant windows represent behaviors the task explicitly excluded (e.g., `EXPLICITLY EXCLUDED` section of task prompt lists a behavior, but the real system still emits those events). The windows are about behavior the task decided not to evaluate.
     - Fix: exclude those windows, document in `reports/trace_compliance.md` with explicit reference to the task's exclusion rule.

     **Type C — Benchmark-setup error**: all or most windows violate the contract uniformly, system fundamentally cannot emit at the required granularity (e.g., task requires PreCandidate but etcd was configured without PreVote).
     - Fix: halt evaluation, report to user. This is a benchmark data preparation problem.

5. **Choice of action** must be justified in `reports/trace_compliance.md`:
   - State which Type (A/B/C) applies and cite evidence (which windows, what pattern, reference to task prompt).
   - If Type A: show the re-instrumentation (patch / new emit points) and confirm regenerated traces are compliant.
   - If Type B: cite the task's exclusion rule verbatim.
   - If Type C: stop.

**Under no circumstance** should non-compliant windows be silently excluded without Type classification. An agent that does so is producing misleading scores.

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

### Step 9 — Audit (downgrade-only, per-action bug finding)

Why this step exists: WV windows only test what the trace covers. A spec that passes 100% WV may still have bugs WV cannot catch — trivially-true guards, wrong post-state assignments, missing preconditions, etc. This step is your chance to **read spec against source code** and find such bugs.

**Audit rules**:
- **Downgrade-only**: audit can only lower an action's score, never raise it.
- **Only audit actions with WV pass rate = 1.0** (or "clean" 1.0 if defects were excluded). If pass rate < 1.0, WV already penalized the action; no audit needed → save tokens.
- **Zero-tolerance**: a single TLC-verified bug in an action → that action's final score = 0.
- **Evidence required**: every bug claim must be proven by a TLC run that either:
  - (a) Constructs an "impossibility window" — a pre→post transition the REAL system cannot do — and shows spec ACCEPTS it (TLC reports `Invariant NeverPost is violated`), OR
  - (b) States an invariant that the real system maintains, and shows spec VIOLATES it (TLC reports the same violation).
- **No hallucination**: if you suspect a bug but cannot produce a TLC-passing proof, the action stays correct. The filter is mechanical.

**What to look for** (patterns seen in past audits):
- **Vacuous implications**: `(x # NULL => P)` when x = NULL is reachable → the implication is trivially true in those states and can enable actions that should be blocked.
- **Wrong post-state assignment**: primed-variable values that don't match what the real code writes.
- **Missing quorum / ack conditions**: consensus/distributed-system specs that commit/apply without requiring replication.
- **Single-site state updates for multi-node protocols**: e.g., a leader-change action that only updates the new leader's term but not followers' terms.
- **Over-simplified voter/participant sets**: `voters == {s \in Servers : TRUE}` (all servers) when the real protocol requires a specific quorum.
- **Missing preconditions on role/state**: e.g., a command processing action that doesn't check the server's current role.

**For each action to audit**:
1. Read the spec's definition of the action (`<workspace>/spec/<sys>.tla`).
2. Read the corresponding code in `<workspace>/repo/` — find the function/method that implements this behavior.
3. Ask yourself: does the spec's action enable the SAME set of pre→post transitions the real code does? Is there any transition spec enables that real code cannot?
4. For each suspected bug:
   - Design an impossibility window or invariant.
   - Write a `<workspace>/audit/Audit_<Action>_<BugName>.tla` + `.cfg`, following the WV module pattern (INSTANCE the spec, Init from synthesized pre, Next fires the spec's action, Invariant `NeverPost`).
   - Run TLC. If `Invariant NeverPost is violated` → bug confirmed.
   - If TLC says spec is correct → your hypothesis was wrong, move on.
5. Record each action's verdict to `<workspace>/reports/audit.md`:
   - `correct` + one-line justification (even when no bug found, say what you checked)
   - `wrong(bug_name, evidence_path, tlc_exit_code)` if a bug was TLC-verified

**Final scoring**:
- `final(A_i) = 0` if audit verdict for A_i is wrong
- `final(A_i) = WV pass rate of A_i` otherwise (which is 1.0 for audited actions, since only 100%-WV actions are audited)
- `Phase3_score = mean(final(A_i))`

**Report format** (append to `<workspace>/reports/final_report.md`):

```
## Phase 3 Final Score (after audit)

| Action | WV rate | Audited | Verdict | Final |
|---|---|---|---|---|
| ActionA | 1.0 | yes | correct | 1.0 |
| ActionB | 1.0 | yes | wrong (NullLeader bug, see audit/Audit_ActionB_NullLeader.tla) | 0 |
| ActionC | 0.85 | no (WV already <1.0) | — | 0.85 |

Phase3_score = (1.0 + 0 + 0.85) / 3 = 0.62
```

The audit section of `audit.md` should contain the full reasoning and TLC transcripts for each `wrong` verdict. `correct` verdicts need only one line each.

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
