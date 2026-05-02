# Spec-Repair Skill — TLA+ Spec Repair

You are the repairer. Given a model-generated TLA+ spec that fails **P1 (SANY parse)** or **P2 (TLC from Init)**, repair it so both pass. The goal is to unblock downstream P3 (TV) and P4 (invariant) scoring.

## Core principle

> **Make it run. Preserve the model's intent, not its broken syntax.**

A spec that doesn't run gives us zero signal about modeling quality. A spec that runs but has a real modeling bug will be caught by P3 (TV) per-action. So: **default to fixing**, and reserve refusal for the narrow cases where fixing would actually change what the model said about the system — adding actions the model didn't write, weakening guards, changing which variables an action writes to, etc.

**"Intent" means**:
- Which actions exist (the Next disjunct set)
- Which guard each action fires from (substantive meaning, not exact syntax)
- Which variables each action writes to (and roughly in what direction)
- Which invariants the model claimed

**"Mechanics" means** everything else: how to spell a guard, how to bound an infinite quantifier, how to express a conditional update, how to declare a Null sentinel, how to import stdlib. Fix these freely.

When a repair is borderline — e.g., the model's action body is syntactically broken and you have to reconstruct its effect — lean toward **fixing with a justification**, not halting. Document your interpretation so a human can audit it.

**Autonomous by default.** Batch repair runs without confirmation. `unrepairable` is a last resort, not a safe default.

---

## Inputs

The skill operates on one **cell** — a `(model, system)` pair — at a time. Input layout follows `docs/leaderboard/specs/`:

| Item | Path / form |
|---|---|
| Input cell dir | `<in_dir>/` (e.g., `docs/leaderboard/specs/gpt52/locksvc/`) |
| Failing spec | `<in_dir>/<module>.tla` (e.g., `locksvc.tla`) |
| Failing config | `<in_dir>/<module>.cfg` |
| P1 error log (optional) | SANY stdout+stderr from a prior run, or run it yourself |
| P2 error log (optional) | TLC stdout+stderr from a prior run, or run it yourself |
| Task contract | `tla_eval/tasks/<task>/task.yaml` — authoritative for CONSTANTS and `target_actions` |
| Task prompt | `tla_eval/tasks/<task>/prompts/` — authoritative for what the spec is supposed to model |

The cell's own `scores.json` may be used to discover which phase(s) failed (`phase1_compilation.status`, `phase2_runtime.status`). You do not edit it.

## Outputs

Write to `<out_dir>/` (e.g., `docs/leaderboard/specs_repaired/gpt52/locksvc/`). **The output cell mirrors the input cell structure and keeps the original filenames** — the downstream TV pipeline expects `<module>.tla` / `<module>.cfg`, so do not rename.

| File | Purpose |
|---|---|
| `<module>.tla` | Repaired spec — same filename as input |
| `<module>.cfg` | Repaired config — same filename as input |
| `repair_report.md` | Human-readable per-edit justification |
| `repair_manifest.json` | Structured log: `{applied: bool, ops: [...], edit_count: int, status: "none"\|"minor"\|"moderate"\|"heavy"\|"unrepairable", p1_passed: bool, p2_passed: bool}` |
| `repair_logs/sany_before.log`, `repair_logs/sany_after.log` | SANY output before and after repair |
| `repair_logs/tlc_before.log`, `repair_logs/tlc_after.log` | TLC output before and after repair |

If a spec already passes both P1 and P2, still **copy `<module>.tla` and `<module>.cfg` verbatim to `<out_dir>/`** so the output folder is a complete, self-contained mirror, then write `repair_manifest.json` with `applied: false, status: "none", p1_passed: true, p2_passed: true` and exit. Don't touch the spec content.

---

## Allow-list (operations you MAY apply)

These are all the operations allowed. Together they cover essentially every case where a TLA+ spec is written incorrectly but the model's **intent** is recoverable. Use them liberally. When in doubt, repair.

### Structural / bookkeeping
| # | Operation | Typical trigger |
|---|---|---|
| A1 | Add missing `EXTENDS` (TLC, Naturals, Integers, Sequences, SequencesExt, FiniteSets, FiniteSetsExt, Bags) | SANY "Unknown operator" for a stdlib symbol |
| A2 | Add/remove variables in `UNCHANGED` tuples so every variable is determined every step | SANY/TLC complains variable not primed; or UNCHANGED references undeclared var |
| A3 | Fix `vars` tuple to exactly list declared VARIABLES | SANY unknown identifier in `vars`; or `vars` missing from WF_vars |
| A7 | Rename module header to match filename | SANY "Module name does not match filename" |
| A8 | Add missing `Spec`, `Next`, or `vars` definition when structure is obvious | SANY reports them undefined, and Next is clearly the disjunction of declared actions |
| A9 | Fix `.cfg` syntax (missing SPECIFICATION/INIT/NEXT/CONSTANT directives, wrong operator names) | TLC rejects the .cfg |

### Constants / initialization
| # | Operation | Typical trigger |
|---|---|---|
| A4 | Bind missing CONSTANTS in `.cfg` per `task.yaml` | TLC "Constant C is not assigned a value" |
| A5 | Widen `Init` to a satisfiable, non-empty state so actions can fire | TLC "Initial predicate has no solution" or immediate deadlock; keep the binding shape, just fill values |

### Identifier / scope fixes
| # | Operation | Typical trigger |
|---|---|---|
| A6 | Fix identifier typos, case mismatches, off-by-one prime placement when context makes the intent unambiguous | SANY "Unknown operator" resolvable from nearby declarations |
| A13 | Remove a local definition that duplicates stdlib (`Min`, `Max`, `Head`, `Tail`, `Len`, etc.) | SANY "symbol already declared" from a stdlib import |
| A14 | Rename a parameter that shadows a global VARIABLE / CONSTANT | SANY/TLC errors about shadowed identifier |

### Making unrunnable TLA+ runnable (expanded — apply freely)
| # | Operation | Typical trigger |
|---|---|---|
| A10 | **Bound an infinite quantifier domain**. `\E x \in Seq(T)` → `\E x \in BoundedSeq(T, K)`; `\A s \in SUBSET INF : P(s)` → bounded subset; `[f \in INF \|-> ...]` → bounded. Pick K = the smallest value from task.yaml that still exercises the action (usually 2-4). | TLC "Attempted to enumerate infinite set" / OOM / hang on quantifier |
| A11 | **Promote an unbounded `CHOOSE` sentinel to a CONSTANT**. `Null == CHOOSE v : v \notin D` → `CONSTANT Null` + bind a fresh string like `"NULL"` in .cfg. This is the standard TLA+ idiom for Null sentinels. | TLC cannot evaluate unbounded CHOOSE |
| A12 | **Rewrite a syntactically broken action body** to the standard idiom that expresses the same intent. Examples: `x''` (double-prime) → `x' = IF cond THEN newVal ELSE x`; missing `IN` after `LET`; unclosed records; malformed `CASE`; misplaced `\/` / `/\`. **Hard requirements**: keep the action's name, the substantive guard intent (what pre-states enable it), and which variables it writes to. Document the before/after in repair_report.md so a human can audit. | SANY parse error inside an action, or TLC rejects a malformed expression |
| A15 | Add `UNCHANGED` lines for variables an action clearly doesn't touch (when omitted) | TLC "value of v is not determined" after the action |
| A16 | Fix malformed TLA+ expressions (wrong operator precedence, missing parens, wrong arity in standard operators) when the intended meaning is unambiguous | SANY parse error in a local expression |
| A17 | Replace an unrunnable construct with its runnable equivalent when the semantic meaning is preserved at finite TLC-model-checking scale (e.g., unbounded RECURSIVE → bounded via K-depth; non-constructive `CHOOSE` over a condition → explicit value) | TLC crash, hang, or non-termination traced to the construct |

**Every edit cites a rule (A1-A17) in `repair_report.md`**. If your reasoning doesn't map to any rule, check the forbid-list; if it's not forbidden either, you may still apply it — but document your interpretation of intent carefully.

---

## Forbid-list (narrow — these would actually compromise fairness)

Halt with `repair_status: unrepairable` ONLY if the fix requires one of these. Do not treat the forbid-list as a trip-wire to dodge hard cases — read it strictly.

| # | Forbidden | Why it matters for fairness |
|---|---|---|
| F1 | **Add or remove an entire action** (disjunct of `Next`). | The set of actions is the model's answer to "what can happen?" TV scores per-action; inventing an action gives the model credit for code it didn't write. |
| F2 | **Change a guard's substantive logical condition**. E.g., `r = leader` → `TRUE`, `x > 0` → `x >= 0`, negating a conjunct, dropping a conjunct. | Guards decide from which states an action can fire. Weakening them hides real bugs in the model's precondition reasoning. |
| F3 | **Change which variables an action writes to**. If the model's `Propose` only writes `clientMsgs`, don't add writes to `log`. (Adjusting UNCHANGED per A2 is fine — that's saying "doesn't write", not "writes to".) | TV checks post-state per variable; changing write-set changes the answer. |
| F4 | **Add, rename, or remove VARIABLES declarations**. | Changes the state vocabulary the model chose. |
| F5 | **Add invariants or `PROPERTY` clauses the model didn't write**. | We score what the model claimed, not what it should have claimed. |
| F6 | **Remove or weaken invariants the model did write**, even if they cause P2 to fail. If the model wrote a wrong invariant, that's a real modeling bug and honest to preserve. | Weakening would launder a bug. |
| F7 | **Change CONSTANTS cardinality to hide multi-actor behavior**. `Server = {s1, s2, s3}` → `{s1}` to dodge a distributed bug is not OK. But adjusting the *value* of a constant per task.yaml is fine (A4). | Changing cardinality changes which interleavings get explored. |

If none of F1-F7 applies, you can repair. "I'm worried the edit is semantic" is not a sufficient reason to halt — the forbid-list is the definition of what "unfair" means, and only those are unfair.

**When applying A11 (CHOOSE sentinel → CONSTANT)**: this is explicitly *not* F4 — you are not adding a variable, you are re-expressing a model constant the model already intended. Similarly A10 (bounding infinite quantifiers) is *not* F2 — the model's guard intent is preserved, only the enumeration domain is made finite.

---

## Workflow

### Phase 0 — Baseline

1. Create `<out_dir>/` and `<out_dir>/repair_logs/`. Copy `<in_dir>/<module>.tla` and `<in_dir>/<module>.cfg` to `<out_dir>/<module>.tla` and `<out_dir>/<module>.cfg`. **All subsequent edits happen on the copies in `<out_dir>/`.** The input cell is read-only.
2. Run SANY and TLC on the input spec (the baseline measurement) and save logs as `<out_dir>/repair_logs/sany_before.log` and `<out_dir>/repair_logs/tlc_before.log`.
   ```bash
   # P1 — parse check
   java -cp <repo>/lib/tla2tools.jar tla2sany.SANY -error-codes <in_dir>/<module>.tla

   # P2 — runtime check (matches the benchmark pipeline: 60s wall clock, -deadlock on)
   cd <in_dir> && timeout 60 java -cp <repo>/lib/tla2tools.jar tlc2.TLC \
       -config <module>.cfg -deadlock <module>.tla
   ```
   **P2 pass semantics** (matches `tla_eval/evaluation/semantics/runtime_check.py`):
   - TLC completes within 60s with no error → PASS
   - TLC hits the 60s timeout **and** the partial output contains no violation / no deadlock / no TLC error → PASS (large state space, still exploring safely)
   - Any invariant violation, deadlock (with `-deadlock` on), parse error, type error, assertion failure → FAIL
3. Classify baseline:
   - P1 pass, P2 pass → copy `<module>.tla`/`.cfg` verbatim to `<out_dir>/`, write manifest with `status: none, applied: false, p1_passed: true, p2_passed: true` and exit.
   - P1 fail → go to Phase 1.
   - P1 pass, P2 fail → go to Phase 3.

### Phase 1 — Diagnose P1

Read `sany_before.log`. For **each** distinct SANY error:

1. Quote the error line and the offending source location.
2. Ask: *what is the root cause?* Match it to an allow-list rule (A1…A9) or identify that the only fix is a forbidden op.
3. Record the diagnosis in `repair_report.md` under `## Phase 1 — P1 Diagnosis`, one subsection per error.

If any error requires a forbidden op, halt now with `repair_status: unrepairable`. Don't attempt partial repair.

### Phase 2 — Apply P1 fixes, iterate

For each diagnosed error:

1. Apply the allow-list edit to `<out_dir>/<module>.tla` / `<out_dir>/<module>.cfg`. Keep the edit **minimal** — change only what the rule covers.
2. Write a justification in `repair_report.md`:
   > **Edit #k** (rule **A2**): Added `q` to `UNCHANGED` in `ClientLockRequest` (line 28).
   > **Why**: SANY flagged `q` as undetermined after the action. All other branches either prime it or list it in UNCHANGED. This is bookkeeping; the model's intent (q is not modified by this action) is preserved.
3. After each batch of edits, re-run SANY. If new errors appear, loop back to Phase 1 with the new log. If the same error persists after an attempted fix, halt with `repair_status: unrepairable` — your fix didn't address the root cause and retrying the same thing is not allowed.

Exit Phase 2 when SANY passes cleanly. Save `repair_logs/sany_after.log`.

### Phase 3 — Diagnose P2

Run TLC on `<out_dir>/<module>.tla` / `<out_dir>/<module>.cfg` with the same command used in Phase 0 (60s timeout, `-deadlock`). Read the output.

Common P2 failure modes:

| Symptom | Likely rule | Action |
|---|---|---|
| `Attempted to compute the value of X but could not determine it` | A2 / A4 | Missing UNCHANGED, or CONSTANTS unbound |
| `Initial predicate ... has no satisfying assignments` | A4 / A5 | CONSTANTS wrong, or Init too restrictive |
| `Deadlock reached` on Init (no successor) | A5 | Init contradicts every action's guard |
| `Error: In computing next states, the value of ... was not a ...` | A2 | Usually an UNCHANGED/missing prime issue |
| Invariant violation at Init (if an invariant is already present) | Forbidden — F6/F8 | **Halt. Do not weaken.** |
| Type error inside an action body | Forbidden — F3/F7/F8 | **Halt.** |

Record diagnosis in `repair_report.md` under `## Phase 3 — P2 Diagnosis`.

### Phase 4 — Apply P2 fixes, iterate

Same loop as Phase 2, but for P2. Apply allow-list edit → re-run TLC → re-diagnose if needed.

**Extra strictness at P2**: it is tempting to "fix" a deadlock by rewriting a guard (F2) or adding a fallback action (F1). Don't. If the only way to get past a P2 error is a forbidden op, halt.

Exit Phase 4 when TLC passes per the P2 semantics above (completes, or hits 60s timeout with no violations/deadlock/errors). Save `repair_logs/tlc_after.log`.

### Phase 5 — Finalize

1. Compute `edit_count` (total allow-list operations applied).
2. Classify:
   - `none` → applied = false (spec was already OK)
   - `minor` → 1–3 edits
   - `moderate` → 4–10 edits
   - `heavy` → 11+ edits
3. Write `repair_manifest.json`:
   ```json
   {
     "applied": true,
     "status": "minor",
     "p1_passed": true,
     "p2_passed": true,
     "edit_count": 3,
     "ops": [
       {"rule": "A1", "location": "line 2", "summary": "Added EXTENDS Bags"},
       {"rule": "A2", "location": "ClientLockRequest line 28", "summary": "Added q to UNCHANGED"},
       {"rule": "A4", "location": "locksvc.cfg", "summary": "Bound NumClients = 2 per task.yaml"}
     ]
   }
   ```
4. Finalize `repair_report.md` — it should read top-to-bottom as a coherent account: baseline errors → diagnoses → edits → verification result.

### Phase 6 — Unrepairable exit

If at any point a forbidden op is the only way forward:

1. Leave `<out_dir>/<module>.tla` / `<out_dir>/<module>.cfg` in their current state (partial edits OK — they document how far repair got before hitting the wall).
2. Write `repair_manifest.json` with `status: "unrepairable", applied: <true if any edits were made before halt>, p1_passed: <current>, p2_passed: <current>`.
3. In `repair_report.md`, add a final section `## Halted — Unrepairable` that states:
   - The exact SANY/TLC error that required the forbidden op.
   - Which forbid-list rule (F1…F8) applies.
   - Why no allow-list rule could address the root cause.
4. The downstream pipeline will treat unrepairable cells as P2 = 0 and P3/P4 = N/A — this is the intended, honest outcome.

---

## Critical rules

1. **Preserve intent.** A repair that makes the spec pass P1/P2 by changing what the spec *says* is a failed repair. Better to halt unrepairable than to launder a modeling bug.
2. **Every edit cites an allow-list rule.** No rule ↔ no edit.
3. **Minimal edits.** Don't clean up unrelated whitespace, don't "improve" unrelated definitions, don't add comments the model didn't write. Touch the minimum bytes needed.
4. **Never add invariants or properties.** You are unblocking the scoring pipeline, not enriching the spec.
5. **Do not re-try the same fix.** If an edit didn't resolve the error, the root cause was different — re-diagnose, don't retry.
6. **Halt on forbidden ops, don't tiptoe.** An unrepairable status is a valid, expected outcome — it tells the downstream pipeline to mark P3/P4 as N/A honestly.
7. **Keep originals.** The input cell (`<in_dir>/`) is read-only. Repaired spec lives at `<out_dir>/<module>.tla` / `<out_dir>/<module>.cfg` with the **original filenames** — the TV pipeline expects them unchanged.
8. **Logs are evidence.** `repair_logs/sany_before.log`, `repair_logs/sany_after.log`, `repair_logs/tlc_before.log`, `repair_logs/tlc_after.log` must exist even on unrepairable exits — they are what distinguishes an honest halt from an opaque failure.

---

## Checklist before writing the manifest

- [ ] `<out_dir>/<module>.tla` compiles under SANY (or unrepairable is declared)
- [ ] `<out_dir>/<module>.tla` passes TLC per the 60s `-deadlock` rule (or unrepairable is declared)
- [ ] Output filenames are `<module>.tla` and `<module>.cfg` (NOT `repaired.tla`) — verify by listing `<out_dir>/`
- [ ] Every edit in `repair_report.md` names an allow-list rule (A1…A9)
- [ ] No edit touched a Next disjunct, a guard, or an action body (beyond A2-style UNCHANGED)
- [ ] No VARIABLES were added or removed
- [ ] No invariants or properties added
- [ ] `repair_manifest.json` fields all populated, `status` matches `edit_count`
- [ ] All four logs saved under `<out_dir>/repair_logs/`
