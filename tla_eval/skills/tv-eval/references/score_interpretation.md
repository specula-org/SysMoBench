# Score Interpretation Guide

Every score you report needs an explanation. No mystery numbers. This guide lists the common score patterns and how to diagnose each.

## The golden rule

**Before blaming the spec, rule out trace issues.**

When a TV score looks bad, default suspicion should be:
1. Trace non-compliant with task contract (Step 0 should have caught this)
2. TV Init aux-var defaults wrong
3. TV value mapping wrong
4. *Then* actual spec quality issue

Most "unfair" scores come from (1)–(3), not (4).

## Score patterns

### "Spec broken" — no score at all (detected in Step 1)

Examples from the prototype:
- **Symbol collision** (etcd `20260125_115416`): spec defined `Min(s) == ...` and `Max(s) == ...`. Community Modules' `FiniteSetsExt` (transitively loaded) also defines them. TLC refuses to load → `Multiple declarations or definitions for symbol Min`.
- **Parse error** (etcd `20260126_182328`): line 159 has an expression TLC can't parse → `Encountered "Beginning of definition" at line 159`.
- **Double-priming** (etcd `20260126_182746`): `DeliverProp` has `/\ msgs' = RemoveAt(...)` and `/\ msgs' = FoldSet(...)` — two assignments to `msgs'` in one action. Semantic error.

**How to report**: "Evaluation failed — spec does not compile. Error: <quote>. Severity: cannot evaluate. Recommend: 0 on all actions, or mark as DNQ (did not qualify)."

Don't try to patch the spec. The AI's job was to produce valid TLA+; it didn't.

### 100% pass — "spec reproduces observed behavior"

Interpretation: the spec's action definition, when applied from every observed pre-state, reaches every observed post-state.

Caveats:
- **Doesn't mean spec is complete.** The spec might model more states than the trace exercises. That's coverage, not correctness.
- **Sanity-check** a few passing windows: dump TLC counter-example, verify the reached state actually matches post. If TLC is finding pass via a bizarre path, something's off.

### 0% pass — usually a mismatch, not a spec bug

Diagnostic steps:
1. Re-run Step 0 contract check mentally.
2. Take one failing window. Pre-state, Post-state, expected action.
3. Manually trace through the spec's action: starting from pre, what does the spec say happens?
4. Compare to window's post.
5. If spec produces a DIFFERENT post than the window: 
   - Is it systematic? All 11 failures the same kind of mismatch? → trace granularity issue
   - Is each failure different? → spec quality issue (rare)

**Known pattern: spec finer than trace.** E.g., etcd ElectionTimeout goes `Follower → PreCandidate` (one step) but trace's Timeout event shows `Follower → Candidate` (which takes multiple spec steps: ElectionTimeout → [handle prevote responses] → BecomeCandidate). This produces 0/N because every window wants the end state of a multi-step sequence.

Fix: trace needs finer instrumentation, OR task needs to agree on coarser granularity. Not the spec's fault.

### Middle score (e.g. 50-80%) — split analysis required

This is the most informative pattern. Almost always the failing windows share a common feature.

Diagnostic procedure:
1. List failing windows.
2. For each failing window, extract pre-state pattern (which var values are distinctive).
3. Cluster: do all failures have a shared attribute?
   - Same actor value?
   - Same pre-state for some field?
   - Same pre-post transition shape?
4. If yes → that's the diagnostic. Explain it.
5. If no → iterate: dump TLC traces for sample failures and look for pattern.

**Known pattern from spin prototype**: 123/210 pass on AcquireLock. All 123 pass had `pc[actor] = "trying"` in pre. All 87 fail had `pc[actor] = "idle"`. Reason: trace skipped the intermediate `trying` state in no-contention paths. Trace non-compliant in some windows.

Fix: filter out the non-compliant windows at Step 0 (per-window contract check), OR report the split clearly.

### Random failures (no pattern)

Rare. If 43% fail with no obvious commonality:
1. Sample 3–5 failing windows.
2. For each, dump TLC explanation of why.
3. Compare. If they're all different reasons, the spec genuinely has coverage gaps.

This IS a spec quality signal. Document it.

## Contract violation signals (when Step 0 was missed)

Red flags that suggest you should have stopped at Step 0:
- An action's FAIL count equals exactly the count of windows with a specific pre-state feature.
- A required state value (per task) appears 0 times in all traces.
- An action's precondition cannot be satisfied by ANY window's pre (systematic block).

Seeing any of these late in the process → rewind, re-do Step 0, reject non-compliant data.

## What to write in the report

For each action, produce a block like:

```markdown
### Action: <Name>

**Pass rate**: X / N (Y%)

**Explanation**: <one paragraph>

**Evidence**:
- Example passing window: window_id=K (<trace>, <actor>). Pre <..>. Post <..>.
  Spec's action produces exactly this post.
- Example failing window: window_id=M. Pre <..>. Post <..>.
  Spec's action produces <actual post> instead. Cause: <diagnostic>.

**Classification**:
- [ ] Spec reproduces system (score 100%, verified by sanity check)
- [ ] Spec correct, some windows non-compliant (middle score, pattern identified)
- [ ] Spec incorrect (investigated, genuine quality issue)
- [ ] Trace non-compliant (should have been caught in Step 0)
```

Never produce a score without the Classification line filled in.
