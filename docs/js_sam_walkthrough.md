# JS-SAM in SysMoBench — A Walkthrough From Scratch

*What was built, what was tested, and what it actually demonstrates.*

---

## 1. The project this plugs into

**SysMoBench** (github.com/specula-org/SysMoBench) is a benchmark that asks one
question: *can an AI model read the source code of a real system and write a
correct formal specification of its behavior?*

It ships 11 real systems as tasks — OS synchronization primitives from the
Asterinas kernel (`spin`, `mutex`, `rwmutex`, `ringbuffer`) and distributed
systems (`etcd`, `zookeeper`, `redisraft`, and others). A benchmark run works
like this:

1. **Generate.** An LLM is shown the system's source code (e.g. the Rust
   spinlock from Asterinas) plus a task prompt, and asked to produce a
   specification.
2. **Phase 1 — Syntax.** Does the spec parse and type-check?
3. **Phase 2 — Runtime.** A model checker explores the spec's own state
   space. Does exploration terminate cleanly (no crashes, no deadlocks)?
4. **Phase 3 — Trace validation.** The benchmark has traces captured from
   the *real running system* — sequences of `(state before, action, state
   after)`. Does the spec admit each real transition? This is the "is the
   spec faithful to reality" test, weighted 35% of the score.
5. **Phase 4 — Invariant verification.** Experts wrote invariants per system
   in plain language ("at most one thread holds the lock"). They get
   translated into the spec's vocabulary and checked. Also 35%.

Everything language-specific lives behind one plugin interface called
`LanguageBackend`. Before this work there were three backends, all **formal
specification languages**: TLA+ (checked with TLC), Alloy (bounded relational
checker), and PAT (CSP# process algebra). Adding a language means writing one
backend subclass — the four phase evaluators never change.

## 2. What we added and why

We added a fourth language: **JS-SAM** — JavaScript using the
[SAM pattern](https://sam.js.org) (State-Action-Model), via the
`@cognitive-fab/sam-pattern` and `@cognitive-fab/sam-fsm` npm packages.

SAM is a JavaScript programming pattern deliberately built on TLA+'s
semantics: **actions** compute *proposals*, the **model** accepts or rejects
each proposal in a single synchronized step, and **state** is a pure function
of the model. So it has the same conceptual skeleton as a TLA+ spec — but the
artifact is an executable `.js` module, not formal-language source.

That makes it the benchmark's first *non-formal-language* backend, and it
tests a genuinely different hypothesis:

> LLMs have seen JavaScript constantly in training and TLA+/Alloy/CSP#
> rarely. If models score much better in JS-SAM, formal *syntax* is the
> bottleneck. If they score the same or worse, the hard part is *modeling*,
> not syntax. Either result is informative.

A practical bonus: `sam-pattern` ships its own bounded model checker (a
"behavior explorer" that tries all action permutations to a depth limit), so
Phases 2 and 4 need no external tool — and because a SAM model is literally
`model.present(action(data))`, Phase 3 trace replay becomes trivial: put the
model in the trace's pre-state, fire the action, diff the result against the
trace's post-state.

## 3. What was built

### The module contract (what a generated spec must look like)

A JS-SAM spec is a CommonJS module that must export six things:

```js
module.exports = {
  instance,       // the SAM instance (synchronous mode is mandatory)
  init,           // () => void — reset to the initial state
  actions,        // { AcquireLock: (data) => void, ReleaseLock: (data) => void }
  getState,       // () => plain JSON snapshot of the model
  setState,       // (snapshot) => void — force the model into a given state
  checkerIntents, // input domains for the explorer, e.g. every {thread, callType} combo
};
```

Each export exists for a phase: `getState`/`setState` make Phase 3 replay
possible, `instance` + `checkerIntents` are what the bundled explorer needs
for Phases 2/4, and the structural check of this whole shape *is* Phase 1's
semantic check. The action names (`AcquireLock`, `ReleaseLock` for the
spinlock) are a hard contract from the task config — same rule as the other
three languages.

### The pieces (mirroring how Alloy and PAT are integrated)

| Piece | File(s) | What it does |
|---|---|---|
| Node helper | `tools/js-sam/cli.mjs` | The "checker binary". Five subcommands (`ping`, `validate`, `check`, `transitions`, `invariants`), each reading one JSON request on stdin and writing one JSON response on stdout. |
| Python backend | `tla_eval/languages/js_sam.py` | The `LanguageBackend` subclass. Maps each phase to a helper subcommand; auto-registered, so `--language JS-SAM` just works. |
| Direct Phase 3 path | `tla_eval/evaluation/semantics/trace_loader.py` + the previously-stubbed branch in `transition_validation.py` | **First use in the bench.** TLA+/Alloy/PAT validate traces by launching a coding agent (30 min–hours, $1–4 per run). JS-SAM replays windows directly in seconds with no LLM involved. The trace loader (NDJSON → windows) is language-neutral so any future backend gets it for free. |
| Prompts | `tla_eval/tasks/spin/prompts/js-sam/` | The generation prompt (with the contract and a complete **Rocket Launcher** example — the SAM community's canonical small system), a correction prompt, and the invariant-translation prompt. |
| Invariant templates | `data/js_sam_invariant_templates/spin/invariants.yaml` | 3 starter safety invariants reused from the Alloy library: MutualExclusion, LockStatusConsistency, NoDeadlock. |

Two incidental fixes the integration surfaced:

- `scripts/run_benchmark.py` demanded Java + TLA+ tools for *every* language.
  It now gates on them only when TLA+ is selected — a Node-only machine can
  run JS-SAM end to end.
- The docs and code disagreed about who loads traces for the direct Phase 3
  path (a stub that had never been exercised). Resolved: the evaluator loads
  and windows traces; the backend validates them.

Two library quirks discovered by reading `sam-pattern`'s source, both handled
in the helper: acceptor exceptions surface as unhandled promise rejections
(which would otherwise kill the process — the helper captures and attributes
them), and a rejected/gated action reports itself through the same error slot
as a crash (the helper distinguishes "the model said no", which is legal,
from "the model blew up", which is a failure).

## 4. The samples

### The reference spinlock model — `tests/fixtures/js_sam/specs/spin-good.js`

A complete, hand-written JS-SAM spec of the Asterinas spinlock — effectively
the "gold answer" for the spin task. Its state:

```js
{
  lockHeld: false,                              // is the lock taken?
  lockHolder: null,                             // which thread holds it (0, 1, or null)
  threadStatus: { 0: 'idle', 1: 'idle' },       // idle | trying | locked
  callType: { 0: null, 1: null },               // pending blocking call, if any
}
```

It models the semantic distinction the benchmark cares about most:
**`lock()` blocks** (a thread that loses the CAS race goes to `'trying'` and
spins) while **`try_lock()` never blocks** (a loser goes straight back to
`'idle'`). Guards make invalid operations no-ops — e.g. a release by a thread
that doesn't hold the lock changes nothing, which is exactly how you want a
model to behave under exhaustive exploration.

### Five deliberately broken samples

Each exists to prove one failure path is *detected and correctly classified*:

| Fixture | The defect | What must catch it |
|---|---|---|
| `spin-syntax-error.js` | unbalanced brace | Phase 1, as a **syntax** error |
| `spin-load-error.js` | throws while being imported | Phase 1, as a **load** error |
| `spin-missing-export.js` | no `setState`, no `checkerIntents` | Phase 1, as **contract** violations (one message per missing export) |
| `spin-bad-release.js` | `ReleaseLock` frees the lock but never clears `lockHolder`/status | Phase 3 — and *only* on release windows |
| `spin-throwing.js` | an acceptor throws on a reachable state | Phase 2, classified `runtime_error` |

### Synthetic traces

Two small NDJSON files emulate what the real instrumentation harness
produces: a **stream form** (one state snapshot per line; consecutive lines
form windows) and an **event form** (each line carries its own pre/post
state). The real captured traces for `spin` are *not* in the repository —
they come from a Docker/QEMU harness — which is why synthetic ones were
needed and why that gap is flagged to the maintainers (more in §6).

### The Rocket Launcher

Embedded in the generation prompt as the in-context example the LLM sees: a
countdown (10 → 0), `Start` / `Decrement` / `Abort` actions, and a reactor
that flips status to `launched` at zero. Small enough not to leak the
spinlock solution, complete enough to teach the whole module contract.

## 5. What was run, and what each run demonstrated

### Direct helper smoke tests (run during development)

- **Phase 2 on the good model:** explored **326,592 states in ~2.7 s** at the
  bench depth bound of 6, twice (the second pass is a determinism check),
  with zero violations. Demonstrates the bundled explorer is fast enough that
  depth 6 costs seconds, not minutes.
- **Phase 3 with a deliberately wrong expectation:** fed a window claiming a
  failed `try_lock` leaves the thread `'trying'`. The replay failed exactly
  that window with the pinpoint diff
  `threadStatus.1: expected "trying", got "idle"` — demonstrating failures
  come with actionable, field-level reasons.
- **Phase 4 with one true and one false invariant:** `MutualExclusion`
  passed; a deliberately false "the lock is never held" failed **with a
  counterexample behavior** — the action sequence that reaches the violating
  state — demonstrating the explorer produces evidence, not just a verdict.

### The automated test suite — 29 tests, all passing

`tests/test_languages/test_js_sam.py` (20 tests) and
`tests/test_evaluation/test_trace_loader.py` (9 tests):

- **Registration:** `js-sam`, `JS-SAM`, `SAM`, `jssam` all resolve to the
  backend through the existing registry; both ```` ```javascript ```` and
  ```` ```js ```` fences are extracted from model output.
- **Phase 1:** the good spec passes; each broken fixture fails with the
  *right class* of error (syntax vs load vs contract).
- **Phase 2:** good spec explores cleanly; the throwing fixture is classified
  `runtime_error` with the actual exception message surfaced.
- **Phase 3:** the good spec passes all windows at 100%; the buggy-release
  spec scores `AcquireLock: 1.0, ReleaseLock: 0.0` — the bug is detected
  *and attributed to the right action*. Unknown actions count as failed
  windows (the naming contract is enforced), and an empty window set is an
  error rather than a silent vacuous pass.
- **Phase 4:** pass/fail discrimination with counterexample metadata; a
  template with no translation is reported as a translation failure, not
  skipped.
- **Trace loader:** both NDJSON forms parse; windows chain correctly
  (each window's pre-state is the previous window's post-state); off-target
  actions are filtered; missing/empty trace folders raise actionable errors
  instead of producing an empty (and misleadingly perfect) result.
- **Direct-path evaluator:** the full wiring — evaluator → trace loader →
  backend → Node helper → score — produces correct totals and pass rates;
  and the realistic failure mode (no captured traces in the repo) is
  reported cleanly instead of crashing.

### End-to-end CLI checkpoint (the integration doc's §6 acceptance test)

```
python scripts/run_benchmark.py --task spin --method direct_call --model claude \
  --metric compilation_check --language JS-SAM \
  --spec-file tests/fixtures/js_sam/specs/spin-good.js
```

produced the expected sequence — prompt resolved for `language=JS-SAM`,
`Evaluating compilation (JS-SAM): spin/direct_call/claude`, experiment
directory created under `output/compilation_check/js-sam/...`, and
`Metric 'compilation_check': PASS`. This demonstrates the backend is wired
into the real benchmark pipeline, not just unit-tested in isolation.

### Regression check

The full pre-existing suite was run with and without our changes: the same
2 failures + 1 error exist in `tests/test_models/` on a clean checkout
(model-provider config tests, unrelated to this work). Nothing we touched
regressed.

### A live benchmark run with Claude (2026-06-09)

After the fixture-based validation above, the pipeline was exercised with a
real model: **claude-sonnet-4-20250514** generated a JS-SAM spec for the
`spin` task from the actual Asterinas Rust source
(`run_benchmark.py --language JS-SAM`, real generation, no `--spec-file`).

| Stage | Result | Detail |
|---|---|---|
| Generation | valid on attempt 1 of 3 | 15.0 s, 3,922 prompt + 1,293 completion tokens, no correction loop |
| Phase 1 — syntax/contract | PASS | 0 syntax errors, 0 semantic errors, 0.09 s |
| Phase 2 — exploration | PASS | depth 6, ~4 s, no runtime errors, no nondeterminism |
| Phase 4 — invariants | PASS 3/3 | live translation by Claude (3.3 s) + explorer checks (5.5 s): MutualExclusion, LockStatusConsistency, NoDeadlock |
| Phase 3 — traces | not run | captured `spin` traces not in the repository (open question on the PR) |

The generated spec was the model's own modeling, not a copy of the prompt's
Rocket Launcher example. Its most interesting decision: a **reactor** that
grants the freed lock to a spinning thread the instant `ReleaseLock` fires —
a defensible reading of "the thread spins until acquisition," but it makes
lock hand-off atomic with the release step. In the real system's traces a
release leaves the waiting thread `trying` until its *own* next CAS attempt
wins. Phases 1, 2, and 4 cannot see that divergence; trace validation exists
precisely to catch it — concrete evidence for why the trace-availability
question matters most for the JS-SAM leaderboard number.

The run also surfaced one fix now included in this branch:
`config/models.yaml` requested `max_tokens: 200000` for the `claude` entry,
which the Anthropic API rejects for claude-sonnet-4 (output cap 64,000).
Lowered to 64,000 with a dated probe note, matching the style of the
existing `deepseek_tencent` comment. The same `claude` entry is the
translator Alloy's and PAT's Phase 4 remaps rely on, so the fix is not
JS-SAM-specific.

## 6. What is demonstrated — and what is not yet

**Demonstrated:**

- A non-formal, executable language fits the four-phase architecture with
  zero changes to the evaluators' interfaces — one backend class + one
  helper, exactly like Alloy and PAT.
- The evaluation pipeline has *discriminative power* for JS-SAM: every
  seeded defect class is caught, correctly classified, and attributed
  (right phase, right action, right field, with counterexamples).
- The direct Phase 3 path — designed into the architecture but never used —
  works end to end, and cuts trace validation from "30 min–hours and $1–4
  of agent spend" to seconds at zero marginal cost.
- The whole thing runs on a machine with only Node and Python (no Java, no
  mono).

**Also demonstrated (live run, see §5):** one real model (claude-sonnet-4)
generating a spec that passes Phases 1, 2, and 4 first-try, including the
live invariant-translation API path.

**Not yet demonstrated (the honest gaps):**

- **The hypothesis itself is untested.** One model passing one task is an
  existence proof that the pipeline works end to end with real generation —
  not the comparison. The actual experiment is multiple models, multiple
  runs, JS-SAM scores side by side with TLA+/Alloy/PAT on the same task.
- **Phase 3 leaderboard numbers are blocked on real traces.** The captured
  `spin` traces aren't in the repository; our tests use synthetic ones.
  Raised with the maintainers as an open question on the PR. The live run
  makes this concrete: the generated spec's instant-handoff reactor would
  plausibly diverge from real traces, and only Phase 3 can adjudicate that.
- Three invariants vs Alloy's six for spin; the depth bound (6) is a single
  bench-wide constant. Both flagged as deliberate first-cut choices.

## 7. Where everything stands

- Branch `full-stack`, commit `0094a35` — 26 files, ~2,900 lines.
- Pull request: **specula-org/SysMoBench#17**, opened from the fork
  `jdubray/SysMoBench-1` (you lack write access to specula-org).
- Design record: `docs/js_sam_backend_spec.md` (user stories, acceptance
  criteria, decision log, implementation map).

Try it yourself:

```bash
# tooling sanity check
python -c "from tla_eval.languages import get; print(get('js-sam').check_available())"

# the test suite
python -m pytest tests/test_languages tests/test_evaluation -v

# one helper call by hand (Phase 2 on the reference model)
echo '{"specPath":"tests/fixtures/js_sam/specs/spin-good.js","depthMax":6}' | node tools/js-sam/cli.mjs check
```
