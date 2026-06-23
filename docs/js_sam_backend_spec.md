# JS-SAM Language Backend — Specification

**Status:** Implemented (first cut) — see §11 for the implementation map
**Author:** JJ Dubray (jdubray@gmail.com)
**Date:** 2026-06-09
**Depends on:** `docs/add_new_spec_language.md` (the integration contract this spec instantiates)

---

## 1. Summary

Add a fourth specification language to SysMoBench: **JavaScript using the SAM
pattern** ([sam.js.org](https://sam.js.org)), via the
[`@cognitive-fab/sam-pattern`](https://www.npmjs.com/package/@cognitive-fab/sam-pattern)
and [`@cognitive-fab/sam-fsm`](https://www.npmjs.com/package/@cognitive-fab/sam-fsm)
npm packages.

SAM is grounded in TLA+ semantics — actions present proposals, a model accepts
or rejects them, state is a pure function of the model — but the artifact is an
**executable `.js` module**, not formal-spec source. This is the bench's first
non-formal-language backend, and it tests a distinct hypothesis:

> Can LLMs produce correct system specifications in a pattern they see
> constantly in training data (JavaScript), rather than in formal languages
> (TLA+, Alloy, CSP) they rarely see?

### Hypothesis & success measure

The backend is informative if, on the same task with the same models, JS-SAM
scores diverge measurably from the formal-language scores in either direction.
Either outcome is a publishable signal:

- JS-SAM ≫ formal languages → familiarity with the host language dominates
  spec quality; formal syntax is the bottleneck.
- JS-SAM ≈ or ≪ formal languages → the hard part is modeling, not syntax.

---

## 2. Scope

### In scope (first cut)

| Item | Decision |
|------|----------|
| Target task | `spin` only (Asterinas spinlock) — parity with Alloy and PAT's historical scope |
| Generation method | `direct_call` (one prompt) |
| Invariants | Three starter invariants drawn from the generic spinlock set (see §8) |
| In-context example | Rocket Launcher — the SAM community's canonical small system |
| Phases | All four (Phase 3 via the **direct path**, a bench first — see §7) |
| Agent translators | Remapped to a direct API call, same as Alloy (`alloy.py:358`) and PAT (`pat.py:340`) |

### Out of scope (first cut)

- Tasks beyond `spin`. (Candidate second target: `ringbuffer` or `locksvc`
  for richer control state — deferred until the team answers Open Question 2.)
- A real agent-based invariant translator (no backend has one except TLA+;
  see Open Question 4).
- Extending the TLA+-shaped `tv-eval` agent skill — the direct path makes it
  unnecessary for JS-SAM.
- Leaderboard weighting changes. JS-SAM reuses the existing metric weights
  (syntax 0.15, runtime 0.15, transition 0.35, invariant 0.35).
- TypeScript output, browser execution, or any UI concern. The spec artifact
  is a plain Node-executable module.

---

## 3. Architecture

One `LanguageBackend` subclass plus a small Node helper the Python side shells
out to with JSON-in / JSON-out — the same shape as the Alloy backend (Java
helpers over `lib/alloy.jar`) and the PAT backend (`mono PAT3.Console.exe`).

```
tla_eval/languages/js_sam.py          # JsSamBackend(LanguageBackend) + register()
tools/js-sam/                          # Node helper (the "checker binary")
  package.json                         # pins @cognitive-fab/sam-pattern, @cognitive-fab/sam-fsm
  package-lock.json
  cli.mjs                              # subcommands: validate | check | transitions | invariants
tla_eval/tasks/spin/prompts/js-sam/   # per-language prompts (loader lowercases name)
  direct_call.txt
  agent_correction.txt
data/js_sam_invariant_templates/spin/
  invariants.yaml                      # `javascript_example` snippet field
tests/test_languages/test_js_sam.py   # backend unit tests (TDD)
tests/fixtures/js_sam/                 # known-good / known-bad sample specs, synthetic traces
```

Evaluator-side work (the one exception to "no evaluator changes"):

```
tla_eval/evaluation/semantics/trace_loader.py          # NEW — language-neutral NDJSON loader + windower
tla_eval/evaluation/semantics/transition_validation.py # implement the stubbed direct path (lines 86–94)
```

### Helper protocol

Every `cli.mjs` subcommand reads a single JSON request on stdin and writes a
single JSON response on stdout. Exit code 0 means "the helper ran" — pass/fail
is carried in the JSON (`{"success": bool, "errors": [...], ...}`), mirroring
how the PAT backend text-detects failures because PAT always exits 0. Crashes
(non-zero exit, malformed JSON) are reported as tooling errors, not spec
failures.

### Backend identity fields

| Field | Value | Rationale |
|-------|-------|-----------|
| `name` | `"JS-SAM"` | Distinguishes the pattern from generic JavaScript |
| `aliases` | `("js-sam", "jssam", "sam", "javascript-sam")` | Case-insensitive CLI resolution |
| `fence_label` | `"javascript"` | Models emit ```` ```javascript ```` fences natively; fighting that costs Phase 1 points for the wrong reason |
| `config_fence_label` | `None` | No separate config artifact (like Alloy/PAT) |
| `spec_extension` | `".js"` | CommonJS for the first cut (see §4 module contract) |
| `config_extension` | `None` | — |
| `invariant_template_dirname()` | `"js_sam_invariant_templates"` | Parallel to `alloy_invariant_templates` |
| `invariant_example_field()` | `"javascript_example"` | Default derivation from `fence_label` — no override needed |

`check_available()` verifies: `node --version` ≥ the pinned engine,
`tools/js-sam/node_modules` present (or runs `npm ci` guidance), and returns a
one-line actionable message on failure — same pattern as
`AlloyBackend.check_available` / `PATBackend.check_available`.

---

## 4. The generated-spec module contract

The prompt instructs the model to emit one fenced `javascript` block that is a
CommonJS module with this export shape:

```js
module.exports = {
  instance,       // the SAM instance (createInstance({ hasAsyncActions: false }))
  init,           // () => void — (re)initialize the model to its initial state
  actions,        // { AcquireLock: (data) => void, ReleaseLock: (data) => void, ... }
  getState,       // () => object — pure JSON-serializable snapshot of the model
  setState,       // (snapshot) => void — force the model to a given snapshot (Phase 3)
  checkerIntents, // [{ name, intent, values }] — explorer descriptors (Phases 2 & 4)
};
```

> **Contract update vs the original draft.** The draft proposed four exports;
> validating against `@cognitive-fab/sam-pattern@1.6.1` showed the bundled
> `checker()` needs the SAM `instance` plus intent descriptors with value
> domains (`{ intent, name, values }`, exactly as in the library's Die Hard
> example), so the contract carries six exports. `values` is the action's
> input domain — for spin, every `{thread, callType}` combination — which is
> genuinely part of the model, like a TLA+ constants block.

Contract rules (enforced structurally in Phase 1, semantically in Phases 2–4):

1. `actions` MUST contain one function per name in
   `task.yaml > tv.target_actions` (for `spin`: `AcquireLock`, `ReleaseLock`).
   This is the same hard contract every language's prompt carries
   (`add_new_spec_language.md` §5).
2. `getState()` MUST return a plain JSON-serializable object — no functions,
   no classes, no cycles. This is what trace states are diffed against.
3. `setState(snapshot)` MUST be a legal entry point for any snapshot
   `getState()` could produce. It exists solely so Phase 3 can replay
   `(pre, action, post)` windows without searching for a path to `pre`.
4. The module MUST NOT perform I/O, use timers, or depend on wall-clock time
   or randomness. Violations surface as Phase 2/3 nondeterminism failures.

> **Resolved (was TBD):** the construction idiom was validated against the
> installed packages (`sam-pattern 1.6.1`, `sam-fsm 1.0.0`). The prompt's
> Rocket Launcher example pins it: `createInstance({ hasAsyncActions: false })`
> (synchronous steps are mandatory), plain actions returning proposals with
> `__name` set, guard-style acceptors, `getState` via a sanitizing
> `JSON.stringify` replacer over `instance({}).state()`, and `setState` via
> `instance({ initialState: clone(snapshot) })` (the library's
> `addInitialState` merges into the live model). Notable library behaviors
> the helper accounts for: acceptors run inside an async wrapper, so acceptor
> throws surface as unhandled promise rejections (captured by the helper, not
> fatal); gated actions reject by presenting `__error: 'unexpected action …'`,
> which the helper treats as legal rejection rather than a runtime fault.

---

## 5. User stories — backend

Stories follow the bench's four phases. "Helper" means `tools/js-sam/cli.mjs`.

### US-1 — Backend registration

**Description:** As a bench maintainer, I want `--language JS-SAM` resolved by
the existing registry so that no CLI or evaluator plumbing changes are needed.

**Acceptance criteria:**
- `tla_eval/languages/js_sam.py` subclasses `LanguageBackend`, sets the
  identity fields in §3, and calls `register(JsSamBackend())` at import side.
- `from tla_eval.languages import get; get("js-sam")`, `get("SAM")`, and
  `get("JS-SAM")` all return the backend (auto-discovery via the lazy
  bootstrap — no `__init__.py` edit).
- `get("JS-SAM").check_available()` returns `None` on a machine with Node and
  installed helper deps, and an actionable one-liner (naming the missing tool
  and the install command) otherwise.
- The default `extract_artifacts()` pulls the spec from a
  ```` ```javascript ```` fence with no override.

### US-2 — Phase 1: syntax / structural check

**Description:** As a model evaluator, I want a generated JS-SAM spec
parse-checked and structurally validated so that Phase 1 measures "is this a
well-formed SAM module" and not merely "is this JavaScript".

**Acceptance criteria:**
- `validate_syntax()` writes the spec to `work_dir / spec_filename` (or a
  default name) and invokes `cli.mjs validate`.
- The helper performs, in order, reporting the first failure class:
  1. **Parse check** — load the file; a `SyntaxError` populates
     `SyntaxOutcome.syntax_errors` with message + line number.
  2. **Load check** — `require()` the module in a fresh process; import-time
     throws populate `semantic_errors`.
  3. **Structural check** — exports match the §4 contract: `init`, `actions`,
     `getState`, `setState` present and callable; `actions` contains every
     name from the request's `target_actions` list; missing/extra-typed
     exports populate `semantic_errors` with one entry per violation.
- A known-good fixture spec yields `SyntaxOutcome(success=True)`; fixtures
  with (a) a syntax error, (b) an import-time throw, (c) a missing target
  action each yield `success=False` with the correct error class populated.
- Helper crash or timeout yields `success=False` with `error_message` set and
  empty error lists (tooling failure, not spec failure).

### US-3 — Phase 2: bounded behavior exploration

**Description:** As a model evaluator, I want the spec's own state space
explored by the SAM behavior explorer so that Phase 2 measures whether the
model terminates cleanly under bounded exploration — the same question TLC,
AlloyRuntime, and PAT answer for the other languages.

**Acceptance criteria:**
- `run_model_checker()` invokes `cli.mjs check` with the spec path, the
  depth bound, and the timeout; no separate config artifact is required
  (the base `generate_default_config()` returning `(True, "", None)` is
  used unchanged, like Alloy/PAT).
- The depth bound defaults to **`depthMax: 6`** (the sam-pattern explorer
  default) and is read from a single bench-side constant so changing it later
  is a config edit, not a code change. The effective bound is recorded in
  `ModelCheckOutcome.raw_output`. (Bench-wide vs per-task is Open Question 3;
  the first cut hardcodes bench-wide.)
- Exploration that completes without an acceptor exception, unhandled throw,
  or non-serializable state yields `success=True`.
- Failures populate `classification` with one of:
  `"violation"` (an explorer-detected predicate/acceptor failure),
  `"runtime_error"` (unhandled throw during an action or present),
  `"nondeterminism"` (two runs from the same state diverge),
  `"timeout"`, `"parse_error"` — aligning with the classification vocabulary
  TLA+ already emits so downstream aggregation needs no changes.
- The whole exploration respects the evaluator's `timeout`; the helper is
  killed and `classification="timeout"` reported if exceeded.

### US-4 — Phase 3: direct transition validation

**Description:** As a model evaluator, I want each captured
`(pre_state, action, post_state)` trace window replayed against the SAM model
directly — `setState(pre)`, `actions[name](data)`, diff `getState()` against
`post` — so that Phase 3 needs no coding-agent orchestration and costs
seconds, not the agent path's "30 min to several hours and $1–4".

**Acceptance criteria:**
- `JsSamBackend.supports_direct_transition_validation = True` — the first
  backend in the bench to set it.
- `validate_transitions(spec_path, trace_windows, work_dir, timeout)` batches
  windows to `cli.mjs transitions` and returns a `TransitionOutcome` with
  `per_action_pass_rates`, `total_passed`, `total_windows` populated.
- Per-window semantics: a window **passes** iff `setState(pre_state)` then
  `actions[action](data)` completes without throwing AND the resulting
  `getState()` matches `post_state` under the **projection rule**: every key
  present in the trace's `post_state` must be deeply equal in the model's
  state; model-only auxiliary keys are ignored. (Rationale: SAM models may
  carry bookkeeping the instrumented kernel doesn't emit; the trace is the
  source of truth for the variables it covers. This mirrors how the tv-eval
  agent skill maps trace variables onto spec variables rather than requiring
  bijection.)
- A window whose `action` is absent from the module's `actions` map counts as
  failed for that action group (not a tooling error) — the prompt made the
  action names a hard contract.
- A deterministic synthetic-trace fixture exists in `tests/fixtures/js_sam/`:
  a hand-written correct spin model passes 100% of a hand-written NDJSON
  trace; a deliberately broken model (e.g. `ReleaseLock` that doesn't clear
  the holder) fails exactly the windows that exercise the bug.

### US-5 — Phase 4: invariant translation and checking

**Description:** As a model evaluator, I want the expert spinlock invariants
translated into JavaScript predicates over the spec's own state shape and
checked by the behavior explorer, so that Phase 4 measures whether the
generated model is *verifiable*, not just runnable.

**Acceptance criteria:**
- `translate_invariants()` supports `translator="claude"` and explicit direct
  model names via `model.generate_direct()`; `"claude-code"` and `"codex"`
  are remapped to the direct call with a code comment marking the remap —
  byte-for-byte the same policy as `pat.py:330-369`. Unsupported modes return
  `({}, "translator '<x>' not supported by JS-SAM backend")`.
- A translated invariant is a JavaScript predicate of shape
  `(state) => boolean` (safety) — the translation prompt includes the spec
  source so the predicate is written against the *generated* state shape,
  not an assumed one.
- `check_invariants()` invokes `cli.mjs invariants`, which runs the behavior
  explorer with each predicate installed as a safety check, and returns one
  `InvariantCaseResult` per template: `success`, the `translated` text, the
  explorer's `raw_output`, and on failure a counterexample trace prefix in
  `metadata["counterexample"]`.
- A template with no matching `translated` entry is reported as a translation
  failure (per the `base.py:check_invariants` docstring), not silently
  skipped.
- Liveness templates: the first cut declares **safety-only**, matching
  Alloy's precedent (`data/alloy_invariant_templates/spin/invariants.yaml`
  metadata: "Liveness properties are not included as Alloy performs bounded
  model checking"). The bounded explorer has the same limitation. The
  invariants.yaml metadata note states this explicitly.

### US-6 — Prompts and in-context example

**Description:** As a bench maintainer, I want JS-SAM prompts for `spin` that
carry the same hard contracts as the other languages' prompts so that
generation failures are attributable to the model, not the harness.

**Acceptance criteria:**
- `tla_eval/tasks/spin/prompts/js-sam/direct_call.txt` exists (resolution
  path 1 from `add_new_spec_language.md` §5; no legacy fallback applies to
  non-TLA+ languages — a missing prompt is a hard error, which the test
  suite asserts).
- The prompt specifies: (1) the ```` ```javascript ```` fence, (2) the
  mandatory action names `AcquireLock` / `ReleaseLock` from
  `task.yaml > tv.target_actions`, (3) the §4 export contract verbatim,
  (4) the no-I/O / no-randomness determinism rules, and (5) the complete
  **Rocket Launcher** SAM model as the in-context example (sourced from
  [sam-samples](https://github.com/jdubray/sam-samples), adapted to the §4
  contract).
- `tla_eval/tasks/spin/prompts/js-sam/agent_correction.txt` exists and
  reuses `fence_format_hint()`'s single-block phrasing (the base default —
  no override needed).

### US-7 — Starter invariant templates

**Description:** As a model evaluator, I want a `spin` invariant library for
JS-SAM so that Phase 4 is exercisable on day one.

**Acceptance criteria:**
- `data/js_sam_invariant_templates/spin/invariants.yaml` exists with **three**
  starter invariants, drawn from the generic set the Alloy library already
  defines: `MutualExclusion`, `LockStatusConsistency`, `NoDeadlock`
  (all `type: "safety"`).
- Each entry carries `name`, `type`, `natural_language`,
  `formal_description` (reused verbatim from the Alloy library where
  applicable) and a `javascript_example` reference snippet written against a
  *plausible* state shape, labeled as illustrative — the translator rewrites
  it against the actual generated spec.
- Parity note recorded in the YAML metadata: Alloy ships 6 invariants for
  spin (4 generic + 2 Asterinas-specific: `TryLockNonBlocking`,
  `BlockingVsNonBlockingSemantics`); extending JS-SAM from 3 → 6 is a
  follow-up item, not first-cut scope.

---

## 6. User stories — evaluator (the one cross-cutting change)

### US-8 — Implement the direct-path wrapper in `TransitionValidationEvaluator`

**Description:** As a bench maintainer, I want the stubbed direct path in
`transition_validation.py:86-94` implemented so that any backend declaring
`supports_direct_transition_validation = True` gets traces loaded, windowed,
and dispatched — JS-SAM is the first consumer, but the wrapper must be
language-neutral.

**Acceptance criteria:**
- A new language-neutral `trace_loader.py` module:
  - resolves the trace directory from `task.yaml > traces_folder`
    (for `spin`: `data/sys_traces/spin`),
  - parses NDJSON trace files and yields `(action, pre_state, post_state)`
    windows matching the `base.py:259` signature,
  - filters to `task.yaml > tv.target_actions`,
  - raises a clear error (not an empty iterator) when the trace folder is
    missing or empty — see the dependency note below.
- `TransitionValidationEvaluator.evaluate()` replaces the lines 86–94 stub:
  when `backend.supports_direct_transition_validation`, it loads windows via
  the trace loader, calls `backend.validate_transitions(...)` with the
  configured timeout, and maps `TransitionOutcome` into
  `TransitionValidationResult` (`per_action_pass_rates`, `total_passed` →
  `total_passed`, `total_windows` → `total_windows`, score =
  passed/windows, `overall_success` per the existing policy at line 208).
- The agent path is untouched: TLA+/Alloy/PAT runs through
  `launch_tv_eval.sh` exactly as before (regression-tested).
- **Contract clarification to land with this change:** the code comment at
  `transition_validation.py:87-88` says "Trace loading + windowing is the
  backend's responsibility", but the `base.py:256-262` signature passes
  `trace_windows` *in* — i.e. the evaluator owns loading. This spec resolves
  the contradiction in favor of the signature (loader is shared and
  language-neutral; windows-in is the contract); the stale comment and
  `add_new_spec_language.md` §3.3 are updated in the same PR.

**Dependency / risk:** `data/sys_traces/spin` is referenced by `task.yaml`
but **not checked into the repository** — traces are produced by the Docker/
QEMU instrumentation harness (`tv.harness` in `task.yaml`). Phase 3
end-to-end therefore requires either (a) running the trace-capture harness,
or (b) the maintainers publishing the captured traces. The unit-test story
(US-4) is insulated from this via synthetic fixture traces, but the
*leaderboard* Phase 3 number is blocked on real traces. Flag on the
integration thread alongside the PR.

---

## 7. Why the direct path (and not the agent path)

`add_new_spec_language.md` §3.3 offers two Phase 3 routes. The agent path
requires extending the TLA+-shaped `tv-eval` skill — "a larger project than
writing a backend" per the doc — and costs $1–4 plus up to hours per run.
JS-SAM is the textbook direct-path case the doc anticipates: the language's
own runtime answers "does `pre ∧ action ⇒ post'` hold?" by construction,
because SAM **is** `model.present(action(data))` — trace replay reduces to
`setState(pre)`, `actions[name](data)`, state diff. No orchestration, no
search, no LLM in the loop. This is also why US-8 lands with this backend:
the doc explicitly assigns trace-loader threading to "the first language
that uses it."

---

## 8. Test plan (TDD)

Per project convention, tests precede implementation:

| Layer | Tests | Fixtures |
|-------|-------|----------|
| Helper (`cli.mjs`) | Node-side unit tests per subcommand: good/bad parse, structural violations, explorer pass/violation/timeout, window pass/fail/projection | `tests/fixtures/js_sam/specs/*.js` |
| Backend (`js_sam.py`) | Python unit tests mocking the subprocess boundary: JSON marshalling, outcome mapping, error-class routing, `check_available` messages | canned helper JSON responses |
| Evaluator (US-8) | Trace loader: NDJSON parsing, windowing, action filtering, missing-folder error. Direct-path dispatch with a stub backend. Agent-path regression (TLA+ unaffected) | `tests/fixtures/js_sam/traces/*.ndjson` (synthetic) |
| End-to-end | The §6 checkpoint from `add_new_spec_language.md`: `run_benchmark.py --metric compilation_check --language JS-SAM --spec-file <fixture>` produces the expected log sequence and experiment directory | hand-written correct spin model |

---

## 9. Open questions for the team (decision log)

| # | Question | First-cut position pending decision |
|---|----------|--------------------------------------|
| 1 | Does the bench want a runtime-pattern language at all? The leaderboard story shifts: "did the LLM write a working SAM model" vs "a sound TLA+ spec". | Proceed to spec/scaffold; hold the PR merge on this answer. The architecture (`add_new_spec_language.md`) was built expecting a direct-path language, so the plumbing investment (US-8) benefits any future backend regardless. |
| 2 | Is `spin` the right first target, or would richer control state (>2 target actions) differentiate models better? | `spin` first for Alloy/PAT comparability; `ringbuffer`/`locksvc` as the documented second target. |
| 3 | Depth bound governance: `depthMax: 6` default — per-task, per-language, or bench-wide? | Bench-wide constant recorded in run output (US-3); revisit when a second task lands. |
| 4 | Real agent paths for invariant translation — anyone planning to wire them for Alloy/PAT? If so, share infrastructure instead of three remaps. | JS-SAM remaps to direct calls like Alloy/PAT today; `AgentInvariantTranslator` (TLA+) is the generalization point if/when. |
| 5 | Trace availability: will maintainers publish captured `spin` NDJSON traces, or is harness execution expected per-evaluator? | Blocks leaderboard Phase 3 only; unit tests use synthetic fixtures (§6 risk note). |

---

## 10. Non-functional requirements

- **Determinism:** two runs of any phase on the same spec produce identical
  outcomes (modulo timing fields). The helper sets no ambient state; the spec
  contract (§4 rule 4) bans nondeterminism sources, and Phase 2 actively
  classifies violations.
- **Isolation:** generated specs are untrusted **native code** — the helper
  `require()`s the spec module and `eval()`s its invariant predicates, unlike
  the restricted modelling languages the TLA+/Alloy/PAT checkers consume. Every
  helper invocation therefore runs inside a throwaway Docker container
  (`tla_eval/languages/js_sam.py:_run_helper`): `--network none` (no egress),
  `--read-only` rootfs with only the spec file bind-mounted read-only, a
  non-root `--user`, `--memory`/`--cpus`/`--pids-limit` caps, and the
  evaluator's hard wall-clock `timeout`. The container is the trust boundary;
  the host is never exposed to model code. A hostile spec that tries to write
  the filesystem or open a socket fails inside the container and surfaces as an
  ordinary spec error.
- **Portability:** no mono/Java dependency. The helper runs in Docker (already
  a bench-wide dependency for the trace harnesses) using the `node:20-slim`
  image, with `tools/js-sam` dependencies installed via `npm ci`. Node ≥ 20 is
  pinned in `package.json` `engines` and supplied by the image. `check_available()`
  reports a missing daemon, image, helper, or `node_modules`.
- **Cost:** Phases 1–3 are LLM-free at evaluation time (generation aside).
  Phase 4 makes one direct API call per run for translation, mirroring
  Alloy/PAT cost shape.

---

## 11. Implementation map (first cut, landed with this spec)

| Spec item | Implementation |
|-----------|----------------|
| US-1 backend + registry | `tla_eval/languages/js_sam.py` (auto-discovered; `js-sam`/`sam`/`jssam` aliases) |
| US-2–US-5 helper | `tools/js-sam/cli.mjs` (`ping` / `validate` / `check` / `transitions` / `invariants`), deps pinned in `tools/js-sam/package.json` (incl. unscoped `sam-pattern`/`sam-fsm` aliases for require-resolution resilience) |
| US-6 prompts | `tla_eval/tasks/spin/prompts/js-sam/{direct_call,agent_correction,phase3_invariant_implementation}.txt` |
| US-7 templates | `data/js_sam_invariant_templates/spin/invariants.yaml` (3 starters + parity note) |
| US-8 evaluator | `tla_eval/evaluation/semantics/trace_loader.py` (new) + direct path in `transition_validation.py` (`_evaluate_direct`); stale "backend's responsibility" comment resolved in favor of the `base.py` windows-in signature |
| §8 tests | `tests/test_languages/test_js_sam.py`, `tests/test_evaluation/test_trace_loader.py`, fixtures under `tests/fixtures/js_sam/` (29 tests) |
| Incidental fix | `scripts/run_benchmark.py`: the unconditional TLA+ (java + tla2tools.jar) prerequisite gate is now applied only when `--language` is TLA+; other backends rely on their own `check_available()` |

Phase 2/4 exploration cost on the reference spin model: ~6.5k steps at
`depthMax 4`, ~327k steps (~2.7 s) at the bench-wide `depthMax 6`.
