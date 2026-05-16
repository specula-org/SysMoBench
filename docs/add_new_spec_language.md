# Adding a new spec language

SysMoBench evaluates AI-generated specifications in four phases (syntax,
runtime, trace validation, expert invariants). The phases are
language-neutral; everything specific to TLA+, Alloy, PAT, or any future
language lives behind a single `LanguageBackend` strategy class.

Adding a new language means writing one `LanguageBackend` subclass and
registering it. No evaluator code or CLI plumbing is required.

> **Package-name note.** The Python package is still called `tla_eval/` for
> historical reasons. It is now language-agnostic and will be renamed in a
> future cleanup. Don't read anything language-specific into the path.

---

## 1. What you're plugging into

A SysMoBench run for one system (say `mutex`) is, conceptually:

1. **Generate.** A language model is shown the system's source code and a
   task-specific prompt, and asked to produce a specification.
2. **Phase 1 — Syntax check.** Does the produced spec parse and type-check?
3. **Phase 2 — Runtime check.** Does the spec's own state space, explored
   by a model checker, terminate cleanly?
4. **Phase 3 — Trace validation.** We have NDJSON traces captured from the
   real system. For each `(pre_state, action, post_state)` window, does the
   spec admit this transition? Per-action pass rates are the score.
5. **Phase 4 — Invariant verification.** A per-system library of
   expert-written invariants is translated into the spec's vocabulary and
   each one is checked.

Trace capture (per system, language-agnostic) and the leaderboard pipeline
are not your concern; the evaluators are not your concern either. You only
implement the parts of the four phases that are specific to your language.

---

## 2. The `LanguageBackend` contract

Located in `tla_eval/languages/base.py`. Subclass and register:

```python
from tla_eval.languages.base import LanguageBackend
from tla_eval.languages.registry import register

class MyLangBackend(LanguageBackend):
    name = "MyLang"
    aliases = ("mylang", "ml")
    fence_label = "mylang"
    config_fence_label = None       # set to e.g. "cfg" if your language has a separate config artifact
    spec_extension = ".ml"
    config_extension = None

    # implement validate_syntax, run_model_checker, translate_invariants,
    # check_invariants. See §3 for each.

register(MyLangBackend())
```

Drop the file as `tla_eval/languages/<mylang>.py`. The package's lazy
bootstrap in `tla_eval/languages/__init__.py` auto-discovers and imports
backend modules on first registry lookup. No `__init__.py` edit is required.
Import errors are not swallowed; a broken backend module fails fast.

The runtime resolves backends from `--language` (case-insensitive, `+`
stripped). After registration `--language MyLang` selects yours; no other
CLI changes are needed.

### Identity fields

| Field | Purpose |
|-------|---------|
| `name` | Canonical name shown in logs and CLI (e.g. `"TLA+"`, `"Alloy"`, `"PAT"`). |
| `aliases` | Other accepted spellings (`("tla+", "tla")`, `("alloy", "als")`). |
| `fence_label` | Markdown fence the model is told to use for the spec block (e.g. `tla`, `alloy`, `csp`). |
| `config_fence_label` | Fence for an optional separate config block, or `None`. TLA+ uses `cfg`; Alloy/PAT have no separate config. |
| `spec_extension` | File extension used when the evaluator writes the spec to disk (`.tla`, `.als`, `.csp`). |
| `config_extension` | Same for config, or `None`. |

The default `extract_artifacts()` and `fence_format_hint()` use these to
parse the model response and build correction-prompt hints. Override only
if your language doesn't fit the "one mandatory block + one optional
block" shape.

### Optional tool diagnostics

```python
def check_available(self) -> Optional[str]:
    """Return None when tools are ready, or a one-line explanation of what's missing."""
```

`scripts/run_benchmark.py` calls this for the selected backend before
generation/evaluator setup, so missing tools are reported as environment
problems instead of being fed into a correction loop as spec errors. Both
`AlloyBackend.check_available` and `PATBackend.check_available` are good
references — they surface tool-install gaps with actionable messages.

---

## 3. The four phase methods

Each phase returns a small typed result from `tla_eval/languages/result_types.py`:
`SyntaxOutcome`, `ModelCheckOutcome`, `InvariantOutcome` (which contains
`InvariantCaseResult` items), `TransitionOutcome`. These are intermediate
shapes; the language-neutral evaluators translate them into the persistent
`SyntaxEvaluationResult` / `SemanticEvaluationResult` types written to
`result.json`.

### 3.1 Phase 1 — Syntax / static check

```python
def validate_syntax(
    self,
    spec: str,
    config: Optional[str],
    work_dir: Path,
    timeout: int,
    spec_filename: Optional[str] = None,
) -> SyntaxOutcome: ...
```

Run your parser / static checker. Populate `success`, `syntax_errors`,
`semantic_errors`, `raw_output`, `elapsed_seconds`. `work_dir` is an
output directory the evaluator owns — feel free to drop temp files there.
If `spec_filename` is provided, write/check the spec at
`work_dir / spec_filename` so filename-sensitive parsers see the same name
the evaluator will persist. TLA+ uses this to preserve SANY's module-name vs
filename check.

References: `tla_plus.py:validate_syntax` (delegates to `TLAValidator`),
`alloy.py:validate_syntax` (shells out to `AlloyCliValidator` Java helper),
`pat.py:validate_syntax` (shells out to `mono PAT3.Console.exe`,
text-detects errors because PAT always returns exit 0).

### 3.2 Phase 2 — Runtime / bounded model check

```python
def run_model_checker(
    self,
    spec_path: Path,
    config_path: Optional[Path],
    work_dir: Path,
    timeout: int,
) -> ModelCheckOutcome: ...
```

Run your model checker on a spec already written to disk. The evaluator
handles three config sources before calling you — model-emitted, file on
disk, or fallback via `generate_default_config()` — so you just consume the
result. Populate `success`, `raw_output`, `elapsed_seconds`,
`error_message`, optional `classification` string.

```python
def generate_default_config(
    self, spec: str, task_name: str, model_name: Optional[str]
) -> Tuple[bool, str, Optional[str]]: ...
```

Optional override. The base default returns `(True, "", None)` — fine for
languages where Phase 2 doesn't need a separate config. TLA+ overrides to
drive its `ConfigGenerator` (LLM-generated `.cfg`). Override if your
language needs a fallback for when the model didn't emit a config.

References: `tla_plus.py:run_model_checker` (delegates to `TLCRunner`),
`alloy.py:run_model_checker`, `pat.py:run_model_checker`.

### 3.3 Phase 3 — Trace / transition validation

Two paths; pick one:

**Default — agent path.** Leave `supports_direct_transition_validation = False`
(the base class default). `TransitionValidationEvaluator` then shells out to
`scripts/launch_tv_eval.sh`, which spins up a coding agent (Claude Code or
Codex) that knows how to read the spec, slice traces into
`(pre, action, post)` windows, and check each one. The agent skill at
`tla_eval/skills/tv-eval/` is currently TLA+-shaped; reusing it for a new
language means extending the skill, which is a larger project than writing
a backend. Plan accordingly.

**Direct path.** If your language's model checker has a CLI that answers
"is `pre ∧ action ⇒ post'` valid?" directly without orchestration, set
`supports_direct_transition_validation = True` and implement:

```python
def validate_transitions(
    self,
    spec_path: Path,
    trace_windows,                # iterable of (action, pre_state, post_state)
    work_dir: Path,
    timeout: int,
) -> TransitionOutcome: ...
```

Populate `per_action_pass_rates`, `total_passed`, `total_windows`. The
evaluator-side wrapper for this path is stubbed in
`transition_validation.py` — you'll need to thread the trace loader through
on the first language that uses it; flag it on the integration thread.

Today neither TLA+, Alloy, nor PAT exposes a direct API, so all three
inherit the agent path.

### 3.4 Phase 4 — Invariant verification

```python
def translate_invariants(
    self,
    templates: List[InvariantTemplate],
    spec: str,
    task_name: str,
    translator: str = "claude-code",
    agent_timeout: Optional[int] = None,
) -> Tuple[Dict[str, str], Optional[str]]: ...

def check_invariants(
    self,
    spec_path: Path,
    config_path: Optional[Path],
    templates: List[InvariantTemplate],
    translated: Dict[str, str],
    work_dir: Path,
    timeout: int,
) -> InvariantOutcome: ...
```

`InvariantTemplate` (from `base.py`) carries `name`, `type` ("safety" /
"liveness"), `natural_language`, `formal_description`, and `example`
(the language-specific reference snippet — see §4 for the YAML field).

**`translate_invariants`** maps each template into a string of code in
your language using whichever translator backend the caller asked for:

- `"claude-code"` (default) — Claude Code agent
- `"codex"` — Codex agent
- `"claude"` — single Claude API call
- any other value is treated as a model name for a direct API call

You don't have to support all four; return `({}, "translator '<x>' not
supported by <Lang> backend")` for modes you skip. TLA+ supports
`claude-code`, `codex`, `claude`, and explicit direct-model names. Alloy
maps the generic agent defaults to its historical GPT-5 direct call; PAT maps
them to its historical Claude direct call.

`agent_timeout` is passed through by Phase 4 for agent-based translators.
Ignore it for direct API translators unless your backend has an equivalent
timeout knob.

**`check_invariants`** runs each translated invariant through the model
checker and returns one `InvariantCaseResult` per template. The `templates`
list carries the per-invariant `type` ("safety" vs "liveness"), which
matters for languages like TLA+ that splice them into INVARIANT vs PROPERTY
slots in the config — read `template.type`, don't hardcode `"safety"`.

References:
- `tla_plus.py:translate_invariants` / `check_invariants` (reuses
  `InvariantTranslator` / `AgentInvariantTranslator` plus per-invariant
  TLC runs).
- `alloy.py:translate_invariants` / `check_invariants` (GPT-5 JSON
  translation, splices via append, runs AlloyRuntime).
- `pat.py:translate_invariants` / `check_invariants` (Claude JSON
  translation, strips existing `#assert`/`#define` lines, appends new
  assertion, runs PAT Console).

### 3.5 Optional lifecycle hook

```python
def finalize_run(
    self, work_dir: Path, task_name: str, method_name: str, model_name: str
) -> None: ...
```

Called once per `evaluator.evaluate()` invocation, after all phases.
Default no-op. TLA+ overrides to dump TLC error-classification stats to
`error_statistics.yaml`. Override if you want per-run reports.

---

## 4. Invariant templates

Per-language templates live at:

```
data/<dirname>/<task>/invariants.yaml
```

where `<dirname>` defaults to `invariant_templates` and can be overridden
per backend:

```python
def invariant_template_dirname(self) -> str:
    return "mylang_invariant_templates"

def invariant_example_field(self) -> str:
    return "mylang_example"   # the YAML key that holds the per-language snippet
```

Each entry:

```yaml
- name: "MutualExclusion"
  type: "safety"             # or "liveness"
  natural_language: "At most one thread holds the lock."
  formal_description: "|{ t : state[t] = HOLDING }| <= 1"
  mylang_example: "..."      # the field your backend declares
```

You only need to populate entries for the systems you intend to evaluate.
TLA+ templates exist for all 11 systems; Alloy and PAT historically
covered only `spin`.

---

## 5. Tasks and prompts

Tasks live at `tla_eval/tasks/<system>/task.yaml`. Most fields
(`repository`, `source_files`, `tv.target_actions`, etc.) describe the
system under test and are language-agnostic — your backend ignores them.

Per-language prompts go under `tla_eval/tasks/<system>/prompts/<lang>/`
(see `tla_eval/tasks/spin/prompts/alloy/` for the layout). The task
loader normalizes `<lang>` to lowercase and strips `+`, then resolves prompts
in this order:

1. `tla_eval/tasks/<system>/prompts/<lang>/<method>.txt`
2. `tla_eval/tasks/<system>/prompts/<method>_<lang>.txt`

For TLA+ only, the legacy prompt
`tla_eval/tasks/<system>/prompts/<method>.txt` is also accepted. Non-TLA+
languages do not fall back to that legacy TLA+ prompt; a missing
per-language prompt is a hard error.

Tell the model in your prompt:

1. The fence to use (matches your `fence_label`).
2. Mandatory action names from `task.yaml > tv.target_actions` — this is
   a hard contract used downstream by Phase 3.
3. Any naming conventions your tooling assumes (state variables, file
   structure, etc.).

---

## 6. End-to-end checkpoint

Once you've registered your backend and added prompts for one system:

```bash
# Diagnose tool availability
python3 -c "from tla_eval.languages import get; print(get('MyLang').check_available())"

# Phase 1 against a hand-written spec
python3 scripts/run_benchmark.py \
  --task <system> --method direct_call --model <any> \
  --metric compilation_check --language MyLang \
  --spec-file path/to/sample.ml
```

The expected sequence in the log output:

```
Evaluating compilation (MyLang): <system>/direct_call/<model>
Created experiment directory: output/compilation_check/mylang/<system>/...
```

If those land but the validator itself fails for a tooling reason, your
backend wiring is correct.

Phase 2 / Phase 4 follow the same pattern, swap `--metric`.

---

## 7. Reference implementations

| Backend  | File                                | Notes |
|----------|-------------------------------------|-------|
| TLA+     | `tla_eval/languages/tla_plus.py`    | Wraps existing TLAValidator/TLCRunner/InvariantTranslator. |
| Alloy    | `tla_eval/languages/alloy.py`       | Java helper (`AlloyCliValidator`, `AlloyRuntime`) over `lib/alloy.jar`. |
| PAT      | `tla_eval/languages/pat.py`         | `mono lib/PAT3.Console.exe`; PAT always returns exit 0 so failures are text-detected. |

Other files of interest:

| Concern | Path |
|---------|------|
| Abstract contract | `tla_eval/languages/base.py` |
| Registry | `tla_eval/languages/registry.py` |
| Intermediate result types | `tla_eval/languages/result_types.py` |
| Phase 1 evaluator | `tla_eval/evaluation/syntax/compilation_check.py` |
| Phase 2 evaluator | `tla_eval/evaluation/semantics/runtime_check.py` |
| Phase 3 evaluator | `tla_eval/evaluation/semantics/transition_validation.py` |
| Phase 4 evaluator | `tla_eval/evaluation/semantics/manual_invariant_evaluator.py` |
| Generation method | `tla_eval/methods/direct_call/method.py` |
| Metric registry | `tla_eval/evaluation/base/metric_registry.py` |
| CLI runner | `scripts/run_benchmark.py` |

When this doc and the code disagree, the code wins — please open an issue
so we can update the doc.
