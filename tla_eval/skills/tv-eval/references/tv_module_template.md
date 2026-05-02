# TV Module Template

Every `TV_<Action>.tla` follows this skeleton. Two files needed per action:
- `TV_<Action>.tla` — the validator module
- `TV_<Action>.cfg` — fixed constants, never changes per window

## Template

```tla
---- MODULE TV_<Action> ----
\* Window Validator for <Action>. Reads windows from JSON at runtime.
\* WINDOW_INDEX env var selects which window to check.

EXTENDS Naturals, FiniteSets, Sequences, TLC, Json, IOUtils

\* === Original spec constants (copy from spec.tla / spec.cfg) ===
CONSTANTS <list them here>

\* === Original spec variables (copy from spec.tla) ===
VARIABLES <schema and aux vars>, step

\* === Import the spec under evaluation ===
S == INSTANCE <spec_module>

\* === Load windows and pick one ===
AllWindows == ndJsonDeserialize("windows_<Action>.ndjson")
w == AllWindows[atoi(IOEnv.WINDOW_INDEX)]

\* === Helpers for complex types (if needed) ===
\* For log abstraction: MakeLog(len) == [i \in 1..len |-> [term |-> 0, data |-> "dummy"]]

vars == <<<all vars + step>>>

\* === Init: schema vars from window, aux vars from plausible defaults ===
Init ==
    /\ <schema_var_1> = w.pre.<schema_var_1>
    /\ <schema_var_2> = w.pre.<schema_var_2>
    \* ... one line per schema variable
    /\ <aux_var_1> = <plausible default>
    /\ <aux_var_2> = <plausible default>
    \* ... one line per aux variable
    /\ step = 0

\* === Next: fire the target action exactly once ===
Next ==
    /\ step = 0
    /\ \E <params> : S!<Action>(<args>)
    /\ step' = 1

\* === Post-condition: did we reach the target post-state? ===
PostReached ==
    /\ step = 1
    /\ <schema_var_1> = w.post.<schema_var_1>
    /\ <schema_var_2> = w.post.<schema_var_2>
    \* ... one per schema variable
    \* For log: /\ \A n \in Nodes : Len(log[n]) = w.post.logLen[n]

NeverPost == ~PostReached

Spec == Init /\ [][Next]_vars

====
```

## Cfg template

```
SPECIFICATION Spec
INVARIANT NeverPost

CONSTANTS
    \* Copy from the spec's original .cfg
    Nodes = {1, 2, 3}
    MaxTerm = 10
    ...
```

## Key rules

1. **`S == INSTANCE <spec_module>`** — never copy action bodies. The INSTANCE pattern delegates all logic to the original spec, so you're testing exactly what the spec says.

2. **WINDOW_INDEX via env, not cfg.** Don't bake window data into cfg constants. That makes cfg generation per-window, fragile, and limited by cfg's poor syntax.

3. **Schema vars from window, aux from defaults.**
   - Schema vars: `<var> = w.pre.<var>` — fixes the part the window cares about.
   - Aux vars: pick values that satisfy TypeOK and the action's precondition without forcing trouble. See `examples/etcd/ai_spec_1/TV_ElectionTimeout.tla` for a case with 4 aux vars.

4. **Abstract complex types.** Log is a sequence of records. The window only tracks length. Construct: `log = [n \in Nodes |-> MakeLog(w.pre.logLen[n])]` using dummy entries. PostReached checks `Len(log[n]) = w.post.logLen[n]`, not contents.

5. **Check only schema vars in PostReached.** Aux var values post-action don't matter.

## How to adapt for a new action

1. Look at the action in the spec. Note:
   - Its parameters (e.g., `(t)` for a node-scoped action, `(m)` for message-handling)
   - Its precondition (what must be true in pre-state)
   - Its effects (what changes, what's UNCHANGED)

2. Copy the template. Fill in:
   - Constants and variables (from spec)
   - Init's aux defaults: pick values that make the precondition satisfiable (e.g., `electionElapsed = [n |-> 1]` because the action requires `>= 1`)
   - Init's log construction if applicable
   - Next's `\E` bindings and the call `S!<Action>(...)`

3. Write the cfg with the spec's standard constants.

4. Test with one window first (`WINDOW_INDEX=1 java ... -config ... TV_*.tla`), verify exit code 12 or 0 as expected.

5. Run `tla_eval/tv_tools/runner.py` on full window set.

## Input-carrying actions (message handlers)

For actions like `HandleVoteRequest(m)` that take a message:

```tla
Init ==
    /\ <schema vars from w.pre>
    /\ messages = {w.input}           \* inject triggering message as singleton
    /\ <other aux defaults>
    /\ step = 0

Next ==
    /\ step = 0
    /\ \E m \in messages : S!HandleVoteRequest(m)
    /\ step' = 1
```

The trigger message goes into `messages` via the window's `input` field. Let TLC pick it from the set (there's only one, so deterministic). See the etcd examples when that worked example is added.
