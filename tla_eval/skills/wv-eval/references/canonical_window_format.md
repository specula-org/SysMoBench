# Canonical Window Format

All `generate_windows.py` implementations output this schema, one record per line (NDJSON).

## Schema

```json
{
  "window_id": <int>,              // 1-based, unique within this file
  "trace_id": "<str>",             // source trace identifier (for debugging)
  "action": "<str>",               // task-level action name (from the trace)
  "actor": <int|str|null>,         // which node/thread triggered (null for global)
  "pre_state": {
    "<schema_var>": <value>        // scalar (global) or {"<node>": <value>} (per-node function)
  },
  "post_state": { ... },            // same structure as pre_state
  "input": <object|null>            // OPTIONAL: for message-handling actions
}
```

## Rules

1. **Var-first, not node-first.** Group state by schema variable, not by node.

   ```json
   ✓ "pre_state": {"currentTerm": {"1": 0, "2": 0}, "state": {"1": "Follower", "2": "Follower"}}
   ✗ "pre_state": {"1": {"currentTerm": 0, "state": "Follower"}, "2": {...}}
   ```

2. **Node keys are strings** in JSON (can be integer-looking or model-value-like). Value types are whatever makes sense for the variable.

3. **`actor` refers to the node id** that triggered the action, for node-scoped actions. It may be `null` for globally-scoped actions.

4. **`input` is optional** and used only for actions that consume external data (e.g., `HandleVoteRequest(m)` where `m` is a message). Its schema is free-form — whatever captures enough of the message to determine which spec action to fire.

## Example: spin

```json
{"window_id": 1, "trace_id": "trace_01", "action": "AcquireSuccess", "actor": 1, "pre_state": {"lockState": "unlocked", "pc": {"0": "idle", "1": "acquiring", "2": "idle"}}, "post_state": {"lockState": "locked", "pc": {"0": "idle", "1": "locked", "2": "idle"}}}
```

## Example: etcd

```json
{"window_id": 1, "trace_id": "normal_election", "action": "ClientRequest", "actor": 1, "pre_state": {"currentTerm": {"1": 1, "2": 1, "3": 1}, "state": {"1": "StateLeader", "2": "StateFollower", "3": "StateFollower"}, "votedFor": {"1": 1, "2": 1, "3": 1}, "commitIndex": {"1": 1, "2": 1, "3": 1}, "logLen": {"1": 1, "2": 1, "3": 1}}, "post_state": {...}}
```

## What happens next

`make_windows.py` (per-spec) reads this canonical format and produces a spec-ready JSON where:
- Node-keyed dicts become arrays indexed 1..N (matching TLA+ function representation).
- String values are mapped if the spec uses different names (e.g., `"acquiring"` → `"trying"`).
- Auxiliary-like fields (e.g., `votedFor=0` for "None") are translated to spec values.

See `examples/spin/ai_spec_1/make_windows.py` and `examples/etcd/ai_spec_1/make_windows.py` for concrete implementations.
