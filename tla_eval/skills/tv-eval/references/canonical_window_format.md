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

4. **`input` is required for message-handling actions** (e.g., `HandleVoteRequest(m)`). It captures the triggering message as a TLA+-compatible record. Recommended schema:

   ```json
   "input": {
     "type": "MsgVote",           // message type (task-level name)
     "from": <node_id>,            // sender
     "to": <node_id>,              // receiver (same as actor)
     "term": <int>,                // message's term
     "<other_field>": <value>      // payload (logIndex, prevLogIndex, entries, etc.)
   }
   ```

   The fields required depend on the action. For `HandleVoteRequest`: `type, from, to, term, logIndex, logTerm`. For `HandleAppendEntriesRequest`: `type, from, to, term, prevLogIndex, prevLogTerm, entries, commitIndex`. Look at the spec's action to know what fields it reads.

   If the trace doesn't carry a field, reconstruct it from context (e.g., sender's last log metadata from per-event state tracking).

## Example: spin

```json
{"window_id": 1, "trace_id": "trace_01", "action": "AcquireSuccess", "actor": 1, "pre_state": {"lockState": "unlocked", "pc": {"0": "idle", "1": "acquiring", "2": "idle"}}, "post_state": {"lockState": "locked", "pc": {"0": "idle", "1": "locked", "2": "idle"}}}
```

## Example: etcd

```json
{"window_id": 1, "trace_id": "normal_election", "action": "ClientRequest", "actor": 1, "pre_state": {"currentTerm": {"1": 1, "2": 1, "3": 1}, "state": {"1": "StateLeader", "2": "StateFollower", "3": "StateFollower"}, "votedFor": {"1": 1, "2": 1, "3": 1}, "commitIndex": {"1": 1, "2": 1, "3": 1}, "logLen": {"1": 1, "2": 1, "3": 1}, "logLastTerm": {"1": 1, "2": 1, "3": 1}}, "post_state": {...}}
```

## Log abstraction (logLen + logLastTerm)

TLA+ specs model `log` as a sequence of entries. Faithfully serializing full log contents into windows is wasteful and often impossible (trace doesn't record every entry). Instead, we use a **two-field abstraction**:

- **`logLen`**: integer, length of each node's log.
- **`logLastTerm`**: integer, the `term` of the last log entry (or 0 if log is empty).

Together these capture what spec actions typically read from the log:
- `Len(log[n])` — matches `logLen[n]`
- `log[n][Len(log[n])].term` (i.e., `LastLogTerm(n)`) — matches `logLastTerm[n]`

For actions doing deep log inspection (e.g., `HandleAppendEntries` checking `log[n][prevLogIndex].term`), this abstraction is lossy in the middle entries. In practice `prevLogIndex = logLen` covers the common case and `logLastTerm` is sufficient. Document any further relaxations in the TV module itself.

State reconstruction rules (in `generate_windows.py`):
- On log-growing events (ClientRequest, HandleAppendEntriesRequest accept): `logLen` += delta; `logLastTerm` = term of newly-appended entry (typically the event's `term` field or the message's `term`).
- On events that don't change log: unchanged.
- On truncation (rare): best-effort reset; flag in window metadata if unavoidable.

## What happens next

`make_windows.py` (per-spec) reads this canonical format and produces a spec-ready JSON where:
- Node-keyed dicts become arrays indexed 1..N (matching TLA+ function representation).
- String values are mapped if the spec uses different names (e.g., `"acquiring"` → `"trying"`).
- Auxiliary-like fields (e.g., `votedFor=0` for "None") are translated to spec values.

See `examples/spin/ai_spec_1/make_windows.py` and `examples/etcd/ai_spec_1/make_windows.py` for concrete implementations.
