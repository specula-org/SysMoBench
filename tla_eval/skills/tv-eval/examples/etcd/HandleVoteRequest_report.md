# Worked Example: HandleVoteRequest (Message-handling Action)

Validates that the TV framework handles actions consuming external messages by carrying an `input` field in each window.

## Results

| Spec | Windows | Pass Rate | Explanation |
|------|---------|-----------|-------------|
| ai_spec_1 | 21 | 21/21 (100%) | Grant logic permissive, reproduces all observed transitions |
| ai_spec_2 | 21 | 14/21 (66.7%) | Grant logic too restrictive: rejects same-term re-votes |

Execution: ~2s each via `tv_tools.run_tv_batch` with 8 workers.

## ai_spec_2 Failure Analysis (agent-visible signal)

All 7 failures share a common pre/input pattern:

- `pre.votedFor[receiver] = 1`
- `input.from = 2 or 3` (not 1)
- `input.term = pre.currentTerm[receiver]` (same term, not higher)
- `post.votedFor[receiver] = input.from` (trace shows re-vote happened)

ai_spec_2's grant condition:
```tla
grant == /\ m.term >= currentTerm[n]
         /\ logOk
         /\ \/ votedFor[n] = m.from                 \* already voted for sender
            \/ m.term > currentTerm[n] /\ votedFor[n] = CHOOSE x \in Nodes : TRUE
```

For the failing windows:
- First disjunct fails: `votedFor[n] (=1) ≠ m.from (=2 or 3)`
- Second disjunct fails: `m.term = currentTerm[n]`, not strictly greater

So `grant = FALSE`, spec keeps `votedFor` unchanged, but the window expects it to change to `m.from`. Post-state mismatch → FAIL.

**Interpretation**: ai_spec_2 enforces stricter Raft vote semantics (one vote per term) than what the trace actually shows. Either:
(a) the trace contains a Raft invariant violation (real bug in the system being traced), or
(b) ai_spec_2's grant logic is too conservative compared to real etcd behavior.

Either way, TV surfaces the divergence precisely and reproducibly.

## Framework Features Demonstrated

1. **`input` field in canonical windows** — triggering message carried per-window.
2. **Value reconstruction from trace fields** — `generate_windows.py::extract_input` builds MsgVote record from `src`, `node`, `msgTerm`, and prior state.
3. **Signature adaptation** — ai_spec_1's `HandleVoteRequest(m)` vs ai_spec_2's `HandleVoteRequest(n, m)` both supported by writing the appropriate `\E n, m : S!Action(...)` in Next.
4. **Field-name adaptation** — ai_spec_2 uses `granted` instead of `voteGranted`; message construction in TV adapts accordingly.

## Limitations Exposed

- `logTerm` in reconstructed message is hardcoded to 0 (we don't track per-log-entry terms in windows). For early-election windows (logLen=0 across cluster), this is fine. Later windows with non-empty logs would need richer tracking.

## How to reproduce

```bash
cd tla_eval/skills/tv-eval/examples/etcd
python3 generate_windows.py HandleRequestVoteRequest
python3 ai_spec_1/make_windows.py HandleRequestVoteRequest HandleVoteRequest
python3 ai_spec_2/make_windows.py HandleRequestVoteRequest HandleVoteRequest
# Run TV via tv_tools.run_tv_batch on both ai_spec_*/TV_HandleVoteRequest
```
