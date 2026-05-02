#!/usr/bin/env python3
"""
Convert canonical windows → spec-ready JSON for ai_spec_2 (20260125_131046).

Differences vs canonical/ai_spec_1:
  - State values: StateFollower → "Follower" (no "State" prefix)
  - votedFor=0 (trace) → map to node 1 (this spec requires votedFor in Nodes;
    ClientRequest doesn't care about votedFor's value, so any valid node works)
"""

import json
import sys
from pathlib import Path

SRC = Path(__file__).parent.parent / "action_windows.jsonl"
TRACE_EVENT = sys.argv[1] if len(sys.argv) > 1 else "ClientRequest"
SPEC_ACTION = sys.argv[2] if len(sys.argv) > 2 else TRACE_EVENT
DST = Path(__file__).parent / f"windows_{SPEC_ACTION}.ndjson"

NODES = ["1", "2", "3"]

# ai_spec_2 drops the "State" prefix from role names
STATE_MAP = {
    "StateFollower":     "Follower",
    "StateCandidate":    "Candidate",
    "StateLeader":       "Leader",
    "StatePreCandidate": "PreCandidate",
}


def map_votedfor(v):
    # ai_spec_2 requires votedFor \in Nodes (no None). Since ClientRequest
    # doesn't read votedFor, any valid node id works. Use 1 as placeholder.
    return 1 if v == 0 else v


def convert_state(s):
    return {
        "currentTerm": [s[n]["currentTerm"] for n in NODES],
        "state":       [STATE_MAP.get(s[n]["state"], s[n]["state"]) for n in NODES],
        "votedFor":    [map_votedfor(s[n]["votedFor"]) for n in NODES],
        "commitIndex": [s[n]["commitIndex"] for n in NODES],
        "logLen":      [s[n]["logLen"] for n in NODES],
        "logLastTerm": [s[n].get("logLastTerm", 0) for n in NODES],
    }


def main():
    out = []
    with open(SRC) as f:
        for line in f:
            w = json.loads(line)
            if w["event"] != TRACE_EVENT:
                continue
            rec = {
                "id": len(out) + 1,
                "trace_id": w["trace_id"],
                "node": int(w["node"]),
                "pre": convert_state(w["pre_state"]),
                "post": convert_state(w["post_state"]),
            }
            if "input" in w:
                # pass through: TV TLA+ module is responsible for adapting
                # field names if the spec uses different ones
                rec["input"] = w["input"]
            out.append(rec)
    with open(DST, "w") as f:
        for w in out:
            f.write(json.dumps(w) + "\n")
    print(f"Wrote {len(out)} windows (action={SPEC_ACTION}) to {DST}")


if __name__ == "__main__":
    main()
