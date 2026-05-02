#!/usr/bin/env python3
"""
Convert trace-format windows into spec-ready JSON for the AI spec (etcdraft).

Mappings (this spec):
  - Node id: trace str "1","2","3" → spec int 1,2,3 (array indexed 1..3)
  - votedFor: trace 0 → spec "None"
  - role: keep as string (spec uses string model values)
  - log: trace logLen → spec needs array of length logLen; use dummy entries
"""

import json
import sys
from pathlib import Path

SRC = Path(__file__).parent.parent / "action_windows.jsonl"
# args: trace_event_name [spec_action_name]
TRACE_EVENT = sys.argv[1] if len(sys.argv) > 1 else "ClientRequest"
SPEC_ACTION = sys.argv[2] if len(sys.argv) > 2 else TRACE_EVENT
DST = Path(__file__).parent / f"windows_{SPEC_ACTION}.ndjson"

NODES = ["1", "2", "3"]


def map_votedfor(v):
    return "None" if v == 0 else v


def convert_state(s):
    return {
        "currentTerm": [s[n]["currentTerm"] for n in NODES],
        "state":       [s[n]["state"] for n in NODES],
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
                # pass through; TV module expects spec-compatible field types
                rec["input"] = w["input"]
            out.append(rec)

    with open(DST, "w") as f:
        for w in out:
            f.write(json.dumps(w) + "\n")
    print(f"Wrote {len(out)} {TRACE_EVENT} windows to {DST} (for spec action {SPEC_ACTION})")


if __name__ == "__main__":
    main()
