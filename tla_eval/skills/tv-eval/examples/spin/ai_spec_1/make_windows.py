#!/usr/bin/env python3
"""
Convert trace-format windows into spec-ready JSON for the AI spec.

Mappings (this spec):
  - Thread id: trace '0','1','2' → spec indices 1,2,3 (array)
  - pc value: trace 'acquiring' → spec 'trying' (same for idle/locked)
"""

import json
from pathlib import Path

SRC = Path(__file__).parent.parent / "action_windows.jsonl"
DST = Path(__file__).parent / "windows_AcquireLock.ndjson"

PC_VALUE_MAP = {"idle": "idle", "acquiring": "trying", "locked": "locked"}
TARGET_ACTION = "AcquireSuccess"


def convert_state(s):
    """Convert pre/post_state to spec-ready form."""
    return {
        "lockState": s["lockState"],
        "pc": [PC_VALUE_MAP[s["pc"][str(i)]] for i in range(3)],
    }


def main():
    out = []
    with open(SRC) as f:
        for line in f:
            w = json.loads(line)
            if w["action"] != TARGET_ACTION:
                continue
            out.append({
                "id": len(out) + 1,      # 1-indexed for TLA+
                "trace_id": w["trace_id"],
                "actor": int(w["actor"]) + 1,  # trace 0/1/2 → spec 1/2/3
                "pre": convert_state(w["pre_state"]),
                "post": convert_state(w["post_state"]),
            })

    with open(DST, "w") as f:
        for w in out:
            f.write(json.dumps(w) + "\n")
    print(f"Wrote {len(out)} {TARGET_ACTION} windows to {DST}")


if __name__ == "__main__":
    main()
