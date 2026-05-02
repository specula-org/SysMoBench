#!/usr/bin/env python3
"""Convert locksvc PGo-native trace to SysMoBench TV-ready NDJSON.

Action mapping (locksvc-specific):
  AClient.acquireLock      → ClientLockRequest
  AClient.criticalSection  → ClientCriticalSection
  AClient.unlock           → ClientUnlockRequest
  AServer.serverRespond WHERE a GrantMsg (=3) is written to network[x]
                          → ServerGrantLock     (also fires on Unlock-driven grant)
  AServer.serverRespond without such write → internal (action=null)
  Everything else → internal (action=null)

Usage:
  parse_traces.py <input.ndjson> <output_dir>
"""

import json
import sys
from pathlib import Path

GRANT_MSG = 3  # from locksvc.tla: GrantMsg == 3


def _var_key(el):
    name = el["name"]
    prefix = name.get("prefix", "")
    base = name["name"]
    full = f"{prefix}.{base}" if prefix else base
    idx = el.get("indices", [])
    if idx:
        full += "[" + ",".join(idx) + "]"
    return full


def _val(v):
    if isinstance(v, str):
        s = v.strip()
        if s.startswith('"') and s.endswith('"') and len(s) >= 2:
            return s[1:-1]
        if s.lstrip("-").isdigit():
            try:
                return int(s)
            except ValueError:
                pass
    return v


def map_action(label, writes):
    simple = {
        "AClient.acquireLock": "ClientLockRequest",
        "AClient.criticalSection": "ClientCriticalSection",
        "AClient.unlock": "ClientUnlockRequest",
    }
    if label in simple:
        return simple[label]
    if label == "AServer.serverRespond":
        # Fires for both Lock-with-empty-q (direct grant) and Unlock (next-in-queue grant).
        # Both write GrantMsg to network[<client>].
        for key, val in writes.items():
            if "network" in key and val == GRANT_MSG:
                return "ServerGrantLock"
    return None


def extract_event(raw):
    if raw.get("isAbort"):
        return None
    cs = raw.get("csElements", [])
    pc_write = next(
        (e for e in cs if e.get("tag") == "write" and e.get("name", {}).get("name") == ".pc"),
        None,
    )
    if pc_write is None:
        return None
    label = pc_write["oldValue"].strip('"')
    next_label = pc_write["value"].strip('"')

    reads, writes = {}, {}
    for el in cs:
        if el.get("name", {}).get("name") == ".pc":
            continue
        key = _var_key(el)
        if el.get("tag") == "read":
            reads[key] = _val(el.get("value"))
        elif el.get("tag") == "write":
            writes[key] = _val(el.get("value"))

    action = map_action(label, writes)
    return {
        "tag": "trace",
        "event": {
            "name": action if action else label,
            "action": action,
            "label": label,
            "next_label": next_label,
            "pid": raw["self"],
            "archetype": raw["archetypeName"],
            "clock": raw.get("clock", []),
            "start": raw.get("startTime"),
            "end": raw.get("endTime"),
            "reads": reads,
            "writes": writes,
        },
    }


def main():
    if len(sys.argv) != 3:
        print(__doc__, file=sys.stderr)
        sys.exit(2)

    src = Path(sys.argv[1])
    out_dir = Path(sys.argv[2])
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "trace_01.ndjson"

    with out_path.open("w") as outf:
        outf.write("# locksvc PGo trace (PGo-native → SysMoBench NDJSON)\n")
        outf.write("# Spec actions: ClientLockRequest, ServerGrantLock, ClientCriticalSection, ClientUnlockRequest\n")
        for line in src.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            ev = extract_event(raw)
            if ev is None:
                continue
            outf.write(json.dumps(ev) + "\n")

    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
