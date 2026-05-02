#!/usr/bin/env python3
"""Convert dqueue PGo-native trace to SysMoBench TV-ready NDJSON.

PGo emits one JSON line per MPCal block execution, including aborted steps
(failed attempts to take a label). Each line has:
  archetypeName, self, startTime, endTime, isAbort, csElements
where csElements is a list of reads/writes against state variables (incl.
the per-archetype .pc value).

We extract, for each NON-aborted event:
  - `label`: the MPCal label that just executed (from .pc oldValue)
  - `action`: mapped spec action name (Request|Produce|None)
  - `pid`:   process id (`self`)
  - `state`: flat dict of post-state for touched vars (excl. .pc)
  - `reads`, `writes`: raw csElements for traceability

Mapping (dqueue-specific):
  Consumer.c1 → Request   (net[PRODUCER] := self)
  Producer.p2 → Produce   (net[requester] := stream value)
  Everything else → no spec action (internal transition)

Output: one `{"tag":"trace", "event":{...}}` line per input event. Same
NDJSON schema the tv-eval skill already handles for redisraft/spin.

Usage:
  parse_traces.py <input.ndjson> <output_dir>
"""

import json
import sys
from pathlib import Path

ACTION_MAP = {
    "AConsumer.c1": "Request",
    "AProducer.p2": "Produce",
}


def extract_event(raw):
    """Turn one PGo-native event into our NDJSON form, or None if aborted."""
    if raw.get("isAbort"):
        return None
    cs = raw.get("csElements", [])

    # .pc write records the label that just executed (oldValue)
    pc_write = next(
        (e for e in cs if e.get("tag") == "write" and e.get("name", {}).get("name") == ".pc"),
        None,
    )
    if pc_write is None:
        return None

    label = pc_write["oldValue"].strip('"')    # e.g. AConsumer.c1
    next_label = pc_write["value"].strip('"')  # e.g. AConsumer.c2
    action = ACTION_MAP.get(label)

    reads = {}
    writes = {}
    for el in cs:
        if el.get("name", {}).get("name") == ".pc":
            continue
        key = _var_key(el)
        if el.get("tag") == "read":
            reads[key] = _val(el.get("value"))
        elif el.get("tag") == "write":
            writes[key] = _val(el.get("value"))

    return {
        "tag": "trace",
        "event": {
            "name": action if action else label,
            "action": action,          # None if out-of-scope; agent filters
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
    """Unwrap JSON-quoted TLA values to plain strings/numbers when possible."""
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


def main():
    if len(sys.argv) != 3:
        print(__doc__, file=sys.stderr)
        sys.exit(2)

    src = Path(sys.argv[1])
    out_dir = Path(sys.argv[2])
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / "trace_01.ndjson"
    with out_path.open("w") as outf:
        outf.write("# dqueue PGo trace (PGo-native → SysMoBench NDJSON)\n")
        outf.write("# action=Request/Produce mapped from MPCal label; others kept as-is\n")
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
