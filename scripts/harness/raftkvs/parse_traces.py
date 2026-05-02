#!/usr/bin/env python3
"""Convert raftkvs PGo-native trace to SysMoBench TV-ready NDJSON.

Action mapping (raftkvs-specific):

  AClient.sndReq
    → ClientRequest

  AServerRequestVote.serverRequestVoteLoop
    → ElectionTimeout   (block runs when leaderTimeout fires and the server
                         transitions to Candidate)

  AServer.handleMsg
    Dispatches by the received message's mtype (from the event's reads).
    Msg-type codes per raftkvs.tla:
      RequestVoteRequest   == "rvq"
      RequestVoteResponse  == "rvp"
      AppendEntriesRequest == "apq"
      AppendEntriesResponse== "app"
    Mapping:
      mtype=="rvq" → HandleRequestVoteRequest
      mtype=="apq" → HandleAppendEntriesRequest
      mtype=="app" → HandleAppendEntriesResponse
      mtype=="rvp" → (internal — out of target scope)
      mtype=="cpq"/"cgq" (client requests) → (internal — handled on leader only)

  Everything else → internal (action=null).

Usage:
  parse_traces.py <input.ndjson> <output_dir>
"""

import json
import re
import sys
from pathlib import Path

MTYPE_MAP = {
    "rvq": "HandleRequestVoteRequest",
    "apq": "HandleAppendEntriesRequest",
    "app": "HandleAppendEntriesResponse",
    # cpq/cgq are the spec's ClientRequest(i) action — server side handler.
    # AClient.sndReq is NOT mapped here; it is client-side only and does not
    # correspond to any spec action (the server-side processing is what matters).
    "cpq": "ClientRequest",
    "cgq": "ClientRequest",
}

# These labels are tight busy-waiting loops that only touch local archetype
# variables (newCommitIndex etc.) and fire tens of thousands of times per
# trace. Drop at parse time to keep output files tractable without losing
# any spec-relevant state transition.
NOISE_LABELS = {
    "AServerAdvanceCommitIndex.applyLoop",
    "AServerAdvanceCommitIndex.serverAdvanceCommitIndexLoop",
}
# PGo emits TLA records via function-construction syntax:
#   (("mtype") :> ("rvq") @@ ("mterm") :> (2) @@ ...)
# Fallback to PlusCal record syntax `|->` for robustness.
MTYPE_RE = re.compile(r'"mtype"\s*\)?\s*(?::>|\|->)\s*\(?\s*"([a-z]+)"')


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


def find_mtype(reads, writes):
    """Scan read/write values for an mtype field inside a TLA record literal."""
    for v in list(reads.values()) + list(writes.values()):
        if not isinstance(v, str):
            continue
        m = MTYPE_RE.search(v)
        if m:
            return m.group(1)
    return None


def map_action(label, reads, writes):
    if label == "AServerRequestVote.serverRequestVoteLoop":
        return "ElectionTimeout"
    if label == "AServer.handleMsg":
        mtype = find_mtype(reads, writes)
        if mtype and mtype in MTYPE_MAP:
            return MTYPE_MAP[mtype]
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
    if label in NOISE_LABELS:
        return None

    reads, writes = {}, {}
    for el in cs:
        if el.get("name", {}).get("name") == ".pc":
            continue
        key = _var_key(el)
        if el.get("tag") == "read":
            # Take the FIRST read of each variable (= pre-write value).
            # PGo sometimes reads a variable again after writing it within the
            # same atomic step (e.g., currentTerm is incremented then re-read
            # for the console print), which would overwrite the pre-state value.
            if key not in reads:
                reads[key] = _val(el.get("value"))
        elif el.get("tag") == "write":
            writes[key] = _val(el.get("value"))

    action = map_action(label, reads, writes)
    return {
        "tag": "trace",
        "event": {
            "name": action if action else label,
            "action": action,
            "label": label,
            "next_label": next_label,
            "pid": raw["self"],
            "archetype": raw["archetypeName"],
            "mtype": find_mtype(reads, writes),
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
        outf.write("# raftkvs PGo trace (PGo-native → SysMoBench NDJSON)\n")
        outf.write("# Spec actions: ElectionTimeout, HandleRequestVoteRequest,\n")
        outf.write("#               HandleAppendEntriesRequest, HandleAppendEntriesResponse,\n")
        outf.write("#               ClientRequest\n")
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
