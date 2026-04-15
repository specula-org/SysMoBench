#!/usr/bin/env python3
"""
Generate action windows from etcd-raft NDJSON traces.

State schema (per-node):
  - currentTerm: int (from trace 'term')
  - state: role string (from trace 'role')
  - votedFor: int (from trace 'votedFor', 0 means None)
  - commitIndex: int (from trace 'commitIndex')
  - logLen: int (from trace 'logLen') — log length abstraction
"""

import json
import copy
import sys
from pathlib import Path

TRACES_DIR = Path(__file__).parent / "traces"
OUTPUT_FILE = Path(__file__).parent / "action_windows.jsonl"

NODES = [1, 2, 3]

SCHEMA_FIELDS = {
    "currentTerm": "term",
    "state": "role",
    "votedFor": "votedFor",
    "commitIndex": "commitIndex",
    "logLen": "logLen",
}


def initial_cluster_state():
    return {
        str(n): {
            "currentTerm": 0,
            "state": "StateFollower",
            "votedFor": 0,
            "commitIndex": 0,
            "logLen": 0,
            "logLastTerm": 0,
        }
        for n in NODES
    }


def update_node_state(node_state, event):
    """Subject node's post-state = event's schema fields (fall back to current).

    Also maintains logLastTerm:
    - When logLen grows (ClientRequest or accepted HandleAppendEntriesRequest),
      the newly-appended entry's term is the event's `term` (for ClientRequest =
      leader's currentTerm; for MsgApp accept = message/leader's term).
    - Otherwise logLastTerm is unchanged.
    """
    new = dict(node_state)
    old_len = new["logLen"]
    for schema_key, trace_key in SCHEMA_FIELDS.items():
        if trace_key in event:
            new[schema_key] = event[trace_key]
    new_len = new["logLen"]
    if new_len > old_len and "term" in event:
        new["logLastTerm"] = event["term"]
    elif new_len == 0:
        new["logLastTerm"] = 0
    # else: unchanged
    return new


def extract_input(event, pre_cluster):
    """Reconstruct the triggering message for message-handling events.
    Returns None for non-message events."""
    ev = event["event"]
    src = event.get("src")
    node = str(event["node"])

    if ev == "HandleRequestVoteRequest":
        sender = pre_cluster.get(str(src), {})
        return {
            "type": "MsgVote",
            "from": src,
            "to": int(node),
            "term": event.get("msgTerm", 0),
            "logIndex": sender.get("logLen", 0),
            "logTerm": 0,  # approximation: last log term not tracked
        }
    if ev == "HandleRequestVoteResponse":
        return {
            "type": "MsgVoteResp",
            "from": src,
            "to": int(node),
            "term": event.get("msgTerm", event.get("term", 0)),
            "voteGranted": not event.get("reject", False),
        }
    if ev == "HandleAppendEntriesRequest":
        sender = pre_cluster.get(str(src), {})
        receiver = pre_cluster.get(node, {})
        # Reconstruct MsgApp: receiver's pre-state tells us where leader picks up
        # (assumes leader appends at the receiver's current end — the common case).
        pre_len = receiver.get("logLen", 0)
        pre_last_term = receiver.get("logLastTerm", 0)
        # Derive entries from logLen delta observed in this event
        post_len = event.get("logLen", pre_len)
        num_entries = max(0, post_len - pre_len)
        entry_term = event.get("term", 0)
        entries = [{"term": entry_term, "data": "dummy"}] * num_entries
        return {
            "type": "MsgApp",
            "from": src,
            "to": int(node),
            "term": event.get("msgTerm", event.get("term", 0)),
            "prevLogIndex": pre_len,
            "prevLogTerm": pre_last_term,
            "entries": entries,
            "commitIndex": event.get("leaderCommit", sender.get("commitIndex", 0)),
        }
    if ev == "HandleAppendEntriesResponse":
        return {
            "type": "MsgAppResp",
            "from": src,
            "to": int(node),
            "term": event.get("term", 0),
            "success": not event.get("reject", False),
            "matchIndex": event.get("matchIndex", 0),
        }
    if ev == "HandleHeartbeat":
        return {
            "type": "MsgHeartbeat",
            "from": src,
            "to": int(node),
            "term": event.get("term", 0),
            "commitIndex": event.get("leaderCommit", 0),
        }
    return None


def generate_windows_for_trace(trace_id, events, target_event):
    """Emit windows for events matching target_event."""
    cluster = initial_cluster_state()
    windows = []

    for e in events:
        node = str(e["node"])
        pre_cluster = copy.deepcopy(cluster)
        cluster[node] = update_node_state(cluster[node], e)
        post_cluster = copy.deepcopy(cluster)

        if e["event"] == target_event:
            window = {
                "trace_id": trace_id,
                "ts": e["ts"],
                "event": e["event"],
                "node": node,
                "pre_state": pre_cluster,
                "post_state": post_cluster,
            }
            inp = extract_input(e, pre_cluster)
            if inp is not None:
                window["input"] = inp
            windows.append(window)

    return windows


def parse_trace(path):
    events = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            events.append(json.loads(line))
    return events


def main():
    target_event = sys.argv[1] if len(sys.argv) > 1 else "Timeout"

    all_windows = []
    trace_files = sorted(TRACES_DIR.glob("*.ndjson"))
    if not trace_files:
        print(f"ERROR: no trace files in {TRACES_DIR}")
        sys.exit(1)

    for tf in trace_files:
        trace_id = tf.stem
        events = parse_trace(tf)
        windows = generate_windows_for_trace(trace_id, events, target_event)
        all_windows.extend(windows)
        print(f"  {trace_id}: {len(events)} events → {len(windows)} '{target_event}' windows")

    # Assign global IDs
    for i, w in enumerate(all_windows):
        w["window_id"] = i

    with open(OUTPUT_FILE, "w") as f:
        for w in all_windows:
            f.write(json.dumps(w) + "\n")

    print(f"\nTotal: {len(all_windows)} windows written to {OUTPUT_FILE}")

    if all_windows:
        print("\nFirst window:")
        print(json.dumps(all_windows[0], indent=2))


if __name__ == "__main__":
    main()
