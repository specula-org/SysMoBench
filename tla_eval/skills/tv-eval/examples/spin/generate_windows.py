#!/usr/bin/env python3
"""
Generate action windows from spinlock system traces.

Reads raw trace JSONL files, reconstructs full state at each step,
and outputs (pre_state, post_state) windows for transition validation.

State variables tracked:
  - lockState: "locked" | "unlocked"
  - pc: per-thread program counter {"idle", "acquiring", "locked"}
"""

import json
import copy
import sys
from pathlib import Path

TRACES_DIR = Path(__file__).parent.parent.parent / "data" / "sys_traces" / "spin"
OUTPUT_FILE = Path(__file__).parent / "action_windows.jsonl"

THREAD_IDS = ["0", "1", "2"]

# How each trace action updates the state
ACTION_EFFECTS = {
    "TryAcquireBlocking": lambda state, actor: _set_pc(state, actor, "acquiring"),
    "TryAcquireNonBlocking": lambda state, actor: _set_pc(state, actor, "acquiring"),
    "AcquireSuccess": lambda state, actor: _acquire(state, actor),
    "StopSpinning": lambda state, actor: state,  # no change to tracked vars
    "Release": lambda state, actor: _release(state, actor),
}


def _set_pc(state, actor, value):
    new = copy.deepcopy(state)
    new["pc"][actor] = value
    return new


def _acquire(state, actor):
    new = copy.deepcopy(state)
    new["lockState"] = "locked"
    new["pc"][actor] = "locked"
    return new


def _release(state, actor):
    new = copy.deepcopy(state)
    new["lockState"] = "unlocked"
    new["pc"][actor] = "idle"
    return new


def initial_state():
    return {
        "lockState": "unlocked",
        "pc": {tid: "idle" for tid in THREAD_IDS},
    }


def parse_trace(trace_path):
    """Parse a trace JSONL file, skipping comment lines."""
    events = []
    with open(trace_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            events.append(json.loads(line))
    return events


def generate_windows_from_trace(trace_id, events):
    """Generate action windows from a sequence of trace events."""
    windows = []
    current_state = initial_state()
    window_id = 0

    for event in events:
        action = event["action"]
        actor = str(event["actor"])

        if action not in ACTION_EFFECTS:
            print(f"  WARNING: unknown action '{action}' in {trace_id}, skipping")
            continue

        pre_state = copy.deepcopy(current_state)
        post_state = ACTION_EFFECTS[action](current_state, actor)

        # Skip no-op windows (pre == post), e.g. StopSpinning
        if pre_state == post_state:
            current_state = post_state
            continue

        # Sanity check: lockState in post should match trace 'state' field
        trace_lock_state = event.get("state")
        if trace_lock_state and post_state["lockState"] != trace_lock_state:
            print(
                f"  WARNING: state mismatch in {trace_id} seq {event['seq']}: "
                f"computed={post_state['lockState']}, trace={trace_lock_state}"
            )

        windows.append({
            "window_id": window_id,
            "trace_id": trace_id,
            "seq": event["seq"],
            "action": action,
            "actor": actor,
            "pre_state": pre_state,
            "post_state": post_state,
        })

        current_state = post_state
        window_id += 1

    return windows


def main():
    trace_files = sorted(TRACES_DIR.glob("trace_*.jsonl"))
    if not trace_files:
        print(f"ERROR: no trace files found in {TRACES_DIR}")
        sys.exit(1)

    all_windows = []
    global_id = 0

    for trace_file in trace_files:
        trace_id = trace_file.stem  # e.g. "trace_01"
        events = parse_trace(trace_file)
        windows = generate_windows_from_trace(trace_id, events)

        # Re-number with global IDs
        for w in windows:
            w["window_id"] = global_id
            global_id += 1

        all_windows.extend(windows)
        print(f"  {trace_id}: {len(events)} events → {len(windows)} windows")

    # Write output
    with open(OUTPUT_FILE, "w") as f:
        for w in all_windows:
            f.write(json.dumps(w) + "\n")

    print(f"\nTotal: {len(all_windows)} windows written to {OUTPUT_FILE}")

    # Summary stats
    actions = {}
    for w in all_windows:
        a = w["action"]
        actions[a] = actions.get(a, 0) + 1
    print("\nWindows by action type:")
    for a, count in sorted(actions.items()):
        print(f"  {a}: {count}")


if __name__ == "__main__":
    main()
