#!/usr/bin/env python3
"""Fold Asterinas spin-trace serial events into JS-SAM (pre, post) NDJSON windows.

Input: the docker/QEMU run log (mixed build output + kernel serial JSON events of
the form emitted by ostd/src/sync/spin_trace.rs):

    {"seq":N,"thread":T,"lock":0,"state":"locked|unlocked","action":A,"actor":T}

Output: one NDJSON file of event-form windows (see
tla_eval/evaluation/semantics/trace_loader.py), each line:

    {"action": "AcquireLock"|"ReleaseLock", "data": {...},
     "pre_state": {...}, "post_state": {...}}

The window state is the *objective, model-independent* spinlock state
{lockHeld, lockHolder} — the real lock ownership observed by the kernel. JS-SAM's
Phase-3 projection rule only requires the trace's post_state keys to match the
model, so the model's internal threadStatus/callType bookkeeping is ignored and
the traces stay faithful to the system rather than tailored to any one model.

Event -> model action mapping:
  TryAcquireBlocking(t)  pre-step of a blocking lock(); folded, emits no window
  AcquireSuccess(t)      AcquireLock: lock becomes held by t
                         (callType "lock" if preceded by TryAcquireBlocking, else "try")
  AcquireFail(t)         AcquireLock (callType "try"): contention — ownership unchanged
  Release(t)             ReleaseLock: lock becomes free
  StopSpinning(t)        internal spin step; skipped

Usage:
  parse_traces.py <input_log> <output_ndjson>
"""

import json
import re
import sys
from pathlib import Path

EVENT_RE = re.compile(r'\{"seq":\d+[^{}]*"action":"[^"]*"[^{}]*\}')


def extract_events(log_text):
    events = []
    for match in EVENT_RE.findall(log_text):
        try:
            event = json.loads(match)
        except json.JSONDecodeError:
            continue
        if "action" in event and "thread" in event:
            events.append(event)
    return events


def fold_windows(events):
    lock_held = False
    lock_holder = None
    blocking = {}  # thread -> True after a TryAcquireBlocking, cleared on its acquire
    windows = []

    for event in events:
        action = event["action"]
        thread = event["thread"]

        if action == "TryAcquireBlocking":
            blocking[thread] = True
            continue
        if action == "StopSpinning":
            continue

        pre = {"lockHeld": lock_held, "lockHolder": lock_holder}

        if action == "AcquireSuccess":
            call_type = "lock" if blocking.pop(thread, False) else "try"
            lock_held, lock_holder = True, thread
            windows.append({
                "action": "AcquireLock",
                "data": {"thread": thread, "callType": call_type},
                "pre_state": pre,
                "post_state": {"lockHeld": lock_held, "lockHolder": lock_holder},
            })
        elif action == "AcquireFail":
            blocking.pop(thread, None)
            # Failed contention: the lock stays with its current holder.
            windows.append({
                "action": "AcquireLock",
                "data": {"thread": thread, "callType": "try"},
                "pre_state": pre,
                "post_state": {"lockHeld": lock_held, "lockHolder": lock_holder},
            })
        elif action == "Release":
            lock_held, lock_holder = False, None
            windows.append({
                "action": "ReleaseLock",
                "data": {"thread": thread},
                "pre_state": pre,
                "post_state": {"lockHeld": lock_held, "lockHolder": lock_holder},
            })

    return windows


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(2)
    log_path = Path(sys.argv[1])
    out_path = Path(sys.argv[2])

    events = extract_events(log_path.read_text(encoding="utf-8", errors="replace"))
    windows = fold_windows(events)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for window in windows:
            f.write(json.dumps(window) + "\n")

    print(f"wrote {len(windows)} windows from {len(events)} events -> {out_path}")


if __name__ == "__main__":
    main()
