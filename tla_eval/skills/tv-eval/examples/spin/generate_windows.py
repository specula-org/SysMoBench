#!/usr/bin/env python3
"""Reconstruct single-lock action windows from complete spin trace events."""

import argparse
import copy
import json
from collections import Counter
from pathlib import Path

EXAMPLE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EXAMPLE_DIR.parents[4]
TRACES_DIR = PROJECT_ROOT / "data/sys_traces/spin"
OUTPUT_FILE = EXAMPLE_DIR / "action_windows.jsonl"
THREAD_IDS = ["0", "1", "2"]


def initial_state():
    return {"lockState": "unlocked", "pc": {tid: "idle" for tid in THREAD_IDS}}


def parse_trace(trace_path):
    events = []
    for number, line in enumerate(trace_path.read_text().splitlines(), 1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        try:
            event = json.loads(line)
            if not isinstance(event, dict):
                raise ValueError("expected an event object")
        except ValueError as exc:
            raise ValueError(f"{trace_path}:{number}: {exc}") from exc
        events.append(event)
    if not events:
        raise ValueError(f"{trace_path}: no events")
    return events


def generate_windows_from_trace(trace_id, events, *, allow_legacy_direct_acquire=False):
    """Reject incomplete/inconsistent traces instead of scoring invented states.

    The legacy switch only reproduces the archived coarse example; it does not
    make missing request events compliant with the task's single-step contract.
    """
    windows = []
    state = initial_state()
    requests = {}
    previous_seq = -1
    lock_id = None
    for event in events:
        try:
            action, actor, seq = event["action"], str(event["actor"]), event["seq"]
            observed = event["state"]
            if actor not in THREAD_IDS:
                raise ValueError(f"unknown actor {actor}")
            if type(seq) is not int or seq <= previous_seq:
                raise ValueError("sequence numbers must increase within a trace")
            previous_seq = seq
            if observed not in ("locked", "unlocked"):
                raise ValueError("state must be locked or unlocked")
            if "lock" in event:
                if lock_id is not None and event["lock"] != lock_id:
                    raise ValueError("one lock per trace is required")
                lock_id = event["lock"]
            post = copy.deepcopy(state)
            pc = state["pc"][actor]
            held = state["lockState"] == "locked"
            if action in ("TryAcquireBlocking", "TryAcquireNonBlocking"):
                if pc != "idle":
                    raise ValueError("request requires an idle actor")
                post["pc"][actor] = "acquiring"
                requests[actor] = action
            elif action == "AcquireSuccess":
                if held:
                    raise ValueError("acquisition requires a free lock")
                if pc != "acquiring" and not (
                    allow_legacy_direct_acquire and pc == "idle"
                ):
                    raise ValueError(
                        "AcquireSuccess is missing its request event; re-instrument and recapture"
                    )
                post["lockState"] = "locked"
                post["pc"][actor] = "locked"
                requests.pop(actor, None)
            elif action == "AcquireFail":
                if (
                    pc != "acquiring"
                    or not held
                    or requests.get(actor) != "TryAcquireNonBlocking"
                ):
                    raise ValueError(
                        "failure requires a pending nonblocking request and a held lock"
                    )
                post["pc"][actor] = "idle"
                requests.pop(actor)
            elif action == "Release":
                if pc != "locked" or not held:
                    raise ValueError("release requires the owning actor")
                post["lockState"] = "unlocked"
                post["pc"][actor] = "idle"
            elif action == "StopSpinning":
                if pc != "acquiring" or requests.get(actor) != "TryAcquireBlocking":
                    raise ValueError("StopSpinning requires a pending blocking request")
            else:
                raise ValueError(f"unknown action {action!r}")
            # Check even no-op events, before deciding whether to emit a window.
            if post["lockState"] != observed:
                raise ValueError(
                    f"state mismatch: computed={post['lockState']}, observed={observed}"
                )
        except (KeyError, ValueError) as exc:
            raise ValueError(f"{trace_id} seq {event.get('seq', '?')}: {exc}") from exc
        if post != state:
            windows.append(
                {
                    "window_id": len(windows),
                    "trace_id": trace_id,
                    "seq": seq,
                    "action": action,
                    "actor": actor,
                    "pre_state": state,
                    "post_state": post,
                }
            )
        state = post
    return windows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traces-dir", type=Path, default=TRACES_DIR)
    parser.add_argument("--output", type=Path, default=OUTPUT_FILE)
    parser.add_argument(
        "--allow-legacy-direct-acquire",
        action="store_true",
        help="Reproduce historical coarse windows; not valid benchmark input",
    )
    args = parser.parse_args()
    files = sorted(args.traces_dir.glob("trace_*.jsonl"))
    if not files:
        parser.error(f"no trace_*.jsonl files in {args.traces_dir}")
    windows = []
    try:
        for path in files:
            for window in generate_windows_from_trace(
                path.stem,
                parse_trace(path),
                allow_legacy_direct_acquire=args.allow_legacy_direct_acquire,
            ):
                window["window_id"] = len(windows)
                windows.append(window)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    args.output.write_text("".join(json.dumps(w) + "\n" for w in windows))
    print(f"Wrote {len(windows)} windows to {args.output}")
    print(dict(Counter(w["action"] for w in windows)))


if __name__ == "__main__":
    main()
