#!/usr/bin/env python3
"""Extract the successful spin TV kernel test without synthesizing events."""

import argparse
import json
import re
from pathlib import Path

BEGIN = "=== SPIN_TV_BEGIN ==="
END = "=== SPIN_TV_END ==="


def extract_trace(text):
    text = re.sub(r"\x1b\[[0-9;]*m", "", text)
    if text.count(BEGIN) != 1 or text.count(END) != 1:
        raise ValueError("expected exactly one complete spin TV test")
    if "test result: ok. 1 passed; 0 failed;" not in text:
        raise ValueError("kernel test did not report success")
    body = text.split(BEGIN, 1)[1].split(END, 1)[0]
    events = []
    for line in body.splitlines():
        if not line.strip():
            continue
        event = json.loads(line)
        if (
            not isinstance(event, dict)
            or not {"seq", "actor", "action", "state", "lock"} <= event.keys()
        ):
            raise ValueError("incomplete kernel event")
        events.append(event)
    if not events:
        raise ValueError("no kernel events")
    if [event["seq"] for event in events] != list(range(len(events))):
        raise ValueError("kernel event sequence is incomplete or out of order")
    return events


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    try:
        events = extract_trace(args.log.read_text())
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(event) + "\n" for event in events))
    print(f"Captured {len(events)} kernel events in {args.output}")


if __name__ == "__main__":
    main()
