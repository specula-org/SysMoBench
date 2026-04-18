#!/usr/bin/env python3
"""Parse Asterinas kernel serial output into per-scenario NDJSON trace files.

test_rb_trace_randomized emits several scenarios; each starts with a line
marker like `=== TRACE_RANDOM_<n> ===`. We split on those markers.

Unlike mutex/rwmutex, ring_buffer_trace.rs does NOT reset the seq counter
per scenario — it's a single monotonic counter across the whole run.

Usage:
  parse_traces.py <input_log> <output_dir>
"""

import json
import re
import sys
from pathlib import Path
from datetime import datetime

JSON_RE = re.compile(r'\{"seq":\d+[^{}]*"action":"[^"]*"[^{}]*\}')
BANNER_RE = re.compile(r'=== TRACE_RANDOM_(\d+) ===')


def parse_log(log_text):
    scenarios = []
    current_events = None
    current_title = None
    for line in log_text.splitlines():
        banner = BANNER_RE.search(line)
        if banner:
            if current_events is not None:
                scenarios.append((current_title, current_events))
            current_title = f"TRACE_RANDOM_{banner.group(1)}"
            current_events = []
            continue
        if current_events is None:
            continue
        for m in JSON_RE.findall(line):
            try:
                ev = json.loads(m)
            except json.JSONDecodeError:
                continue
            if 'seq' in ev and 'action' in ev:
                if 'actor' not in ev and 'thread' in ev:
                    ev['actor'] = ev['thread']
                current_events.append(ev)
    if current_events is not None:
        scenarios.append((current_title, current_events))
    return scenarios


def main():
    if len(sys.argv) != 3:
        print(__doc__, file=sys.stderr)
        sys.exit(2)
    log_path = Path(sys.argv[1])
    out_dir = Path(sys.argv[2])
    out_dir.mkdir(parents=True, exist_ok=True)
    log_text = log_path.read_text(errors='replace')
    scenarios = [s for s in parse_log(log_text) if s[1]]
    if not scenarios:
        print('ERROR: no scenario banners found in log', file=sys.stderr)
        sys.exit(1)
    stamp = datetime.now().isoformat()
    for idx, (title, events) in enumerate(scenarios, 1):
        path = out_dir / f'trace_{idx:02d}.jsonl'
        with path.open('w') as f:
            f.write(f'# {title}: Asterinas RingBuffer randomized scenario\n')
            f.write(f'# Generated from test_rb_trace_randomized, Timestamp: {stamp}\n')
            for ev in events:
                f.write(json.dumps(ev) + '\n')
    total = sum(len(e) for _, e in scenarios)
    print(f'Wrote {len(scenarios)} scenarios, {total} events total, to {out_dir}')


if __name__ == '__main__':
    main()
