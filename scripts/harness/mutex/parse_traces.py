#!/usr/bin/env python3
"""Parse Asterinas kernel serial output into per-scenario NDJSON trace files.

Reads the combined docker stdout log (mixed build output + kernel serial JSON
blobs from test_mutex_trace) and splits into trace_01.jsonl .. trace_NN.jsonl.

The ktest emits 20 scenarios; each starts by resetting TRACE_SEQUENCE to 0,
so we split on seq==0 boundaries (after the first event).

Usage:
  parse_traces.py <input_log> <output_dir>
"""

import json
import re
import sys
from pathlib import Path
from datetime import datetime

JSON_RE = re.compile(r'\{"seq":\d+[^{}]*"action":"[^"]*"[^{}]*\}')


def extract_events(log_text):
    events = []
    for chunk in log_text.split('\n'):
        for m in JSON_RE.findall(chunk):
            try:
                ev = json.loads(m)
            except json.JSONDecodeError:
                continue
            if 'seq' in ev and 'action' in ev:
                if 'actor' not in ev and 'thread' in ev:
                    ev['actor'] = ev['thread']
                events.append(ev)
    return events


def split_by_seq_reset(events):
    """Split events into scenarios. A new scenario starts when seq goes back to 0
    (or to a small number after a large one). Preserves original seq per scenario."""
    traces = []
    current = []
    last_seq = -1
    for ev in events:
        seq = ev.get('seq', 0)
        if seq == 0 and current:
            traces.append(current)
            current = []
        elif seq < last_seq and last_seq > 10:
            traces.append(current)
            current = []
        current.append(ev)
        last_seq = seq
    if current:
        traces.append(current)
    return traces


def main():
    if len(sys.argv) != 3:
        print(__doc__, file=sys.stderr)
        sys.exit(2)

    log_path = Path(sys.argv[1])
    out_dir = Path(sys.argv[2])
    out_dir.mkdir(parents=True, exist_ok=True)

    log_text = log_path.read_text(errors='replace')
    events = extract_events(log_text)
    if not events:
        print('ERROR: no JSON events found in log', file=sys.stderr)
        sys.exit(1)

    traces = split_by_seq_reset(events)
    stamp = datetime.now().isoformat()

    for idx, tr in enumerate(traces, 1):
        path = out_dir / f'trace_{idx:02d}.jsonl'
        with path.open('w') as f:
            f.write(f'# TRACE_{idx}: Asterinas MutexTrace kernel scenario\n')
            f.write(f'# Generated from test_mutex_trace, Timestamp: {stamp}\n')
            for ev in tr:
                f.write(json.dumps(ev) + '\n')

    total_events = sum(len(t) for t in traces)
    print(f'Wrote {len(traces)} traces, {total_events} events total, to {out_dir}')


if __name__ == '__main__':
    main()
