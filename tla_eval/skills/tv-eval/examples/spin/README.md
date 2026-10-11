# Spin transition-validation example

Each window checks **one specification action** from its recorded pre-state to
its recorded post-state. The spin task requires a request step (`idle → trying`)
before acquisition, including a successful nonblocking `try_lock()`.

The default example uses a small Asterinas kernel capture. It exercises blocking
acquisition, successful nonblocking acquisition, nonblocking failure while another
actor holds the lock, and release. The three actors execute on one CPU; this is
not a concurrent stress test and does not exercise the blocking spin loop.

## Run

From the repository root, install the package and tools as described in the root
README (`pip install -e .`, then `sysmobench-setup`). Both TLC and CommunityModules
are required for this example.

```sh
EXAMPLE=tla_eval/skills/tv-eval/examples/spin
python "$EXAMPLE/ai_spec_1/run_tv.py"
python "$EXAMPLE/reference_spec/run_tv.py"
```

Expected results:

| Specification | Windows | Result |
|---|---|---|
| `ai_spec_1` | Three successful acquisitions | 3/3 pass |
| `reference_spec` | All captured request, acquisition, failure and release events | 11/11 pass |

The candidate `ai_spec_1/spin.tla` is unchanged. The reference is a **coarse
example abstraction**, not a complete implementation of the task's blocking /
nonblocking / spinning contract. In particular, it retains `AcquireLockDirect`
for examining the historical coarse windows.

Use `--tla-jar PATH --community-jar PATH` for tools installed elsewhere.
`--report PATH` writes per-window outcomes. Runner exit status is 0 when all
windows pass, 1 for a model mismatch, and 2 for setup errors, TLC errors or timeouts.
Internally, TLC exit 12 means the `NeverPost` invariant was violated: a matching
post-state exists. TLC exit 0 means the window did not match.

## Rebuild the windows

```sh
python "$EXAMPLE/generate_windows.py" --traces-dir "$EXAMPLE/traces"
python "$EXAMPLE/ai_spec_1/make_windows.py"
```

`traces/trace_01.jsonl` contains the captured events, `kernel-test.txt` contains
the kernel test output, and `manifest.json` records the source commit, container
image and checksums. The generator rejects missing request events, unknown
actions, invalid ownership and inconsistent state observations. It never inserts
missing transitions or silently filters bad events. Its default input location
for newly collected task traces is `<repository>/data/sys_traces/spin`.

## Capture again from Asterinas

Requires Git, Docker, Python and network access for the pinned source and build
dependencies:

```sh
bash scripts/harness/spin_tv/run.sh
python "$EXAMPLE/generate_windows.py" --traces-dir "$EXAMPLE/traces"
python "$EXAMPLE/ai_spec_1/make_windows.py"
```

The script creates a separate source/build directory under `artifacts/`, applies
`data/patches/asterinas_tla_trace.patch`, and runs `test_spin_tv_contract` under
QEMU TCG. It records executed operations on the instrumented `SpinTrace`; this
test emits no hand-written contention events. An optional first argument selects
another output directory. Build logs remain in the printed capture workspace.

## Historical incomplete traces

`legacy/` retains the original window files byte-for-byte. They contain 87
acquisitions with `idle` pre-states because the old nonblocking instrumentation
omitted its request event. They are a diagnostic fixture, **not valid input for
benchmark scoring**.

```sh
python "$EXAMPLE/ai_spec_1/run_tv.py" --windows "$EXAMPLE/legacy/windows_AcquireLock.ndjson"
# Expected: 123 pass, 87 fail; runner exit 1.
python "$EXAMPLE/reference_spec/run_tv.py" --windows "$EXAMPLE/legacy/action_windows.jsonl"
# Expected: 543/543 pass with the coarse reference abstraction.
```

The reference previously rejected 104 requests made while the lock was free.
Starting a request does not require a held lock; its `TryAcquire` guard now
reflects that. Passing the coarse reference does not repair missing events or
establish compliance with the finer task contract.
