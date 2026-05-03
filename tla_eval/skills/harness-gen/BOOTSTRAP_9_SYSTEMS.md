# Bootstrap Harnesses for the 9 Remaining SysMoBench Tasks

You are doing one-time harness setup for 9 tasks so the benchmark's
transition validation (TV) phase can eventually score them. When you
finish, the `tv-eval` skill will produce real window pass-rates for each
task's AI-generated specs (not just "Cannot evaluate — no harness").

Only `spin` and `etcd` already have harnesses. Everything else is greenfield.

## Inputs per task

For every task in the table below:

- **Prompt contract** at `tla_eval/tasks/<task>/prompts/direct_call.txt` — this
  tells you which variables/actions a spec *must* declare. Your harness must
  emit events whose pre/post state lines up with this contract.
- **TV scope** in `tla_eval/tasks/<task>/task.yaml` → `tv.target_actions`.
  Only these actions need to be exercisable from the traces you collect.
- **Upstream repo** in `tla_eval/tasks/<task>/task.yaml` → `repository.url`.

## 9 tasks, their upstream, and suggested reuse

| Task | Upstream | Reuse hint | Category |
|---|---|---|---|
| `curp` | github.com/xline-kv/Xline | fresh clone | A (distributed RPC) |
| `dqueue` | github.com/DistCompiler/pgo | one clone shared with locksvc/raftkvs | A |
| `locksvc` | github.com/DistCompiler/pgo | share pgo clone | A |
| `raftkvs` | github.com/DistCompiler/pgo | share pgo clone | A |
| `mutex` | github.com/asterinas/asterinas | **reuse `artifacts/spin/`** — same repo, different files | B (concurrent kernel primitive) |
| `ringbuffer` | github.com/asterinas/asterinas | reuse `artifacts/spin/` | B (lock-free SPSC queue) |
| `rwmutex` | github.com/asterinas/asterinas | reuse `artifacts/spin/` | B |
| `redisraft` | github.com/RedisLabs/redisraft | already cloned at `/home/ubuntu/Specula/case-studies/redisraft` — symlink or copy into `artifacts/redisraft/` | A |
| `zookeeper` | github.com/Lingzhi-Ouyang/Remix v3.9.1 | fresh clone; ZK FLE trace via Remix | A |

"Category" is from the decision tree in `guide.md` Step 0:
- **A** = distributed / message-passing (ms-level ops) → standard mutex+NDJSON
- **B** = concurrent lock-free / kernel primitive (ns-level ops) → timebox
  approach, per-thread trace files — see `references/concurrent-timebox-guide.md`

## What DONE looks like (per task)

A task is done when ALL of these are true:

1. `artifacts/<task>/` exists (directory containing the cloned + instrumented
   source tree). Reuse across tasks that share an upstream repo is
   encouraged — `mutex`, `ringbuffer`, `rwmutex` can all live under
   `artifacts/asterinas-kernel/` if you prefer.
2. `tla_eval/tasks/<task>/task.yaml` → `tv.repo_path` points at the cloned
   tree (absolute path or path relative to project root).
3. `tla_eval/tasks/<task>/task.yaml` → `tv.harness` has:
   - `instrumentation_file`: the header or source file you added (e.g.
     `rte_ring_tla_trace.h`, `tla_spec_trace.go`).
   - `test_file`: the test driver you wrote or patched.
   - `run_command`: the exact shell command that produces traces under
     `traces_output_env`.
   - `traces_output_env`: the env var your `run.sh` obeys for trace
     destination (default `TRACES_DIR`).
4. `artifacts/<task>/INSTRUMENTATION.md` exists, documenting:
   - Which events you emit, their schema (JSON shape).
   - Which lines of the original repo you changed, and why.
   - How to rebuild / re-run the harness from a clean checkout.
5. **Smoke verification** — run the harness once:
   ```
   cd artifacts/<task> && bash run.sh
   ```
   Confirm at least 5 NDJSON trace files land in `traces_output_env`'s
   directory, each non-empty, parseable JSON per line.
6. **End-to-end smoke** — launch TV against an existing spec for this task:
   ```
   bash scripts/launch_tv_eval.sh --task=<task> \
     --spec=tla_eval/tasks/<task>/reference_spec.tla   # or any existing spec
   ```
   The agent's `final_report.md` must contain an actual pass rate (e.g.,
   `AcquireLock 103/103`), NOT "Cannot evaluate — spec broken" or
   "no workspace". If the spec itself is broken, the agent will still say
   "cannot evaluate" — that's fine, re-run on a known-good spec for the
   smoke test (use `experiments/best_specs/<task>.tla` if available).
7. Commit per-task: source changes go into `artifacts/<task>/` as a git
   patch (not tracked in the SysMoBench repo's git; the `artifacts/` tree
   is gitignored). The new task.yaml fields DO get committed.

## Order of operations (recommended)

Start with the **easiest** tasks first so the toolchain shakes out before
you hit the complex Raft-family systems:

1. **`mutex`** — reuse `artifacts/spin/` (Asterinas kernel already set up
   for spin). Add a `test_mutex_trace` ktest with minor changes; most of
   spin's infrastructure transfers.
2. **`ringbuffer`** — same Asterinas clone. Category B (ns-level), needs
   the timebox pattern from `references/concurrent-timebox-guide.md`.
3. **`rwmutex`** — same Asterinas clone. Category B.
4. **`redisraft`** — already cloned in Specula dir; copy/symlink and
   instrument. Category A.
5. **`dqueue`, `locksvc`, `raftkvs`** — share one `artifacts/pgo-distcompiler/`
   clone (PGo generates all three). Each is a different TLA+ spec
   compiled out of PGo, so the harness per-task is mostly "run the PGo
   output with instrumentation" — one shared instrumentation helper,
   three different entry points.
6. **`curp`** — Xline is a large Rust project; budget extra time.
7. **`zookeeper`** — Remix project for FLE tracing; probably the most
   involved because ZK runs as a JVM cluster.

## Cost / time estimates per task

Budget per task (from past work on `spin` + `etcd`):
- Read prompt + understand action semantics: ~30 min
- Set up build environment + first trace: 1–3 hours
- Instrument, validate schema against prompt contract: 1–2 hours
- Write run.sh, verify 5 fresh traces: 30 min
- TV smoke (one known-good spec): 30 min

Estimate **3–6 hours per task**, **~1–2 days for all 9** when done
in parallel across a single agent session.

## What to read first

1. `SKILL.md` — SysMoBench conventions.
2. `guide.md` Step 0 — category decision (A vs B).
3. For any Category B task: `references/concurrent-timebox-guide.md`.
4. `examples/cometbft/` — Category A worked example.
5. `examples/dpdk-ring/` — Category B worked example.
6. The done harnesses in the repo, for SysMoBench-specific patterns:
   - `artifacts/spin/` (Asterinas kernel ktest harness)
   - `/home/ubuntu/Specula/case-studies/etcd-raft/artifact/raft/` (Go
     `with_tla` build-tag harness — used by etcd task)

## Out of scope

- Do NOT touch `spin` or `etcd`. They are already done.
- Do NOT modify task prompts in `tla_eval/tasks/<task>/prompts/` — the
  harness must match the prompt contract, not the other way around.
- Do NOT generate new TLA+ specs. Your output is traces + harness
  infrastructure.
- Do NOT chase TV agent bugs; that's the tv-eval skill's job. If the TV
  smoke fails on ALL reasonable specs for a task, that's a real signal
  the harness emits non-compliant events — fix the harness.

## Reporting back

For each task, produce a short status entry (one block):

```
## <task>
- Category: A / B
- artifacts/<task>/: <absolute path>
- Repo commit: <sha>
- Events emitted: <list>
- Smoke: <N> traces, <M> events total
- TV smoke: PASS / FAIL (last-report-line)
- Open issues: <any known limitations>
```

Append these to `docs/dev/harness_bootstrap_status.md` as you complete each
task. When all 9 are done, the user can launch full TV scoring on every
model's batch runs.
