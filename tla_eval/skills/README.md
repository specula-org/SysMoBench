# SysMoBench Skills

Two agent-driven skills that together cover the TV (transition validation) pipeline:

## `harness-gen/` — bootstrap a task's harness (one-time per task)

Input: a task name (e.g. `etcd`) with repository URL in `task.yaml` but no local artifact yet.

Agent work:
1. Clone the upstream repo under `artifacts/<task>/`.
2. Read the task prompt (`tla_eval/tasks/<task>/prompts/`) to learn what behaviors must be observable.
3. Write instrumentation (e.g. `tla_spec_trace.go` for Go, a patch for C, etc.) that emits NDJSON events for each required behavior.
4. Write test scenarios that exercise each of the `tv.target_actions` at least once.
5. Write `harness/run.sh` so anyone (or the next skill) can regenerate traces with one command.
6. Produce `artifacts/<task>/INSTRUMENTATION.md` documenting where each event is emitted.
7. Update `tla_eval/tasks/<task>/task.yaml`:
   - `tv.repo_path` ← path to the clone
   - `tv.harness.{instrumentation_file, test_file, run_command}` ← what we just wrote

Output: a `artifacts/<task>/` tree with working instrumentation. Subsequent evaluations reuse it.

## `tv-eval/` — evaluate one spec (run many times per task)

Input: a specific AI-generated TLA+ spec + the already-bootstrapped harness.

Agent work:
1. Step 0: trace compliance check. If traces violate task contract, use `tv.harness.run_command` to regenerate.
2. Steps 1-8: analyze spec, cut windows, write TV modules, run TLC batch, write report.

Output: per-action pass rates with evidence in `reports/final_report.md`.

## Pipeline for a new task

```
(once per task)                          (many times per spec)
┌───────────────┐  harness-gen skill  ┌──────────────────┐  tv-eval skill  ┌────────────────┐
│ empty task/   │ ─────────────────▶ │ instrumented     │ ─────────────▶ │ per-spec       │
│ task.yaml     │                     │ artifact +       │                 │ pass-rate      │
│ + prompts     │                     │ harness run.sh   │                 │ report         │
└───────────────┘                     └──────────────────┘                 └────────────────┘
```

## Current status

| Task | Repo cloned | Instrumentation ready |
|---|---|---|
| etcd | ✓ (via Specula) | ✓ (via Specula + agent extended) |
| spin | — | Docker-based (existing SysMoBench pipeline) |
| redisraft | ✓ (via Specula) | TBD |
| curp, raftkvs, zookeeper, mutex, rwmutex, dqueue, ringbuffer, locksvc | — | TBD (run harness-gen skill) |
