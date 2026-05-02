---
name: harness-gen
description: "Trace harness generation for SysMoBench. Use when bootstrapping a new task: clone the system into artifacts/<task>/, instrument it to emit NDJSON traces at the task-required granularity, write a run.sh, and produce INSTRUMENTATION.md. One-time work per task; the resulting harness is reused for every spec evaluation via the tv-eval skill."
---

Read `guide.md` for the full workflow methodology.

## SysMoBench conventions (overrides Specula paths in guide.md)

- Cloned source lives at `artifacts/<task>/` (see `artifacts/README.md`).
- The task's prompt at `tla_eval/tasks/<task>/prompts/` defines what must be modeled — the harness must emit events for each required behavior at the right granularity.
- After instrumenting, update `tla_eval/tasks/<task>/task.yaml`'s `tv.repo_path` to point at the cloned tree, and fill `tv.harness` (`instrumentation_file`, `test_file`, `run_command`).
- Trace output goes to `artifacts/<task>/traces/` by default (or a path defined by `tv.harness.traces_output_env`).
- Downstream consumer: the `tv-eval` skill will re-run this harness when spec evaluation needs fresh/compliant traces.

## Orchestration: bootstrap the 9 missing tasks

If you were handed this skill as part of the SysMoBench harness-bootstrap
project, read `BOOTSTRAP_9_SYSTEMS.md` next — it lists the 9 tasks still
needing harnesses, repo reuse hints, ordering, and per-task done-criteria.
