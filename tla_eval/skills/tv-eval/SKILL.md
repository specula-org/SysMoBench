---
name: tv-eval
description: "Transition validation (TV) for TLA+ specs. Use when: scoring how faithfully an AI-generated spec models a real system, producing per-action pass rates with defensible explanations. The agent acts as the examiner (考官) — writes instrumentation, runs harness, writes TV modules, runs TLC, and interprets scores."
---

Read `guide.md` for the full workflow.

Reference docs:
- `references/canonical_window_format.md` — the one true window file schema
- `references/tv_module_template.md` — how to write TV_<Action>.tla
- `references/score_interpretation.md` — how to explain pass rates

Worked examples:
- `examples/spin/` — simple case (spinlock), 1 aux variable
- `examples/etcd/` — complex case (etcd-raft), 4 aux variables, log abstraction
