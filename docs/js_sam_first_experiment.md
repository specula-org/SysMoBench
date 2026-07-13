# First End-to-End JS-SAM Experiment on SysMoBench

**Task:** `spin` (Asterinas OS spinlock) · **Backend:** JS-SAM · **Model:** Claude
Opus 4.8 · **Method:** `direct_call`
**Date:** 2026-06-30 · **Host:** Windows 11, Docker Desktop 4.80 (engine 29.6.1)

---

## Abstract

We ran the first complete, four-phase SysMoBench evaluation of the **JS-SAM**
backend — JavaScript specifications written in the [SAM
pattern](https://sam.js.org) — on the `spin` task, using Claude Opus 4.8 to
generate the specification. All four phases produced real results: syntax
(Phase 1), bounded model checking (Phase 2), transition validation against
**real kernel-captured traces** (Phase 3), and invariant verification (Phase 4).

The headline result is a **Phase 3 score of 87.5% (7/8 trace windows)**. The
single failing window is diagnostic rather than incidental: the generated model
reproduces blocking `lock()` acquisitions correctly but its non-blocking
`try_lock` acquire never actually takes the lock — a defect observable only when
acquiring from a free state, and caught precisely because the model was replayed
against transitions captured from the real Asterinas spinlock.

Reaching a real Phase 3 number required standing up a kernel trace-capture
harness from scratch (the trace corpus is not shipped with the benchmark) and
authoring a 2-thread instrumentation scenario matching what the JS-SAM prompt
models. Several cross-platform defects in the harness were fixed along the way,
making JS-SAM runnable on a Windows host.

---

## 1. Background and objective

SysMoBench evaluates AI systems on formally modeling real concurrent/distributed
systems across four automated phases. JS-SAM is the benchmark's first
**non-formal-language** backend: instead of TLA+/Alloy/CSP, the model emits an
executable JavaScript module following the SAM pattern (actions propose, a model
accepts, state is a pure function of the model). It tests a distinct hypothesis:

> Can an LLM produce a correct system specification in a pattern it sees
> constantly in training (JavaScript) more readily than in formal languages it
> rarely sees?

This experiment is the first data point toward that question: a full run of one
model on one task, establishing that the JS-SAM pipeline works end-to-end and
producing a per-phase score profile for later comparison against the formal
backends.

### Integration decisions (from the maintainers)

Four open questions from the JS-SAM spec were resolved by the upstream
maintainer prior to this run, and shaped it:

1. **In scope** — JS-SAM is a valid leaderboard backend.
2. **Traces** — trace sets are not fixed data; they are generated on demand by
   instrumenting the real system. The maintainers' `spin` captures use three
   threads; the JS-SAM prompt models two. The two cannot share a trace set, so
   **JS-SAM must generate its own traces at the granularity it models** (two
   threads), using the existing instrumentation
   (`data/patches/asterinas_tla_trace.patch`, `task.yaml > tv.harness`) as a
   reference. This directly motivated the Phase 3 work below.
3. **Depth bound** — `depthMax` is a per-task config value with a documented
   default; 6 is the default for `spin`.
4. **Shared invariant translation** — only TLA+ has the agent-based invariant
   translator; Alloy, PAT, and JS-SAM fall back to a direct API call.
   Generalizing the agent translator is deferred future work.

---

## 2. Environment and configuration

| Component | Value |
|---|---|
| Host OS | Windows 11, Git Bash + PowerShell |
| Container runtime | Docker Desktop 4.80, engine 29.6.1 (WSL2 backend) |
| JS-SAM sandbox image | `node:20-slim` (throwaway container per helper call) |
| Generation model | `claude` → `claude-opus-4-8` (Anthropic API via LiteLLM) |
| Task | `spin` — Asterinas `ostd/src/sync/spin.rs` |
| Generation method | `direct_call` (single prompt) |
| Exploration depth (Phase 2/4) | `depthMax = 6` |

Two configuration corrections were required before any run:

- The `claude` model entry pointed at `claude-sonnet-4-20250514`, which
  **retired 2026-06-15** and returned `not_found_error`. Repointed to
  `claude-opus-4-8`.
- Opus 4.8 rejects `temperature`/`top_p`/`top_k` (HTTP 400); the LiteLLM adapter
  was sending them unconditionally. Both fixes are described in §5.

---

## 3. Method

Each phase evaluates the **same** generated specification (`spin.js`, produced
once by Opus 4.8 in Phase 1) so the four scores describe one artifact. The
JS-SAM backend shells out to a Node helper (`tools/js-sam/cli.mjs`) inside a
locked-down Docker container (`--network none`, read-only rootfs, non-root user,
CPU/memory/PID caps); the container is the trust boundary for model-generated
JavaScript.

- **Phase 1 — compilation_check:** parse + module-load + structural contract
  check of the generated module.
- **Phase 2 — runtime_check:** bounded exploration of the model's own state
  space by the sam-pattern explorer at `depthMax 6`; a second identical
  exploration detects nondeterminism.
- **Phase 3 — transition_validation (direct path):** each captured
  `(pre_state, action, post_state)` window is replayed —
  `setState(pre)`, `actions[name](data)`, then `getState()` is diffed against
  `post` under the projection rule (only keys present in the trace's post-state
  must match; model-only bookkeeping is ignored). This is the direct path — no
  coding-agent orchestration — because SAM *is* `model.present(action(data))`.
- **Phase 4 — invariant_verification:** three expert spinlock invariants are
  translated to JavaScript predicates (direct API call) and checked by the
  explorer.

---

## 4. Phase 3: capturing real spinlock traces

Phase 3 is the only phase requiring data external to the model, and no `spin`
trace corpus ships with the benchmark. Producing a legitimate trace set meant
building and running the instrumented Asterinas kernel and folding its output
into the JS-SAM state schema. This section documents that pipeline because it is
the experiment's main methodological contribution.

### 4.1 Pipeline

1. **Source at a compatible revision.** The reference instrumentation patch does
   **not** apply to Asterinas `main`; it applies cleanly at tag **v0.16.0**,
   which also matches the toolchain bundled in the
   `asterinas/asterinas:0.16.0-20250822` image. We clone v0.16.0.
2. **Instrumentation.** Apply `data/patches/asterinas_tla_trace.patch` (the
   reference TLA+-trace instrumentation, which adds `ostd/src/sync/spin_trace.rs`
   emitting serial JSON events) plus a new **2-thread ktest** (§4.2).
3. **Build + run.** Inside the image: install `cargo-osdk`, `make initramfs`,
   and `cargo osdk test --features tla-trace --qemu-args='-accel tcg'
   test_spin_2thread`. The kernel boots under QEMU (software TCG) and the ktest
   emits trace events over the serial port.
4. **Parse.** `scripts/harness/spin/parse_traces.py` folds the event stream into
   `(pre, post)` NDJSON windows.

### 4.2 Authoring a 2-thread contention scenario

The reference spin ktests are **single-actor** (all operations attributed to
thread 0), so they never exercise the contention (`trying`/failed-acquire)
behavior the JS-SAM model represents across two threads. A blocking `lock()`
from a second actor in a single-CPU ktest would spin forever (deadlock), so the
new `test_spin_2thread` expresses the second actor's contention with a
**non-blocking `try_lock`** that fails without blocking:

```
actor0.lock()      -> TryAcquireBlocking(0), AcquireSuccess(0)   // 0 holds
actor1.try_lock()  -> AcquireFail(1)                             // contention, no deadlock
actor0.release()   -> Release(0)
actor1.lock()      -> TryAcquireBlocking(1), AcquireSuccess(1)   // 1 holds
actor0.try_lock()  -> AcquireFail(0)                             // contention
actor1.release()   -> Release(1)
actor0.try_lock()  -> AcquireSuccess(0)                          // try from free succeeds
actor0.release()   -> Release(0)
```

### 4.3 Event → state mapping

The kernel emits low-level events
(`TryAcquireBlocking`, `AcquireSuccess`, `AcquireFail`, `Release`). The parser
folds them into windows keyed on the **objective, model-independent** spinlock
state `{lockHeld, lockHolder}` — the real observable lock ownership — and lets
the projection rule ignore the model's internal `threadStatus`/`callType`
bookkeeping. This keeps the traces faithful to the *system* rather than tailored
to any one model:

| Kernel event | Model window |
|---|---|
| `TryAcquireBlocking(t)` | folded (pre-step of a blocking acquire; emits no window) |
| `AcquireSuccess(t)` | `AcquireLock` → lock held by `t` (`callType` `lock` if preceded by `TryAcquireBlocking`, else `try`) |
| `AcquireFail(t)` | `AcquireLock` (`callType` `try`) → ownership unchanged (contention does not steal the lock) |
| `Release(t)` | `ReleaseLock` → lock free |

The 10 emitted events fold to **8 windows** at
`data/sys_traces/spin/spin_2thread.ndjson`.

The whole capture is reproducible via `scripts/harness/spin/run.sh`.

---

## 5. Engineering contributions

The following defects were found and fixed to make the run possible; each is a
self-contained commit on the `full-stack` branch.

1. **Cross-platform JS-SAM Docker sandbox** (`tla_eval/languages/js_sam.py`).
   The sandbox was POSIX-host-only: it called `os.getuid()`/`getgid()` (absent
   on Windows) and mounted the helper and spec at their host paths (invalid
   container destinations for Windows `C:\` paths). Fixed by decoupling host
   paths from fixed in-container mount points (`/opt/js-sam`, `/work`),
   normalizing `-v` sources to forward slashes, rewriting `specPath` to the
   container path, and selecting a non-root `--user` behind a
   `hasattr(os, "getuid")` guard. Behavior is unchanged on Linux/macOS; the
   backend now runs on a Windows host. (Upstreamable.)
2. **Model refresh** (`config/models.yaml`, `tla_eval/models/litellm_adapter.py`).
   Repointed `claude` to `claude-opus-4-8` and added
   `_should_omit_sampling_params()` so `temperature`/`top_p`/`top_k` are dropped
   for models that reject them (Opus 4.7+, Fable 5, Mythos 5) — LiteLLM's
   `drop_params` does not yet cover these models.
3. **UTF-8 console output** (`scripts/run_benchmark.py`). The result summary
   printed `✓`/`✗`, which raised `UnicodeEncodeError` on a Windows cp1252
   console after a successful evaluation; std streams are now reconfigured to
   UTF-8.
4. **`states_explored` for non-TLA+ backends** (`ModelCheckOutcome`,
   `js_sam.py`, `runtime_check.py`, `manual_invariant_evaluator.py`). The metric
   was hardcoded to 0 for any non-TLA+ backend even though the JS-SAM helper
   already reports `stepsExplored`; it is now threaded through (and the invariant
   evaluator's top-level field, previously 0 for all backends, is populated).
5. **Spin Phase-3 trace harness** (`scripts/harness/spin/*`,
   `data/patches/spin_2thread_ktest.patch`, `data/sys_traces/spin/*`). §4.

Cross-platform frictions resolved during the capture: patch pinned to v0.16.0;
CRLF normalization of both the Asterinas clone (`core.autocrlf`) and the patch
files; and building on an ext4 docker volume because the Windows bind-mount does
not support `fallocate` (needed to build the initramfs image).

---

## 6. Results

All four phases evaluate the same Opus 4.8 `spin.js`.

| Phase | Metric | Result | Detail |
|---|---|---|---|
| 1 | compilation_check | **PASS** | 0 syntax errors, 0 semantic errors |
| 2 | runtime_check | **PASS** | 326,592 states explored (`depthMax 6`), 0 violations, no deadlock |
| 3 | transition_validation | **PASS** | **87.5% (7/8 windows)** — AcquireLock 80% (4/5), ReleaseLock 100% (3/3) |
| 4 | invariant_verification | **PASS** | 3/3 invariants: MutualExclusion, LockStatusConsistency, NoDeadlock |

### The Phase 3 discrepancy

The single failing window is an `AcquireLock` where the trace expects
`{lockHeld: true, lockHolder: 0}` but the model produced `{lockHeld: false,
lockHolder: null}` — i.e., the model failed to acquire a **free** lock via the
`try_lock` (`callType: "try"`) path. The pattern is precise:

- Blocking `lock()` acquisitions (`callType: "lock"`) reproduce correctly.
- A failed `try_lock` under contention passes (ownership is unchanged either
  way, so the mismatch is masked).
- A `try_lock` that **should succeed** from a free lock exposes the defect: the
  generated model's `try` acquire never takes the lock.

This is a genuine modeling gap in the Opus 4.8 specification, surfaced only
because the model was checked against transitions observed in the real kernel —
the intended value of transition validation.

---

## 7. Discussion

- **The JS-SAM pipeline is viable end-to-end**, including on a non-Linux host,
  and produces a full four-phase score profile suitable for later comparison
  against the formal-language backends.
- **Phase 3 has teeth.** Even a small, deterministic trace set (8 windows)
  discriminated correct from incorrect transitions and localized a specific
  defect in the generated model. The `try_lock`-from-free case is a good example
  of a transition that syntactic and self-consistency checks (Phases 1–2) and
  invariant checking (Phase 4) all pass while the model is still wrong about
  real behavior.
- **Objective-state windows are robust.** Keying traces on `{lockHeld,
  lockHolder}` and relying on the projection rule kept the trace corpus
  independent of the model under test, which is the correct stance for a
  ground-truth oracle.

---

## 8. Limitations and threats to validity

- **Single model, single task, single generation.** No cross-model or
  cross-language comparison yet; the central JS-SAM hypothesis is not answered by
  one data point.
- **Small, single-scenario trace set.** Eight windows from one deterministic
  ktest. Contention is expressed via `try_lock` on a single CPU rather than true
  SMP concurrency, so scenarios like a thread blocking-then-acquiring on release
  are not represented. Broader coverage (more scenarios, more interleavings)
  would strengthen Phase 3.
- **Invariant translation used the direct API call**, per JS-SAM's current
  policy, not the agent-based translator TLA+ uses (Open Question 4).
- **Non-blocking-only second actor.** The 2-thread ktest models contention
  faithfully for lock ownership but does not capture a real waiter's blocking
  wakeup.

---

## 9. Future work

1. **Broaden the spin trace corpus** — additional `test_spin_*` scenarios
   (blocking waiter released and acquiring, longer interleavings) for richer
   Phase 3 coverage.
2. **Run additional models and record the score profiles** to begin answering
   the JS-SAM hypothesis (e.g., Sonnet 4.6, Haiku 4.5).
3. **A second JS-SAM task** with richer control state (`ringbuffer`/`locksvc`)
   once trace harnesses exist for them.
4. **Generalize the agent invariant translator** so JS-SAM (and Alloy/PAT) can
   share the TLA+ approach instead of falling back to a direct call.

---

## 10. Reproducibility

```bash
# Phases 1/2/4 (generation + evaluation), one metric per invocation:
python scripts/run_benchmark.py --task spin --method direct_call \
  --model claude --language JS-SAM --metric compilation_check
#   ... runtime_check / invariant_verification

# Phase 3: capture traces, then validate.
bash scripts/harness/spin/run.sh          # -> data/sys_traces/spin/spin_2thread.ndjson
python scripts/run_benchmark.py --task spin --method direct_call \
  --model claude --language JS-SAM --metric transition_validation --yes \
  --spec-file <generated spin.js>
```

Requirements: Docker (with the `node:20-slim` and
`asterinas/asterinas:0.16.0-20250822` images), Node ≥ 20, `ANTHROPIC_API_KEY` in
`.env`. The JS-SAM sandbox and the Phase-3 capture both run under Docker Desktop
on Windows after the fixes in §5.
