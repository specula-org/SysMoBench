# SysMoBench

SysMoBench is a benchmark for evaluating AI on formally modeling complex real-world systems. It targets TLA+, the de facto specification language for concurrent and distributed systems, and automates four kinds of evaluation: syntax checking with SANY, runtime model checking with TLC, transition validation against captured system traces, and verification of expert-written invariants. Eleven systems are included, ranging from kernel-level synchronization primitives in the Asterinas operating system to industrially deployed consensus implementations such as etcd Raft, Redis Raft, and Xline CURP.

The corresponding paper appears at ICLR 2026: ["SysMoBench: Evaluating AI on Formally Modeling Complex Real-World Systems"](https://openreview.net/forum?id=SAeaTz8YoM). Up-to-date scores are at [sysmobench.com](https://sysmobench.com).

## Highlights

- **End-to-end automation.** Generation, compilation, model checking, transition validation against real traces, and invariant verification all run as a single pipeline, with no human in the loop.
- **Real systems with real traces.** Each task is built around the upstream system's actual source code, paired with an instrumentation harness that emits NDJSON traces from a real execution and a hand-written invariant template.

## Setup

Required on the host:

- Python 3.8+
- Java 11+ (for SANY and TLC, downloaded by the setup script below)
- Docker (for the Asterinas-based harnesses: `spin`, `mutex`, `rwmutex`)
- Go 1.26+ (for the `etcd` harness)
- Maven and a JDK build chain (for the `zookeeper` and `redisraft` harnesses)
- A coding-agent CLI — either [`claude-code`](https://github.com/anthropics/claude-code) or [`codex`](https://github.com/openai/codex) — used by transition validation and by the agent-driven invariant translator

Then install (a virtual environment is recommended on Python 3.12+ hosts that enforce PEP 668):

```
git clone https://github.com/specula-org/SysMoBench.git
cd SysMoBench
python3 -m venv .venv && source .venv/bin/activate
pip install -e .
sysmobench-setup
```

Add the models you intend to evaluate to `config/models.yaml` (the file ships with example entries) and export the corresponding API keys.

Alternatively, pull the prebuilt image (published per release tag):

```
docker pull ghcr.io/specula-org/sysmobench:latest
docker run --rm -it -e ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY ghcr.io/specula-org/sysmobench:latest
```

Or build it locally:

```
docker build -t sysmobench .
docker run --rm -it -e ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY sysmobench
```

The image bundles Python, Java, Maven, Go, and the `claude-code` CLI. Asterinas-based tasks (`spin`, `mutex`, `rwmutex`, `ringbuffer`) launch their own containers — pass `-v /var/run/docker.sock:/var/run/docker.sock` to forward the host Docker socket when running those.

## Running

A single (system, model, metric) cell:

```
sysmobench --task spin --method direct_call --model claude --metric compilation_check
```

A full sweep across all 11 systems:

```
python3 scripts/run_batch_experiment.py --all --model claude
```

See [`docs/Usage.md`](docs/Usage.md) for details.

## Tasks

`sysmobench --list-tasks` enumerates the live set.

| System | Type |
|---|---|
| `spin`, `mutex`, `rwmutex` | Asterinas OS synchronization primitives |
| `ringbuffer` | Concurrent queue |
| `etcd`, `redisraft` | Raft consensus |
| `curp` | Xline CURP replication |
| `zookeeper` | Distributed coordination |
| `dqueue`, `locksvc`, `raftkvs` | PGo-compiled distributed systems |

## Metrics

| Stage | What it measures |
|---|---|
| Syntax | The spec compiles (`compilation_check`, `action_decomposition`) |
| Runtime | TLC can execute it (`runtime_check`, `coverage`, `runtime_coverage`) |
| Transition validation | Per-action conformance to captured system traces (`transition_validation`) |
| Invariant verification | The spec satisfies expert invariants (`invariant_verification`) |

`sysmobench --list-metrics` gives the full catalog. Canonical aggregate weights are 0.15, 0.15, 0.35, and 0.35 for the four stages above.

## Leaderboard

Up-to-date scores live at [sysmobench.com](https://sysmobench.com).

## Adding a new system

See [`docs/add_new_system.md`](docs/add_new_system.md). A system is declared by `tla_eval/tasks/<name>/task.yaml`, paired with prompts, an instrumentation harness, and an invariant template; once those pieces are in place the rest of the pipeline picks the system up automatically.

## Citation

```bibtex
@inproceedings{cheng2026sysmobench,
  title     = {SysMoBench: Evaluating AI on Formally Modeling Complex Real-World Systems},
  author    = {Cheng, Qian and Tang, Ruize and Ma, Emilie and Hackett, Finn and
               He, Peiyang and Su, Yiming and Beschastnikh, Ivan and Huang, Yu and
               Ma, Xiaoxing and Xu, Tianyin},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2026},
  url       = {https://arxiv.org/abs/2509.23130}
}
```

## License

Apache License 2.0 — see [`LICENSE`](LICENSE).
