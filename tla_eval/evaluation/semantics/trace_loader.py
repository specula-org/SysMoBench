"""
Language-neutral NDJSON trace loading + windowing for the direct Phase 3 path.

Backends that set `supports_direct_transition_validation = True` receive an
iterable of `(action, pre_state, post_state)` windows (see
languages/base.py:validate_transitions). This module produces those windows
from the task's captured system traces.

Two NDJSON record shapes are accepted, and may be mixed within a file:

  Event form — one self-contained window per line:
    {"action": "AcquireLock", "data": {...}, "pre_state": {...}, "post_state": {...}}

  Stream form — a state snapshot per line; consecutive lines form windows:
    {"state": {...}}                                       # seeds the stream
    {"action": "AcquireLock", "data": {...}, "state": {...}}   # post-state of the action

`action` in a yielded window is the plain action name when the record has no
`data` payload, or `{"name": ..., "data": ...}` when it does. Windows whose
action is not in the task's `tv.target_actions` are skipped (the stream still
advances through them).
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import yaml

logger = logging.getLogger(__name__)

TRACE_GLOBS = ("*.ndjson", "*.jsonl")

Action = Union[str, Dict[str, Any]]
Window = Tuple[Action, Dict[str, Any], Dict[str, Any]]


def _load_task_config(task_name: str) -> Dict[str, Any]:
    from ...tasks.loader import get_task_loader

    task_yaml = Path(get_task_loader().tasks_dir) / task_name / "task.yaml"
    if not task_yaml.exists():
        raise FileNotFoundError(f"task.yaml not found for task '{task_name}': {task_yaml}")
    return yaml.safe_load(task_yaml.read_text(encoding="utf-8")) or {}


def _make_action(name: str, data: Optional[Dict[str, Any]]) -> Action:
    return {"name": name, "data": data} if data else name


def load_trace_windows(
    task_name: str,
    traces_dir: Optional[Path] = None,
    target_actions: Optional[List[str]] = None,
) -> List[Window]:
    """
    Load all trace windows for a task.

    `traces_dir` and `target_actions` override the task.yaml values
    (`traces_folder` and `tv.target_actions`) when given — mainly for tests.

    Raises FileNotFoundError when the trace folder is missing or holds no
    trace files: captured traces are produced by the task's instrumentation
    harness (task.yaml > tv.harness) and are not checked into the repository.
    """
    if traces_dir is None or target_actions is None:
        config = _load_task_config(task_name)
        if traces_dir is None:
            folder = config.get("traces_folder")
            if not folder:
                raise FileNotFoundError(
                    f"Task '{task_name}' declares no traces_folder in task.yaml."
                )
            traces_dir = Path(folder)
        if target_actions is None:
            target_actions = (config.get("tv") or {}).get("target_actions") or []

    traces_dir = Path(traces_dir)
    if not traces_dir.is_absolute():
        from ...tasks.loader import get_task_loader

        # task.yaml paths are repo-root-relative; resolve against the same
        # root the task loader uses (its tasks_dir is <root>/tla_eval/tasks).
        root = Path(get_task_loader().tasks_dir).resolve().parent.parent
        traces_dir = root / traces_dir

    if not traces_dir.exists():
        raise FileNotFoundError(
            f"Trace folder not found: {traces_dir}. Captured traces are generated "
            f"by the task's instrumentation harness (see task.yaml > tv.harness) "
            f"and are not checked into the repository."
        )

    trace_files: List[Path] = []
    for pattern in TRACE_GLOBS:
        trace_files.extend(sorted(traces_dir.glob(pattern)))
    if not trace_files:
        raise FileNotFoundError(f"Trace folder {traces_dir} contains no {TRACE_GLOBS} files.")

    windows: List[Window] = []
    wanted = set(target_actions or [])
    skipped = 0

    for trace_file in trace_files:
        prev_state: Optional[Dict[str, Any]] = None
        with open(trace_file, encoding="utf-8") as f:
            for line_number, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as e:
                    logger.warning("%s:%d unparseable trace line: %s", trace_file, line_number, e)
                    continue
                if not isinstance(record, dict):
                    continue

                action_name = record.get("action")

                if "pre_state" in record and "post_state" in record:
                    # Event form: a self-contained window.
                    if action_name and (not wanted or action_name in wanted):
                        windows.append(
                            (
                                _make_action(action_name, record.get("data")),
                                record["pre_state"],
                                record["post_state"],
                            )
                        )
                    elif action_name:
                        skipped += 1
                    prev_state = record["post_state"]
                elif "state" in record:
                    # Stream form: window from the previous snapshot to this one.
                    if action_name and prev_state is not None:
                        if not wanted or action_name in wanted:
                            windows.append(
                                (
                                    _make_action(action_name, record.get("data")),
                                    prev_state,
                                    record["state"],
                                )
                            )
                        else:
                            skipped += 1
                    prev_state = record["state"]
                else:
                    logger.warning(
                        "%s:%d trace record has neither pre/post_state nor state; skipped",
                        trace_file, line_number,
                    )

    logger.info(
        "Loaded %d trace window(s) for %s from %d file(s) in %s (%d off-target skipped)",
        len(windows), task_name, len(trace_files), traces_dir, skipped,
    )
    return windows
