"""
Tests for the language-neutral trace loader and the direct Phase 3 path in
TransitionValidationEvaluator.
"""

import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

from tla_eval.evaluation.semantics.trace_loader import load_trace_windows

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_TRACES = PROJECT_ROOT / "tests" / "fixtures" / "js_sam" / "traces"

TARGETS = ["AcquireLock", "ReleaseLock"]


class TestTraceLoaderStreamForm:
    def test_windows_from_consecutive_states(self, tmp_path):
        shutil.copy(FIXTURE_TRACES / "spin-stream.ndjson", tmp_path / "t.ndjson")
        windows = load_trace_windows("spin", traces_dir=tmp_path, target_actions=TARGETS)
        # 7 lines: 1 seed state + 6 action records -> 6 windows
        assert len(windows) == 6
        action, pre, post = windows[0]
        assert action == {"name": "AcquireLock", "data": {"thread": 0, "callType": "lock"}}
        assert pre["lockHeld"] is False
        assert post["lockHeld"] is True

    def test_pre_state_chains_through_the_stream(self, tmp_path):
        shutil.copy(FIXTURE_TRACES / "spin-stream.ndjson", tmp_path / "t.ndjson")
        windows = load_trace_windows("spin", traces_dir=tmp_path, target_actions=TARGETS)
        for i in range(1, len(windows)):
            assert windows[i][1] == windows[i - 1][2]


class TestTraceLoaderEventForm:
    def test_self_contained_windows_and_filtering(self, tmp_path):
        shutil.copy(FIXTURE_TRACES / "spin-events.ndjson", tmp_path / "t.ndjson")
        windows = load_trace_windows("spin", traces_dir=tmp_path, target_actions=TARGETS)
        # 3 records, one of which ("Noise") is off-target.
        assert len(windows) == 2
        assert [w[0]["name"] for w in windows] == ["AcquireLock", "ReleaseLock"]

    def test_no_filter_keeps_everything(self, tmp_path):
        shutil.copy(FIXTURE_TRACES / "spin-events.ndjson", tmp_path / "t.ndjson")
        windows = load_trace_windows("spin", traces_dir=tmp_path, target_actions=[])
        assert len(windows) == 3
        # The Noise record carries no data payload -> plain string action.
        assert windows[2][0] == "Noise"


class TestTraceLoaderErrors:
    def test_missing_folder_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="Trace folder not found"):
            load_trace_windows("spin", traces_dir=tmp_path / "nope", target_actions=TARGETS)

    def test_empty_folder_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="contains no"):
            load_trace_windows("spin", traces_dir=tmp_path, target_actions=TARGETS)

    def test_unparseable_lines_are_skipped(self, tmp_path):
        (tmp_path / "t.ndjson").write_text(
            'not json\n{"action": "AcquireLock", "pre_state": {}, "post_state": {}}\n',
            encoding="utf-8",
        )
        windows = load_trace_windows("spin", traces_dir=tmp_path, target_actions=TARGETS)
        assert len(windows) == 1


@pytest.mark.skipif(
    shutil.which("node") is None,
    reason="Node.js is required for the direct-path evaluator test",
)
class TestDirectPathEvaluator:
    def test_direct_path_dispatches_to_backend(self, tmp_path, monkeypatch):
        from tla_eval.evaluation.semantics import trace_loader
        from tla_eval.evaluation.semantics.transition_validation import (
            TransitionValidationEvaluator,
        )

        spec = PROJECT_ROOT / "tests" / "fixtures" / "js_sam" / "specs" / "spin-good.js"
        pre = {
            "lockHeld": False, "lockHolder": None,
            "threadStatus": {"0": "idle", "1": "idle"},
            "callType": {"0": None, "1": None},
        }
        post = {
            "lockHeld": True, "lockHolder": 0,
            "threadStatus": {"0": "locked", "1": "idle"},
            "callType": {"0": None, "1": None},
        }
        windows = [({"name": "AcquireLock", "data": {"thread": 0, "callType": "lock"}}, pre, post)]
        monkeypatch.setattr(trace_loader, "load_trace_windows", lambda task_name: windows)

        evaluator = TransitionValidationEvaluator(
            language="JS-SAM", workspace_root=str(tmp_path)
        )
        result = evaluator.evaluate(
            generation_result=SimpleNamespace(metadata={}),
            task_name="spin",
            method_name="direct_call",
            model_name="fixture",
            spec_file_path=str(spec),
        )

        assert result.error_message is None
        assert result.total_windows == 1
        assert result.total_passed == 1
        assert result.score == 1.0
        assert result.overall_success
        assert result.per_action_pass_rates == {"AcquireLock": 1.0}

    def test_missing_traces_reported_cleanly(self, tmp_path, monkeypatch):
        from tla_eval.evaluation.semantics import trace_loader
        from tla_eval.evaluation.semantics.transition_validation import (
            TransitionValidationEvaluator,
        )

        def raise_missing_traces(task_name):
            raise FileNotFoundError(f"Trace folder not found for task '{task_name}'")

        monkeypatch.setattr(trace_loader, "load_trace_windows", raise_missing_traces)

        spec = PROJECT_ROOT / "tests" / "fixtures" / "js_sam" / "specs" / "spin-good.js"
        evaluator = TransitionValidationEvaluator(
            language="JS-SAM", workspace_root=str(tmp_path)
        )
        result = evaluator.evaluate(
            generation_result=SimpleNamespace(metadata={}),
            task_name="spin",
            method_name="direct_call",
            model_name="fixture",
            spec_file_path=str(spec),
        )
        assert not result.overall_success
        assert result.error_message
        assert "Trace folder" in result.error_message or "traces_folder" in result.error_message
