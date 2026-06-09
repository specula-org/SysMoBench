"""
JS-SAM backend tests.

Integration-style: they exercise the real Node helper (tools/js-sam/cli.mjs)
against the fixtures in tests/fixtures/js_sam/. Skipped when Node or the
helper's node_modules are unavailable.
"""

import json
import shutil
from pathlib import Path

import pytest

from tla_eval.languages import get
from tla_eval.languages.base import InvariantTemplate
from tla_eval.languages.js_sam import (
    HELPER_NODE_MODULES,
    JsSamBackend,
    _parse_js_sam_translations,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIXTURES = PROJECT_ROOT / "tests" / "fixtures" / "js_sam"
SPECS = FIXTURES / "specs"

node_required = pytest.mark.skipif(
    shutil.which("node") is None or not HELPER_NODE_MODULES.exists(),
    reason="Node.js or the JS-SAM helper dependencies are not installed",
)


@pytest.fixture()
def backend():
    return get("js-sam")


class TestRegistration:
    def test_registry_resolution(self):
        backend = get("js-sam")
        assert backend.name == "JS-SAM"
        assert get("JS-SAM") is backend
        assert get("SAM") is backend
        assert get("jssam") is backend

    def test_identity_fields(self):
        backend = get("js-sam")
        assert backend.fence_label == "javascript"
        assert backend.spec_extension == ".js"
        assert backend.config_fence_label is None
        assert backend.supports_direct_transition_validation is True

    def test_extract_artifacts_accepts_js_and_javascript_fences(self):
        backend = get("js-sam")
        body = "module.exports = {};"
        for label in ("javascript", "js"):
            artifacts = backend.extract_artifacts(f"text\n```{label}\n{body}\n```\ntail")
            assert artifacts.spec == body
            assert artifacts.config is None


@node_required
class TestCheckAvailable:
    def test_tools_ready(self, backend):
        assert backend.check_available() is None


@node_required
class TestPhase1Syntax:
    def _validate(self, backend, fixture, tmp_path):
        spec = (SPECS / fixture).read_text(encoding="utf-8")
        return backend.validate_syntax(spec, None, tmp_path, timeout=60)

    def test_good_spec_passes(self, backend, tmp_path):
        outcome = self._validate(backend, "spin-good.js", tmp_path)
        assert outcome.success, outcome.raw_output
        assert outcome.syntax_errors == []
        assert outcome.semantic_errors == []

    def test_syntax_error_reported_as_syntax(self, backend, tmp_path):
        outcome = self._validate(backend, "spin-syntax-error.js", tmp_path)
        assert not outcome.success
        assert outcome.syntax_errors
        assert outcome.semantic_errors == []

    def test_import_throw_reported_as_semantic(self, backend, tmp_path):
        outcome = self._validate(backend, "spin-load-error.js", tmp_path)
        assert not outcome.success
        assert outcome.syntax_errors == []
        assert any("failed to load" in e for e in outcome.semantic_errors)

    def test_contract_violations_reported(self, backend, tmp_path):
        outcome = self._validate(backend, "spin-missing-export.js", tmp_path)
        assert not outcome.success
        joined = " ".join(outcome.semantic_errors)
        assert "setState" in joined
        assert "checkerIntents" in joined


@node_required
class TestPhase2ModelCheck:
    def test_good_spec_explores_cleanly(self, backend, tmp_path):
        outcome = backend.run_model_checker(SPECS / "spin-good.js", None, tmp_path, timeout=300)
        assert outcome.success, outcome.error_message
        assert outcome.classification is None

    def test_throwing_acceptor_is_runtime_error(self, backend, tmp_path):
        outcome = backend.run_model_checker(SPECS / "spin-throwing.js", None, tmp_path, timeout=300)
        assert not outcome.success
        assert outcome.classification == "runtime_error"
        assert "release by non-holder" in outcome.error_message


@node_required
class TestPhase3Transitions:
    PRE_FREE = {
        "lockHeld": False, "lockHolder": None,
        "threadStatus": {"0": "idle", "1": "idle"},
        "callType": {"0": None, "1": None},
    }
    POST_HELD_0 = {
        "lockHeld": True, "lockHolder": 0,
        "threadStatus": {"0": "locked", "1": "idle"},
        "callType": {"0": None, "1": None},
    }

    def _windows(self):
        return [
            (
                {"name": "AcquireLock", "data": {"thread": 0, "callType": "lock"}},
                self.PRE_FREE,
                self.POST_HELD_0,
            ),
            (
                {"name": "ReleaseLock", "data": {"thread": 0}},
                self.POST_HELD_0,
                self.PRE_FREE,
            ),
        ]

    def test_good_spec_passes_all_windows(self, backend, tmp_path):
        outcome = backend.validate_transitions(
            SPECS / "spin-good.js", self._windows(), tmp_path, timeout=120
        )
        assert outcome.error_message is None
        assert outcome.total_windows == 2
        assert outcome.total_passed == 2
        assert outcome.per_action_pass_rates == {"AcquireLock": 1.0, "ReleaseLock": 1.0}

    def test_buggy_release_fails_only_release_windows(self, backend, tmp_path):
        outcome = backend.validate_transitions(
            SPECS / "spin-bad-release.js", self._windows(), tmp_path, timeout=120
        )
        assert outcome.error_message is None
        assert outcome.per_action_pass_rates["AcquireLock"] == 1.0
        assert outcome.per_action_pass_rates["ReleaseLock"] == 0.0
        failures = json.loads((tmp_path / "transition_failures.json").read_text(encoding="utf-8"))
        assert any(f["action"] == "ReleaseLock" for f in failures)

    def test_unknown_action_counts_as_failed_window(self, backend, tmp_path):
        windows = [("NotAnAction", self.PRE_FREE, self.PRE_FREE)]
        outcome = backend.validate_transitions(
            SPECS / "spin-good.js", windows, tmp_path, timeout=120
        )
        assert outcome.total_windows == 1
        assert outcome.total_passed == 0
        assert outcome.per_action_pass_rates["NotAnAction"] == 0.0

    def test_empty_windows_is_an_error(self, backend, tmp_path):
        outcome = backend.validate_transitions(SPECS / "spin-good.js", [], tmp_path, timeout=120)
        assert outcome.error_message


@node_required
class TestPhase4Invariants:
    def _templates(self):
        return [
            InvariantTemplate(
                name="MutualExclusion", type="safety",
                natural_language="At most one thread holds the lock",
                formal_description="", example="",
            ),
            InvariantTemplate(
                name="AlwaysFree", type="safety",
                natural_language="The lock is never held (deliberately false)",
                formal_description="", example="",
            ),
        ]

    def test_check_invariants_pass_and_fail(self, backend, tmp_path):
        translated = {
            "MutualExclusion":
                "(state) => Object.values(state.threadStatus || {})"
                ".filter((s) => s === 'locked').length <= 1",
            "AlwaysFree": "(state) => state.lockHeld === false",
        }
        outcome = backend.check_invariants(
            SPECS / "spin-good.js", None, self._templates(), translated, tmp_path, timeout=300
        )
        by_name = {c.name: c for c in outcome.cases}
        assert by_name["MutualExclusion"].success, by_name["MutualExclusion"].error_message
        assert not by_name["AlwaysFree"].success
        assert by_name["AlwaysFree"].metadata.get("counterexample")

    def test_missing_translation_is_a_failure_case(self, backend, tmp_path):
        outcome = backend.check_invariants(
            SPECS / "spin-good.js", None, self._templates()[:1], {}, tmp_path, timeout=120
        )
        assert len(outcome.cases) == 1
        assert not outcome.cases[0].success
        assert "No translated invariant" in outcome.cases[0].error_message


class TestTranslationParsing:
    def _templates(self):
        return [
            InvariantTemplate(
                name="MutualExclusion", type="safety",
                natural_language="", formal_description="", example="",
            ),
        ]

    def test_parses_plain_json(self):
        text = json.dumps({
            "invariants": [{"name": "MutualExclusion", "predicate": "(state) => true"}]
        })
        out = _parse_js_sam_translations(text, self._templates())
        assert out == {"MutualExclusion": "(state) => true"}

    def test_parses_fenced_json_and_case_insensitive_names(self):
        text = '```json\n{"invariants": [{"name": "mutualexclusion", "predicate": "(s) => true"}]}\n```'
        out = _parse_js_sam_translations(text, self._templates())
        assert out == {"MutualExclusion": "(s) => true"}

    def test_rejects_garbage(self):
        assert _parse_js_sam_translations("not json", self._templates()) == {}
        assert _parse_js_sam_translations('{"foo": 1}', self._templates()) == {}

    def test_unknown_translator_unsupported(self):
        backend = JsSamBackend()
        # An explicit unknown model name is forwarded to the model registry,
        # which fails as "not configured" — exercised via the error return.
        translated, error = backend.translate_invariants(
            self._templates(), "module.exports = {}", "spin",
            translator="definitely-not-a-model",
        )
        assert translated == {}
        assert error
