"""Behavioral regressions for the spin example's trace contract and TLC checks."""

import copy
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = ROOT / "tla_eval/skills/tv-eval/examples/spin"
_spec = importlib.util.spec_from_file_location(
    "spin_windows", EXAMPLE / "generate_windows.py"
)
windows = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(windows)
TLA_JAR = ROOT / "lib/tla2tools.jar"
COMM_JAR = ROOT / "lib/CommunityModules-deps.jar"
requires_tlc = pytest.mark.skipif(
    not (TLA_JAR.is_file() and COMM_JAR.is_file() and shutil.which("java")),
    reason="Run sysmobench-setup to install TLC and CommunityModules",
)


def event(seq, action, actor=0, state="unlocked"):
    return {"seq": seq, "action": action, "actor": actor, "state": state, "lock": 0}


def read_windows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def test_request_is_valid_with_free_or_held_lock():
    events = [
        event(0, "TryAcquireBlocking"),
        event(1, "AcquireSuccess", state="locked"),
        event(2, "TryAcquireBlocking", actor=1, state="locked"),
    ]
    result = windows.generate_windows_from_trace("requests", events)
    assert result[0]["post_state"]["pc"]["0"] == "acquiring"
    assert result[2]["post_state"]["pc"]["1"] == "acquiring"
    assert (
        result[2]["pre_state"]["lockState"]
        == result[2]["post_state"]["lockState"]
        == "locked"
    )


def test_failed_try_lock_preserves_holder_and_allows_retry():
    events = [
        event(0, "TryAcquireBlocking"),
        event(1, "AcquireSuccess", state="locked"),
        event(2, "TryAcquireNonBlocking", actor=1, state="locked"),
        event(3, "AcquireFail", actor=1, state="locked"),
        event(4, "Release"),
        event(5, "TryAcquireNonBlocking", actor=1),
        event(6, "AcquireSuccess", actor=1, state="locked"),
    ]
    result = windows.generate_windows_from_trace("failure", events)
    assert result[3]["post_state"] == {
        "lockState": "locked",
        "pc": {"0": "locked", "1": "idle", "2": "idle"},
    }
    assert result[-1]["post_state"]["pc"]["1"] == "locked"


@pytest.mark.parametrize(
    "events, message",
    [
        ([event(0, "AcquireSuccess", state="locked")], "missing its request event"),
        ([event(0, "Unknown")], "unknown action"),
        ([event(0, "TryAcquireBlocking", state="locked")], "state mismatch"),
        ([event(0, "Release")], "owning actor"),
        (
            [event(0, "TryAcquireBlocking"), event(1, "StopSpinning", state="locked")],
            "state mismatch",
        ),
        (
            [
                event(0, "TryAcquireBlocking"),
                event(0, "AcquireSuccess", state="locked"),
            ],
            "sequence numbers",
        ),
        (
            [
                event(0, "TryAcquireBlocking"),
                {**event(1, "AcquireSuccess", state="locked"), "lock": 1},
            ],
            "one lock",
        ),
    ],
)
def test_bad_trace_is_rejected(events, message):
    with pytest.raises(ValueError, match=message):
        windows.generate_windows_from_trace("invalid", events)


def test_generator_rejects_before_overwriting_output(tmp_path):
    (tmp_path / "trace_01.jsonl").write_text(
        json.dumps(event(0, "AcquireSuccess", state="locked")) + "\n"
    )
    output = tmp_path / "windows.jsonl"
    output.write_text("keep existing results\n")
    result = subprocess.run(
        [
            sys.executable,
            str(EXAMPLE / "generate_windows.py"),
            "--traces-dir",
            str(tmp_path),
            "--output",
            str(output),
        ],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 2
    assert "missing its request" in result.stderr
    assert output.read_text() == "keep existing results\n"


def test_captured_data_regenerates_without_inventing_events():
    generated = []
    for path in sorted((EXAMPLE / "traces").glob("trace_*.jsonl")):
        for window in windows.generate_windows_from_trace(
            path.stem, windows.parse_trace(path)
        ):
            window["window_id"] = len(generated)
            generated.append(window)
    assert generated and generated == read_windows(EXAMPLE / "action_windows.jsonl")
    assert {w["action"] for w in generated} == {
        "TryAcquireBlocking",
        "TryAcquireNonBlocking",
        "AcquireSuccess",
        "AcquireFail",
        "Release",
    }


def test_missing_tools_is_not_a_model_failure(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            str(EXAMPLE / "ai_spec_1/run_tv.py"),
            "--tla-jar",
            str(tmp_path / "missing.jar"),
        ],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 2
    assert "sysmobench-setup" in result.stderr
    assert '"failed"' not in result.stdout


def run_validator(tmp_path, spec, source):
    report = tmp_path / "report.json"
    result = subprocess.run(
        [
            sys.executable,
            str(EXAMPLE / spec / "run_tv.py"),
            "--windows",
            str(source),
            "--report",
            str(report),
            "--workers",
            "2",
        ],
        capture_output=True,
        text=True,
        # The historical reference fixture starts 543 JVMs. Allow the whole
        # batch to finish on hosted runners; each window still has its own
        # 30-second TLC timeout. Limit concurrency to avoid CPU oversubscription.
        timeout=600,
        env={**os.environ, "JAVA_TOOL_OPTIONS": "-XX:ActiveProcessorCount=2 -Xmx256m"},
    )
    assert report.exists(), result.stdout + result.stderr
    return result.returncode, json.loads(report.read_text())


@requires_tlc
@pytest.mark.parametrize(
    "spec,path,passed,failed",
    [
        ("ai_spec_1", "ai_spec_1/windows_AcquireLock.ndjson", 3, 0),
        ("reference_spec", "action_windows.jsonl", 11, 0),
        ("ai_spec_1", "legacy/windows_AcquireLock.ndjson", 123, 87),
        ("reference_spec", "legacy/action_windows.jsonl", 543, 0),
    ],
)
def test_tlc_positive_and_historical_windows(tmp_path, spec, path, passed, failed):
    code, report = run_validator(tmp_path, spec, EXAMPLE / path)
    assert code == (1 if failed else 0)
    assert report["summary"]["passed"] == passed
    assert report["summary"]["failed"] == failed
    assert report["summary"]["errored"] == report["summary"]["timeout"] == 0


@requires_tlc
@pytest.mark.parametrize(
    "mutation", ["held_lock", "wrong_actor", "wrong_post", "missing_request"]
)
def test_tlc_rejects_invalid_acquisition(tmp_path, mutation):
    window = copy.deepcopy(
        read_windows(EXAMPLE / "ai_spec_1/windows_AcquireLock.ndjson")[0]
    )
    if mutation == "held_lock":
        window["pre"]["lockState"] = "locked"
    elif mutation == "wrong_actor":
        window["actor"] = 3
    elif mutation == "wrong_post":
        window["post"]["lockState"] = "unlocked"
    else:
        window["pre"]["pc"][window["actor"] - 1] = "idle"
    source = tmp_path / "invalid.ndjson"
    source.write_text(json.dumps(window) + "\n")
    code, report = run_validator(tmp_path, "ai_spec_1", source)
    assert code == 1
    assert report["summary"]["failed"] == 1
    assert report["summary"]["errored"] == 0


@requires_tlc
def test_malformed_tlc_input_is_an_error(tmp_path):
    source = tmp_path / "invalid.ndjson"
    source.write_text('{"missing": "pre/post state"}\n')
    code, report = run_validator(tmp_path, "ai_spec_1", source)
    assert code == 2
    assert report["summary"]["errored"] == 1
    assert report["summary"]["failed"] == 0


def test_capture_parser_requires_successful_complete_kernel_output():
    spec = importlib.util.spec_from_file_location(
        "spin_capture", ROOT / "scripts/harness/spin_tv/parse_trace.py"
    )
    parser = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(parser)
    log = (EXAMPLE / "traces/kernel-test.txt").read_text()
    assert parser.extract_trace(log) == read_windows(EXAMPLE / "traces/trace_01.jsonl")
    for invalid in (
        log.replace("=== SPIN_TV_END ===", ""),
        log.replace("1 passed; 0 failed;", "0 passed; 1 failed;"),
        log.replace('"seq":5,', '"seq":50,'),
    ):
        with pytest.raises(ValueError):
            parser.extract_trace(invalid)


def test_capture_manifest_matches_patch_and_trace():
    import hashlib

    manifest = json.loads((EXAMPLE / "traces/manifest.json").read_text())
    for source, field in (
        (ROOT / manifest["patch"], "patch_sha256"),
        (EXAMPLE / "traces/trace_01.jsonl", "trace_sha256"),
    ):
        assert hashlib.sha256(source.read_bytes()).hexdigest() == manifest[field]


@requires_tlc
def test_reference_rejects_release_by_nonowner(tmp_path):
    window = copy.deepcopy(
        next(
            w
            for w in read_windows(EXAMPLE / "action_windows.jsonl")
            if w["action"] == "Release"
        )
    )
    window["actor"] = "2"
    source = tmp_path / "invalid.ndjson"
    source.write_text(json.dumps(window) + "\n")
    code, report = run_validator(tmp_path, "reference_spec", source)
    assert code == 1
    assert report["summary"]["failed"] == 1
    assert report["summary"]["errored"] == 0
