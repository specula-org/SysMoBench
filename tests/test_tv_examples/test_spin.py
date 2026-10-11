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


def blocking_contention_events():
    return [
        event(0, "TryAcquireBlocking"),
        event(1, "AcquireSuccess", state="locked"),
        event(2, "TryAcquireBlocking", actor=1, state="locked"),
        event(3, "Release"),
        event(4, "AcquireSuccess", actor=1, state="locked"),
        event(5, "StopSpinning", actor=1, state="locked"),
        event(6, "Release", actor=1),
    ]


def test_blocking_contention_records_one_acquisition_after_release():
    result = windows.generate_windows_from_trace(
        "contention", blocking_contention_events()
    )
    assert len(result) == 6
    acquisition = result[4]
    assert acquisition["action"] == "AcquireSuccess"
    assert acquisition["pre_state"] == {
        "lockState": "unlocked",
        "pc": {"0": "idle", "1": "acquiring", "2": "idle"},
    }
    assert acquisition["post_state"] == {
        "lockState": "locked",
        "pc": {"0": "idle", "1": "locked", "2": "idle"},
    }
    assert result[-1]["post_state"] == windows.initial_state()


@pytest.fixture(scope="module")
def executed_contention_events(tmp_path_factory):
    if not shutil.which("rustc"):
        pytest.skip("rustc is required to execute the patched spin methods")
    patch = (ROOT / "data/patches/asterinas_tla_trace.patch").read_text()
    section = patch.split(
        "diff --git a/ostd/src/sync/spin_trace.rs b/ostd/src/sync/spin_trace.rs\n", 1
    )[1].split("\ndiff --git ", 1)[0]
    source = "\n".join(
        line[1:]
        for line in section.splitlines()
        if line.startswith("+") and not line.startswith("+++")
    )
    methods = []
    for name in ("acquire_lock_with_id", "try_acquire_lock_internal", "release_lock"):
        start = source.index(f"    fn {name}(")
        end = source.index("\n    }", start) + len("\n    }")
        methods.append(source[start:end])
    template = Path(__file__).with_name("spin_contention.rs").read_text()
    work = tmp_path_factory.mktemp("spin-contention")
    native_source = work / "main.rs"
    native_source.write_text(
        template.replace("// SPIN_TRACE_METHODS", "\n".join(methods))
    )
    binary = work / "spin-contention"
    subprocess.run(
        ["rustc", "--edition=2021", str(native_source), "-o", str(binary)],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    result = subprocess.run(
        [str(binary)], check=True, capture_output=True, text=True, timeout=10
    )
    return [json.loads(line) for line in result.stdout.splitlines()]


def test_executed_contention_matches_windows(executed_contention_events):
    generated = windows.generate_windows_from_trace(
        "native", executed_contention_events
    )
    expected = windows.generate_windows_from_trace(
        "native", blocking_contention_events()
    )
    assert generated == expected


@pytest.mark.parametrize("mutation", ["unlocked", "nonblocking", "wrong_actor"])
def test_invalid_stop_spinning_is_rejected(mutation):
    events = blocking_contention_events()
    if mutation == "unlocked":
        events[5]["state"] = "unlocked"
    elif mutation == "nonblocking":
        events[2]["action"] = "TryAcquireNonBlocking"
    else:
        events[5]["actor"] = 2
    with pytest.raises(ValueError):
        windows.generate_windows_from_trace("invalid-contention", events)


@pytest.mark.parametrize("missing", [{5, 6, 7}, set(range(5))])
def test_incomplete_sequence_is_rejected_before_overwriting_output(tmp_path, missing):
    events = read_windows(EXAMPLE / "traces/trace_01.jsonl")
    events = [e for e in events if e["seq"] not in missing]
    (tmp_path / "trace_01.jsonl").write_text(
        "".join(json.dumps(e) + "\n" for e in events)
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
    assert "sequence numbers" in result.stderr
    assert output.read_text() == "keep existing results\n"


def test_legacy_mode_preserves_coarse_spin_markers_and_sequence_gaps():
    events = [
        event(0, "AcquireSuccess", state="locked"),
        event(2, "TryAcquireBlocking", actor=1, state="locked"),
        event(3, "StopSpinning", actor=1, state="locked"),
        event(4, "Release"),
        event(5, "AcquireSuccess", actor=1, state="locked"),
        event(6, "Release", actor=1),
    ]
    result = windows.generate_windows_from_trace(
        "legacy", events, allow_legacy_direct_acquire=True
    )
    assert len(result) == 5
    assert result[-1]["post_state"] == windows.initial_state()


@pytest.mark.parametrize(
    "events, message",
    [
        ([event(0, "AcquireSuccess", state="locked")], "missing its request event"),
        ([event(0, "Unknown")], "unknown action"),
        ([event(0, "TryAcquireBlocking", state="locked")], "state mismatch"),
        ([event(0, "Release")], "owning actor"),
        (
            [event(0, "TryAcquireBlocking"), event(1, "StopSpinning", state="locked")],
            "held blocking acquisition",
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
@pytest.mark.parametrize("spec, count", [("reference_spec", 6), ("ai_spec_1", 2)])
def test_tlc_executed_blocking_contention(
    tmp_path, executed_contention_events, spec, count
):
    generated = windows.generate_windows_from_trace(
        "native", executed_contention_events
    )
    source = tmp_path / "windows.jsonl"
    source.write_text("".join(json.dumps(w) + "\n" for w in generated))
    if spec == "ai_spec_1":
        converted = tmp_path / "acquire.ndjson"
        subprocess.run(
            [
                sys.executable,
                str(EXAMPLE / "ai_spec_1/make_windows.py"),
                "--source",
                str(source),
                "--output",
                str(converted),
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        source = converted
    code, report = run_validator(tmp_path, spec, source)
    assert code == 0
    assert report["summary"]["total"] == report["summary"]["passed"] == count


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
