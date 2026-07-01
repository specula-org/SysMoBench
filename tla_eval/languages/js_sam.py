"""
JS-SAM language backend.

Evaluates specifications written as executable JavaScript modules following
the SAM pattern (https://sam.js.org), built on @cognitive-fab/sam-pattern and
@cognitive-fab/sam-fsm. SAM is grounded in TLA+ semantics — actions present
proposals, the model accepts them, state is a pure function of the model —
but the artifact is a runnable .js module rather than formal-spec source.

The Python side runs a small Node helper (tools/js-sam/cli.mjs) with
JSON-in / JSON-out — the same shape as the Alloy (Java helper) and PAT (mono
console) backends.

SECURITY: unlike TLA+/Alloy/PAT, whose artifacts are restricted modelling
languages, a SAM spec is executable JavaScript that this helper loads via
require() and whose invariant predicates it eval()s. Model-generated code is
therefore untrusted native code. Every helper invocation runs inside a
throwaway Docker container (see _run_helper): no network (--network none),
read-only rootfs with the spec mounted read-only, a non-root user, and
CPU / memory / PID limits. The container is the trust boundary; the host is
never exposed to model-generated code.

This is the first backend to set `supports_direct_transition_validation`:
SAM is literally `model.present(action(data))`, so Phase 3 trace replay
reduces to `setState(pre); actions[name](data); diff(getState(), post)` with
no agent orchestration.
"""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import time
from pathlib import Path
from string import Template
from typing import Any, Dict, List, Optional, Tuple

from .base import InvariantTemplate, LanguageBackend, SpecArtifacts
from .registry import register
from .result_types import (
    InvariantCaseResult,
    InvariantOutcome,
    ModelCheckOutcome,
    SyntaxOutcome,
    TransitionOutcome,
)

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
HELPER_DIR = PROJECT_ROOT / "tools" / "js-sam"
HELPER_CLI = HELPER_DIR / "cli.mjs"
HELPER_NODE_MODULES = HELPER_DIR / "node_modules"

MIN_NODE_MAJOR = 20

# Model-generated SAM specs are executable JavaScript, so the helper runs
# inside a locked-down Docker container rather than directly on the host.
# Docker is already a bench-wide dependency (the trace harnesses use it).
DOCKER_IMAGE = "node:20-slim"
_SANDBOX_MEMORY = "4g"
_SANDBOX_CPUS = "2"
_SANDBOX_PIDS = "256"

# Fixed in-container mount points, kept independent of the host paths so the
# sandbox works regardless of host path shape (POSIX paths and Windows C:\
# paths alike). cli.mjs resolves node_modules from its own location, so the
# whole helper dir is mounted and the spec is mounted under a separate dir.
_CONTAINER_HELPER_DIR = "/opt/js-sam"
_CONTAINER_WORK_DIR = "/work"

# Bench-wide exploration depth for the sam-pattern behavior explorer
# (Phases 2 and 4). Deliberately a single constant rather than per-task
# config: revisit when a second task lands (spec Open Question 3).
DEFAULT_DEPTH_MAX = 6


def _docker_available() -> Optional[str]:
    """Return an error string if the Docker daemon can't be reached, else None."""
    try:
        r = subprocess.run(
            ["docker", "version", "--format", "{{.Server.Version}}"],
            capture_output=True, text=True, timeout=20,
        )
    except FileNotFoundError:
        return (
            "Docker not found in PATH. JS-SAM runs model-generated JavaScript "
            "inside a sandboxed container and requires Docker."
        )
    except subprocess.TimeoutExpired:
        return "Docker daemon did not respond within 20s."
    if r.returncode != 0:
        return f"Docker daemon not reachable: {(r.stderr or r.stdout).strip()[:200]}"
    return None


def _image_present() -> bool:
    try:
        r = subprocess.run(
            ["docker", "image", "inspect", DOCKER_IMAGE],
            capture_output=True, text=True, timeout=20,
        )
        return r.returncode == 0
    except Exception:
        return False


def _force_remove_container(name: str) -> None:
    try:
        subprocess.run(["docker", "rm", "-f", name], capture_output=True, timeout=20)
    except Exception:
        pass


# Monotonic-ish suffix so concurrent/sequential helper calls get distinct
# container names (used for forced cleanup on timeout).
_helper_call_counter = 0


def _docker_mount_source(path: Path) -> str:
    """Host path for a ``docker -v`` source, normalized to forward slashes.

    Docker accepts forward-slash paths on every platform; on Windows this
    avoids backslash ambiguity in the ``-v src:dst`` argument (the drive-letter
    colon, e.g. ``C:``, is handled by Docker Desktop's volume parser).
    """
    return str(path).replace("\\", "/")


def _sandbox_user_args() -> List[str]:
    """``--user`` args that run the sandbox as a non-root user.

    On POSIX hosts, match the host uid:gid so the read-only bind mounts are
    guaranteed readable. Windows has no ``os.getuid``/``getgid`` and Docker
    Desktop maps bind mounts readable to any container user, so fall back to a
    fixed non-root uid there.
    """
    if hasattr(os, "getuid"):
        return ["--user", f"{os.getuid()}:{os.getgid()}"]
    return ["--user", "1000:1000"]


def _run_helper(
    subcommand: str, request: Dict[str, Any], timeout: int
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    Invoke one helper subcommand inside a locked-down Docker container.
    Returns (response, error_message); exactly one of the two is None.
    Helper-level failures (crash, bad JSON, ok=False) are tooling errors,
    not spec failures.

    The container is the trust boundary for model-generated JavaScript:
      --network none      no egress
      --read-only         immutable rootfs; only --tmpfs /tmp is writable
      --user <uid:gid>    non-root, no privileges
      ro bind mounts      the helper and the single spec file, read-only
      memory/cpu/pid caps bound a runaway or malicious spec
    """
    global _helper_call_counter
    _helper_call_counter += 1
    container_name = f"sysmobench-jssam-{os.getpid()}-{_helper_call_counter}"

    # Mount the helper (cli.mjs + node_modules) at a fixed container path; the
    # in-container layout is then independent of the host path shape, and the
    # spec is mounted under a separate dir below. The helper only reads these;
    # all output is JSON on stdout. cli.mjs resolves node_modules from its own
    # location (its NODE_PATH setup), so any consistent mount point works.
    mounts = ["-v", f"{_docker_mount_source(HELPER_DIR)}:{_CONTAINER_HELPER_DIR}:ro"]

    # Rewrite specPath to where the spec is mounted *inside* the container.
    # Copy the request so the caller's dict is not mutated.
    request = dict(request)
    spec_path = request.get("specPath")
    if spec_path:
        sp = Path(spec_path).resolve()
        container_spec = f"{_CONTAINER_WORK_DIR}/{sp.name}"
        mounts += ["-v", f"{_docker_mount_source(sp)}:{container_spec}:ro"]
        request["specPath"] = container_spec

    cmd = [
        "docker", "run", "--rm", "-i",
        "--name", container_name,
        "--network", "none",
        *_sandbox_user_args(),
        "--read-only",
        "--tmpfs", "/tmp",
        "-e", "HOME=/tmp",
        "--memory", _SANDBOX_MEMORY,
        "--cpus", _SANDBOX_CPUS,
        "--pids-limit", _SANDBOX_PIDS,
        *mounts,
        "-w", _CONTAINER_HELPER_DIR,
        DOCKER_IMAGE,
        "node", f"{_CONTAINER_HELPER_DIR}/cli.mjs", subcommand,
    ]
    try:
        proc = subprocess.run(
            cmd,
            input=json.dumps(request),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        # The docker client was killed, but --rm may not fire — remove explicitly.
        _force_remove_container(container_name)
        return None, f"JS-SAM helper timed out after {timeout}s ({subcommand})"
    except FileNotFoundError as e:
        return None, f"Cannot run docker: {e}"

    try:
        response = json.loads(proc.stdout)
    except json.JSONDecodeError:
        tail = ((proc.stderr or "") + (proc.stdout or ""))[-1500:]
        return None, (
            f"JS-SAM helper produced no JSON (exit {proc.returncode}, {subcommand}). "
            f"Output tail:\n{tail}"
        )

    if not response.get("ok", False):
        return None, f"JS-SAM helper error ({subcommand}): {response.get('error', 'unknown')}"
    return response, None


def _format_js_templates_for_prompt(templates: List[InvariantTemplate]) -> str:
    out = []
    for t in templates:
        out.append(
            f"""
### {t.name} ({t.type.upper()})
**Description**: {t.natural_language}

**Formal**: {t.formal_description}

**JavaScript Example** (illustrative state shape — adapt to the actual spec):
```javascript
{t.example.strip()}
```
"""
        )
    return "\n".join(out)


def _parse_js_sam_translations(
    generated_text: str, templates: List[InvariantTemplate]
) -> Dict[str, str]:
    """
    Parse {"invariants": [{"name": ..., "predicate": "(state) => ..."}]}
    out of a model response, tolerating a markdown fence around the JSON.
    """
    out: Dict[str, str] = {}
    clean = generated_text.strip()
    fence = re.match(r"^```(?:json)?\s*\n(.*)\n```$", clean, re.DOTALL)
    if fence:
        clean = fence.group(1)

    try:
        data = json.loads(clean)
    except json.JSONDecodeError as e:
        logger.error(f"JS-SAM translator JSON parse failed: {e}")
        return out

    if not (isinstance(data, dict) and isinstance(data.get("invariants"), list)):
        logger.error(f"Unexpected JS-SAM translation JSON structure: {data}")
        return out

    by_name = {t.name.lower(): t.name for t in templates}
    for entry in data["invariants"]:
        if not isinstance(entry, dict):
            continue
        name = str(entry.get("name", "")).strip()
        predicate = entry.get("predicate")
        if not name or not isinstance(predicate, str) or not predicate.strip():
            continue
        canonical = by_name.get(name.lower())
        if canonical:
            out[canonical] = predicate.strip()
    return out


class JsSamBackend(LanguageBackend):
    name = "JS-SAM"
    aliases = ("js-sam", "jssam", "sam", "javascript-sam")
    fence_label = "javascript"
    config_fence_label = None
    spec_extension = ".js"
    config_extension = None

    def check_available(self) -> Optional[str]:
        docker_error = _docker_available()
        if docker_error:
            return docker_error
        if not _image_present():
            return (
                f"JS-SAM sandbox image '{DOCKER_IMAGE}' not found. "
                f"Run: docker pull {DOCKER_IMAGE}"
            )
        if not HELPER_CLI.exists():
            return f"JS-SAM helper not found at {HELPER_CLI}."
        if not HELPER_NODE_MODULES.exists():
            return (
                f"JS-SAM helper dependencies missing. "
                f"Run: npm ci --prefix {HELPER_DIR}"
            )
        # Exercise the full sandboxed path end to end.
        response, error = _run_helper("ping", {}, timeout=60)
        if error:
            return error
        version = str(response.get("node", ""))
        match = re.match(r"v(\d+)\.", version)
        if not match or int(match.group(1)) < MIN_NODE_MAJOR:
            return (
                f"JS-SAM sandbox node {version or '?'} < required v{MIN_NODE_MAJOR} "
                f"(image {DOCKER_IMAGE})."
            )
        logger.debug(
            "JS-SAM sandbox ready (node %s, sam-pattern %s, image %s)",
            response.get("node"), response.get("samPattern"), DOCKER_IMAGE,
        )
        return None

    def extract_artifacts(self, raw_text: str) -> SpecArtifacts:
        """Accept both ```javascript and ```js fences (models emit either)."""
        spec_pattern = re.compile(
            r"```\s*(?:javascript|js)\s*\n(.*?)\n```",
            re.DOTALL | re.IGNORECASE,
        )
        spec_match = spec_pattern.search(raw_text)
        spec = spec_match.group(1) if spec_match else raw_text
        return SpecArtifacts(spec=spec, config=None)

    # ---- Phase 1 --------------------------------------------------------

    def validate_syntax(
        self,
        spec: str,
        config: Optional[str],
        work_dir: Path,
        timeout: int,
        spec_filename: Optional[str] = None,
    ) -> SyntaxOutcome:
        work_dir.mkdir(parents=True, exist_ok=True)
        spec_file = work_dir / (spec_filename or "_validate_input.js")
        spec_file.write_text(spec, encoding="utf-8")

        start = time.time()
        try:
            # Target-action enforcement happens in Phase 3 (the evaluator API
            # doesn't expose the task here); Phase 1 checks parse + load +
            # the module contract shape.
            response, error = _run_helper(
                "validate", {"specPath": str(spec_file.resolve())}, timeout
            )
        finally:
            if spec_filename is None:
                try:
                    spec_file.unlink()
                except FileNotFoundError:
                    pass

        elapsed = time.time() - start
        if error:
            return SyntaxOutcome(
                success=False,
                syntax_errors=[error],
                raw_output=error,
                elapsed_seconds=elapsed,
            )
        return SyntaxOutcome(
            success=bool(response.get("success")),
            syntax_errors=list(response.get("syntaxErrors", [])),
            semantic_errors=list(response.get("semanticErrors", [])),
            raw_output=json.dumps(response, indent=2),
            elapsed_seconds=elapsed,
        )

    # ---- Phase 2 --------------------------------------------------------

    def run_model_checker(
        self,
        spec_path: Path,
        config_path: Optional[Path],
        work_dir: Path,
        timeout: int,
    ) -> ModelCheckOutcome:
        start = time.time()
        response, error = _run_helper(
            "check",
            {"specPath": str(spec_path.resolve()), "depthMax": DEFAULT_DEPTH_MAX},
            timeout,
        )
        elapsed = time.time() - start
        if error:
            classification = "timeout" if "timed out" in error else None
            return ModelCheckOutcome(
                success=False,
                error_message=error,
                elapsed_seconds=elapsed,
                classification=classification,
            )
        errors = response.get("errors", [])
        return ModelCheckOutcome(
            success=bool(response.get("success")),
            raw_output=json.dumps(response, indent=2),
            elapsed_seconds=elapsed,
            error_message="; ".join(errors) if errors else None,
            classification=response.get("classification"),
        )

    # ---- Phase 3 (direct path) ------------------------------------------

    supports_direct_transition_validation = True

    def validate_transitions(
        self,
        spec_path: Path,
        trace_windows,
        work_dir: Path,
        timeout: int,
    ) -> TransitionOutcome:
        windows = []
        for window in trace_windows:
            action, pre_state, post_state = window
            if isinstance(action, dict):
                name = action.get("name")
                data = action.get("data") or {}
            else:
                name = action
                data = {}
            windows.append(
                {"action": name, "data": data, "preState": pre_state, "postState": post_state}
            )

        if not windows:
            return TransitionOutcome(error_message="No trace windows to validate.")

        response, error = _run_helper(
            "transitions", {"specPath": str(spec_path.resolve()), "windows": windows}, timeout
        )
        if error:
            return TransitionOutcome(error_message=error)
        if not response.get("success", False):
            return TransitionOutcome(error_message=response.get("error", "unknown helper failure"))

        work_dir.mkdir(parents=True, exist_ok=True)
        failures = response.get("failures", [])
        if failures:
            (work_dir / "transition_failures.json").write_text(
                json.dumps(failures, indent=2), encoding="utf-8"
            )

        per_action = response.get("perAction", {})
        return TransitionOutcome(
            per_action_pass_rates={name: stats.get("rate", 0.0) for name, stats in per_action.items()},
            total_passed=int(response.get("totalPassed", 0)),
            total_windows=int(response.get("totalWindows", 0)),
        )

    # ---- Phase 4 --------------------------------------------------------

    def invariant_template_dirname(self) -> str:
        return "js_sam_invariant_templates"

    def invariant_example_field(self) -> str:
        return "javascript_example"

    def translate_invariants(
        self,
        templates: List[InvariantTemplate],
        spec: str,
        task_name: str,
        translator: str = "claude-code",
        agent_timeout: Optional[int] = None,
    ) -> Tuple[Dict[str, str], Optional[str]]:
        # No agent-translator exists for JS-SAM — same policy as Alloy/PAT,
        # which remap the generic agent defaults to a direct API call
        # (spec Open Question 4 tracks sharing real agent infrastructure).
        if translator in ("claude-code", "codex", "claude"):
            model_name = "claude"
        else:
            model_name = translator

        from ..config import get_configured_model
        from ..models.base import GenerationConfig
        from ..tasks.loader import get_task_loader

        task_loader = get_task_loader()
        prompt_file = (
            task_loader.tasks_dir / task_name / "prompts" / "js-sam"
            / "phase3_invariant_implementation.txt"
        )
        if not prompt_file.exists():
            return {}, f"JS-SAM invariant prompt not found: {prompt_file}"

        try:
            template_text = prompt_file.read_text(encoding="utf-8")
            prompt = Template(template_text).substitute(
                javascript_specification=spec,
                invariant_templates=_format_js_templates_for_prompt(templates),
            )
        except Exception as e:
            return {}, f"JS-SAM prompt format failed: {e}"

        try:
            model = get_configured_model(model_name)
        except Exception as e:
            return {}, f"JS-SAM translator model '{model_name}' not configured: {e}"

        result = model.generate_direct(prompt, GenerationConfig(use_json_mode=True))
        if not result.success:
            return {}, result.error_message
        return _parse_js_sam_translations(result.generated_text, templates), None

    def check_invariants(
        self,
        spec_path: Path,
        config_path: Optional[Path],
        templates: List[InvariantTemplate],
        translated: Dict[str, str],
        work_dir: Path,
        timeout: int,
    ) -> InvariantOutcome:
        outcome = InvariantOutcome()

        requested = [
            {"name": t.name, "predicate": translated[t.name]}
            for t in templates
            if t.name in translated
        ]

        results_by_name: Dict[str, Dict[str, Any]] = {}
        helper_error: Optional[str] = None
        if requested:
            response, error = _run_helper(
                "invariants",
                {
                    "specPath": str(spec_path.resolve()),
                    "invariants": requested,
                    "depthMax": DEFAULT_DEPTH_MAX,
                },
                timeout,
            )
            if error:
                helper_error = error
            elif not response.get("success", False):
                helper_error = response.get("error", "unknown helper failure")
            else:
                results_by_name = {r["name"]: r for r in response.get("results", [])}

        for template in templates:
            name = template.name
            trans = translated.get(name)
            if trans is None:
                outcome.cases.append(
                    InvariantCaseResult(
                        name=name, success=False,
                        error_message="No translated invariant produced for this template.",
                    )
                )
                continue
            if helper_error:
                outcome.cases.append(
                    InvariantCaseResult(
                        name=name, success=False, translated=trans,
                        error_message=helper_error,
                    )
                )
                continue

            r = results_by_name.get(name)
            if r is None:
                outcome.cases.append(
                    InvariantCaseResult(
                        name=name, success=False, translated=trans,
                        error_message="Helper returned no result for this invariant.",
                    )
                )
                continue

            metadata: Dict[str, Any] = {}
            if r.get("counterexample"):
                metadata["counterexample"] = r["counterexample"]
            if r.get("stepsExplored") is not None:
                metadata["steps_explored"] = r["stepsExplored"]

            outcome.cases.append(
                InvariantCaseResult(
                    name=name,
                    success=bool(r.get("success")),
                    translated=trans,
                    raw_output=json.dumps(r, indent=2),
                    elapsed_seconds=float(r.get("elapsedMs", 0)) / 1000.0,
                    error_message=r.get("error"),
                    metadata=metadata,
                )
            )

        return outcome


register(JsSamBackend())
