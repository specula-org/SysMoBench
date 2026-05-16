"""
PAT (Process Analysis Toolkit) language backend.

Wraps PAT3.Console.exe via mono. Notes inherited from the historical
implementation:
  * PAT always returns exit code 0, even on parsing errors — failures are
    detected by scanning stdout/stderr for "Parsing Error:" etc.
  * Runtime output is written to a separate file passed on the command line.
  * Each assertion section is delimited by "====...===" and contains
    "is VALID" / "is INVALID" markers we parse out.

Restored from d7f7bc1^ snapshot of the deleted `pat_validator.py`,
`pat_runtime_executor.py`, `pat_invariant_translator.py`,
`pat_invariant_check.py`, `pat_invariant_loader.py`. Behavior preserved;
rehoused behind LanguageBackend.
"""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import tempfile
import time
from pathlib import Path
from string import Template
from typing import Any, Dict, List, Optional, Tuple

from .base import InvariantTemplate, LanguageBackend
from .registry import register
from .result_types import (
    InvariantCaseResult,
    InvariantOutcome,
    ModelCheckOutcome,
    SyntaxOutcome,
)

logger = logging.getLogger(__name__)

PAT_CONSOLE = Path("lib/PAT3.Console.exe")
PARSE_ERROR_MARKERS = ("Parsing Error:", "parsing error:")
GENERIC_ERROR_KEYWORDS = (
    "error occurred:", "compilation error", "syntax error", "semantic error",
)


# ---- PAT Console wrappers ---------------------------------------------------


def _run_pat_console(spec_file: Path, timeout: int) -> Tuple[int, str, str, str]:
    """
    Returns (returncode, stdout, stderr, output_file_content).
    PAT always returns 0; check output text for errors.
    """
    fd, output_file = tempfile.mkstemp(suffix=".txt", prefix="pat_output_")
    os.close(fd)
    output_path = Path(output_file)
    try:
        cmd = [
            "mono", str(PAT_CONSOLE),
            "-csp",
            str(spec_file.resolve()),
            str(output_path),
        ]
        logger.debug(f"Running: {' '.join(cmd)}")
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout + 10,
        )
        content = output_path.read_text(encoding="utf-8") if output_path.exists() else ""
        return proc.returncode, proc.stdout, proc.stderr, content
    finally:
        if output_path.exists():
            try:
                output_path.unlink()
            except OSError:
                pass


def _parse_validator_output(stdout: str, stderr: str) -> SyntaxOutcome:
    combined = stdout + "\n" + stderr
    if any(m in combined for m in PARSE_ERROR_MARKERS):
        errors = [
            line.strip() for line in combined.split("\n")
            if line.strip() and ("parsing error:" in line.lower() or "error" in line.lower())
        ]
        if not errors:
            errors = ["Parsing error detected (details not available)"]
        return SyntaxOutcome(success=False, syntax_errors=errors, raw_output=combined)
    if any(k in combined.lower() for k in GENERIC_ERROR_KEYWORDS):
        errors = [
            line.strip() for line in combined.split("\n")
            if line.strip() and any(k in line.lower() for k in GENERIC_ERROR_KEYWORDS)
        ]
        return SyntaxOutcome(success=False, syntax_errors=errors, raw_output=combined)
    return SyntaxOutcome(success=True, raw_output=combined)


def _parse_runtime_output(output_content: str, stdout: str, stderr: str) -> Dict[str, Any]:
    combined_streams = stdout + "\n" + stderr
    if any(m in combined_streams for m in PARSE_ERROR_MARKERS):
        return {"success": False, "error": f"Parsing error: {combined_streams}"}

    sections = output_content.split("=======================================================")
    assertions: List[Dict[str, Any]] = []
    passed = 0
    failed = 0
    failed_names: List[str] = []

    for sec in sections:
        sec = sec.strip()
        if not sec or "Assertion:" not in sec:
            continue
        a = _parse_assertion_section(sec)
        if not a:
            continue
        assertions.append(a)
        if a["result"] == "VALID":
            passed += 1
        elif a["result"] == "INVALID":
            failed += 1
            failed_names.append(a.get("name", "Unknown"))

    total = len(assertions)
    out: Dict[str, Any] = {
        "success": failed == 0 and total > 0,
        "total_assertions": total,
        "passed_assertions": passed,
        "failed_assertions": failed,
        "assertions": assertions,
        "total_time": sum(a.get("time_used", 0.0) for a in assertions),
        "raw_output": output_content,
        "stderr": stderr,
    }
    if failed:
        head = ", ".join(failed_names[:3]) + ("..." if failed > 3 else "")
        out["error"] = f"{failed} assertion(s) failed: {head}"
    return out


def _parse_assertion_section(section: str) -> Optional[Dict[str, Any]]:
    try:
        out: Dict[str, Any] = {}
        m = re.search(r"Assertion:\s*(.+?)(?:\n|\r|$)", section)
        out["name"] = m.group(1).strip() if m else "Unknown"
        if "is VALID" in section:
            out["result"] = "VALID"
        elif "is INVALID" in section or "is NOT valid" in section:
            out["result"] = "INVALID"
        else:
            out["result"] = "UNKNOWN"
        m = re.search(r"Visited States:\s*(\d+)", section)
        if m:
            out["visited_states"] = int(m.group(1))
        m = re.search(r"Total Transitions:\s*(\d+)", section)
        if m:
            out["transitions"] = int(m.group(1))
        m = re.search(r"Time Used:\s*([\d.]+)\s*s", section)
        if m:
            out["time_used"] = float(m.group(1))
        return out
    except Exception as e:
        logger.warning(f"Failed to parse PAT assertion section: {e}")
        return None


# ---- Invariant translator helpers -------------------------------------------


_GENERIC_PROMPT = """You are an expert in formal verification and PAT (Process Analysis Toolkit) CSP# specifications.

Your task is to translate generic invariant templates into concrete CSP# assertions for a specific specification.

# CSP# Specification
$csp_specification

# Invariant Templates
$invariant_templates

# Instructions
For each invariant template above, generate a concrete CSP# assertion that:
1. Uses the actual variable names, processes, and structures from the specification
2. Follows PAT CSP# syntax (use #define for formulas, #assert for assertions)
3. Preserves the semantic meaning of the invariant
4. Is syntactically correct and verifiable by PAT

# Output Format
Return a JSON object where keys are invariant names and values are complete CSP# assertion code:

{
  "InvariantName1": "#define formula1 = ...; #assert System |= [] formula1;",
  "InvariantName2": "#assert System deadlockfree;"
}

Return ONLY the JSON object, no additional text."""


def _format_pat_templates_for_prompt(templates: List[InvariantTemplate]) -> str:
    lines: List[str] = []
    for i, t in enumerate(templates, 1):
        lines.append(f"## Invariant {i}: {t.name}")
        lines.append(f"Type: {t.type}")
        lines.append(f"Natural Language: {t.natural_language}")
        lines.append(f"Formal Description: {t.formal_description}")
        lines.append(f"CSP# Example:\n```\n{t.example.strip()}\n```\n")
    return "\n".join(lines)


def _strip_existing_assertions(csp: str) -> str:
    out: List[str] = []
    for line in csp.split("\n"):
        s = line.strip()
        if s.startswith("#assert"):
            continue
        if s.startswith("#define") and ("|=" in s or "deadlock" in s.lower()):
            continue
        out.append(line)
    return "\n".join(out).rstrip() + "\n"


# ---- Backend -----------------------------------------------------------------


class PATBackend(LanguageBackend):
    name = "PAT"
    aliases = ("pat", "csp", "csp#")
    fence_label = "csp"
    config_fence_label = None
    spec_extension = ".csp"
    config_extension = None

    def check_available(self) -> Optional[str]:
        if not PAT_CONSOLE.exists():
            return f"PAT Console not found at {PAT_CONSOLE}."
        try:
            r = subprocess.run(["mono", "--version"], capture_output=True, text=True, timeout=10)
            if r.returncode != 0:
                return "mono runtime not available; `mono --version` failed."
        except FileNotFoundError:
            return "mono runtime not installed; PAT3.Console.exe requires mono on Linux."
        except Exception as e:
            return f"mono probe raised: {e}"
        return None

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
        spec_file = work_dir / (spec_filename or "_validate_input.csp")
        spec_file.write_text(spec, encoding="utf-8")
        start = time.time()
        try:
            _rc, stdout, stderr, _content = _run_pat_console(spec_file, timeout)
            out = _parse_validator_output(stdout, stderr)
            out.elapsed_seconds = time.time() - start
            return out
        except subprocess.TimeoutExpired:
            return SyntaxOutcome(
                success=False,
                semantic_errors=[f"Timeout after {timeout}s"],
                elapsed_seconds=float(timeout),
            )
        except FileNotFoundError as e:
            return SyntaxOutcome(
                success=False,
                syntax_errors=[f"Cannot run PAT: {e}"],
                elapsed_seconds=time.time() - start,
            )
        finally:
            if spec_filename is None:
                try:
                    spec_file.unlink()
                except FileNotFoundError:
                    pass

    # ---- Phase 2 --------------------------------------------------------

    def run_model_checker(
        self,
        spec_path: Path,
        config_path: Optional[Path],
        work_dir: Path,
        timeout: int,
    ) -> ModelCheckOutcome:
        start = time.time()
        try:
            _rc, stdout, stderr, content = _run_pat_console(spec_path, timeout)
        except subprocess.TimeoutExpired:
            return ModelCheckOutcome(
                success=False,
                error_message=f"Timeout after {timeout}s",
                elapsed_seconds=float(timeout),
            )
        except FileNotFoundError as e:
            return ModelCheckOutcome(
                success=False,
                error_message=f"Cannot run PAT: {e}",
                elapsed_seconds=time.time() - start,
            )
        rt = _parse_runtime_output(content, stdout, stderr)
        elapsed = time.time() - start
        if rt.get("error") and not rt.get("assertions"):
            return ModelCheckOutcome(
                success=False, error_message=rt["error"],
                raw_output=rt.get("stderr", ""), elapsed_seconds=elapsed,
            )
        return ModelCheckOutcome(
            success=bool(rt.get("success")),
            raw_output=rt.get("raw_output", ""),
            elapsed_seconds=elapsed,
            error_message=rt.get("error"),
        )

    # ---- Phase 4 --------------------------------------------------------

    def invariant_template_dirname(self) -> str:
        return "pat_invariant_templates"

    def invariant_example_field(self) -> str:
        return "csp_example"

    def translate_invariants(
        self,
        templates: List[InvariantTemplate],
        spec: str,
        task_name: str,
        translator: str = "claude-code",
        agent_timeout: Optional[int] = None,
    ) -> Tuple[Dict[str, str], Optional[str]]:
        # Legacy hardcoded "claude" direct call. Other modes not implemented.
        if translator in ("claude-code", "codex"):
            model_name = "claude"
        elif translator == "claude":
            model_name = "claude"
        else:
            model_name = translator

        from ..config import get_configured_model
        from ..models.base import GenerationConfig

        prompt_template = _load_pat_translation_prompt(task_name)
        prompt = Template(prompt_template).substitute(
            csp_specification=spec,
            invariant_templates=_format_pat_templates_for_prompt(templates),
        )

        try:
            model = get_configured_model(model_name)
        except Exception as e:
            return {}, f"PAT translator model '{model_name}' not configured: {e}"

        result = model.generate_direct(prompt, GenerationConfig(use_json_mode=True))
        if not result.success:
            return {}, result.error_message
        try:
            data = json.loads(result.generated_text)
        except json.JSONDecodeError as e:
            return {}, f"Failed to parse PAT translation JSON: {e}"
        if not isinstance(data, dict):
            return {}, "PAT translation JSON is not a top-level object"
        return {str(k): str(v) for k, v in data.items()}, None

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
        csp_content = spec_path.read_text(encoding="utf-8")
        base = _strip_existing_assertions(csp_content)

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

            inv_dir = work_dir / name
            inv_dir.mkdir(parents=True, exist_ok=True)
            augmented = base + "\n" + trans.strip() + "\n"
            aug_file = inv_dir / f"{spec_path.stem}_{name}.csp"
            aug_file.write_text(augmented, encoding="utf-8")

            start = time.time()
            try:
                _rc, stdout, stderr, content = _run_pat_console(aug_file, timeout)
            except subprocess.TimeoutExpired:
                outcome.cases.append(
                    InvariantCaseResult(
                        name=name, success=False, translated=trans,
                        elapsed_seconds=float(timeout),
                        error_message=f"Timeout after {timeout}s",
                    )
                )
                continue
            except FileNotFoundError as e:
                outcome.cases.append(
                    InvariantCaseResult(
                        name=name, success=False, translated=trans,
                        elapsed_seconds=time.time() - start,
                        error_message=f"Cannot run PAT: {e}",
                    )
                )
                continue

            rt = _parse_runtime_output(content, stdout, stderr)
            elapsed = time.time() - start
            assertions = rt.get("assertions", [])
            visited = sum(a.get("visited_states", 0) for a in assertions)
            if rt.get("success"):
                ok = True
                err = None
            else:
                ok = False
                err = rt.get("error", "PAT check failed")
            outcome.cases.append(
                InvariantCaseResult(
                    name=name, success=ok, translated=trans,
                    elapsed_seconds=elapsed,
                    raw_output=rt.get("raw_output", ""),
                    error_message=err,
                    metadata={"visited_states": visited, "assertions": assertions},
                )
            )

        return outcome


def _load_pat_translation_prompt(task_name: str) -> str:
    """Task-specific prompt at data/prompts/pat_invariant_translation/<task>.txt; else generic."""
    p = Path("data/prompts/pat_invariant_translation") / f"{task_name}.txt"
    if p.exists():
        return p.read_text(encoding="utf-8")
    return _GENERIC_PROMPT


register(PATBackend())
