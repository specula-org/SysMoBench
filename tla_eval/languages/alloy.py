"""
Alloy language backend.

Wraps the official Alloy compiler via a small Java helper layer
(`AlloyCliValidator`, `AlloyRuntime`). lib/alloy.jar must be present and
the helper classes must be on the classpath alongside it.

Restored from the d7f7bc1^ snapshot of the deleted `alloy_validator.py`,
`alloy_runtime_executor.py`, `alloy_invariant_translator.py`,
`alloy_invariant_check.py`. Behavior preserved; rehoused behind
LanguageBackend.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
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


ALLOY_JAR = Path("lib/alloy.jar")
VALIDATOR_CLASS = "AlloyCliValidator"
RUNTIME_CLASS = "AlloyRuntime"


# ---- Java wrapper helpers ----------------------------------------------------


def _java_classpath() -> str:
    return os.pathsep.join([str(ALLOY_JAR), str(ALLOY_JAR.parent)])


def _run_validator(spec_file: Path, timeout: int) -> SyntaxOutcome:
    start = time.time()
    cmd = ["java", "-cp", _java_classpath(), VALIDATOR_CLASS, str(spec_file.resolve())]
    logger.debug(f"Running: {' '.join(cmd)}")
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return SyntaxOutcome(
            success=False,
            semantic_errors=[f"Timeout after {timeout}s"],
            raw_output=f"Validation timeout after {timeout}s",
            elapsed_seconds=float(timeout),
        )
    except Exception as e:
        return SyntaxOutcome(
            success=False,
            syntax_errors=[f"Cannot run validator: {e}"],
            raw_output=f"Failed to run validator: {e}",
            elapsed_seconds=time.time() - start,
        )

    elapsed = time.time() - start
    rc, stdout, stderr = proc.returncode, proc.stdout, proc.stderr

    if rc == 0:
        return SyntaxOutcome(
            success=True,
            raw_output=stdout,
            elapsed_seconds=elapsed,
        )

    if rc == 1:
        error_type = None
        error_msg = None
        for line in stderr.split("\n"):
            if line.startswith("ERROR_TYPE:"):
                error_type = line.split(":", 1)[1].strip()
            elif line.startswith("ERROR_MSG:"):
                error_msg = line.split(":", 1)[1].strip()
        syntax_errors: List[str] = []
        semantic_errors: List[str] = []
        if error_type == "SYNTAX":
            syntax_errors.append(error_msg or stderr)
        elif error_type == "SEMANTIC":
            semantic_errors.append(error_msg or stderr)
        else:
            syntax_errors.append(error_msg or stderr)
        return SyntaxOutcome(
            success=False,
            syntax_errors=syntax_errors,
            semantic_errors=semantic_errors,
            raw_output=stderr,
            elapsed_seconds=elapsed,
        )

    return SyntaxOutcome(
        success=False,
        syntax_errors=[f"Internal validator error (code {rc}): {stderr}"],
        raw_output=stderr,
        elapsed_seconds=elapsed,
    )


def _run_runtime(spec_file: Path, timeout: int) -> Dict[str, Any]:
    cmd = [
        "java",
        "-cp",
        _java_classpath(),
        RUNTIME_CLASS,
        str(spec_file.resolve()),
        "--timeout",
        str(timeout),
    ]
    logger.debug(f"Running: {' '.join(cmd)}")
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout + 10)
    except subprocess.TimeoutExpired:
        return {"success": False, "error": f"Timeout after {timeout}s"}
    except Exception as e:
        return {"success": False, "error": f"Cannot run runtime checker: {e}"}

    rc, stdout, stderr = proc.returncode, proc.stdout, proc.stderr

    if rc not in (0, 1):
        return {"success": False, "error": f"Runtime checker internal error (code {rc}): {stderr}"}

    commands: List[Dict[str, Any]] = []
    total = 0
    passed = 0
    failed = 0
    current: Optional[Dict[str, Any]] = None
    for raw in stdout.split("\n"):
        line = raw.strip()
        if not line:
            continue
        if line.startswith("COMMANDS:"):
            try:
                total = int(line.split(":", 1)[1].strip())
            except ValueError:
                pass
        elif line.startswith("=== COMMAND_"):
            if current:
                commands.append(current)
            current = {}
        elif line.startswith("LABEL:") and current is not None:
            current["label"] = line.split(":", 1)[1].strip()
        elif line.startswith("TYPE:") and current is not None:
            current["type"] = line.split(":", 1)[1].strip()
        elif line.startswith("RESULT:") and current is not None:
            r = line.split(":", 1)[1].strip()
            current["result"] = r
            if "PASS" in r.upper():
                passed += 1
            else:
                failed += 1
        elif line.startswith("STATUS:") and current is not None:
            current["status"] = line.split(":", 1)[1].strip()
        elif line.startswith("EXEC_TIME:") and current is not None:
            t = line.split(":", 1)[1].strip().replace("ms", "")
            try:
                current["exec_time_ms"] = int(t)
            except ValueError:
                pass
    if current:
        commands.append(current)

    return {
        "success": rc == 0,
        "total_commands": total,
        "successful_commands": passed,
        "failed_commands": failed,
        "commands": commands,
        "raw_output": stdout,
        "stderr": stderr,
    }


# ---- Invariant translator helpers -------------------------------------------


def _format_alloy_templates_for_prompt(templates: List[InvariantTemplate]) -> str:
    out = []
    for t in templates:
        out.append(
            f"""
### {t.name} ({t.type.upper()})
**Description**: {t.natural_language}

**Formal**: {t.formal_description}

**Alloy Example**:
```
{t.example.strip()}
```
"""
        )
    return "\n".join(out)


def _parse_alloy_translations(
    generated_text: str, templates: List[InvariantTemplate]
) -> Dict[str, str]:
    out: Dict[str, str] = {}
    clean = generated_text.strip()
    if clean.startswith("```json"):
        lines = clean.split("\n")
        if lines[0].strip() == "```json" and lines[-1].strip() == "```":
            clean = "\n".join(lines[1:-1])
    elif clean.startswith("```"):
        lines = clean.split("\n")
        if lines[0].strip() == "```" and lines[-1].strip() == "```":
            clean = "\n".join(lines[1:-1])

    try:
        data = json.loads(clean)
    except json.JSONDecodeError as e:
        logger.error(f"Alloy translator JSON parse failed: {e}")
        return out

    if not (isinstance(data, dict) and isinstance(data.get("invariants"), list)):
        logger.error(f"Unexpected Alloy JSON structure: {data}")
        return out

    for inv in data["invariants"]:
        if not isinstance(inv, str) or not inv.strip():
            continue
        name = None
        if "assert" in inv:
            i = inv.find("assert")
            j = inv.find("{", i)
            if i != -1 and j != -1:
                name = inv[i + 6 : j].strip()
        if not name:
            continue
        for t in templates:
            if t.name.lower() == name.lower():
                out[t.name] = inv
                break
    return out


# ---- Backend -----------------------------------------------------------------


class AlloyBackend(LanguageBackend):
    name = "Alloy"
    aliases = ("alloy", "als")
    fence_label = "alloy"
    config_fence_label = None
    spec_extension = ".als"
    config_extension = None

    def check_available(self) -> Optional[str]:
        if not ALLOY_JAR.exists():
            return f"Alloy JAR not found at {ALLOY_JAR}. Please install alloy.jar."
        # Check Java available
        try:
            r = subprocess.run(["java", "-version"], capture_output=True, text=True, timeout=10)
            if r.returncode != 0:
                return "Java not available; `java -version` failed."
        except Exception as e:
            return f"Java check failed: {e}"
        # Check helper class is loadable.
        try:
            r = subprocess.run(
                ["java", "-cp", _java_classpath(), VALIDATOR_CLASS, "--help"],
                capture_output=True, text=True, timeout=10,
            )
        except Exception as e:
            return f"Helper class probe raised: {e}"
        if "ClassNotFoundException" in (r.stderr or "") or "Could not find or load main class" in (r.stderr or ""):
            return (
                f"Alloy Java helper class '{VALIDATOR_CLASS}' is not on the classpath. "
                f"Build/install it next to {ALLOY_JAR}."
            )
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
        spec_file = work_dir / (spec_filename or "_validate_input.als")
        spec_file.write_text(spec, encoding="utf-8")
        try:
            return _run_validator(spec_file, timeout)
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
        rt = _run_runtime(spec_path, timeout)
        elapsed = time.time() - start
        if rt.get("error"):
            return ModelCheckOutcome(
                success=False,
                error_message=rt["error"],
                raw_output=rt.get("stderr", ""),
                elapsed_seconds=elapsed,
            )
        return ModelCheckOutcome(
            success=bool(rt.get("success")),
            raw_output=rt.get("raw_output", ""),
            elapsed_seconds=elapsed,
            error_message=None if rt.get("success") else (
                f"{rt.get('failed_commands', 0)} command(s) did not pass"
            ),
        )

    # ---- Phase 4 --------------------------------------------------------

    def invariant_template_dirname(self) -> str:
        return "alloy_invariant_templates"

    def invariant_example_field(self) -> str:
        return "alloy_example"

    def translate_invariants(
        self,
        templates: List[InvariantTemplate],
        spec: str,
        task_name: str,
        translator: str = "claude-code",
        agent_timeout: Optional[int] = None,
    ) -> Tuple[Dict[str, str], Optional[str]]:
        # Legacy Alloy translator was hardcoded to GPT-5 because Alloy syntax
        # is small and GPT-5 returned the cleanest JSON. Honor that as the
        # default when caller asks for "claude-code" (the generic default).
        # If caller passes an explicit model name, use it.
        if translator in ("claude-code", "codex"):
            # No agent-translator was ever built for Alloy.
            model_name = "gpt5"
        elif translator == "claude":
            model_name = "claude"
        else:
            model_name = translator

        from ..config import get_configured_model
        from ..models.base import GenerationConfig
        from ..tasks.loader import get_task_loader

        task_loader = get_task_loader()
        prompt_file = (
            task_loader.tasks_dir / task_name / "prompts" / "alloy" / "phase3_invariant_implementation.txt"
        )
        if not prompt_file.exists():
            return {}, f"Alloy invariant prompt not found: {prompt_file}"

        try:
            template_text = prompt_file.read_text(encoding="utf-8")
            prompt = Template(template_text).substitute(
                alloy_specification=spec,
                invariant_templates=_format_alloy_templates_for_prompt(templates),
            )
        except Exception as e:
            return {}, f"Alloy prompt format failed: {e}"

        try:
            model = get_configured_model(model_name)
        except Exception as e:
            return {}, f"Alloy translator model '{model_name}' not configured: {e}"

        result = model.generate_direct(prompt, GenerationConfig(use_json_mode=True))
        if not result.success:
            return {}, result.error_message
        return _parse_alloy_translations(result.generated_text, templates), None

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
        alloy_content = spec_path.read_text(encoding="utf-8")

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

            modified = alloy_content + f"\n\n// ---- Invariant: {name} ----\n" + trans
            modified_file = inv_dir / f"{spec_path.stem}_{name}.als"
            modified_file.write_text(modified, encoding="utf-8")

            start = time.time()
            rt = _run_runtime(modified_file, timeout)
            elapsed = time.time() - start

            if rt.get("error"):
                outcome.cases.append(
                    InvariantCaseResult(
                        name=name, success=False, translated=trans,
                        elapsed_seconds=elapsed,
                        error_message=rt["error"], raw_output=rt.get("stderr", ""),
                    )
                )
                continue

            commands = rt.get("commands", [])
            check_cmd = next(
                (c for c in commands if str(c.get("type", "")).lower() == "check"),
                commands[0] if commands else None,
            )
            if not check_cmd:
                outcome.cases.append(
                    InvariantCaseResult(
                        name=name, success=False, translated=trans,
                        elapsed_seconds=elapsed,
                        error_message="No check command executed",
                        raw_output=rt.get("raw_output", ""),
                    )
                )
                continue

            r = str(check_cmd.get("result", ""))
            ok = r.upper().startswith("PASS")
            outcome.cases.append(
                InvariantCaseResult(
                    name=name, success=ok, translated=trans,
                    elapsed_seconds=elapsed,
                    raw_output=str(check_cmd),
                    error_message=None if ok else f"Check failed: {r}",
                )
            )

        return outcome


register(AlloyBackend())
