"""
TLA+ language backend.

Delegates to the existing TLA+ machinery (TLAValidator, TLCRunner,
InvariantTranslator, StaticConfigGenerator). Behavior matches the
pre-refactor evaluators byte-for-byte; this module just rehouses the
dispatch behind the LanguageBackend interface.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .base import InvariantTemplate, LanguageBackend
from .registry import register
from .result_types import (
    InvariantCaseResult,
    InvariantOutcome,
    ModelCheckOutcome,
    SyntaxOutcome,
)

logger = logging.getLogger(__name__)


class TLAPlusBackend(LanguageBackend):
    name = "TLA+"
    aliases = ("tla+", "tla", "tlaplus", "tla_plus")
    fence_label = "tla"
    config_fence_label = "cfg"
    spec_extension = ".tla"
    config_extension = ".cfg"

    def check_available(self) -> Optional[str]:
        try:
            from ..utils.setup_utils import get_tla_tools_path, check_java_available
        except Exception as e:
            return f"setup_utils import failed: {e}"
        if not get_tla_tools_path().exists():
            return (
                "TLA+ tools not found. Run `python3 -m tla_eval.setup` to download them."
            )
        if not check_java_available():
            return "Java not found in PATH; TLA+ tools require Java."
        return None

    # -- Phase 1 -----------------------------------------------------------

    def validate_syntax(
        self,
        spec: str,
        config: Optional[str],
        work_dir: Path,
        timeout: int,
    ) -> SyntaxOutcome:
        from ..core.verification.validators import TLAValidator

        validator = TLAValidator(timeout=timeout)
        start = time.time()
        try:
            result = validator.validate_specification(spec)
        except Exception as e:
            return SyntaxOutcome(
                success=False,
                error_message=f"TLAValidator raised: {e}",
                elapsed_seconds=time.time() - start,
            )
        return SyntaxOutcome(
            success=result.success,
            syntax_errors=list(result.syntax_errors),
            semantic_errors=list(result.semantic_errors),
            raw_output=result.output,
            elapsed_seconds=result.compilation_time or (time.time() - start),
        )

    # -- Phase 2 -----------------------------------------------------------

    def run_model_checker(
        self,
        spec_path: Path,
        config_path: Optional[Path],
        work_dir: Path,
        timeout: int,
    ) -> ModelCheckOutcome:
        from ..evaluation.semantics.runtime_check import TLCRunner

        if config_path is None:
            return ModelCheckOutcome(
                success=False,
                error_message="TLA+ runtime check requires a .cfg config; none was provided.",
            )

        runner = TLCRunner(timeout=timeout)
        start = time.time()
        try:
            ok, output, exit_code = runner.run_model_checking(
                str(spec_path), str(config_path), record_stats=True, use_deadlock_flag=False
            )
        except Exception as e:
            return ModelCheckOutcome(
                success=False,
                error_message=f"TLCRunner raised: {e}",
                elapsed_seconds=time.time() - start,
            )
        return ModelCheckOutcome(
            success=ok,
            raw_output=output,
            elapsed_seconds=time.time() - start,
            classification=None if ok else "tlc_failure",
            error_message=None if ok else f"TLC exit {exit_code}",
        )

    # -- Phase 4 -----------------------------------------------------------

    def invariant_template_dirname(self) -> str:
        return "invariant_templates"

    def invariant_example_field(self) -> str:
        return "tla_example"

    def translate_invariants(
        self,
        templates: List[InvariantTemplate],
        spec: str,
        task_name: str,
        translator: str = "claude-code",
    ) -> Tuple[Dict[str, str], Optional[str]]:
        from ..evaluation.semantics.manual_invariant_evaluator import (
            AgentInvariantTranslator,
            InvariantTemplate as TLAInvariantTemplate,
            InvariantTranslator,
        )

        legacy = [
            TLAInvariantTemplate(
                name=t.name,
                type=t.type,
                natural_language=t.natural_language,
                formal_description=t.formal_description,
                tla_example=t.example,
            )
            for t in templates
        ]

        if translator == "claude-code":
            agent = AgentInvariantTranslator()
            success, translated, error = agent.translate_all_invariants(
                legacy, spec, task_name, "claude"
            )
        elif translator == "codex":
            return {}, "translator 'codex' not yet supported by TLA+ backend (AgentInvariantTranslator is claude-code only)"
        else:
            # "claude" or an explicit model name → direct API call
            direct = InvariantTranslator()
            success, translated, error = direct.translate_all_invariants(
                legacy, spec, task_name, translator
            )

        if not success:
            return {}, error
        return translated, None

    def check_invariants(
        self,
        spec_path: Path,
        config_path: Optional[Path],
        templates: List[InvariantTemplate],
        translated: Dict[str, str],
        work_dir: Path,
        timeout: int,
    ) -> InvariantOutcome:
        from ..evaluation.semantics.manual_invariant_evaluator import (
            ManualInvariantEvaluator,
        )
        from ..evaluation.semantics.runtime_check import TLCRunner

        if config_path is None:
            return InvariantOutcome(
                cases=[], translation_error="TLA+ invariant check requires a base .cfg config."
            )

        spec_text = spec_path.read_text(encoding="utf-8")
        base_config = config_path.read_text(encoding="utf-8")

        evaluator = ManualInvariantEvaluator(tlc_timeout=timeout)
        evaluator.tlc_runner = TLCRunner(timeout=timeout)

        outcome = InvariantOutcome()

        for template in templates:
            name = template.name
            translated_invariant = translated.get(name)
            if translated_invariant is None:
                outcome.cases.append(
                    InvariantCaseResult(
                        name=name,
                        success=False,
                        error_message="No translated invariant produced for this template.",
                    )
                )
                continue

            inv_dir = work_dir / name
            inv_dir.mkdir(parents=True, exist_ok=True)

            modified_spec = evaluator._add_invariant_to_spec(spec_text, translated_invariant, name)
            spec_file = inv_dir / spec_path.name
            spec_file.write_text(modified_spec, encoding="utf-8")

            cfg_ok, cfg_content, cfg_err = (
                evaluator.static_config_generator.generate_config_for_invariant_from_base(
                    base_config, name, template.type
                )
            )
            if not cfg_ok:
                outcome.cases.append(
                    InvariantCaseResult(
                        name=name,
                        success=False,
                        translated=translated_invariant,
                        error_message=f"Config generation failed: {cfg_err}",
                    )
                )
                continue

            cfg_file = inv_dir / config_path.name
            cfg_file.write_text(cfg_content, encoding="utf-8")

            start = time.time()
            try:
                tlc_ok, tlc_output, _exit = evaluator.tlc_runner.run_model_checking(
                    str(spec_file), str(cfg_file), record_stats=False, use_deadlock_flag=True
                )
                violations, deadlock, _states = evaluator.tlc_runner.parse_tlc_output(tlc_output)
                success = tlc_ok and not violations and not deadlock
                outcome.cases.append(
                    InvariantCaseResult(
                        name=name,
                        success=success,
                        translated=translated_invariant,
                        raw_output=tlc_output,
                        elapsed_seconds=time.time() - start,
                        error_message=None
                        if success
                        else f"TLC: {len(violations)} violations, deadlock={deadlock}",
                    )
                )
            except Exception as e:
                outcome.cases.append(
                    InvariantCaseResult(
                        name=name,
                        success=False,
                        translated=translated_invariant,
                        elapsed_seconds=time.time() - start,
                        error_message=f"TLC raised: {e}",
                    )
                )

        return outcome


register(TLAPlusBackend())
