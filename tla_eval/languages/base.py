"""
LanguageBackend abstract base class.

A backend bundles the language-specific operations the four evaluation phases
need, behind one stable interface. The phase evaluators are language-neutral
and dispatch through `registry.get(language)`.

Adding a new language: subclass `LanguageBackend`, implement the abstract
methods, and call `register(MyBackend())` from your module's import side
(see `tla_eval/languages/tla_plus.py` for the reference).
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
import re
from typing import List, Optional, Dict, Tuple

from .result_types import (
    SyntaxOutcome,
    ModelCheckOutcome,
    InvariantOutcome,
    TransitionOutcome,
)


@dataclass
class SpecArtifacts:
    """What `extract_artifacts` pulls out of a model's raw response."""

    spec: str
    config: Optional[str] = None


@dataclass
class InvariantTemplate:
    """Generic invariant template loaded from data/invariant_templates/<lang>/<task>/invariants.yaml."""

    name: str
    type: str  # "safety" | "liveness"
    natural_language: str
    formal_description: str
    example: str  # the per-language reference snippet (e.g. tla_example, alloy_example)


class LanguageBackend(ABC):
    """Abstract backend for one specification language."""

    # ---- identity ----------------------------------------------------------

    name: str = ""
    """Canonical language name as it appears in CLI / config (e.g. 'TLA+')."""

    aliases: Tuple[str, ...] = ()
    """Other accepted spellings (case-insensitive). E.g. ('tla+', 'tla')."""

    fence_label: str = ""
    """Markdown fence label the model is told to use for the spec, e.g. 'tla'."""

    config_fence_label: Optional[str] = None
    """Fence label for an optional config block (e.g. 'cfg' for TLA+). None if the language has no separate config artifact."""

    spec_extension: str = ""
    """File extension for the spec when written to disk, including the dot. E.g. '.tla'."""

    config_extension: Optional[str] = None
    """File extension for the config artifact, or None."""

    # ---- artifact extraction ----------------------------------------------

    def extract_artifacts(self, raw_text: str) -> SpecArtifacts:
        """
        Pull the spec (and optional config) out of a model response.

        Default implementation locates fenced blocks by `fence_label` /
        `config_fence_label`. Override if your language uses something else.
        """
        spec_pattern = re.compile(
            rf"```\s*{re.escape(self.fence_label)}\s*\n(.*?)\n```",
            re.DOTALL | re.IGNORECASE,
        )
        spec_match = spec_pattern.search(raw_text)
        spec = spec_match.group(1) if spec_match else raw_text

        config: Optional[str] = None
        if self.config_fence_label:
            cfg_pattern = re.compile(
                rf"```\s*{re.escape(self.config_fence_label)}\s*\n(.*?)\n```",
                re.DOTALL | re.IGNORECASE,
            )
            cfg_match = cfg_pattern.search(raw_text)
            config = cfg_match.group(1) if cfg_match else None

        return SpecArtifacts(spec=spec, config=config)

    # ---- correction prompt context ----------------------------------------

    def fence_format_hint(self) -> str:
        """
        Short instruction to inject into the correction prompt so the model
        re-emits the same fenced shape. Default covers the common case
        (one mandatory spec block + optional config block).
        """
        if self.config_fence_label:
            return (
                f"Output EXACTLY TWO fenced code blocks: ```{self.fence_label} ... ``` "
                f"and ```{self.config_fence_label} ... ```. Nothing outside the fences."
            )
        return (
            f"Output EXACTLY ONE fenced code block: ```{self.fence_label} ... ```. "
            f"Nothing outside the fence."
        )

    # ---- tool availability -------------------------------------------------

    def check_available(self) -> Optional[str]:
        """
        Return None if the backend's external tools are installed and ready,
        or a human-readable error message describing what's missing. Used by
        the registry/CLI for early-fail diagnostics.
        """
        return None

    # ---- Phase 1: syntax / static check -----------------------------------

    @abstractmethod
    def validate_syntax(
        self,
        spec: str,
        config: Optional[str],
        work_dir: Path,
        timeout: int,
        spec_filename: Optional[str] = None,
    ) -> SyntaxOutcome:
        """
        Parse / type-check the spec.

        If `spec_filename` is given, the backend should write the spec to
        `work_dir / spec_filename` before running the parser, so that any
        filename / module-name consistency check the parser does (e.g.
        TLA+'s SANY: MODULE name must match filename) is actually exercised.
        Otherwise the backend picks an arbitrary filename.
        """

    # ---- Phase 2: bounded model checking ----------------------------------

    @abstractmethod
    def run_model_checker(
        self,
        spec_path: Path,
        config_path: Optional[Path],
        work_dir: Path,
        timeout: int,
    ) -> ModelCheckOutcome:
        """Run the language's model checker on a spec already on disk."""

    def generate_default_config(
        self,
        spec: str,
        task_name: str,
        model_name: Optional[str],
    ) -> Tuple[bool, str, Optional[str]]:
        """
        Produce a fallback config when the model didn't emit one.

        Returns (success, config_text, error_message). Default: success with
        empty config — for languages where Phase 2 doesn't need a separate
        config file (Alloy, PAT). TLA+ overrides to drive ConfigGenerator.
        """
        return True, "", None

    # ---- Phase 4: expert-invariant verification ---------------------------

    def invariant_template_dirname(self) -> str:
        """
        Directory under data/ where this language's invariant templates live.
        Default: `invariant_templates` (TLA+ legacy). Override for Alloy/PAT
        which use `alloy_invariant_templates`, `pat_invariant_templates`.
        """
        return "invariant_templates"

    def invariant_example_field(self) -> str:
        """
        YAML key that holds the per-language reference snippet inside an
        invariant template (e.g. `tla_example`, `alloy_example`).
        """
        return f"{self.fence_label}_example"

    @abstractmethod
    def translate_invariants(
        self,
        templates: List[InvariantTemplate],
        spec: str,
        task_name: str,
        translator: str = "claude-code",
        agent_timeout: Optional[int] = None,
    ) -> Tuple[Dict[str, str], Optional[str]]:
        """
        Translate each template into the spec's vocabulary.

        `translator` selects how the translation is performed:
          - "claude-code" (default) — Claude Code agent
          - "codex" — Codex agent
          - "claude" — direct Claude API call
          - any other value is treated as a model name for a direct API call

        Backends that don't support a requested mode should return
        ({}, "translator '<name>' not supported by <Lang> backend").

        Returns ({invariant_name: translated_text}, error_message_or_None).
        On a wholesale translation failure return ({}, "...").
        """

    @abstractmethod
    def check_invariants(
        self,
        spec_path: Path,
        config_path: Optional[Path],
        templates: List[InvariantTemplate],
        translated: Dict[str, str],
        work_dir: Path,
        timeout: int,
    ) -> InvariantOutcome:
        """
        Check each translated invariant against the spec.

        `templates` carries the per-invariant type ("safety" / "liveness")
        which the backend may need to splice in correctly (e.g. TLA+ puts
        safety into INVARIANT and liveness into PROPERTY in the .cfg).
        `translated` maps invariant name → translated text. Both inputs
        are aligned by name; an entry in `templates` without a matching
        `translated` entry should be reported as a translation failure.
        """

    # ---- per-run lifecycle (optional) -------------------------------------

    def finalize_run(
        self,
        work_dir: Path,
        task_name: str,
        method_name: str,
        model_name: str,
    ) -> None:
        """
        Called once per evaluator.evaluate() invocation, after all phases.
        Backends may persist per-run reports here (e.g. TLA+ writes
        error_statistics.yaml). Default: no-op.
        """
        return None

    # ---- Phase 3: transition validation (optional) ------------------------

    supports_direct_transition_validation: bool = False
    """If True, `validate_transitions` is implemented and can be called directly. If False, callers fall back to the agent-driven path (TLA+ today)."""

    def validate_transitions(
        self,
        spec_path: Path,
        trace_windows,  # iterable of (action, pre_state, post_state)
        work_dir: Path,
        timeout: int,
    ) -> TransitionOutcome:
        """
        Direct-invocation Phase 3. Default raises; override and set
        `supports_direct_transition_validation = True` to opt in.
        """
        raise NotImplementedError(
            f"{self.name} does not implement direct transition validation; "
            "the agent-driven path (scripts/launch_tv_eval.sh) should be used."
        )
