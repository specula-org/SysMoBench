"""
Intermediate result types produced by LanguageBackend methods.

These are deliberately small and language-agnostic. The phase evaluators
translate them into the richer SyntaxEvaluationResult / SemanticEvaluationResult
types defined in tla_eval/evaluation/base/result_types.py.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class SyntaxOutcome:
    """Outcome of a Phase 1 (syntax / static check) run on a single spec."""

    success: bool
    syntax_errors: List[str] = field(default_factory=list)
    semantic_errors: List[str] = field(default_factory=list)
    raw_output: str = ""
    elapsed_seconds: float = 0.0
    error_message: Optional[str] = None


@dataclass
class ModelCheckOutcome:
    """Outcome of a Phase 2 (bounded model checking) run on a single spec."""

    success: bool
    raw_output: str = ""
    elapsed_seconds: float = 0.0
    error_message: Optional[str] = None
    classification: Optional[str] = None  # e.g. "violation", "deadlock", "timeout", "parse_error"


@dataclass
class InvariantCaseResult:
    """Per-invariant outcome inside a Phase 4 run."""

    name: str
    success: bool
    translated: str = ""
    raw_output: str = ""
    elapsed_seconds: float = 0.0
    error_message: Optional[str] = None
    metadata: Dict[str, object] = field(default_factory=dict)
    """Backend-specific extras (e.g. TLA+ puts `states_explored` here)."""


@dataclass
class InvariantOutcome:
    """Aggregated Phase 4 result across all invariants for one spec."""

    cases: List[InvariantCaseResult] = field(default_factory=list)
    translation_error: Optional[str] = None

    @property
    def passed(self) -> int:
        return sum(1 for c in self.cases if c.success)

    @property
    def total(self) -> int:
        return len(self.cases)


@dataclass
class TransitionOutcome:
    """Outcome of a Phase 3 (transition validation) direct-invocation run."""

    per_action_pass_rates: Dict[str, float] = field(default_factory=dict)
    total_passed: int = 0
    total_windows: int = 0
    error_message: Optional[str] = None
