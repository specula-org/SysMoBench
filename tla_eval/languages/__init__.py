"""
Language backends for SysMoBench.

A `LanguageBackend` encapsulates everything that varies between specification
languages (TLA+, Alloy, PAT, ...): the parser/static-checker, the model
checker, the invariant translator, the fence label used in model output, etc.

Phase evaluators (CompilationCheckEvaluator, RuntimeCheckEvaluator,
ManualInvariantEvaluator, TransitionValidationEvaluator) are language-neutral
and dispatch through a backend. Adding a new language means writing one
backend subclass and registering it.
"""

from .base import LanguageBackend, SpecArtifacts
from .registry import register, get, available_languages
from .result_types import (
    SyntaxOutcome,
    ModelCheckOutcome,
    InvariantOutcome,
    InvariantCaseResult,
    TransitionOutcome,
)

__all__ = [
    "LanguageBackend",
    "SpecArtifacts",
    "register",
    "get",
    "available_languages",
    "SyntaxOutcome",
    "ModelCheckOutcome",
    "InvariantOutcome",
    "InvariantCaseResult",
    "TransitionOutcome",
]


def _bootstrap():
    """Register built-in backends. Called lazily on first registry lookup."""
    from . import tla_plus  # noqa: F401  (registers on import)
    try:
        from . import alloy  # noqa: F401
    except ImportError:
        pass
    try:
        from . import pat  # noqa: F401
    except ImportError:
        pass
