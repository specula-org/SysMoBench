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
    """
    Auto-discover and import every backend module in this package.

    Each `tla_eval/languages/<name>.py` other than the framework files
    (base / registry / result_types / __init__) is imported, which gives
    each backend module a chance to `register()` itself. ImportError is
    NOT swallowed — a broken backend file should surface immediately.
    """
    import importlib
    import pkgutil

    framework_modules = {"base", "registry", "result_types"}
    for mod_info in pkgutil.iter_modules(__path__):
        if mod_info.ispkg or mod_info.name.startswith("_"):
            continue
        if mod_info.name in framework_modules:
            continue
        importlib.import_module(f"{__name__}.{mod_info.name}")
