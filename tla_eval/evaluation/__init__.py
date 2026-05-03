"""
Evaluation modules for SysMoBench.

Phases:
- Phase 1 (Syntax): Compilation checking (SANY)
- Phase 2 (Semantics): Runtime correctness via TLC bounded model checking
- Phase 3 (Conformance): Transition validation against captured system traces
- Phase 4 (Invariant): Agent-translated invariants verified by TLC
"""

from .syntax.compilation_check import CompilationCheckEvaluator
from .syntax.action_decomposition import ActionDecompositionEvaluator
from .semantics.runtime_check import RuntimeCheckEvaluator
from .semantics.manual_invariant_evaluator import ManualInvariantEvaluator
from .semantics.transition_validation import TransitionValidationEvaluator

from .base.evaluator import BaseEvaluator
from .base.result_types import (
    EvaluationResult,
    SyntaxEvaluationResult,
    SemanticEvaluationResult,
    TransitionValidationResult,
)

__all__ = [
    "CompilationCheckEvaluator",
    "ActionDecompositionEvaluator",
    "RuntimeCheckEvaluator",
    "ManualInvariantEvaluator",
    "TransitionValidationEvaluator",
    "BaseEvaluator",
    "EvaluationResult",
    "SyntaxEvaluationResult",
    "SemanticEvaluationResult",
    "TransitionValidationResult",
]