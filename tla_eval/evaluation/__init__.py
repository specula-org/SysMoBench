"""
Evaluation modules for TLA+ benchmark framework.

Phases:
- Phase 1 (Syntax): Compilation checking (SANY)
- Phase 2 (Semantics): Runtime correctness via TLC bounded model checking
- Phase 3 (Conformance): Window verification — see scripts/launch_tv_eval.sh
- Phase 4 (Invariant): Agent-translated invariants verified by TLC
"""

from .syntax.compilation_check import CompilationCheckEvaluator
from .syntax.action_decomposition import ActionDecompositionEvaluator
from .semantics.runtime_check import RuntimeCheckEvaluator
from .semantics.manual_invariant_evaluator import ManualInvariantEvaluator

from .base.evaluator import BaseEvaluator
from .base.result_types import (
    EvaluationResult,
    SyntaxEvaluationResult,
    SemanticEvaluationResult,
)

__all__ = [
    "CompilationCheckEvaluator",
    "ActionDecompositionEvaluator",
    "RuntimeCheckEvaluator",
    "ManualInvariantEvaluator",
    "BaseEvaluator",
    "EvaluationResult",
    "SyntaxEvaluationResult",
    "SemanticEvaluationResult",
]