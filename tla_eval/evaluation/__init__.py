"""
Evaluation modules for TLA+ benchmark framework.

This package contains different dimensions of evaluation:
- Syntax: Compilation checking (can the generated TLA+ be compiled?)
- Semantics: Model checking (can the specification be model-checked?)
- Consistency: Trace validation (does the specification match the system behavior?)
"""

# New structured evaluators
from .syntax.compilation_check import CompilationCheckEvaluator
from .semantics.invariant_verification import InvariantVerificationEvaluator
from .consistency.trace_validation import TraceValidationEvaluator

# Base classes and result types
from .base.evaluator import BaseEvaluator
from .base.result_types import (
    EvaluationResult, 
    SyntaxEvaluationResult, 
    SemanticEvaluationResult, 
    ConsistencyEvaluationResult
)

# Backward compatibility aliases (deprecated)
# Note: Legacy Phase classes are deprecated, use new structured evaluators instead

__all__ = [
    # New structured evaluators
    "CompilationCheckEvaluator",
    "InvariantVerificationEvaluator", 
    "TraceValidationEvaluator",
    
    # Base classes and result types
    "BaseEvaluator",
    "EvaluationResult",
    "SyntaxEvaluationResult",
    "SemanticEvaluationResult", 
    "ConsistencyEvaluationResult",
    
    # Legacy compatibility removed - use new structured evaluators
]