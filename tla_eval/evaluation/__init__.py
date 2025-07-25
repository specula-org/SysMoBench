"""
Evaluation modules for TLA+ benchmark framework.

This package contains different phases of evaluation:
- Phase 1: Compilation checking (can the generated TLA+ be compiled?)
- Phase 2: Runtime checking (can the specification be model-checked?)
- Phase 3: Consistency checking (does the specification match the source code behavior?)
"""

from .phase1 import Phase1Evaluator, Phase1EvaluationResult, create_phase1_evaluator
from .phase2 import Phase2Evaluator, Phase2EvaluationResult

__all__ = [
    "Phase1Evaluator",
    "Phase1EvaluationResult", 
    "create_phase1_evaluator",
    "Phase2Evaluator",
    "Phase2EvaluationResult"
]