"""
TLA+ Evaluation Framework

A comprehensive framework for evaluating Large Language Models' capabilities 
in generating industrial-grade TLA+ specifications from real-world distributed 
and concurrent systems.
"""

__version__ = "0.1.0"
__author__ = "TLA+ Benchmark Team"

from .models import ModelAdapter
from .config import get_configured_model
from .core.verification.validators import *
from .evaluation import Phase1Evaluator, Phase1EvaluationResult, Phase2Evaluator, Phase3Evaluator

__all__ = [
    "ModelAdapter", 
    "get_configured_model",
    "Phase1Evaluator",
    "Phase1EvaluationResult",
    "Phase2Evaluator", 
    "Phase3Evaluator"
]