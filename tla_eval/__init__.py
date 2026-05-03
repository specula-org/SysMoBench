"""
SysMoBench: Evaluating AI on formally modeling complex real-world systems.
"""

__version__ = "1.0.0"
__author__ = "SysMoBench Authors"

from .models import ModelAdapter
from .config import get_configured_model
from .core.verification.validators import *
from .evaluation import (
    CompilationCheckEvaluator,
    RuntimeCheckEvaluator,
    ManualInvariantEvaluator,
    TransitionValidationEvaluator,
)

__all__ = [
    "ModelAdapter",
    "get_configured_model",
    "CompilationCheckEvaluator",
    "RuntimeCheckEvaluator",
    "ManualInvariantEvaluator",
    "TransitionValidationEvaluator",
]