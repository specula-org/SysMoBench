"""
Semantic-level evaluation module for TLA+ specifications.

This module contains evaluators for checking the semantic correctness
and model checking capabilities of generated TLA+ specifications.
"""

from .invariant_verification import InvariantVerificationEvaluator

__all__ = ['InvariantVerificationEvaluator']