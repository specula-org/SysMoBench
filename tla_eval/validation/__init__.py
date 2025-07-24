"""
TLA+ validation utilities.

This package contains tools for validating TLA+ specifications,
including compilation checking and syntax validation.
"""

from .validator import TLAValidator, ValidationResult, validate_tla_specification, validate_tla_file

__all__ = [
    "TLAValidator",
    "ValidationResult", 
    "validate_tla_specification",
    "validate_tla_file"
]