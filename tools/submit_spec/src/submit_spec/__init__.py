"""
submit_spec MCP Tool

This MCP tool allows code agents to submit TLA+ specifications for validation.
It returns phase 1 (syntax) and phase 2 (runtime) verification results.
"""

from .server import create_server
from .evaluator import SpecEvaluator

__all__ = ["create_server", "SpecEvaluator"]
