"""
Specification Processing Module

LLM-based configuration generation and trace format conversion for TLA+ specs.
"""

from .spec_converter import SpecTraceGenerator, generate_config_from_tla
from .config_generation import generate_config_from_tla as config_gen

__all__ = [
    'SpecTraceGenerator',
    'generate_config_from_tla',
]