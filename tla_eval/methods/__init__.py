"""
TLA+ generation methods.

This module contains different approaches for generating TLA+ specifications
from source code, including direct LLM calls and agent-based methods.
"""

from .base import TLAGenerationMethod
from .direct_call import DirectCallMethod

__all__ = [
    "TLAGenerationMethod",
    "DirectCallMethod",
]