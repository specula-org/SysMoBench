"""
TLA+ generation methods.

This module contains different approaches for generating TLA+ specifications
from source code.
"""

from .base import TLAGenerationMethod
from .direct_call import DirectCallMethod
from .agent_based import AgentBasedMethod

_METHODS = {
    "direct_call": DirectCallMethod,
    "agent_based": AgentBasedMethod,
}


def get_method(method_name: str, **kwargs) -> TLAGenerationMethod:
    """Get a method instance by name."""
    if method_name not in _METHODS:
        raise ValueError(f"Unknown method '{method_name}'. Available: {list(_METHODS.keys())}")
    return _METHODS[method_name]()


def list_available_methods() -> list:
    """List all available method names."""
    return list(_METHODS.keys())


__all__ = [
    "TLAGenerationMethod",
    "DirectCallMethod",
    "AgentBasedMethod",
    "get_method",
    "list_available_methods",
]
