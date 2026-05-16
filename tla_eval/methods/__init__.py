"""
TLA+ generation methods.
"""

from .base import TLAGenerationMethod
from .direct_call import DirectCallMethod

_METHODS = {
    "direct_call": DirectCallMethod,
}


def get_method(method_name: str, **kwargs) -> TLAGenerationMethod:
    """Get a method instance by name. kwargs are forwarded to the constructor."""
    if method_name not in _METHODS:
        raise ValueError(f"Unknown method '{method_name}'. Available: {list(_METHODS.keys())}")
    return _METHODS[method_name](**kwargs)


def list_available_methods() -> list:
    """List all available method names."""
    return list(_METHODS.keys())


__all__ = [
    "TLAGenerationMethod",
    "DirectCallMethod",
    "get_method",
    "list_available_methods",
]
