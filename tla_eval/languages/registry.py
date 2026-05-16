"""
Registry mapping language names to LanguageBackend instances.

Lookup is case-insensitive and ignores '+' so 'TLA+', 'tla+', 'tla',
'TLA' all resolve to the same backend.
"""

import logging
from typing import Dict, List

from .base import LanguageBackend

logger = logging.getLogger(__name__)

_BACKENDS: Dict[str, LanguageBackend] = {}
_BOOTSTRAPPED = False


def _normalize(name: str) -> str:
    return name.lower().replace("+", "").strip()


def register(backend: LanguageBackend) -> None:
    """Register a backend under its canonical name and aliases."""
    keys = {_normalize(backend.name), *(_normalize(a) for a in backend.aliases)}
    for key in keys:
        if not key:
            continue
        existing = _BACKENDS.get(key)
        if existing is not None and existing is not backend:
            logger.warning(
                "Language backend key '%s' was registered to %s; overwriting with %s",
                key, type(existing).__name__, type(backend).__name__,
            )
        _BACKENDS[key] = backend
    logger.debug("Registered %s under keys: %s", backend.name, sorted(keys))


def _ensure_bootstrapped() -> None:
    global _BOOTSTRAPPED
    if _BOOTSTRAPPED:
        return
    _BOOTSTRAPPED = True  # set first to avoid recursion if a backend triggers another lookup
    from . import _bootstrap  # noqa: WPS433  (lazy import for circular-safety)
    _bootstrap()


def get(language: str) -> LanguageBackend:
    """
    Resolve a language name to its backend. Raises ValueError if unknown.
    """
    _ensure_bootstrapped()
    key = _normalize(language)
    backend = _BACKENDS.get(key)
    if backend is None:
        raise ValueError(
            f"No LanguageBackend registered for '{language}'. "
            f"Known: {sorted(set(b.name for b in _BACKENDS.values()))}"
        )
    return backend


def available_languages() -> List[str]:
    """Canonical names of all registered backends."""
    _ensure_bootstrapped()
    return sorted({b.name for b in _BACKENDS.values()})
