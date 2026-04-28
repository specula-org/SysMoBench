"""
Model adapters for different LLM providers.

All hosted providers route through LiteLLM (`LiteLLMAdapter`).
`ExistSpecAdapter` is a special "model" used when scoring an existing spec
file without invoking an LLM.

Usage:
    from tla_eval.models import get_model_adapter

    model = get_model_adapter(
        "litellm",
        model_name="anthropic/claude-sonnet-4-20250514",
        temperature=0.2,
    )
    result = model.generate_tla_specification(source_code, prompt_template)
"""

from .base import (
    ModelAdapter,
    GenerationConfig,
    GenerationResult,
    ModelError,
    ModelUnavailableError,
    GenerationError,
    RateLimitError,
)
from .litellm_adapter import LiteLLMAdapter
from .factory import ModelFactory, get_model_adapter

__all__ = [
    "ModelAdapter",
    "GenerationConfig",
    "GenerationResult",
    "ModelError",
    "ModelUnavailableError",
    "GenerationError",
    "RateLimitError",
    "LiteLLMAdapter",
    "ModelFactory",
    "get_model_adapter",
]
