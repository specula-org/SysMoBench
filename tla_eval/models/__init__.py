"""
Model adapters for different LLM providers.

This module provides adapters for various Large Language Model providers,
enabling uniform access to different APIs and local models for TLA+ generation.

Available adapters:
- LiteLLMAdapter: Unified adapter for most hosted chat models
- OpenAIAdapter: Legacy native OpenAI adapter
- AnthropicAdapter: Legacy native Anthropic adapter
- GenAIAdapter: Legacy native Google GenAI adapter

Features:
- Automatic retry on 503 Service Unavailable errors (up to 3 retries with 30s delay)
- Uniform interface across all providers
- Comprehensive error handling and logging

Usage:
    from tla_eval.models import get_model_adapter
    
    # Using predefined model configuration
    model = get_model_adapter("openai_gpt4")
    
    # Using recommended LiteLLM configuration
    model = get_model_adapter(
        "litellm",
        model_name="anthropic/claude-sonnet-4-20250514",
        temperature=0.2,
    )
    
    # All adapters automatically retry on service unavailable errors
    result = model.generate_tla_specification(source_code, prompt_template)
"""

from .base import (
    ModelAdapter,
    GenerationConfig, 
    GenerationResult,
    ModelError,
    ModelUnavailableError,
    GenerationError,
    RateLimitError
)

from .litellm_adapter import LiteLLMAdapter
from .openai_adapter import OpenAIAdapter
from .anthropic_adapter import AnthropicAdapter
from .genai_adapter import GenAIAdapter
from .factory import ModelFactory, get_model_adapter

__all__ = [
    # Base classes and types
    "ModelAdapter",
    "GenerationConfig", 
    "GenerationResult",
    
    # Exceptions
    "ModelError",
    "ModelUnavailableError", 
    "GenerationError",
    "RateLimitError",
    
    # Adapter implementations
    "LiteLLMAdapter",
    "OpenAIAdapter",
    "AnthropicAdapter", 
    "GenAIAdapter",
    
    # Factory and utilities
    "ModelFactory",
    "get_model_adapter",
]
