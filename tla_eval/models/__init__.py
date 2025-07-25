"""
Model adapters for different LLM providers.

This module provides adapters for various Large Language Model providers,
enabling uniform access to different APIs and local models for TLA+ generation.

Available adapters:
- OpenAIAdapter: OpenAI GPT models (GPT-4, GPT-3.5, etc.)
- AnthropicAdapter: Anthropic Claude models (Claude 3 family)

Usage:
    from tla_eval.models import get_model_adapter
    
    # Using predefined model configuration
    model = get_model_adapter("openai_gpt4")
    
    # Using custom configuration
    model = get_model_adapter("openai", model_name="gpt-4", temperature=0.2)
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

from .openai_adapter import OpenAIAdapter
from .anthropic_adapter import AnthropicAdapter
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
    "OpenAIAdapter",
    "AnthropicAdapter",
    
    # Factory and utilities
    "ModelFactory",
    "get_model_adapter",
]