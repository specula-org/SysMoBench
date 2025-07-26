"""
OpenAI API adapter for TLA+ specification generation.

This module provides integration with OpenAI's GPT models through their API.
Supports GPT-4, GPT-4 Turbo, GPT-3.5 Turbo, and other chat completion models.
"""

import os
import time
from typing import Dict, Any, Optional
import logging

try:
    import openai
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

from .base import (
    ModelAdapter, 
    GenerationConfig, 
    GenerationResult,
    ModelError,
    ModelUnavailableError,
    GenerationError,
    RateLimitError
)

logger = logging.getLogger(__name__)


class OpenAIAdapter(ModelAdapter):
    """
    Adapter for OpenAI GPT models via API.
    
    Supports all OpenAI chat completion models including GPT-4, GPT-4 Turbo,
    and GPT-3.5 Turbo.
    
    Configuration parameters:
        - api_key: OpenAI API key (or set OPENAI_API_KEY env var)
        - model_name: Model identifier (e.g., "gpt-4", "gpt-4-turbo")
        - organization: OpenAI organization ID (optional)
        - base_url: Custom API base URL (optional, for Azure OpenAI)
        - max_retries: Maximum number of retry attempts (default: 3)
        - timeout: Request timeout in seconds (default: 60)
    """
    
    # Default model configurations
    MODEL_CONFIGS = {
        "gpt-4": {"max_tokens": 4096, "context_length": 8192},
        "gpt-4-turbo": {"max_tokens": 4096, "context_length": 128000},
        "gpt-4-turbo-preview": {"max_tokens": 4096, "context_length": 128000},
        "gpt-3.5-turbo": {"max_tokens": 4096, "context_length": 16385},
        "gpt-3.5-turbo-16k": {"max_tokens": 4096, "context_length": 16385},
    }
    
    def _setup_model(self):
        """Initialize OpenAI client and validate configuration."""
        if not OPENAI_AVAILABLE:
            raise ModelUnavailableError(
                "OpenAI library not installed. Run: pip install openai"
            )
        
        # Get API key from config or environment
        api_key = self.config.get("api_key") or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ModelUnavailableError(
                "OpenAI API key not found. Set OPENAI_API_KEY environment variable "
                "or pass api_key in configuration."
            )
        
        # Initialize OpenAI client
        client_config = {
            "api_key": api_key,
            "max_retries": self.config.get("max_retries", 3),
            "timeout": self.config.get("timeout", 300),  # Increased to 5 minutes for large source files
        }
        
        # Optional organization and base URL (for custom APIs)
        if self.config.get("organization"):
            client_config["organization"] = self.config["organization"]
        if self.config.get("base_url") or self.config.get("url"):
            client_config["base_url"] = self.config.get("base_url") or self.config.get("url")
            
        self.client = OpenAI(**client_config)
        
        # Validate model name
        if self.model_name not in self.MODEL_CONFIGS:
            logger.warning(f"Unknown model {self.model_name}, using default settings")
    
    def generate_tla_specification(
        self, 
        source_code: str, 
        prompt_template: str,
        generation_config: Optional[GenerationConfig] = None
    ) -> GenerationResult:
        """
        Generate TLA+ specification using OpenAI API.
        
        Args:
            source_code: Source code to convert to TLA+
            prompt_template: Prompt template with {source_code} placeholder
            generation_config: Generation parameters
            
        Returns:
            GenerationResult with generated TLA+ specification
            
        Raises:
            GenerationError: If API call fails
            RateLimitError: If rate limit is exceeded
        """
        if not self.is_available():
            raise ModelUnavailableError("OpenAI adapter is not properly configured")
        
        # Use default config if not provided
        if generation_config is None:
            generation_config = GenerationConfig()
        
        # Format prompt - check if prompt_template already contains the content
        if "{source_code}" in prompt_template:
            prompt = prompt_template.format(source_code=source_code)
        else:
            # Prompt is already formatted, use as-is
            prompt = prompt_template
        
        # Prepare API call parameters
        api_params = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "user", 
                    "content": prompt
                }
            ],
            "max_tokens": generation_config.max_tokens,
            "temperature": generation_config.temperature,
            "top_p": generation_config.top_p,
        }
        
        # Add optional parameters
        if generation_config.stop_sequences:
            api_params["stop"] = generation_config.stop_sequences
        if generation_config.seed is not None:
            api_params["seed"] = generation_config.seed
        
        # Enable streaming for better responsiveness with large requests
        api_params["stream"] = False  # TODO: Implement streaming support
        
        # Make API call with error handling
        start_time = time.time()
        try:
            response = self.client.chat.completions.create(**api_params)
            end_time = time.time()
            
            # Extract generated text
            generated_text = response.choices[0].message.content
            if not generated_text:
                raise GenerationError("Empty response from OpenAI API")
            
            # Prepare metadata
            metadata = {
                "model": self.model_name,
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                },
                "latency_seconds": end_time - start_time,
                "finish_reason": response.choices[0].finish_reason,
                "response_id": response.id,
            }
            
            return GenerationResult(
                generated_text=generated_text,
                metadata=metadata,
                timestamp=end_time,
                success=True
            )
            
        except openai.RateLimitError as e:
            raise RateLimitError(f"OpenAI rate limit exceeded: {e}")
        except openai.APIError as e:
            raise GenerationError(f"OpenAI API error: {e}")
        except Exception as e:
            raise GenerationError(f"Unexpected error during generation: {e}")
    
    def is_available(self) -> bool:
        """
        Check if OpenAI adapter is available and properly configured.
        
        Returns:
            True if adapter can be used, False otherwise
        """
        try:
            # Check if OpenAI library is available
            if not OPENAI_AVAILABLE:
                return False
            
            # Check if client is initialized
            if not hasattr(self, 'client'):
                return False
            
            # Check if API key is set
            api_key = self.config.get("api_key") or os.getenv("OPENAI_API_KEY")
            if not api_key:
                return False
            
            return True
            
        except Exception:
            return False
    
    def validate_config(self) -> list[str]:
        """
        Validate OpenAI adapter configuration.
        
        Returns:
            List of validation error messages
        """
        errors = super().validate_config()
        
        # Check OpenAI library
        if not OPENAI_AVAILABLE:
            errors.append("OpenAI library not installed")
        
        # Check API key
        api_key = self.config.get("api_key") or os.getenv("OPENAI_API_KEY")
        if not api_key:
            errors.append("OpenAI API key not found")
        
        # For custom APIs, we don't validate model name format
        if not self.config.get("url") and not self.config.get("base_url"):
            # Only validate OpenAI model names for official OpenAI API
            if self.model_name and not self.model_name.startswith(("gpt-", "text-")):
                errors.append(f"Invalid OpenAI model name: {self.model_name}")
        
        return errors
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get detailed information about the OpenAI model."""
        info = super().get_model_info()
        
        # Add OpenAI-specific information
        model_config = self.MODEL_CONFIGS.get(self.model_name, {})
        info.update({
            "provider": "openai",
            "model_type": "chat_completion",
            "max_tokens": model_config.get("max_tokens", "unknown"),
            "context_length": model_config.get("context_length", "unknown"),
            "api_version": getattr(openai, "__version__", "unknown") if OPENAI_AVAILABLE else "not_installed",
        })
        
        return info