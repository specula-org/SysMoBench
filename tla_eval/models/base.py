"""
Base adapter interface for LLM models.

This module defines the abstract base class that all model adapters must implement.
It ensures compatibility between different model types (API-based, local, etc.).
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass
import time
import logging


@dataclass
class GenerationConfig:
    """Configuration for text generation parameters."""
    max_tokens: int = 32000
    temperature: float = 0.1
    top_p: float = 0.9
    stop_sequences: Optional[List[str]] = None
    seed: Optional[int] = None
    use_json_mode: bool = False


@dataclass
class GenerationResult:
    """Result of text generation from a model."""
    generated_text: str
    metadata: Dict[str, Any]  # Model-specific metadata (tokens used, latency, etc.)
    timestamp: float
    success: bool = True
    error_message: Optional[str] = None

    def __post_init__(self):
        if self.timestamp == 0:
            self.timestamp = time.time()


logger = logging.getLogger(__name__)


class ModelAdapter(ABC):
    """
    Abstract base class for all model adapters.
    
    This interface ensures that all model types (API-based, local, etc.) 
    can be used interchangeably in the benchmark framework.
    """
    
    def __init__(self, model_name: str, **kwargs):
        """
        Initialize the model adapter.
        
        Args:
            model_name: Name/identifier of the model
            **kwargs: Model-specific configuration parameters
        """
        self.model_name = model_name
        self.config = kwargs
        self._setup_model()
    
    def _retry_on_service_unavailable(self, func: Callable, *args, **kwargs):
        """
        One-shot call. Any exception propagates immediately.

        Previously this wrapped the adapter call in a 4-attempt retry loop
        (30 s between attempts) that fired on 500 / 503 / 502 / 504 /
        "rate limit" / "quota" error substrings. That hidden loop billed the
        user 4× per apparent call and stacked with direct_call's own retry
        loop (up to 12 billable requests per logical "attempt"). Observed
        2026-04-17 on MiniMax etcd: 431s of Phase 0 with "Generation failed"
        was 4 billable API errors with 30 s sleeps in between, invisible
        because subprocess stdout was not persisted.

        The user's rule: NO SILENT RETRY. If a call fails, fail fast, surface
        the error to the caller, let the caller decide.
        """
        return func(*args, **kwargs)
    
    @abstractmethod
    def _setup_model(self):
        """Setup model-specific initialization. Called during __init__."""
        pass
    
    @abstractmethod
    def _generate_tla_specification_impl(
        self, 
        source_code: str, 
        prompt_template: str,
        generation_config: Optional[GenerationConfig] = None
    ) -> GenerationResult:
        """
        Internal implementation of TLA+ specification generation.
        
        This method should be implemented by each adapter and contains
        the actual API call logic.
        
        Args:
            source_code: The source code to convert to TLA+
            prompt_template: Template for formatting the prompt
            generation_config: Generation parameters
            
        Returns:
            GenerationResult containing the generated TLA+ specification
            
        Raises:
            ModelError: If generation fails
        """
        pass
    
    def generate_tla_specification(
        self, 
        source_code: str, 
        prompt_template: str,
        generation_config: Optional[GenerationConfig] = None
    ) -> GenerationResult:
        """
        Generate TLA+ specification from source code with automatic retry on service unavailable.
        
        This method wraps the internal implementation with retry logic for handling
        503 Service Unavailable errors and similar temporary issues.
        
        Args:
            source_code: The source code to convert to TLA+
            prompt_template: Template for formatting the prompt
            generation_config: Generation parameters
            
        Returns:
            GenerationResult containing the generated TLA+ specification
            
        Raises:
            ModelError: If generation fails after all retries
        """
        return self._retry_on_service_unavailable(
            self._generate_tla_specification_impl,
            source_code,
            prompt_template,
            generation_config
        )
    
    def generate_direct(
        self, 
        complete_prompt: str,
        generation_config: Optional[GenerationConfig] = None
    ) -> GenerationResult:
        """
        Generate content using a complete, pre-formatted prompt.
        
        This method is for cases where the prompt has already been fully formatted
        (e.g., using Template.substitute()) and doesn't need further processing.
        
        Args:
            complete_prompt: Complete, ready-to-use prompt text
            generation_config: Generation parameters
            
        Returns:
            GenerationResult containing the generated content
            
        Raises:
            ModelError: If generation fails after all retries
        """
        return self._retry_on_service_unavailable(
            self._generate_direct_impl,
            complete_prompt,
            generation_config
        )
    
    @abstractmethod
    def _generate_direct_impl(
        self, 
        complete_prompt: str,
        generation_config: Optional[GenerationConfig] = None
    ) -> GenerationResult:
        """
        Internal implementation of direct generation from complete prompt.
        
        This method should be implemented by each adapter and contains
        the actual API call logic for pre-formatted prompts.
        
        Args:
            complete_prompt: Complete, ready-to-use prompt text
            generation_config: Generation parameters
            
        Returns:
            GenerationResult containing the generated content
        """
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """
        Check if the model is available and properly configured.
        
        Returns:
            True if model can be used, False otherwise
        """
        pass
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the model.
        
        Returns:
            Dictionary containing model metadata
        """
        safe_config = self.config.copy()
        if "api_key" in safe_config:
            safe_config["api_key"] = "***redacted***"

        return {
            "model_name": self.model_name,
            "adapter_type": self.__class__.__name__,
            "config": safe_config,
        }

    def validate_config(self) -> List[str]:
        """
        Validate model configuration.
        
        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []
        if not self.model_name:
            errors.append("model_name is required")
        return errors


class ModelError(Exception):
    """Custom exception for model-related errors."""
    pass


class ModelUnavailableError(ModelError):
    """Raised when a model is not available or not properly configured."""
    pass


class GenerationError(ModelError):
    """Raised when text generation fails."""
    pass


class RateLimitError(ModelError):
    """Raised when API rate limit is exceeded."""
    pass
