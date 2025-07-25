"""
Base adapter interface for LLM models.

This module defines the abstract base class that all model adapters must implement.
It ensures compatibility between different model types (API-based, local, etc.).
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import time


@dataclass
class GenerationConfig:
    """Configuration for text generation parameters."""
    max_tokens: int = 4096
    temperature: float = 0.1
    top_p: float = 1.0
    stop_sequences: Optional[List[str]] = None
    seed: Optional[int] = None


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
    
    @abstractmethod
    def _setup_model(self):
        """Setup model-specific initialization. Called during __init__."""
        pass
    
    @abstractmethod
    def generate_tla_specification(
        self, 
        source_code: str, 
        prompt_template: str,
        generation_config: Optional[GenerationConfig] = None
    ) -> GenerationResult:
        """
        Generate TLA+ specification from source code.
        
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
        return {
            "model_name": self.model_name,
            "adapter_type": self.__class__.__name__,
            "config": self.config
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