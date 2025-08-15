"""
Google GenAI API adapter for TLA+ specification generation.

This module provides integration with Google's GenAI models through their API.
Supports Gemini models and other Google AI models.
"""

import os
import time
from typing import Dict, Any, Optional
import logging

try:
    from google import genai
    from google.genai import types
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False

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


class GenAIAdapter(ModelAdapter):
    """
    Adapter for Google GenAI models via API.
    
    Supports Gemini models and other Google AI chat completion models.
    
    Configuration parameters:
        - api_key: Google GenAI API key (or set GOOGLE_AI_API_KEY env var)
        - model_name: Model identifier (e.g., "gemini-2.5-flash", "gemini-pro")
        - max_retries: Maximum number of retry attempts (default: 3)
        - timeout: Request timeout in seconds (default: 300)
        - thinking_budget: Thinking budget for models that support it (default: 0 to disable)
    """
    
    # Default model configurations
    MODEL_CONFIGS = {
        "gemini-2.5-flash": {"max_tokens": 64000, "context_length": 1000000},
        "gemini-2.5-pro": {"max_tokens": 64000, "context_length": 1000000},
        "gemini-1.5-pro": {"max_tokens": 64000, "context_length": 2000000},
        "gemini-1.5-flash": {"max_tokens": 64000, "context_length": 1000000},  
        "gemini-pro": {"max_tokens": 64000, "context_length": 30720},
        "gemini-pro-vision": {"max_tokens": 64000, "context_length": 30720},
    }
    
    def _setup_model(self):
        """Initialize Google GenAI client and validate configuration."""
        if not GENAI_AVAILABLE:
            raise ModelUnavailableError(
                "Google GenAI library not installed. Run: pip install google-genai"
            )
        
        # Get API key from config or environment
        api_key = self.config.get("api_key") or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ModelUnavailableError(
                "Gemini API key not found. Set GEMINI_API_KEY environment variable "
                "or pass api_key in configuration."
            )
        
        # Initialize GenAI client
        try:
            self.client = genai.Client(api_key=api_key)
        except Exception as e:
            raise ModelUnavailableError(f"Failed to initialize GenAI client: {e}")
        
        # Validate model name
        if self.model_name not in self.MODEL_CONFIGS:
            logger.warning(f"Unknown GenAI model {self.model_name}, using default settings")
    
    def _generate_tla_specification_impl(
        self, 
        source_code: str, 
        prompt_template: str,
        generation_config: Optional[GenerationConfig] = None
    ) -> GenerationResult:
        """
        Generate TLA+ specification using Google GenAI API with retry for empty responses.
        
        This method wraps the actual implementation with retry logic to handle
        Gemini's occasional empty responses (STOP with no content).
        """
        return self._retry_on_empty_response(
            self._generate_tla_specification_core,
            source_code,
            prompt_template,
            generation_config
        )
    
    def _retry_on_empty_response(self, func, *args, **kwargs):
        """
        Retry wrapper specifically for handling Gemini's empty response issues.
        
        Gemini sometimes returns STOP with empty content due to temporary issues
        or content filtering. This wrapper retries up to 3 times with delays.
        """
        max_retries = 3
        retry_delay = 10  # 10 seconds between retries for empty responses
        
        for attempt in range(max_retries + 1):
            try:
                return func(*args, **kwargs)
                
            except GenerationError as e:
                error_str = str(e).lower()
                
                # Check if this is a retryable empty response error
                is_empty_response = (
                    "empty text response" in error_str and 
                    ("finish_reason: stop" in error_str or "stop" in error_str)
                )
                
                if is_empty_response and attempt < max_retries:
                    logger.warning(f"Empty response from Gemini (attempt {attempt + 1}/{max_retries + 1}). Retrying in {retry_delay}s...")
                    logger.debug(f"Error details: {e}")
                    time.sleep(retry_delay)
                    continue
                else:
                    # Non-retryable error or max retries reached
                    if attempt >= max_retries and is_empty_response:
                        logger.error(f"All {max_retries + 1} attempts failed with empty responses. This may indicate persistent content filtering or API issues.")
                    raise
                    
            except Exception as e:
                # For any other exception, don't retry
                raise
        
        # Should not reach here, but just in case
        raise GenerationError("Unexpected error in retry logic")

    def _generate_tla_specification_core(
        self, 
        source_code: str, 
        prompt_template: str,
        generation_config: Optional[GenerationConfig] = None
    ) -> GenerationResult:
        """
        Core TLA+ specification generation using Google GenAI API.
        
        This is the actual implementation that makes the API call.
        It's wrapped by _generate_tla_specification_impl with retry logic.
        
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
            raise ModelUnavailableError("GenAI adapter is not properly configured")
        
        # Use default config if not provided
        if generation_config is None:
            generation_config = GenerationConfig()
        
        # Format prompt - check if prompt_template already contains the content
        if "{source_code}" in prompt_template:
            prompt_content = prompt_template.format(source_code=source_code)
        else:
            # Prompt is already formatted, use as-is
            prompt_content = prompt_template
        
        # Prepare generation config
        # Use model's configured values if available, otherwise use GenerationConfig value
        model_max_tokens = self.config.get("max_tokens", generation_config.max_tokens)
        model_temperature = self.config.get("temperature", generation_config.temperature)
        model_top_p = self.config.get("top_p", generation_config.top_p)
        
        config_params = {
            "temperature": model_temperature,
            "top_p": model_top_p,
            "max_output_tokens": model_max_tokens,
        }
        
        # Add thinking config
        thinking_budget = self.config.get("thinking_budget", 0)
        if thinking_budget > 0:
            config_params["thinking_config"] = types.ThinkingConfig(
                thinking_budget=thinking_budget
            )
        else:
            # Explicitly disable thinking
            config_params["thinking_config"] = types.ThinkingConfig(
                thinking_budget=0
            )
        
        # Add stop sequences if provided
        if generation_config.stop_sequences:
            config_params["stop_sequences"] = generation_config.stop_sequences
            
        genai_config = types.GenerateContentConfig(**config_params)
        
        # Make API call
        start_time = time.time()
        try:
            logger.info(f"Starting generation with model {self.model_name}...")
            
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt_content,
                config=genai_config
            )
            
            end_time = time.time()
            
            # Extract text content with comprehensive error handling
            generated_text = ""
            try:
                # First, log the response structure for debugging
                logger.debug(f"Response type: {type(response)}")
                logger.debug(f"Response has text attr: {hasattr(response, 'text')}")
                
                if hasattr(response, 'text'):
                    try:
                        generated_text = response.text
                        logger.debug(f"Generated text length: {len(generated_text) if generated_text else 0}")
                    except Exception as text_error:
                        logger.error(f"Error accessing response.text: {text_error}")
                        raise GenerationError(f"Failed to access response.text: {text_error}")
                else:
                    logger.error(f"Response missing text attribute")
                    raise GenerationError("Response object missing text attribute")
                
                if not generated_text:
                    # Check for specific reasons why response might be empty
                    error_details = []
                    
                    if hasattr(response, 'candidates') and response.candidates:
                        candidate = response.candidates[0]
                        if hasattr(candidate, 'finish_reason'):
                            error_details.append(f"finish_reason: {candidate.finish_reason}")
                        if hasattr(candidate, 'safety_ratings') and candidate.safety_ratings:
                            safety_info = []
                            for rating in candidate.safety_ratings:
                                safety_info.append(f"{rating.category}:{rating.probability}")
                            error_details.append(f"safety_ratings: [{', '.join(safety_info)}]")
                        # Check for content filtering
                        if hasattr(candidate, 'content') and candidate.content:
                            if hasattr(candidate.content, 'parts'):
                                error_details.append(f"content_parts_count: {len(candidate.content.parts) if candidate.content.parts else 0}")
                    
                    if hasattr(response, 'prompt_feedback') and response.prompt_feedback:
                        if hasattr(response.prompt_feedback, 'block_reason'):
                            error_details.append(f"block_reason: {response.prompt_feedback.block_reason}")
                        if hasattr(response.prompt_feedback, 'safety_ratings'):
                            error_details.append(f"prompt_safety_ratings: {response.prompt_feedback.safety_ratings}")
                    
                    # Log the full response structure for debugging
                    logger.error(f"Full response structure: {response}")
                    
                    error_msg = "Empty text response from GenAI API"
                    if error_details:
                        error_msg += f" ({'; '.join(error_details)})"
                    else:
                        error_msg += " (possible content filtering or model limitations)"
                    
                    # For STOP with empty response, this might be recoverable with retry
                    if any("finish_reason: STOP" in detail for detail in error_details):
                        logger.warning("Got STOP with empty response - this might be due to content filtering or temporary API issues")
                    
                    raise GenerationError(error_msg)
                    
            except GenerationError:
                raise  # Re-raise GenerationError as-is
            except Exception as e:
                logger.error(f"Unexpected error extracting text: {e}")
                logger.error(f"Error type: {type(e).__name__}")
                import traceback
                logger.error(f"Traceback: {traceback.format_exc()}")
                raise GenerationError(f"Unexpected error extracting response text: {e}")
            
            # Prepare metadata with safe attribute access
            usage_data = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            finish_reason = "unknown"
            
            try:
                if hasattr(response, 'usage_metadata') and response.usage_metadata:
                    usage_meta = response.usage_metadata
                    usage_data = {
                        "prompt_tokens": getattr(usage_meta, 'prompt_token_count', 0),
                        "completion_tokens": getattr(usage_meta, 'candidates_token_count', 0),
                        "total_tokens": getattr(usage_meta, 'total_token_count', 0),
                    }
            except Exception as e:
                logger.warning(f"Could not extract usage metadata: {e}")
            
            try:
                if hasattr(response, 'candidates') and response.candidates and len(response.candidates) > 0:
                    finish_reason = getattr(response.candidates[0], 'finish_reason', 'unknown')
            except Exception as e:
                logger.warning(f"Could not extract finish reason: {e}")
            
            metadata = {
                "model": self.model_name,
                "usage": usage_data,
                "latency_seconds": end_time - start_time,
                "finish_reason": str(finish_reason),
                "response_id": f"genai_{int(start_time)}",
                "streaming": False
            }
            
            return GenerationResult(
                generated_text=generated_text,
                metadata=metadata,
                timestamp=end_time,
                success=True
            )
            
        except Exception as e:
            error_msg = str(e)
            logger.error(f"GenAI generation failed: {e}")
            logger.error(f"Exception type: {type(e).__name__}")
            
            # Check for specific error types
            if "rate limit" in error_msg.lower() or "quota" in error_msg.lower():
                raise RateLimitError(f"GenAI rate limit exceeded: {e}")
            elif "api key" in error_msg.lower() or "authentication" in error_msg.lower():
                raise ModelUnavailableError(f"GenAI authentication error: {e}")
            else:
                # Add more context for debugging
                import traceback
                logger.error(f"Full traceback: {traceback.format_exc()}")
                raise GenerationError(f"GenAI API error: {e}")
    
    def _generate_direct_impl(
        self, 
        complete_prompt: str,
        generation_config: Optional[GenerationConfig] = None
    ) -> GenerationResult:
        """
        Generate content using a complete, pre-formatted prompt with retry for empty responses.
        
        This method wraps the actual implementation with retry logic to handle
        Gemini's occasional empty responses.
        """
        return self._retry_on_empty_response(
            self._generate_direct_core,
            complete_prompt,
            generation_config
        )
    
    def _generate_direct_core(
        self, 
        complete_prompt: str,
        generation_config: Optional[GenerationConfig] = None
    ) -> GenerationResult:
        """
        Core direct content generation using Google GenAI API.
        
        Args:
            complete_prompt: Complete, ready-to-use prompt text
            generation_config: Generation parameters
            
        Returns:
            GenerationResult containing the generated content
        """
        # Use default config if not provided
        if generation_config is None:
            generation_config = GenerationConfig()
        
        # Prepare generation config
        # Use model's configured values if available, otherwise use GenerationConfig value
        model_max_tokens = self.config.get("max_tokens", generation_config.max_tokens)
        model_temperature = self.config.get("temperature", generation_config.temperature)
        
        config_params = {
            "temperature": model_temperature,
            "max_output_tokens": model_max_tokens,
        }
        
        # Add top_p if specified and supported
        model_top_p = self.config.get("top_p", generation_config.top_p)
        if model_top_p is not None:
            config_params["top_p"] = model_top_p
        
        # Add JSON mode if requested
        if generation_config.use_json_mode:
            config_params["response_mime_type"] = "application/json"
        
        # Filter out None values
        config_params = {k: v for k, v in config_params.items() if v is not None}
        
        try:
            logger.info("Starting direct generation...")
            start_time = time.time()
            
            # Create generation config
            genai_config = types.GenerateContentConfig(**config_params)
            
            # Generate content using complete prompt directly
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=complete_prompt,
                config=genai_config
            )
            
            end_time = time.time()
            
            # Check if response is valid
            if not response or not response.text:
                raise GenerationError("Empty or invalid response from GenAI")
            
            generated_text = response.text.strip()
            
            # Prepare metadata
            metadata = {
                "model": self.model_name,
                "latency_seconds": end_time - start_time,
                "generation_type": "direct",
                "prompt_length": len(complete_prompt),
                "response_length": len(generated_text)
            }
            
            # Add usage stats if available
            if hasattr(response, 'usage_metadata') and response.usage_metadata:
                metadata["usage"] = {
                    "prompt_tokens": getattr(response.usage_metadata, 'prompt_token_count', 0),
                    "completion_tokens": getattr(response.usage_metadata, 'candidates_token_count', 0),
                    "total_tokens": getattr(response.usage_metadata, 'total_token_count', 0)
                }
            
            logger.info(f"Direct generation completed in {end_time - start_time:.2f}s")
            logger.info(f"Generated text length: {len(generated_text)} characters")
            
            return GenerationResult(
                generated_text=generated_text,
                metadata=metadata,
                timestamp=end_time,
                success=True
            )
            
        except Exception as e:
            logger.error(f"Direct generation failed: {e}")
            # Handle authentication errors specifically
            if "authentication" in str(e).lower() or "credentials" in str(e).lower():
                raise ModelUnavailableError(f"GenAI authentication error: {e}")
            else:
                raise GenerationError(f"GenAI API error: {e}")
    
    def is_available(self) -> bool:
        """
        Check if GenAI adapter is available and properly configured.
        
        Returns:
            True if adapter can be used, False otherwise
        """
        try:
            # Check if GenAI library is available
            if not GENAI_AVAILABLE:
                return False
            
            # Check if client is initialized
            if not hasattr(self, 'client'):
                return False
            
            # Check if API key is set
            api_key = self.config.get("api_key") or os.getenv("GEMINI_API_KEY")
            if not api_key:
                return False
            
            return True
            
        except Exception:
            return False
    
    def validate_config(self) -> list[str]:
        """
        Validate GenAI adapter configuration.
        
        Returns:
            List of validation error messages
        """
        errors = super().validate_config()
        
        # Check GenAI library
        if not GENAI_AVAILABLE:
            errors.append("Google GenAI library not installed")
        
        # Check API key
        api_key = self.config.get("api_key") or os.getenv("GEMINI_API_KEY")
        if not api_key:
            errors.append("Gemini API key not found")
        
        # Check model name format (basic validation)
        if self.model_name and not (self.model_name.startswith("gemini") or "chat" in self.model_name.lower()):
            logger.warning(f"Model name '{self.model_name}' doesn't follow expected GenAI naming pattern")
        
        # Validate thinking budget if provided
        thinking_budget = self.config.get("thinking_budget")
        if thinking_budget is not None:
            if not isinstance(thinking_budget, int) or thinking_budget < 0:
                errors.append("thinking_budget must be a non-negative integer")
        
        return errors
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get detailed information about the GenAI model."""
        info = super().get_model_info()
        
        # Add GenAI-specific information
        model_config = self.MODEL_CONFIGS.get(self.model_name, {})
        info.update({
            "provider": "google_genai",
            "model_type": "chat_completion",
            "max_tokens": model_config.get("max_tokens", "unknown"),
            "context_length": model_config.get("context_length", "unknown"),
            "thinking_budget": self.config.get("thinking_budget", 0),
            "api_version": "google-genai" if GENAI_AVAILABLE else "not_installed",
        })
        
        return info