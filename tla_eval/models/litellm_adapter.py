"""
Unified LiteLLM adapter for multi-provider chat model access.

This adapter consolidates most hosted model providers behind LiteLLM while
keeping the benchmark's existing ModelAdapter interface unchanged.
"""

import os
import time
import logging
from typing import Any, Dict, List, Optional

try:
    import litellm
    from litellm import completion

    LITELLM_AVAILABLE = True
except ImportError:
    litellm = None
    completion = None
    LITELLM_AVAILABLE = False

from .base import (
    ModelAdapter,
    GenerationConfig,
    GenerationResult,
    ModelError,
    ModelUnavailableError,
    GenerationError,
    RateLimitError,
)

logger = logging.getLogger(__name__)


class LiteLLMAdapter(ModelAdapter):
    """
    Adapter for calling multiple providers through LiteLLM.

    Supported configuration parameters:
        - provider: Provider alias used by config/factory
        - api_key: Provider API key
        - api_key_env: Optional environment variable name for the API key
        - model_name: Model identifier, prefixed or unprefixed
        - url / base_url: Optional OpenAI-compatible endpoint
        - timeout: Request timeout in seconds
        - temperature / top_p / max_tokens: Standard generation controls
        - thinking_budget: Optional reasoning budget for Anthropic/Gemini-like models
        - reasoning_effort: Optional OpenAI-style reasoning effort
        - litellm_params: Optional dict of provider-specific passthrough params
    """

    PROVIDER_ALIASES = {
        "openai": "openai",
        "anthropic": "anthropic",
        "genai": "gemini",
        "google_genai": "gemini",
        "gemini": "gemini",
        "deepseek": "deepseek",
        "yunwu": "openai",
    }

    API_KEY_ENV_MAP = {
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "genai": "GEMINI_API_KEY",
        "google_genai": "GEMINI_API_KEY",
        "gemini": "GEMINI_API_KEY",
        "deepseek": "DEEPSEEK_API_KEY",
        "openrouter": "OPENROUTER_API_KEY",
        "xai": "XAI_API_KEY",
        "groq": "GROQ_API_KEY",
        "together_ai": "TOGETHERAI_API_KEY",
        "together": "TOGETHERAI_API_KEY",
    }

    NO_API_KEY_PROVIDERS = {
        "ollama",
        "lm_studio",
        "llamafile",
    }

    def _setup_model(self):
        """Initialize LiteLLM configuration."""
        if not LITELLM_AVAILABLE:
            raise ModelUnavailableError(
                "LiteLLM library not installed. Run: pip install litellm"
            )

        self.provider = str(self.config.get("provider", "litellm")).lower()
        self.api_base = self.config.get("base_url") or self.config.get("url")
        self.effective_provider = self._infer_effective_provider()
        self.api_key = self._resolve_api_key()
        self.litellm_model = self.resolve_model_name(
            self.provider,
            self.model_name,
            self.api_base,
        )

        extra_params = self.config.get("litellm_params")
        if extra_params is not None and not isinstance(extra_params, dict):
            raise ModelUnavailableError("litellm_params must be a dictionary")

    @classmethod
    def resolve_model_name(
        cls,
        provider: Optional[str],
        model_name: str,
        api_base: Optional[str] = None,
    ) -> str:
        """Resolve config-style provider/model fields into a LiteLLM model name."""
        provider_key = str(provider or "litellm").lower()

        if not model_name:
            return model_name

        if "/" in model_name:
            return model_name

        # When the user explicitly pins a provider (e.g. "anthropic"), honor it
        # even with a custom api_base — Anthropic-compatible proxies (like
        # api.minimax.io/anthropic) need the `anthropic/` prefix or LiteLLM
        # builds an OpenAI-format request and gets 404.
        if provider_key not in ("litellm", ""):
            provider_prefix = cls.PROVIDER_ALIASES.get(provider_key, provider_key)
            return f"{provider_prefix}/{model_name}"

        # Default: assume api_base is OpenAI-compatible (gptsapi etc.)
        if api_base:
            return f"openai/{model_name}"

        return model_name

    @staticmethod
    def build_thinking_config(
        thinking_budget: Optional[int],
    ) -> Optional[Dict[str, Any]]:
        """Translate the legacy thinking budget setting into LiteLLM params."""
        if thinking_budget is None or thinking_budget <= 0:
            return None

        return {
            "type": "enabled",
            "budget_tokens": int(thinking_budget),
        }

    @staticmethod
    def _safe_get(obj: Any, key: str, default: Any = None) -> Any:
        if obj is None:
            return default
        if isinstance(obj, dict):
            return obj.get(key, default)
        return getattr(obj, key, default)

    @staticmethod
    def _content_to_text(content: Any) -> str:
        if content is None:
            return ""

        if isinstance(content, str):
            return content

        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                    continue

                if isinstance(item, dict):
                    text = item.get("text") or item.get("content")
                    if isinstance(text, str):
                        parts.append(text)
                    continue

                text = getattr(item, "text", None) or getattr(item, "content", None)
                if isinstance(text, str):
                    parts.append(text)

            return "".join(parts).strip()

        if isinstance(content, dict):
            text = content.get("text") or content.get("content")
            if isinstance(text, str):
                return text

        return str(content)

    @classmethod
    def _extract_generated_text(cls, response: Any) -> str:
        choices = cls._safe_get(response, "choices", [])
        if choices:
            first_choice = choices[0]
            message = cls._safe_get(first_choice, "message")
            if message is not None:
                text = cls._content_to_text(cls._safe_get(message, "content"))
                if text:
                    return text

            delta = cls._safe_get(first_choice, "delta")
            if delta is not None:
                text = cls._content_to_text(cls._safe_get(delta, "content"))
                if text:
                    return text

            text = cls._content_to_text(cls._safe_get(first_choice, "text"))
            if text:
                return text

        output = cls._safe_get(response, "output", [])
        if isinstance(output, list):
            parts = []
            for item in output:
                content = cls._safe_get(item, "content", [])
                if isinstance(content, list):
                    for block in content:
                        text = cls._content_to_text(cls._safe_get(block, "text"))
                        if text:
                            parts.append(text)
            if parts:
                return "".join(parts).strip()

        return ""

    @classmethod
    def _extract_usage(cls, response: Any) -> Dict[str, Any]:
        usage = cls._safe_get(response, "usage", {})
        if usage is None:
            return {}
        if isinstance(usage, dict):
            return usage
        if hasattr(usage, "model_dump"):
            return usage.model_dump()

        extracted = {}
        for field in ("prompt_tokens", "completion_tokens", "total_tokens"):
            value = getattr(usage, field, None)
            if value is not None:
                extracted[field] = value
        return extracted

    def _resolve_api_key(self) -> Optional[str]:
        explicit_env = self.config.get("api_key_env")
        if explicit_env:
            value = os.getenv(explicit_env)
            if value:
                return value

        if self.config.get("api_key"):
            return self.config["api_key"]

        provider_env = self.API_KEY_ENV_MAP.get(self.effective_provider)
        if provider_env:
            value = os.getenv(provider_env)
            if value:
                return value

        if self.api_base:
            return os.getenv("OPENAI_API_KEY")

        return None

    def _infer_effective_provider(self) -> str:
        """Infer the underlying provider for env lookup and local-model handling."""
        provider = self.provider

        if provider != "litellm":
            return self.PROVIDER_ALIASES.get(provider, provider)

        if self.api_base:
            return "openai"

        if "/" in self.model_name:
            return self.model_name.split("/", 1)[0].lower()

        return provider

    def _supports_missing_api_key(self) -> bool:
        return self.effective_provider in self.NO_API_KEY_PROVIDERS

    def _should_omit_top_p(self) -> bool:
        return self.litellm_model.startswith("openai/gpt-5")

    def _build_completion_params(
        self,
        prompt: str,
        generation_config: Optional[GenerationConfig] = None,
    ) -> Dict[str, Any]:
        """Build a LiteLLM completion request."""
        if generation_config is None:
            generation_config = GenerationConfig()

        params: Dict[str, Any] = {
            "model": self.litellm_model,
            "messages": [{"role": "user", "content": prompt}],
            "timeout": self.config.get("timeout", 300),
            "drop_params": True,
            # Disable automatic retries. Reasoning models (grok-4, deepseek-reasoner)
            # can take 10+ minutes per call and cost $1+ each; silent retries on
            # client-side timeouts produce multiple billable calls. A single failure
            # should fail fast, not cascade.
            "num_retries": 0,
            "max_retries": 0,
        }

        if self.api_key:
            params["api_key"] = self.api_key
        if self.api_base:
            params["api_base"] = self.api_base

        model_max_tokens = self.config.get("max_tokens", generation_config.max_tokens)
        if model_max_tokens is not None:
            params["max_tokens"] = model_max_tokens

        model_temperature = self.config.get(
            "temperature",
            generation_config.temperature,
        )
        if model_temperature is not None:
            params["temperature"] = model_temperature

        model_top_p = self.config.get("top_p", generation_config.top_p)
        if model_top_p is not None and not self._should_omit_top_p():
            params["top_p"] = model_top_p

        if generation_config.stop_sequences:
            params["stop"] = generation_config.stop_sequences
        if generation_config.seed is not None:
            params["seed"] = generation_config.seed
        if generation_config.use_json_mode:
            params["response_format"] = {"type": "json_object"}

        thinking = self.build_thinking_config(self.config.get("thinking_budget"))
        if thinking:
            params["thinking"] = thinking

        reasoning_effort = self.config.get("reasoning_effort")
        if reasoning_effort is not None:
            params["reasoning_effort"] = reasoning_effort

        extra_params = self.config.get("litellm_params") or {}
        params.update(extra_params)
        return params

    @staticmethod
    def _is_rate_limit_error(error: Exception) -> bool:
        error_name = type(error).__name__.lower()
        error_text = str(error).lower()
        return (
            "ratelimit" in error_name
            or "rate limit" in error_text
            or "too many requests" in error_text
            or "429" in error_text
        )

    def _generate_from_prompt(
        self,
        prompt: str,
        generation_config: Optional[GenerationConfig],
        generation_type: str,
    ) -> GenerationResult:
        if not self.is_available():
            raise ModelUnavailableError("LiteLLM adapter is not properly configured")

        api_params = self._build_completion_params(prompt, generation_config)
        start_time = time.time()

        # Narrow retry ONLY for provider-side overload signals (HTTP 529 /
        # anthropic "overloaded_error" / generic 503). These are not billable
        # — server returns early without having done the work. Bounded to 2
        # retries with 20/40 s backoff; anything else raises immediately.
        # This is LOUD (warnings to log), unlike the previous silent 4×30s
        # retry-all-errors wrapper that was removed for cost safety.
        max_overload_retries = 2
        attempt = 0
        while True:
            try:
                response = completion(**api_params)
                if attempt > 0:
                    logger.warning(f"LiteLLM 529/overload cleared after {attempt} retry(ies)")
                break
            except Exception as e:
                msg = str(e).lower()
                err_name = type(e).__name__.lower()
                # Recognise transient provider-side rejections. These don't
                # bill (server rejected early without generating output).
                # Match by exception class name and numeric error codes so
                # the check works regardless of the error text's language.
                is_transient = (
                    "529" in msg or "overloaded" in msg or "overload_error" in msg
                    or "ratelimit" in err_name or "rate limit" in msg
                    or "too many requests" in msg or "429" in msg
                    # DashScope cluster-busy returns error code 2064.
                    or "(2064)" in msg
                )
                if is_transient and attempt < max_overload_retries:
                    delay = 20 * (2 ** attempt)  # 20s, 40s
                    logger.warning(
                        f"LiteLLM got provider transient error "
                        f"(attempt {attempt+1}/{max_overload_retries+1}); "
                        f"sleeping {delay}s then retrying. Error: {str(e)[:200]}"
                    )
                    time.sleep(delay)
                    attempt += 1
                    continue
                raise

        try:
            generated_text = self._extract_generated_text(response)
            end_time = time.time()

            if not generated_text:
                raise GenerationError("Empty response from LiteLLM")

            metadata = {
                "model": self.model_name,
                "litellm_model": self.litellm_model,
                "provider": self.provider,
                "usage": self._extract_usage(response),
                "latency_seconds": end_time - start_time,
                "generation_type": generation_type,
                "response_id": self._safe_get(response, "id"),
                "finish_reason": self._safe_get(
                    self._safe_get(response, "choices", [{}])[0],
                    "finish_reason",
                ),
            }

            return GenerationResult(
                generated_text=generated_text,
                metadata=metadata,
                timestamp=end_time,
                success=True,
            )

        except ModelError:
            raise
        except Exception as e:
            if self._is_rate_limit_error(e):
                raise RateLimitError(f"LiteLLM rate limit exceeded: {e}")
            raise GenerationError(f"LiteLLM generation failed: {e}")

    def _generate_tla_specification_impl(
        self,
        source_code: str,
        prompt_template: str,
        generation_config: Optional[GenerationConfig] = None,
    ) -> GenerationResult:
        if "{source_code}" in prompt_template:
            prompt = prompt_template.replace("{source_code}", source_code)
        else:
            prompt = prompt_template

        return self._generate_from_prompt(
            prompt,
            generation_config,
            generation_type="tla_specification",
        )

    def _generate_direct_impl(
        self,
        complete_prompt: str,
        generation_config: Optional[GenerationConfig] = None,
    ) -> GenerationResult:
        return self._generate_from_prompt(
            complete_prompt,
            generation_config,
            generation_type="direct",
        )

    def is_available(self) -> bool:
        """Check whether LiteLLM is installed and the model is configured."""
        if not LITELLM_AVAILABLE:
            return False

        if not self.litellm_model:
            return False

        if not self.api_key and not self._supports_missing_api_key():
            return False

        return True

    def validate_config(self) -> List[str]:
        """Validate LiteLLM adapter configuration."""
        errors = super().validate_config()

        if not LITELLM_AVAILABLE:
            errors.append("LiteLLM library not installed")

        if not self.litellm_model:
            errors.append("Unable to resolve LiteLLM model name")

        if not self.api_key and not self._supports_missing_api_key():
            errors.append(
                f"API key not found for provider '{self.effective_provider}'"
            )

        thinking_budget = self.config.get("thinking_budget")
        if thinking_budget is not None:
            if not isinstance(thinking_budget, int) or thinking_budget < 0:
                errors.append("thinking_budget must be a non-negative integer")

        extra_params = self.config.get("litellm_params")
        if extra_params is not None and not isinstance(extra_params, dict):
            errors.append("litellm_params must be a dictionary")

        return errors

    def get_model_info(self) -> Dict[str, Any]:
        """Get detailed information about the LiteLLM-backed model."""
        info = super().get_model_info()
        info.update(
            {
                "provider": self.provider,
                "effective_provider": self.effective_provider,
                "model_type": "chat_completion",
                "litellm_model": self.litellm_model,
                "api_base": self.api_base,
                "thinking_budget": self.config.get("thinking_budget", 0),
                "api_version": getattr(litellm, "__version__", "unknown"),
            }
        )
        return info
