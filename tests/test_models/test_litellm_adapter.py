from tla_eval.models.base import GenerationConfig, ModelAdapter
from tla_eval.models.factory import ModelFactory
from tla_eval.models.litellm_adapter import LiteLLMAdapter


class DummyAdapter(ModelAdapter):
    def _setup_model(self):
        pass

    def _generate_tla_specification_impl(self, source_code, prompt_template, generation_config=None):
        raise NotImplementedError

    def _generate_direct_impl(self, complete_prompt, generation_config=None):
        raise NotImplementedError

    def is_available(self) -> bool:
        return True


def test_resolve_model_name_for_provider_aliases():
    assert LiteLLMAdapter.resolve_model_name("openai", "gpt-4o") == "openai/gpt-4o"
    assert (
        LiteLLMAdapter.resolve_model_name("anthropic", "claude-sonnet-4-20250514")
        == "anthropic/claude-sonnet-4-20250514"
    )
    assert (
        LiteLLMAdapter.resolve_model_name("genai", "gemini-2.5-pro")
        == "gemini/gemini-2.5-pro"
    )


def test_resolve_model_name_for_custom_openai_compatible_api():
    assert (
        LiteLLMAdapter.resolve_model_name(
            "deepseek",
            "deepseek-reasoner",
            api_base="https://api.deepseek.com",
        )
        == "openai/deepseek-reasoner"
    )
    assert (
        LiteLLMAdapter.resolve_model_name(
            "yunwu",
            "gpt-5-2025-08-07",
            api_base="https://yunwu.ai/v1",
        )
        == "openai/gpt-5-2025-08-07"
    )


def test_prefixed_model_name_is_preserved():
    assert (
        LiteLLMAdapter.resolve_model_name("litellm", "gemini/gemini-2.5-pro")
        == "gemini/gemini-2.5-pro"
    )


def test_effective_provider_can_be_inferred_from_prefixed_model_name():
    class FakeLiteLLMAdapter(LiteLLMAdapter):
        def _setup_model(self):
            self.provider = str(self.config.get("provider", "litellm")).lower()
            self.api_base = self.config.get("base_url") or self.config.get("url")
            self.effective_provider = self._infer_effective_provider()
            self.api_key = None
            self.litellm_model = self.resolve_model_name(
                self.provider,
                self.model_name,
                self.api_base,
            )

        def is_available(self) -> bool:
            return True

    adapter = FakeLiteLLMAdapter("anthropic/claude-sonnet-4-20250514", provider="litellm")
    assert adapter.effective_provider == "anthropic"


def test_build_thinking_config_from_legacy_budget():
    assert LiteLLMAdapter.build_thinking_config(None) is None
    assert LiteLLMAdapter.build_thinking_config(0) is None
    assert LiteLLMAdapter.build_thinking_config(2048) == {
        "type": "enabled",
        "budget_tokens": 2048,
    }


def test_get_model_info_redacts_api_key():
    adapter = DummyAdapter("dummy-model", api_key="super-secret", timeout=30)
    info = adapter.get_model_info()

    assert info["config"]["api_key"] == "***redacted***"
    assert info["config"]["timeout"] == 30


def test_generation_config_still_accepts_none_like_values():
    config = GenerationConfig(max_tokens=None, temperature=0.1, top_p=None)

    assert config.max_tokens is None
    assert config.top_p is None


def test_model_factory_reports_litellm_as_recommended_provider():
    info = ModelFactory.list_available_models()

    assert info["recommended_provider"] == "litellm"
    assert "deepseek" in info["legacy_providers"]
    assert "yunwu" in info["legacy_providers"]
