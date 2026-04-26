"""
TLA+ generation methods.

This module contains different approaches for generating TLA+ specifications
from source code, including direct LLM calls and agent-based methods.
"""

from .base import TLAGenerationMethod
from .direct_call import DirectCallMethod
from .agent_based import AgentBasedMethod
from .code_agent import (
    CodeAgentMethod,
    ClaudeCodeAdapter,
    ClaudeCodeConfig,
    CodexAdapter,
    CodexConfig,
    GeminiAdapter,
    GeminiConfig,
)

# Method registry for simple methods (no special initialization)
_METHODS = {
    "direct_call": DirectCallMethod,
    "agent_based": AgentBasedMethod,
}

# Code agent adapters registry
_CODE_AGENT_ADAPTERS = {
    "claude_code": (ClaudeCodeAdapter, ClaudeCodeConfig),
    "codex": (CodexAdapter, CodexConfig),
    "gemini": (GeminiAdapter, GeminiConfig),
}


def get_method(method_name: str, **kwargs) -> TLAGenerationMethod:
    """
    Get a method instance by name.

    For code_agent methods, use format: "code_agent_<adapter_name>"
    Example: "code_agent_claude_code"

    Args:
        method_name: Method name (e.g., "direct_call", "code_agent_claude_code")
        **kwargs: Additional arguments passed to method constructor

    Returns:
        TLAGenerationMethod instance
    """
    # Check if it's a code agent method
    if method_name.startswith("code_agent_"):
        adapter_name = method_name[len("code_agent_"):]
        return _create_code_agent_method(adapter_name, **kwargs)

    # Regular methods
    if method_name not in _METHODS:
        available = list(_METHODS.keys()) + [f"code_agent_{a}" for a in _CODE_AGENT_ADAPTERS.keys()]
        raise ValueError(f"Unknown method '{method_name}'. Available: {available}")

    method_class = _METHODS[method_name]
    return method_class()


def _create_code_agent_method(adapter_name: str, **kwargs) -> CodeAgentMethod:
    """
    Create a code agent method with the specified adapter.

    Args:
        adapter_name: Name of the adapter (e.g., "claude_code", "codex", "gemini")
        **kwargs: Configuration options

    Returns:
        CodeAgentMethod instance
    """
    if adapter_name not in _CODE_AGENT_ADAPTERS:
        available = list(_CODE_AGENT_ADAPTERS.keys())
        raise ValueError(f"Unknown code agent adapter '{adapter_name}'. Available: {available}")

    adapter_class, config_class = _CODE_AGENT_ADAPTERS[adapter_name]

    # Extract adapter-specific config, using config_class defaults
    # Create a default config to get the default model
    default_config = config_class()
    model = kwargs.pop("model", default_config.model)
    timeout = kwargs.pop("timeout", default_config.timeout)

    config = config_class(model=model, timeout=timeout)
    adapter = adapter_class(config=config)

    # Create method with remaining kwargs
    # Default workspace_base to "workspaces" for persistent storage
    return CodeAgentMethod(
        adapter=adapter,
        max_attempts=kwargs.pop("max_attempts", 3),
        output_dir=kwargs.pop("output_dir", None),
        workspace_base=kwargs.pop("workspace_base", "workspaces"),
        keep_workspace=kwargs.pop("keep_workspace", True),
    )


def list_available_methods() -> list:
    """List all available method names."""
    methods = list(_METHODS.keys())
    methods.extend([f"code_agent_{a}" for a in _CODE_AGENT_ADAPTERS.keys()])
    return methods


__all__ = [
    "TLAGenerationMethod",
    "DirectCallMethod",
    "AgentBasedMethod",
    "CodeAgentMethod",
    "ClaudeCodeAdapter",
    "ClaudeCodeConfig",
    "CodexAdapter",
    "CodexConfig",
    "GeminiAdapter",
    "GeminiConfig",
    "get_method",
    "list_available_methods",
]