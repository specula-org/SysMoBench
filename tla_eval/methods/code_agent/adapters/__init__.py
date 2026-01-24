"""
Code agent adapters for different CLI tools.
"""

from .base import BaseCodeAgentAdapter, ExecutionResult
from .claude_code import ClaudeCodeAdapter, ClaudeCodeConfig
from .codex import CodexAdapter, CodexConfig
from .gemini import GeminiAdapter, GeminiConfig

__all__ = [
    "BaseCodeAgentAdapter",
    "ExecutionResult",
    "ClaudeCodeAdapter",
    "ClaudeCodeConfig",
    "CodexAdapter",
    "CodexConfig",
    "GeminiAdapter",
    "GeminiConfig",
]
