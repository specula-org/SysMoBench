"""
Code Agent method for TLA+ specification generation.

This module provides integration with code agents like Claude Code, Codex CLI,
Gemini CLI, etc. for generating TLA+ specifications through an agentic workflow.
"""

from .method import CodeAgentMethod
from .workspace import TaskWorkspace
from .adapters.base import BaseCodeAgentAdapter, ExecutionResult
from .adapters.claude_code import ClaudeCodeAdapter, ClaudeCodeConfig
from .adapters.codex import CodexAdapter, CodexConfig
from .adapters.gemini import GeminiAdapter, GeminiConfig

__all__ = [
    "CodeAgentMethod",
    "TaskWorkspace",
    "BaseCodeAgentAdapter",
    "ExecutionResult",
    "ClaudeCodeAdapter",
    "ClaudeCodeConfig",
    "CodexAdapter",
    "CodexConfig",
    "GeminiAdapter",
    "GeminiConfig",
]
