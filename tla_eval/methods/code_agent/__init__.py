"""
Code Agent method for TLA+ specification generation.

This module provides integration with code agents like Claude Code, Codex CLI, etc.
for generating TLA+ specifications through an agentic workflow.
"""

from .method import CodeAgentMethod
from .workspace import TaskWorkspace
from .adapters.base import BaseCodeAgentAdapter, ExecutionResult
from .adapters.claude_code import ClaudeCodeAdapter, ClaudeCodeConfig

__all__ = [
    "CodeAgentMethod",
    "TaskWorkspace",
    "BaseCodeAgentAdapter",
    "ExecutionResult",
    "ClaudeCodeAdapter",
    "ClaudeCodeConfig",
]
