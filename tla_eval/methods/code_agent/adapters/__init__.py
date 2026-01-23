"""
Code agent adapters for different CLI tools.
"""

from .base import BaseCodeAgentAdapter, ExecutionResult
from .claude_code import ClaudeCodeAdapter, ClaudeCodeConfig

__all__ = [
    "BaseCodeAgentAdapter",
    "ExecutionResult",
    "ClaudeCodeAdapter",
    "ClaudeCodeConfig",
]
