"""
Base class for code agent adapters.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class ExecutionResult:
    """Result from code agent execution."""
    success: bool
    raw_output: str = ""
    parsed_output: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

    # Execution metadata
    duration_seconds: float = 0.0
    exit_code: Optional[int] = None


class BaseCodeAgentAdapter(ABC):
    """
    Abstract base class for code agent adapters.

    Each code agent (Claude Code, Codex CLI, Gemini CLI, etc.) should
    implement this interface.
    """

    @property
    @abstractmethod
    def agent_name(self) -> str:
        """Return the name of this code agent."""
        pass

    @abstractmethod
    async def execute(
        self,
        workspace_path: Path,
        mcp_config_path: Path,
        model_override: Optional[str] = None,
    ) -> ExecutionResult:
        """
        Execute the code agent in the given workspace.

        The agent should:
        1. Read CLAUDE.md (or equivalent) for task instructions
        2. Read source code from workspace/source_code/
        3. Use MCP tools (submit_spec) for validation
        4. Write output to workspace/output/

        Args:
            workspace_path: Path to prepared workspace directory
            mcp_config_path: Path to MCP configuration file
            model_override: Optional model to use instead of adapter's default

        Returns:
            ExecutionResult with execution outcome and metadata
        """
        pass

    def get_adapter_info(self) -> Dict[str, Any]:
        """Return metadata about this adapter."""
        return {
            "name": self.agent_name,
            "type": "code_agent_adapter",
        }
