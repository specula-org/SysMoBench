"""
Claude Code CLI adapter.
"""

import asyncio
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

from .base import BaseCodeAgentAdapter, ExecutionResult


@dataclass
class ClaudeCodeConfig:
    """Configuration for Claude Code adapter."""
    model: str = "sonnet"              # sonnet / opus
    timeout: int = 1800                # 30 minutes default
    output_format: str = "json"        # text / json / stream-json
    initial_prompt: str = "Read CLAUDE.md and follow the instructions to complete the task."


class ClaudeCodeAdapter(BaseCodeAgentAdapter):
    """
    Adapter for Claude Code CLI.

    Executes Claude Code in a prepared workspace with MCP tools configured.
    """

    def __init__(self, config: Optional[ClaudeCodeConfig] = None):
        """
        Initialize the adapter.

        Args:
            config: Configuration options. Uses defaults if not provided.
        """
        self.config = config or ClaudeCodeConfig()

    @property
    def agent_name(self) -> str:
        return "claude_code"

    async def execute(
        self,
        workspace_path: Path,
        mcp_config_path: Path,
        model_override: Optional[str] = None,
    ) -> ExecutionResult:
        """
        Execute Claude Code in the workspace.

        Args:
            workspace_path: Path to workspace containing CLAUDE.md and source_code/
            mcp_config_path: Path to MCP configuration JSON file
            model_override: Optional model to use instead of config default

        Returns:
            ExecutionResult with execution outcome
        """
        # Use override model if provided, otherwise use config default
        model = model_override or self.config.model

        cmd = [
            "claude",
            "--print",
            "--dangerously-skip-permissions",
            "--mcp-config", str(mcp_config_path),
            "--model", model,
            "--output-format", self.config.output_format,
            # "--no-session-persistence",  # 暂时注释掉，方便调试查看聊天记录
            self.config.initial_prompt,
        ]

        start_time = time.time()

        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=workspace_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=self.config.timeout,
                )
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
                duration = time.time() - start_time
                return ExecutionResult(
                    success=False,
                    raw_output="",
                    error=f"Timeout after {self.config.timeout} seconds",
                    duration_seconds=duration,
                    exit_code=-1,
                )

            duration = time.time() - start_time
            output = stdout.decode("utf-8", errors="replace")
            stderr_output = stderr.decode("utf-8", errors="replace")

            # Try to parse JSON output
            parsed_output = None
            if self.config.output_format == "json" and output.strip():
                try:
                    parsed_output = json.loads(output)
                except json.JSONDecodeError:
                    # Output might not be valid JSON, that's okay
                    pass

            return ExecutionResult(
                success=process.returncode == 0,
                raw_output=output,
                parsed_output=parsed_output,
                error=stderr_output if process.returncode != 0 else None,
                duration_seconds=duration,
                exit_code=process.returncode,
            )

        except FileNotFoundError:
            return ExecutionResult(
                success=False,
                raw_output="",
                error="Claude Code CLI not found. Please ensure 'claude' is installed and in PATH.",
                duration_seconds=0.0,
            )
        except Exception as e:
            duration = time.time() - start_time
            return ExecutionResult(
                success=False,
                raw_output="",
                error=f"Unexpected error: {str(e)}",
                duration_seconds=duration,
            )

    def get_adapter_info(self) -> Dict[str, Any]:
        """Return metadata about this adapter."""
        return {
            "name": self.agent_name,
            "type": "code_agent_adapter",
            "cli": "claude",
            "model": self.config.model,
            "timeout": self.config.timeout,
        }
