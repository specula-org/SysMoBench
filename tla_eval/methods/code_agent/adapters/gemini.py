"""
Google Gemini CLI adapter.
"""

import asyncio
import json
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from .base import BaseCodeAgentAdapter, ExecutionResult


@dataclass
class GeminiConfig:
    """Configuration for Gemini CLI adapter."""
    model: str = ""  # Use Gemini CLI default (configured in ~/.gemini/settings.json)
    timeout: int = 1800                   # 30 minutes default
    output_format: str = "text"           # text / json / stream-json
    initial_prompt: str = "follow the GEMINI.md to complete the task."
    # Polling settings for waiting output files (Gemini may run async)
    poll_interval: int = 10               # Check every 10 seconds
    poll_timeout: int = 600               # Wait up to 10 minutes for output


class GeminiAdapter(BaseCodeAgentAdapter):
    """
    Adapter for Google Gemini CLI.

    Executes Gemini CLI in a prepared workspace with MCP tools configured.
    Gemini CLI requires MCP configuration in ~/.gemini/settings.json.
    """

    def __init__(self, config: Optional[GeminiConfig] = None):
        """
        Initialize the adapter.

        Args:
            config: Configuration options. Uses defaults if not provided.
        """
        self.config = config or GeminiConfig()
        self._config_dir = Path.home() / ".gemini"
        self._config_file = self._config_dir / "settings.json"
        self._backup_file = self._config_dir / "settings.json.sysmobench_backup"

    @property
    def agent_name(self) -> str:
        return "gemini"

    @property
    def instruction_filename(self) -> str:
        """Return the instruction filename for this adapter."""
        return "GEMINI.md"

    def _update_mcp_config(self, mcp_config_path: Path) -> None:
        """
        Add/update MCP server config in Gemini settings.

        Only adds our MCP servers, preserves all existing user config.
        No backup/restore needed - we just update/overwrite our entries.

        Args:
            mcp_config_path: Path to MCP JSON config file
        """
        self._config_dir.mkdir(parents=True, exist_ok=True)

        # Load existing settings or start fresh
        settings = {}
        if self._config_file.exists():
            with open(self._config_file, "r", encoding="utf-8") as f:
                settings = json.load(f)

        # Read MCP config for this task
        with open(mcp_config_path, "r", encoding="utf-8") as f:
            mcp_config = json.load(f)

        # Ensure mcpServers exists
        if "mcpServers" not in settings:
            settings["mcpServers"] = {}

        # Add our MCP servers with trust enabled
        for server_name, server_config in mcp_config.get("mcpServers", {}).items():
            server_config["trust"] = True  # Enable auto-approval
            server_config["timeout"] = 600000  # 10 minutes timeout
            settings["mcpServers"][server_name] = server_config

            # Also add to mcp.allowed list
            if "mcp" not in settings:
                settings["mcp"] = {}
            if "allowed" not in settings["mcp"]:
                settings["mcp"]["allowed"] = []
            if server_name not in settings["mcp"]["allowed"]:
                settings["mcp"]["allowed"].append(server_name)

        # Write updated config
        with open(self._config_file, "w", encoding="utf-8") as f:
            json.dump(settings, f, indent=2)

    def _is_tla_file_complete(self, file_path: Path) -> bool:
        """
        Check if a TLA+ file is complete.

        A complete TLA+ file ends with '====' (possibly followed by whitespace).
        """
        try:
            content = file_path.read_text(encoding="utf-8")
            # TLA+ files end with ==== (4 or more equals signs)
            stripped = content.rstrip()
            return stripped.endswith("====") or stripped.endswith("=============")
        except Exception:
            return False

    async def _wait_for_output_files(
        self,
        workspace_path: Path,
    ) -> bool:
        """
        Wait for output files to be generated and complete.

        Polls for TLA+ files that are complete (end with ====).
        Waits up to poll_timeout seconds (default 10 minutes).

        Args:
            workspace_path: Path to workspace directory

        Returns:
            True if output files found and complete, False if timeout
        """
        output_dir = workspace_path / "output"
        poll_count = self.config.poll_timeout // self.config.poll_interval

        for _ in range(poll_count):
            await asyncio.sleep(self.config.poll_interval)

            # Check for .tla files
            tla_files = list(output_dir.glob("*.tla"))

            if tla_files:
                # Check if all TLA+ files are complete (end with ====)
                all_complete = all(
                    self._is_tla_file_complete(f) for f in tla_files
                )
                if all_complete:
                    return True

        return False

    async def execute(
        self,
        workspace_path: Path,
        mcp_config_path: Path,
        model_override: Optional[str] = None,
    ) -> ExecutionResult:
        """
        Execute Gemini CLI in the workspace.

        Args:
            workspace_path: Path to workspace containing GEMINI.md and source_code/
            mcp_config_path: Path to MCP configuration JSON file
            model_override: Optional model to use instead of config default

        Returns:
            ExecutionResult with execution outcome
        """
        model = model_override or self.config.model

        # Skip MCP config update - assume ~/.gemini/settings.json is already configured
        # self._update_mcp_config(mcp_config_path)

        # Create GEMINI.md from CLAUDE.md if it doesn't exist
        gemini_md = workspace_path / "GEMINI.md"
        claude_md = workspace_path / "CLAUDE.md"
        if not gemini_md.exists() and claude_md.exists():
            # Copy CLAUDE.md to GEMINI.md
            shutil.copy2(claude_md, gemini_md)

        # Build command
        # Use positional prompt (not -p which is deprecated)
        # Use --yolo to auto-approve all tool calls
        cmd = ["gemini", "--yolo"]
        # Only add -m if model is specified and not "default"
        if model and model != "default":
            cmd.extend(["-m", model])
        cmd.append(self.config.initial_prompt)

        start_time = time.time()

        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=workspace_path,
                # Don't capture stdout/stderr - let Gemini run naturally
            )

            try:
                await asyncio.wait_for(
                    process.wait(),
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

            # Gemini CLI main process may exit before files are written.
            # Poll for output files to ensure completion.
            output_dir = workspace_path / "output"
            await self._wait_for_output_files(workspace_path)

            duration = time.time() - start_time

            # Check if output files exist now
            tla_files = list(output_dir.glob("*.tla"))
            has_output = len(tla_files) > 0 and any(
                f.stat().st_size > 0 for f in tla_files
            )

            # Success if process exited cleanly AND output files exist
            success = process.returncode == 0 and has_output

            return ExecutionResult(
                success=success,
                raw_output="",
                parsed_output=None,
                error=None if success else "Gemini failed to generate output",
                duration_seconds=duration,
                exit_code=process.returncode,
            )

        except FileNotFoundError:
            return ExecutionResult(
                success=False,
                raw_output="",
                error="Gemini CLI not found. Please ensure 'gemini' is installed and in PATH. "
                      "Install with: npm install -g @google/gemini-cli",
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
            "cli": "gemini",
            "model": self.config.model,
            "timeout": self.config.timeout,
            "instruction_file": self.instruction_filename,
        }
