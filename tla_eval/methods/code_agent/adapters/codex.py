"""
OpenAI Codex CLI adapter.
"""

import asyncio
import json
import os
import pty
import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from .base import BaseCodeAgentAdapter, ExecutionResult


def update_codex_mcp_config(task: str, spec_module: str, output_dir: str) -> None:
    """
    Update ~/.codex/config.toml with the correct MCP environment variables.

    Args:
        task: The task name (e.g., 'etcd', 'zookeeper')
        spec_module: The TLA+ module name
        output_dir: Output directory for submissions
    """
    config_path = Path.home() / ".codex" / "config.toml"
    if not config_path.exists():
        return

    content = config_path.read_text()

    # Update SYSMOBENCH_TASK
    content = re.sub(
        r'SYSMOBENCH_TASK\s*=\s*"[^"]*"',
        f'SYSMOBENCH_TASK = "{task}"',
        content
    )

    # Update SYSMOBENCH_SPEC_MODULE
    content = re.sub(
        r'SYSMOBENCH_SPEC_MODULE\s*=\s*"[^"]*"',
        f'SYSMOBENCH_SPEC_MODULE = "{spec_module}"',
        content
    )

    # Update SYSMOBENCH_OUTPUT if present
    content = re.sub(
        r'SYSMOBENCH_OUTPUT\s*=\s*"[^"]*"',
        f'SYSMOBENCH_OUTPUT = "{output_dir}"',
        content
    )

    config_path.write_text(content)


@dataclass
class CodexConfig:
    """Configuration for Codex CLI adapter."""
    model: str = ""                     # Use Codex CLI default (configured in ~/.codex/config.toml)
    timeout: int = 1800                 # 30 minutes default
    initial_prompt: str = "follow the CODEX.md to complete the task."


class CodexAdapter(BaseCodeAgentAdapter):
    """
    Adapter for OpenAI Codex CLI.

    Executes Codex in a prepared workspace with MCP tools configured.
    Codex CLI requires MCP configuration in ~/.codex/config.toml.
    """

    def __init__(self, config: Optional[CodexConfig] = None):
        """
        Initialize the adapter.

        Args:
            config: Configuration options. Uses defaults if not provided.
        """
        self.config = config or CodexConfig()

    @property
    def agent_name(self) -> str:
        return "codex"

    def _run_with_pty(self, cmd: list, cwd: Path, timeout: int) -> tuple:
        """
        Run command with a pseudo-terminal to simulate interactive environment.

        Returns:
            Tuple of (return_code, duration)
        """
        start_time = time.time()

        # Create pseudo-terminal
        master_fd, slave_fd = pty.openpty()

        try:
            process = subprocess.Popen(
                cmd,
                cwd=cwd,
                stdin=slave_fd,
                stdout=slave_fd,
                stderr=slave_fd,
                close_fds=True,
            )

            # Close slave in parent process
            os.close(slave_fd)

            # Read and forward output to real stdout
            import select
            while True:
                # Check if process has finished
                ret = process.poll()
                if ret is not None:
                    # Process finished, read any remaining output
                    try:
                        while True:
                            ready, _, _ = select.select([master_fd], [], [], 0.1)
                            if not ready:
                                break
                            data = os.read(master_fd, 1024)
                            if not data:
                                break
                            os.write(1, data)  # Write to stdout
                    except OSError:
                        pass
                    break

                # Check timeout
                if time.time() - start_time > timeout:
                    process.kill()
                    process.wait()
                    return (-1, time.time() - start_time)

                # Read available output
                try:
                    ready, _, _ = select.select([master_fd], [], [], 0.5)
                    if ready:
                        data = os.read(master_fd, 1024)
                        if data:
                            os.write(1, data)  # Write to stdout
                except OSError:
                    break

            return (process.returncode, time.time() - start_time)

        finally:
            os.close(master_fd)

    async def execute(
        self,
        workspace_path: Path,
        mcp_config_path: Path,
        model_override: Optional[str] = None,
    ) -> ExecutionResult:
        """
        Execute Codex CLI in the workspace.

        Args:
            workspace_path: Path to workspace containing CLAUDE.md and source_code/
            mcp_config_path: Path to MCP configuration JSON file
            model_override: Optional model to use instead of config default

        Returns:
            ExecutionResult with execution outcome
        """
        model = model_override or self.config.model

        # Update ~/.codex/config.toml with correct task from mcp_config
        try:
            with open(mcp_config_path) as f:
                mcp_config = json.load(f)
            env_vars = mcp_config.get("mcpServers", {}).get("sysmobench", {}).get("env", {})
            task = env_vars.get("SYSMOBENCH_TASK", "")
            spec_module = env_vars.get("SYSMOBENCH_SPEC_MODULE", "")
            output_dir = env_vars.get("SYSMOBENCH_OUTPUT", "output")
            if task and spec_module:
                update_codex_mcp_config(task, spec_module, output_dir)
        except Exception:
            pass  # If we can't update config, continue anyway

        # Build command
        cmd = [
            "codex",
            "exec",
            "--dangerously-bypass-approvals-and-sandbox",
            "--skip-git-repo-check",
        ]
        # Only add -m if model is specified and not "default"
        if model and model != "default":
            cmd.extend(["-m", model])
        cmd.append(self.config.initial_prompt)

        try:
            # Run in thread pool to avoid blocking event loop
            loop = asyncio.get_event_loop()
            return_code, duration = await loop.run_in_executor(
                None,
                self._run_with_pty,
                cmd,
                workspace_path,
                self.config.timeout,
            )

            if return_code == -1:
                return ExecutionResult(
                    success=False,
                    raw_output="",
                    error=f"Timeout after {self.config.timeout} seconds",
                    duration_seconds=duration,
                    exit_code=-1,
                )

            # Check if output files exist
            output_dir = workspace_path / "output"
            tla_files = list(output_dir.glob("*.tla"))
            has_output = len(tla_files) > 0 and any(
                f.stat().st_size > 0 for f in tla_files
            )

            # Success if output files exist (don't rely on exit code)
            success = has_output

            return ExecutionResult(
                success=success,
                raw_output="",
                parsed_output=None,
                error=None if success else "Codex failed to generate output",
                duration_seconds=duration,
                exit_code=return_code,
            )

        except FileNotFoundError:
            return ExecutionResult(
                success=False,
                raw_output="",
                error="Codex CLI not found. Please ensure 'codex' is installed and in PATH. "
                      "Install with: npm i -g @openai/codex",
                duration_seconds=0.0,
            )
        except Exception as e:
            return ExecutionResult(
                success=False,
                raw_output="",
                error=f"Unexpected error: {str(e)}",
                duration_seconds=0.0,
            )

    def get_adapter_info(self) -> Dict[str, Any]:
        """Return metadata about this adapter."""
        return {
            "name": self.agent_name,
            "type": "code_agent_adapter",
            "cli": "codex",
            "model": self.config.model,
            "timeout": self.config.timeout,
        }
