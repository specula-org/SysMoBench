"""
Task workspace management for code agents.

This module handles the creation and management of workspace directories
where code agents perform their work.
"""

import json
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

from ..base import GenerationTask


class TaskWorkspace:
    """
    Manages the workspace directory for a code agent task.

    The workspace structure:
        workspace_{task}_{timestamp}/
        ├── CLAUDE.md              # Task instructions
        ├── mcp_config.json        # MCP server configuration
        ├── source_code/           # Symlinks to source files
        │   └── ...
        └── output/                # Agent writes spec here
            └── {spec_module}.tla
            └── {spec_module}.cfg
    """

    def __init__(self, base_dir: Path):
        """
        Initialize workspace manager.

        Args:
            base_dir: Base directory for creating workspaces
        """
        self.base_dir = Path(base_dir)
        self.workspace_path: Optional[Path] = None
        self.task: Optional[GenerationTask] = None

    def prepare(
        self,
        task: GenerationTask,
        max_attempts: int,
        source_code_base: Path,
        prompt_content: str,
        instruction_filename: str = "CLAUDE.md",
    ) -> Path:
        """
        Prepare workspace directory for a task.

        Args:
            task: The generation task
            max_attempts: Maximum submission attempts allowed
            source_code_base: Base path for source code repository
            prompt_content: Content of instruction file (task instructions)
            instruction_filename: Name of the instruction file (default: CLAUDE.md)
                                  Use GEMINI.md for Gemini CLI

        Returns:
            Path to the prepared workspace
        """
        self.task = task
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.workspace_path = self.base_dir / f"workspace_{task.task_name}_{timestamp}"
        self.workspace_path.mkdir(parents=True, exist_ok=True)

        # Create directories
        source_dir = self.workspace_path / "source_code"
        source_dir.mkdir(exist_ok=True)
        output_dir = self.workspace_path / "output"
        output_dir.mkdir(exist_ok=True)

        # Create symlinks to source files
        self._link_source_files(task, source_dir, source_code_base)

        # Write instruction file (CLAUDE.md, CODEX.md, GEMINI.md, etc.)
        (self.workspace_path / instruction_filename).write_text(prompt_content, encoding="utf-8")

        return self.workspace_path

    def _link_source_files(
        self,
        task: GenerationTask,
        source_dir: Path,
        source_code_base: Path,
    ) -> None:
        """
        Create symlinks to source code files.

        Args:
            task: The generation task containing source file info
            source_dir: Directory to create symlinks in
            source_code_base: Base path for source code
        """
        if not task.extra_info:
            return

        # Get source files - handle both "source_files" list and "file_path" string/list
        source_files = task.extra_info.get("source_files", [])

        if not source_files:
            # Fallback to file_path field (used by task loader)
            file_path = task.extra_info.get("file_path")
            if file_path:
                if isinstance(file_path, list):
                    source_files = file_path
                else:
                    source_files = [file_path]

        for source_file in source_files:
            if isinstance(source_file, dict):
                file_path = source_file.get("path", "")
            else:
                file_path = str(source_file)

            if not file_path:
                continue

            # Construct full path
            full_source_path = source_code_base / file_path

            if full_source_path.exists():
                # Create symlink with just the filename
                link_name = Path(file_path).name
                link_path = source_dir / link_name

                # Remove existing link if present
                if link_path.exists() or link_path.is_symlink():
                    link_path.unlink()

                # Create relative symlink
                try:
                    # Use relative path for portability
                    rel_path = os.path.relpath(full_source_path, source_dir)
                    link_path.symlink_to(rel_path)
                except OSError:
                    # Fallback to absolute path if relative fails
                    link_path.symlink_to(full_source_path)

    def create_mcp_config(
        self,
        task: GenerationTask,
        max_attempts: int,
        output_dir: Path,
        mcp_server_path: Path,
    ) -> Path:
        """
        Create MCP configuration file for the workspace.

        Args:
            task: The generation task
            max_attempts: Maximum submission attempts
            output_dir: Directory for saving evaluation outputs
            mcp_server_path: Path to submit_spec MCP server script

        Returns:
            Path to the created MCP config file
        """
        if self.workspace_path is None:
            raise RuntimeError("Workspace not prepared. Call prepare() first.")

        mcp_config = {
            "mcpServers": {
                "sysmobench": {
                    "command": "python",
                    "args": [str(mcp_server_path)],
                    "env": {
                        "SYSMOBENCH_TASK": task.task_name,
                        "SYSMOBENCH_SPEC_MODULE": task.spec_module or task.task_name,
                        "SYSMOBENCH_MAX_ATTEMPTS": str(max_attempts),
                        "SYSMOBENCH_OUTPUT": str(output_dir),
                    },
                }
            }
        }

        config_path = self.workspace_path / "mcp_config.json"
        config_path.write_text(json.dumps(mcp_config, indent=2), encoding="utf-8")

        return config_path

    def get_output_spec(self) -> Optional[str]:
        """
        Read the generated specification from output directory.

        Returns:
            Specification content if found, None otherwise
        """
        if self.workspace_path is None:
            return None

        output_dir = self.workspace_path / "output"

        # Try to find .tla file
        tla_files = list(output_dir.glob("*.tla"))
        if tla_files:
            return tla_files[0].read_text(encoding="utf-8")

        return None

    def get_output_config(self) -> Optional[str]:
        """
        Read the generated TLC config from output directory.

        Returns:
            Config content if found, None otherwise
        """
        if self.workspace_path is None:
            return None

        output_dir = self.workspace_path / "output"

        # Try to find .cfg file
        cfg_files = list(output_dir.glob("*.cfg"))
        if cfg_files:
            return cfg_files[0].read_text(encoding="utf-8")

        return None

    def cleanup(self) -> None:
        """Remove the workspace directory."""
        if self.workspace_path and self.workspace_path.exists():
            shutil.rmtree(self.workspace_path)
            self.workspace_path = None

    @property
    def source_code_dir(self) -> Optional[Path]:
        """Get the source code directory path."""
        if self.workspace_path:
            return self.workspace_path / "source_code"
        return None

    @property
    def output_dir(self) -> Optional[Path]:
        """Get the output directory path."""
        if self.workspace_path:
            return self.workspace_path / "output"
        return None
