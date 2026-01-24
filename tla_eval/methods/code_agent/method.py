"""
Code Agent method for TLA+ specification generation.

This method uses external code agents (like Claude Code) to generate
TLA+ specifications through an agentic workflow with iterative validation.
"""

import asyncio
import logging
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

from ..base import TLAGenerationMethod, GenerationTask, GenerationOutput
from .workspace import TaskWorkspace
from .prompt_builder import PromptBuilder
from .adapters.base import BaseCodeAgentAdapter

logger = logging.getLogger(__name__)


class CodeAgentMethod(TLAGenerationMethod):
    """
    Code Agent-based method for TLA+ generation.

    This method:
    1. Prepares a workspace with task instructions (CLAUDE.md)
    2. Launches a code agent (e.g., Claude Code) in the workspace
    3. The agent reads code, generates specs, and validates using submit_spec tool
    4. Collects the final output from the workspace
    """

    def __init__(
        self,
        adapter: BaseCodeAgentAdapter,
        max_attempts: int = 3,
        output_dir: Optional[str] = None,
        workspace_base: Optional[str] = None,
        keep_workspace: bool = True,
    ):
        """
        Initialize the code agent method.

        Args:
            adapter: Code agent adapter (e.g., ClaudeCodeAdapter)
            max_attempts: Maximum submission attempts for validation
            output_dir: Directory for saving evaluation outputs
            workspace_base: Base directory for workspaces (uses temp if not specified)
            keep_workspace: Whether to keep workspace after execution (for debugging)
        """
        super().__init__(f"code_agent_{adapter.agent_name}")
        self.adapter = adapter
        self.max_attempts = max_attempts
        # Ensure output_dir is absolute for MCP server
        self.output_dir = (Path(output_dir) if output_dir else Path("output")).resolve()
        self.workspace_base = Path(workspace_base).resolve() if workspace_base else None
        self.keep_workspace = keep_workspace

        # Path to submit_spec MCP server
        self.mcp_server_path = self._find_mcp_server_path()

        # Prompt builder
        self.prompt_builder = PromptBuilder()

    def _find_mcp_server_path(self) -> Path:
        """Find the path to submit_spec MCP server."""
        # Try relative to this file
        module_dir = Path(__file__).parent
        candidates = [
            module_dir.parent.parent.parent / "tools" / "submit_spec" / "mcp_server.py",
            Path("tools/submit_spec/mcp_server.py"),
        ]

        for candidate in candidates:
            if candidate.exists():
                return candidate.resolve()

        # Fallback - will need to be configured
        return Path("tools/submit_spec/mcp_server.py")

    def generate(self, task: GenerationTask, model_name: str = None) -> GenerationOutput:
        """
        Generate TLA+ specification using code agent.

        Args:
            task: Generation task with source code info
            model_name: Optional model name (used by adapter if supported)

        Returns:
            GenerationOutput with the generated specification
        """
        return asyncio.run(self._async_generate(task, model_name))

    async def _async_generate(
        self,
        task: GenerationTask,
        model_name: Optional[str] = None,
    ) -> GenerationOutput:
        """
        Async implementation of specification generation.

        Args:
            task: Generation task
            model_name: Optional model name

        Returns:
            GenerationOutput with results
        """
        logger.info(f"Starting code agent generation for task: {task.task_name}")

        # Determine workspace base directory
        if self.workspace_base:
            workspace_base = self.workspace_base
            workspace_base.mkdir(parents=True, exist_ok=True)
        else:
            workspace_base = Path(tempfile.mkdtemp(prefix="sysmobench_"))

        # Create workspace manager
        workspace = TaskWorkspace(base_dir=workspace_base)

        try:
            # Get source code base path
            source_code_base = self._get_source_code_base(task)

            # Get source file list for prompt
            source_files = self._get_source_files(task)

            # Build CLAUDE.md content
            prompt_content = self.prompt_builder.build(
                task=task,
                max_attempts=self.max_attempts,
                source_files=source_files,
            )

            # Prepare workspace
            workspace_path = workspace.prepare(
                task=task,
                max_attempts=self.max_attempts,
                source_code_base=source_code_base,
                prompt_content=prompt_content,
            )

            logger.info(f"Workspace prepared at: {workspace_path}")

            # Create MCP config
            mcp_config_path = workspace.create_mcp_config(
                task=task,
                max_attempts=self.max_attempts,
                output_dir=self.output_dir,
                mcp_server_path=self.mcp_server_path,
            )

            logger.info(f"MCP config created at: {mcp_config_path}")

            # Execute code agent
            logger.info(f"Executing {self.adapter.agent_name}...")
            result = await self.adapter.execute(
                workspace_path=workspace_path,
                mcp_config_path=mcp_config_path,
                model_override=model_name,
            )

            logger.info(f"Agent execution completed. Success: {result.success}, Duration: {result.duration_seconds:.2f}s")

            # Collect output
            spec_content = workspace.get_output_spec()
            config_content = workspace.get_output_config()

            # Build metadata
            metadata = {
                "agent": self.adapter.agent_name,
                "adapter_info": self.adapter.get_adapter_info(),
                "workspace_path": str(workspace_path),
                "max_attempts": self.max_attempts,
                "execution_duration_seconds": result.duration_seconds,
                "exit_code": result.exit_code,
                "has_spec_output": spec_content is not None,
                "has_config_output": config_content is not None,
                # Include config content so evaluator can use it
                "config_content": config_content,
            }

            # Save raw output to workspace for debugging
            if result.raw_output:
                metadata["raw_output_length"] = len(result.raw_output)
                raw_output_file = workspace_path / "claude_output.txt"
                raw_output_file.write_text(result.raw_output, encoding="utf-8")
                metadata["raw_output_file"] = str(raw_output_file)

            # Determine success
            success = spec_content is not None and len(spec_content.strip()) > 0

            return GenerationOutput(
                tla_specification=spec_content or "",
                method_name=self.name,
                task_name=task.task_name,
                metadata=metadata,
                success=success,
                error_message=result.error if not success else None,
            )

        except Exception as e:
            logger.error(f"Code agent generation failed: {e}")
            return GenerationOutput(
                tla_specification="",
                method_name=self.name,
                task_name=task.task_name,
                metadata={"error": str(e)},
                success=False,
                error_message=str(e),
            )

        finally:
            # Cleanup workspace if not keeping
            if not self.keep_workspace:
                workspace.cleanup()
                logger.info("Workspace cleaned up")
            else:
                logger.info(f"Workspace kept at: {workspace.workspace_path}")

    def _get_source_code_base(self, task: GenerationTask) -> Path:
        """
        Get the base path for source code.

        Args:
            task: Generation task

        Returns:
            Path to source code base directory
        """
        module_dir = Path(__file__).parent
        project_root = module_dir.parent.parent.parent
        repositories_dir = project_root / "data" / "repositories"

        # Try to get explicit repository_path from task extra_info
        if task.extra_info:
            repo_path = task.extra_info.get("repository_path")
            if repo_path:
                return Path(repo_path)

            # Derive repo name from repository_url (same logic as task loader)
            repo_url = task.extra_info.get("repository_url")
            if repo_url:
                repo_name = repo_url.split("/")[-1].replace(".git", "")
                repo_path = repositories_dir / repo_name
                if repo_path.exists():
                    return repo_path

        # Fallback: try task name as repo name
        task_repo_path = repositories_dir / task.task_name
        if task_repo_path.exists():
            return task_repo_path

        return repositories_dir

    def _get_source_files(self, task: GenerationTask) -> list:
        """
        Get list of source files from task.

        Args:
            task: Generation task

        Returns:
            List of source file paths
        """
        if not task.extra_info:
            return []

        # Try source_files first
        source_files = task.extra_info.get("source_files", [])
        if source_files:
            return [
                sf.get("path") if isinstance(sf, dict) else str(sf)
                for sf in source_files
            ]

        # Fallback to file_path
        file_path = task.extra_info.get("file_path")
        if file_path:
            if isinstance(file_path, list):
                return file_path
            else:
                return [file_path]

        return []

    def get_method_info(self) -> Dict[str, Any]:
        """Get information about this method."""
        return {
            "name": self.name,
            "description": f"Code agent ({self.adapter.agent_name}) with iterative validation",
            "type": "code_agent",
            "agent": self.adapter.agent_name,
            "max_attempts": self.max_attempts,
            "adapter_info": self.adapter.get_adapter_info(),
            "features": [
                "file_system_access",
                "iterative_validation",
                "mcp_tools",
                "source_code_exploration",
            ],
        }
