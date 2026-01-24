"""
Prompt builder for code agent CLAUDE.md files.

This module loads prompts from task-specific code_agent.txt files
and replaces placeholders with actual values.
"""

from pathlib import Path
from typing import List, Optional

from ..base import GenerationTask


class PromptBuilder:
    """
    Builds CLAUDE.md content for code agent tasks.

    Loads prompts from tla_eval/tasks/{task_name}/prompts/code_agent.txt
    and replaces placeholders with task-specific values.
    """

    # Placeholder definitions for reference:
    # {description}   - System description
    # {system_type}   - e.g., "concurrent", "distributed"
    # {language}      - e.g., "rust", "go"
    # {spec_module}   - TLA+ module name
    # {source_files}  - List of source files to read
    # {max_attempts}  - Maximum submission attempts

    def __init__(self):
        """Initialize the prompt builder."""
        self._tasks_dir = Path(__file__).parent.parent.parent / "tasks"

    def build(
        self,
        task: GenerationTask,
        max_attempts: int,
        source_files: Optional[List[str]] = None,
    ) -> str:
        """
        Build CLAUDE.md content for a task.

        Args:
            task: The generation task
            max_attempts: Maximum submission attempts
            source_files: List of source file paths (optional)

        Returns:
            CLAUDE.md content string
        """
        # Load prompt template from file
        template = self._load_prompt_template(task.task_name)

        # Format source files list
        if source_files:
            source_files_str = "\n".join(f"- `{f}`" for f in source_files)
        else:
            source_files_str = "- (Read all files in the source_code directory)"

        # Replace placeholders
        content = template.format(
            description=task.description or f"{task.task_name} system",
            system_type=task.system_type or "concurrent",
            language=task.language or "unknown",
            spec_module=task.spec_module or task.task_name,
            source_files=source_files_str,
            max_attempts=max_attempts,
        )

        return content

    def _load_prompt_template(self, task_name: str) -> str:
        """
        Load prompt template from task-specific file.

        Args:
            task_name: Name of the task

        Returns:
            Prompt template string

        Raises:
            FileNotFoundError: If prompt file doesn't exist
        """
        prompt_path = self._tasks_dir / task_name / "prompts" / "code_agent.txt"

        if not prompt_path.exists():
            raise FileNotFoundError(
                f"Prompt file not found: {prompt_path}\n"
                f"Please create a code_agent.txt prompt for task '{task_name}'"
            )

        return prompt_path.read_text(encoding="utf-8")

    def get_prompt_path(self, task_name: str) -> Path:
        """
        Get the path to the prompt file for a task.

        Args:
            task_name: Name of the task

        Returns:
            Path to the prompt file
        """
        return self._tasks_dir / task_name / "prompts" / "code_agent.txt"

    def prompt_exists(self, task_name: str) -> bool:
        """
        Check if a prompt file exists for a task.

        Args:
            task_name: Name of the task

        Returns:
            True if prompt file exists
        """
        return self.get_prompt_path(task_name).exists()
