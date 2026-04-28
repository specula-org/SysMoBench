"""
Base class for TLA+ generation methods.

This module defines the interface that all TLA+ generation methods must implement.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any
from dataclasses import dataclass


@dataclass
class GenerationTask:
    """A TLA+ generation task with source code and metadata."""
    source_code: str
    task_name: str
    system_type: str  # e.g., "distributed", "concurrent"
    language: str     # source code language, e.g., "go", "java", "c++"
    description: str
    spec_module: str = None  # TLA+ module name for the specification
    extra_info: Dict[str, Any] = None  # Additional task-specific information

    def __post_init__(self):
        if self.extra_info is None:
            self.extra_info = {}


@dataclass  
class GenerationOutput:
    """Output from TLA+ generation method."""
    tla_specification: str
    method_name: str
    task_name: str
    metadata: Dict[str, Any]  # Method-specific metadata
    success: bool = True
    error_message: str = None


class TLAGenerationMethod(ABC):
    """
    Abstract base class for TLA+ generation methods.
    
    All generation methods must inherit from this class.
    """
    
    def __init__(self, name: str):
        """
        Initialize the generation method.
        
        Args:
            name: Name of this method
        """
        self.name = name
    
    @abstractmethod
    def generate(self, task: GenerationTask, model_name: str = None) -> GenerationOutput:
        """Generate a TLA+ specification from `task.source_code` using the configured model."""


def format_prompt_template(prompt_template: str, format_vars: Dict[str, Any]) -> str:
    """
    Safely format a prompt template without requiring callers to escape brace characters.

    This replaces `{field}` placeholders using simple string replacement so that literal
    braces used in prompt instructions (e.g., Alloy/TLA+ snippets) are preserved.
    """
    formatted_prompt = prompt_template
    for key, value in format_vars.items():
        placeholder = f"{{{key}}}"
        if placeholder in formatted_prompt:
            formatted_prompt = formatted_prompt.replace(placeholder, str(value))
    return formatted_prompt
