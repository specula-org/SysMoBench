"""
Trace-based method implementation with automatic error correction.

This method generates TLA+ specifications and automatically detects and corrects
syntax and semantic errors using iterative LLM feedback.
"""

import logging
from random import sample
import time
from typing import Dict, Any
from pathlib import Path

from ..base import TLAGenerationMethod, GenerationTask, GenerationOutput
from ...config import get_configured_model
from ...core.verification.validators import TLAValidator, ValidationResult
from ...models.base import GenerationConfig

logger = logging.getLogger(__name__)


class TraceBasedMethod(TLAGenerationMethod):
    """
    Trace-based method for TLA+ generation with automatic error correction, based on the AgentBasedMethod.
    
    This method implements the same feedback loop, but provides the traces as input instead of the codebase.
    """
    
    def __init__(self, max_correction_attempts: int = 3, validation_timeout: int = 30):
        """
        Initialize trace-based method.
        
        Args:
            max_correction_attempts: Maximum number of correction attempts
            validation_timeout: Timeout for TLA+ validation operations
        """
        super().__init__("trace_based")
        self.max_correction_attempts = max_correction_attempts
        self.validation_timeout = validation_timeout
        self.validator = TLAValidator(timeout=validation_timeout)
        
    def generate(self, task: GenerationTask, model_name: str = None) -> GenerationOutput:
        """
        Generate TLA+ specification with automatic error correction.
        
        Args:
            task: Generation task with traces 
            model_name: Model to use from config
            
        Returns:
            GenerationOutput with corrected TLA+ specification
        """
        logger.info(f"Starting trace-based generation for task: {task.task_name}")
        
        try:
            # Get configured model
            model = get_configured_model(model_name)
            logger.info(f"Using model: {model.model_name}")
            
            # Step 1: Initial generation
            logger.info("Step 1: Initial TLA+ generation")
            initial_result = self._initial_generation(task, model)
            
            if not initial_result.success:
                logger.error(f"Initial generation failed: {initial_result.error_message}")
                return GenerationOutput(
                    tla_specification="",
                    method_name=self.name,
                    task_name=task.task_name,
                    metadata={"initial_generation_failed": True},
                    success=False,
                    error_message=initial_result.error_message
                )
            
            # Skip correction loop - return initial specification directly
            logger.info("Step 2: Skipping internal correction loop (composite evaluator will handle corrections)")
            
            # Compile metadata (no correction metadata)
            total_generation_time = initial_result.metadata.get('latency_seconds', 0)
            
            metadata = {
                "model_info": model.get_model_info(),
                "initial_generation_metadata": initial_result.metadata,
                "total_generation_time": total_generation_time,
                "method_type": "trace_based_no_internal_correction",
                "internal_correction_skipped": True
            }
            
            # Return initial specification without internal correction
            return GenerationOutput(
                tla_specification=initial_result.generated_text,
                method_name=self.name,
                task_name=task.task_name,
                metadata=metadata,
                success=True,
                error_message=None
            )
            
        except Exception as e:
            logger.error(f"Trace-based generation failed with exception: {e}")
            return GenerationOutput(
                tla_specification="",
                method_name=self.name,
                task_name=task.task_name,
                metadata={},
                success=False,
                error_message=str(e)
            )
    
    def _initial_generation(self, task: GenerationTask, model) -> Any:
        """Generate initial TLA+ specification using standard prompt."""
        prompt = self._create_initial_prompt(task)
        
        # Create generation config from model's YAML configuration
        generation_config = GenerationConfig(
            max_tokens=model.config.get('max_tokens'),
            temperature=model.config.get('temperature'),
            top_p=model.config.get('top_p')  # Only if defined in YAML
        )
        
        logger.info(f"Initial generation config from YAML: {model.config}")
        logger.debug(f"Using initial prompt ({len(prompt)} chars)")
        return model.generate_tla_specification("", prompt, generation_config) # set source code to empty, we'll process traces in the prompt

    def _create_initial_prompt(self, task: GenerationTask) -> str:
        """Create initial generation prompt."""
        from ...tasks.loader import get_task_loader
        task_loader = get_task_loader()
        prompt_template = task_loader.get_task_prompt(task.task_name, "trace_based") 

        trace_format = task.extra_info.get("trace_format")
        if not trace_format:
            raise ValueError("trace_format must be provided in task.extra_info for trace_based method")

        traces = task.traces
        if trace_format == "etcd_based":
            traces = sample(traces, 3) # etcd traces are large, so sample a few randomly to avoid overflowing context

        trace_str = ""
        for i, distributed_trace in enumerate(task.traces):
            if isinstance(distributed_trace, list):
                trace_str += f"## Execution #{i+1}:\n"
                for trace_name, trace_content in distributed_trace:
                    trace_str += f"{trace_name}:\n```\n{trace_content}\n```\n"
                trace_str += "\n"
            elif isinstance(distributed_trace, tuple):
                trace_name, trace_content = distributed_trace
                trace_str += f"## Execution #{i+1}:\n{trace_name}:\n```\n{trace_content}\n```\n\n"

        trace_format_file = Path(f"data/trace_based/{trace_format}.txt")
        trace_format_info = trace_format_file.read_text(encoding='utf-8')
        
        # Prepare format variables
        format_vars = {
            'language': task.language,
            'description': task.description,
            'system_type': task.system_type,
            'traces': trace_str,
            'trace_format': trace_format_info,
        }
        
        # Add extra info if available
        if task.extra_info:
            format_vars.update(task.extra_info)
        
        # Format template with task information
        return prompt_template.format(**format_vars)

    def _generate_correction(self, task, current_spec: str, all_errors: list, model_obj):
        from ..agent_based import AgentBasedMethod
        return AgentBasedMethod._generate_correction(self, task, current_spec, all_errors, model_obj)

    
    
    def get_method_info(self) -> Dict[str, Any]:
        """Get information about trace-based method."""
        return {
            "name": self.name,
            "description": "Trace-based LLM generation with automatic error correction",
            "type": "iterative_correction",
            "requires_model": True,
            "supports_iteration": True,
            "max_correction_attempts": self.max_correction_attempts,
            "validation_timeout": self.validation_timeout,
            "features": [
                "automatic_error_detection",
                "iterative_correction",
                "syntax_validation",
                "semantic_validation"
            ]
        }