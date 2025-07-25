"""
Phase 1 Evaluator: TLA+ Compilation Checking

This evaluator focuses on the first phase of benchmarking - whether the generated
TLA+ specifications can be compiled successfully using the TLA tools (SANY).
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import logging

from ...core.verification.validators import TLAValidator, ValidationResult
from ...models.base import GenerationResult

logger = logging.getLogger(__name__)


class Phase1EvaluationResult:
    """Result of Phase 1 evaluation (compilation checking)"""
    
    def __init__(self, task_name: str, method_name: str, model_name: str):
        self.task_name = task_name
        self.method_name = method_name
        self.model_name = model_name
        self.timestamp = time.time()
        
        # Generation results
        self.generation_successful = False
        self.generation_time = 0.0
        self.generation_error = None
        self.generated_specification = None
        
        # Validation results
        self.compilation_successful = False
        self.compilation_time = 0.0
        self.syntax_errors = []
        self.semantic_errors = []
        self.compilation_output = ""
        
        # Legacy compatibility
        self.compilation_errors = []
        
        # Overall metrics
        self.overall_success = False  # Both generation and compilation successful
    
    def set_generation_result(self, generation_result: GenerationResult):
        """Set generation results"""
        self.generation_successful = generation_result.success
        self.generation_time = generation_result.metadata.get('latency_seconds', 0.0)
        self.generated_specification = generation_result.generated_text
        
        if not generation_result.success:
            self.generation_error = generation_result.error_message
    
    def set_validation_result(self, validation_result: ValidationResult):
        """Set validation results"""
        self.compilation_successful = validation_result.success
        self.compilation_time = validation_result.compilation_time
        self.syntax_errors = validation_result.syntax_errors
        self.semantic_errors = validation_result.semantic_errors
        self.compilation_output = validation_result.output
        
        # Legacy compatibility
        self.compilation_errors = validation_result.errors
        
        # Overall success requires both generation and compilation to succeed
        self.overall_success = self.generation_successful and self.compilation_successful
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary for serialization"""
        return {
            "task_name": self.task_name,
            "method_name": self.method_name,
            "model_name": self.model_name,
            "timestamp": self.timestamp,
            "generation": {
                "successful": self.generation_successful,
                "time_seconds": self.generation_time,
                "error": self.generation_error,
                "specification_length": len(self.generated_specification) if self.generated_specification else 0
            },
            "compilation": {
                "successful": self.compilation_successful,
                "time_seconds": self.compilation_time,
                "syntax_errors": self.syntax_errors,
                "semantic_errors": self.semantic_errors,
                "syntax_error_count": len(self.syntax_errors),
                "semantic_error_count": len(self.semantic_errors),
                "total_error_count": len(self.compilation_errors),
                "output_length": len(self.compilation_output)
            },
            "overall": {
                "successful": self.overall_success,
                "total_time_seconds": self.generation_time + self.compilation_time
            }
        }


class Phase1Evaluator:
    """
    Phase 1 Evaluator for TLA+ specification generation benchmarking.
    
    This evaluator checks whether generated TLA+ specifications can be
    compiled successfully using the TLA tools (SANY parser).
    """
    
    def __init__(self, validation_timeout: int = 30):
        """
        Initialize Phase 1 evaluator.
        
        Args:
            validation_timeout: Timeout for TLA+ validation in seconds
        """
        self.validation_timeout = validation_timeout
        self.validator = TLAValidator(timeout=validation_timeout)
        
        logger.info(f"Phase 1 Evaluator initialized with {validation_timeout}s timeout")
    
    def evaluate_generation(self, 
                          generation_result: GenerationResult,
                          task_name: str,
                          method_name: str,
                          model_name: str,
                          spec_module: str = None) -> Phase1EvaluationResult:
        """
        Evaluate a single generation result.
        
        Args:
            generation_result: Result from TLA+ generation
            task_name: Name of the task
            method_name: Name of the generation method
            model_name: Name of the model used
            spec_module: Optional TLA+ module name for the specification
            
        Returns:
            Phase1EvaluationResult with evaluation metrics
        """
        logger.info(f"Evaluating generation: {task_name}/{method_name}/{model_name}")
        
        # Create evaluation result
        eval_result = Phase1EvaluationResult(task_name, method_name, model_name)
        eval_result.set_generation_result(generation_result)
        
        # If generation failed, no need to validate
        if not generation_result.success:
            logger.warning(f"Generation failed, skipping validation: {generation_result.error_message}")
            # Create a dummy validation result
            validation_result = ValidationResult(
                success=False,
                output="Generation failed - no specification to validate",
                syntax_errors=[],
                semantic_errors=["Generation failed"],
                compilation_time=0.0
            )
            eval_result.set_validation_result(validation_result)
            return eval_result
        
        # Validate the generated specification
        try:
            logger.debug("Starting TLA+ specification validation...")
            validation_result = self.validator.validate_specification(
                generation_result.generated_text,
                module_name=spec_module,
                task_name=task_name
            )
            eval_result.set_validation_result(validation_result)
            
            if validation_result.success:
                logger.info("✓ Specification compiled successfully")
            else:
                logger.warning(f"✗ Compilation failed with {len(validation_result.errors)} errors")
                
        except Exception as e:
            logger.error(f"Validation error: {e}")
            # Create error validation result
            validation_result = ValidationResult(
                success=False,
                output=f"Validation error: {e}",
                syntax_errors=[],
                semantic_errors=[str(e)],
                compilation_time=0.0
            )
            eval_result.set_validation_result(validation_result)
        
        logger.info(f"Evaluation complete: success={eval_result.overall_success}")
        return eval_result
    
    def evaluate_batch(self, 
                      results: List[Tuple[GenerationResult, str, str, str]]) -> List[Phase1EvaluationResult]:
        """
        Evaluate multiple generation results in batch.
        
        Args:
            results: List of tuples (generation_result, task_name, method_name, model_name)
            
        Returns:
            List of Phase1EvaluationResult
        """
        logger.info(f"Starting batch evaluation of {len(results)} results")
        
        evaluation_results = []
        
        for i, (generation_result, task_name, method_name, model_name) in enumerate(results):
            logger.info(f"Processing batch item {i+1}/{len(results)}")
            
            try:
                eval_result = self.evaluate_generation(
                    generation_result, task_name, method_name, model_name
                )
                evaluation_results.append(eval_result)
                
            except Exception as e:
                logger.error(f"Batch evaluation error for item {i+1}: {e}")
                # Create error result
                eval_result = Phase1EvaluationResult(task_name, method_name, model_name)
                eval_result.generation_error = f"Batch evaluation error: {e}"
                evaluation_results.append(eval_result)
        
        logger.info(f"Batch evaluation complete: {len(evaluation_results)} results")
        return evaluation_results
    
    def save_results(self, 
                    results: List[Phase1EvaluationResult], 
                    output_file: str,
                    include_specifications: bool = False):
        """
        Save evaluation results to JSON file.
        
        Args:
            results: List of evaluation results
            output_file: Output file path
            include_specifications: Whether to include generated specifications in output
        """
        logger.info(f"Saving {len(results)} results to {output_file}")
        
        # Convert results to dictionaries
        data = []
        for result in results:
            result_dict = result.to_dict()
            
            # Optionally include the generated specification
            if include_specifications and result.generated_specification:
                result_dict["generated_specification"] = result.generated_specification
            
            data.append(result_dict)
        
        # Add summary statistics
        summary = self._calculate_summary(results)
        
        output_data = {
            "evaluation_type": "phase1_compilation",
            "total_evaluations": len(results),
            "timestamp": time.time(),
            "summary": summary,
            "results": data
        }
        
        # Save to file
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Results saved to {output_path}")
    
    def _calculate_summary(self, results: List[Phase1EvaluationResult]) -> Dict[str, Any]:
        """Calculate summary statistics from evaluation results"""
        if not results:
            return {}
        
        total = len(results)
        generation_success = sum(1 for r in results if r.generation_successful)
        compilation_success = sum(1 for r in results if r.compilation_successful)
        overall_success = sum(1 for r in results if r.overall_success)
        
        # Time statistics
        generation_times = [r.generation_time for r in results if r.generation_time > 0]
        compilation_times = [r.compilation_time for r in results if r.compilation_time > 0]
        
        # Error statistics
        total_compilation_errors = sum(len(r.compilation_errors) for r in results)
        
        summary = {
            "total_evaluations": total,
            "success_rates": {
                "generation": generation_success / total if total > 0 else 0.0,
                "compilation": compilation_success / total if total > 0 else 0.0,
                "overall": overall_success / total if total > 0 else 0.0
            },
            "counts": {
                "generation_successful": generation_success,
                "compilation_successful": compilation_success,
                "overall_successful": overall_success
            },
            "timing": {
                "avg_generation_time": sum(generation_times) / len(generation_times) if generation_times else 0.0,
                "avg_compilation_time": sum(compilation_times) / len(compilation_times) if compilation_times else 0.0,
                "max_generation_time": max(generation_times) if generation_times else 0.0,
                "max_compilation_time": max(compilation_times) if compilation_times else 0.0
            },
            "errors": {
                "total_compilation_errors": total_compilation_errors,
                "avg_errors_per_evaluation": total_compilation_errors / total if total > 0 else 0.0
            }
        }
        
        return summary


def create_phase1_evaluator(validation_timeout: int = 30) -> Phase1Evaluator:
    """
    Factory function to create a Phase 1 evaluator.
    
    Args:
        validation_timeout: Timeout for TLA+ validation in seconds
        
    Returns:
        Phase1Evaluator instance
    """
    return Phase1Evaluator(validation_timeout=validation_timeout)