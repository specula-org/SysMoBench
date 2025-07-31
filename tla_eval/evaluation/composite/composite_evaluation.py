"""
Composite Evaluator: Integrated evaluation combining multiple metrics.

This evaluator implements a comprehensive evaluation pipeline that:
1. Generates TLA+ specification once using agent-based method
2. Performs action decomposition evaluation
3. Performs compilation check evaluation 
4. If compilation succeeds, performs invariant verification (3 iterations)
5. Aggregates all results into a unified composite result
"""

import logging
import time
from typing import Optional

from ...models.base import GenerationResult
from ...utils.output_manager import get_output_manager
from ..base.evaluator import BaseEvaluator
from ..base.result_types import CompositeEvaluationResult
from ..syntax.action_decomposition import ActionDecompositionEvaluator
from ..syntax.compilation_check import CompilationCheckEvaluator
from ..semantics.invariant_verification import InvariantVerificationEvaluator

logger = logging.getLogger(__name__)


class CompositeEvaluator(BaseEvaluator):
    """
    Composite evaluator that runs multiple evaluation metrics in sequence.
    
    This evaluator provides a comprehensive evaluation by combining:
    - Action decomposition (syntax)
    - Compilation check (syntax) 
    - Invariant verification (semantics, conditional on compilation success)
    """
    
    def __init__(self, 
                 validation_timeout: int = 30,
                 invariant_iterations: int = 3,
                 keep_temp_files: bool = False):
        """
        Initialize composite evaluator.
        
        Args:
            validation_timeout: Timeout for TLA+ validation in seconds
            invariant_iterations: Number of invariant verification iterations
            keep_temp_files: Whether to keep temporary files for debugging
        """
        super().__init__(timeout=validation_timeout)
        self.invariant_iterations = invariant_iterations
        self.keep_temp_files = keep_temp_files
        
        # Initialize sub-evaluators
        self.action_evaluator = ActionDecompositionEvaluator(
            validation_timeout=validation_timeout,
            keep_temp_files=keep_temp_files
        )
        self.compilation_evaluator = CompilationCheckEvaluator(
            validation_timeout=validation_timeout
        )
        self.invariant_evaluator = InvariantVerificationEvaluator(
            tlc_timeout=validation_timeout
        )
    
    def evaluate(self, 
                generation_result: GenerationResult,
                task_name: str,
                method_name: str,
                model_name: str,
                spec_module: str = None) -> CompositeEvaluationResult:
        """
        Perform comprehensive evaluation using multiple metrics.
        
        Args:
            generation_result: Result from TLA+ generation
            task_name: Name of the task
            method_name: Name of the generation method
            model_name: Name of the model used
            spec_module: Optional TLA+ module name for the specification
            
        Returns:
            CompositeEvaluationResult with aggregated evaluation results
        """
        logger.info(f"Starting composite evaluation: {task_name}/{method_name}/{model_name}")
        start_time = time.time()
        
        # Create composite result
        composite_result = CompositeEvaluationResult(task_name, method_name, model_name)
        
        # Set generation results
        self._set_generation_result(composite_result, generation_result)
        
        # If generation failed, no need to continue
        if not generation_result.success:
            logger.warning(f"Generation failed, skipping all evaluations: {generation_result.error_message}")
            composite_result.overall_success = False
            return composite_result
        
        try:
            # Step 1: Action Decomposition Evaluation (always run)
            logger.info("Step 1/3: Running action decomposition evaluation...")
            action_result = self.action_evaluator.evaluate(
                generation_result, task_name, method_name, model_name, spec_module
            )
            composite_result.action_decomposition_result = action_result
            logger.info(f"Action decomposition: {'✓ PASS' if action_result.overall_success else '✗ FAIL'}")
            
            # Step 2: Compilation Check Evaluation (always run)
            logger.info("Step 2/3: Running compilation check evaluation...")
            compilation_result = self.compilation_evaluator.evaluate(
                generation_result, task_name, method_name, model_name, spec_module
            )
            composite_result.compilation_check_result = compilation_result
            logger.info(f"Compilation check: {'✓ PASS' if compilation_result.overall_success else '✗ FAIL'}")
            
            # Step 3: Invariant Verification (only if compilation succeeded)
            if compilation_result.compilation_successful:
                logger.info(f"Step 3/3: Running invariant verification ({self.invariant_iterations} iterations)...")
                
                for iteration in range(self.invariant_iterations):
                    logger.info(f"Invariant verification iteration {iteration + 1}/{self.invariant_iterations}")
                    
                    try:
                        inv_result = self.invariant_evaluator.evaluate(
                            generation_result, task_name, method_name, model_name, spec_module
                        )
                        composite_result.invariant_verification_results.append(inv_result)
                        
                        success_status = "✓ PASS" if inv_result.overall_success else "✗ FAIL"
                        logger.info(f"Iteration {iteration + 1}: {success_status}")
                        
                    except Exception as e:
                        logger.error(f"Invariant verification iteration {iteration + 1} failed: {e}")
                        # Continue with remaining iterations even if one fails
                        continue
                
                successful_inv = sum(1 for r in composite_result.invariant_verification_results if r.overall_success)
                total_inv = len(composite_result.invariant_verification_results)
                logger.info(f"Invariant verification summary: {successful_inv}/{total_inv} iterations successful")
            else:
                logger.info("Step 3/3: Skipping invariant verification (compilation failed)")
            
            # Calculate overall success
            composite_result.overall_success = self._calculate_overall_success(composite_result)
            
        except Exception as e:
            logger.error(f"Composite evaluation error: {e}")
            composite_result.overall_success = False
        
        finally:
            # Save composite results to output directory
            try:
                self._save_composite_results(composite_result, task_name, method_name, model_name)
            except Exception as save_error:
                logger.error(f"Failed to save composite results: {save_error}")
        
        end_time = time.time()
        total_time = end_time - start_time
        logger.info(f"Composite evaluation complete: success={composite_result.overall_success}, total_time={total_time:.2f}s")
        
        return composite_result
    
    def _set_generation_result(self, composite_result: CompositeEvaluationResult, generation_result: GenerationResult):
        """Set generation results on composite result"""
        composite_result.generation_successful = generation_result.success
        composite_result.generation_time = generation_result.metadata.get('latency_seconds', 0.0)
        composite_result.generated_specification = generation_result.generated_text
        
        if not generation_result.success:
            composite_result.generation_error = generation_result.error_message
    
    def _calculate_overall_success(self, composite_result: CompositeEvaluationResult) -> bool:
        """
        Calculate overall success based on all sub-evaluation results.
        
        Success criteria:
        - Generation must succeed
        - At least one of action decomposition or compilation check must succeed
        - If invariant verification ran, at least one iteration must succeed
        """
        if not composite_result.generation_successful:
            return False
        
        # Check syntax evaluations
        action_success = (composite_result.action_decomposition_result and 
                         composite_result.action_decomposition_result.overall_success)
        compilation_success = (composite_result.compilation_check_result and 
                              composite_result.compilation_check_result.overall_success)
        
        syntax_success = action_success or compilation_success
        
        # Check invariant verification if it ran
        if composite_result.invariant_verification_results:
            inv_success = any(r.overall_success for r in composite_result.invariant_verification_results)
            return syntax_success and inv_success
        
        return syntax_success
    
    def _save_composite_results(self, composite_result: CompositeEvaluationResult, 
                               task_name: str, method_name: str, model_name: str):
        """Save composite evaluation results to output directory"""
        output_manager = get_output_manager()
        output_dir = output_manager.create_experiment_dir(
            metric="composite",
            task=task_name,
            method=method_name,
            model=model_name
        )
        
        # Prepare comprehensive result data
        result_data = composite_result.to_dict()
        
        # Add summary statistics
        result_data["summary"] = {
            "generation_successful": composite_result.generation_successful,
            "action_decomposition_successful": (
                composite_result.action_decomposition_result.overall_success 
                if composite_result.action_decomposition_result else False
            ),
            "compilation_successful": (
                composite_result.compilation_check_result.overall_success 
                if composite_result.compilation_check_result else False
            ),
            "invariant_verification_iterations": len(composite_result.invariant_verification_results),
            "invariant_verification_successful_iterations": sum(
                1 for r in composite_result.invariant_verification_results if r.overall_success
            ),
            "overall_successful": composite_result.overall_success
        }
        
        metadata = {
            "task_name": task_name,
            "method_name": method_name,
            "model_name": model_name,
            "metric": "composite",
            "evaluation_timestamp": time.time(),
            "validation_timeout": self.timeout,
            "invariant_iterations": self.invariant_iterations,
            "keep_temp_files": self.keep_temp_files
        }
        
        # Save specification to output directory
        if composite_result.generated_specification:
            spec_file_path = output_dir / f"{task_name}.tla"
            with open(spec_file_path, 'w', encoding='utf-8') as f:
                f.write(composite_result.generated_specification)
            metadata["specification_file"] = str(spec_file_path)
        
        output_manager.save_result(output_dir, result_data, metadata)
        logger.info(f"Composite results saved to: {output_dir}")
        
        # Store output directory path in result for display
        composite_result.output_directory = str(output_dir)
    
    def _get_evaluation_type(self) -> str:
        """Return the evaluation type identifier"""
        return "composite"


# Convenience function for backward compatibility
def create_composite_evaluator(validation_timeout: int = 30, 
                              invariant_iterations: int = 3,
                              keep_temp_files: bool = False) -> CompositeEvaluator:
    """
    Factory function to create a composite evaluator.
    
    Args:
        validation_timeout: Timeout for TLA+ validation in seconds
        invariant_iterations: Number of invariant verification iterations
        keep_temp_files: Whether to keep temporary files for debugging
        
    Returns:
        CompositeEvaluator instance
    """
    return CompositeEvaluator(
        validation_timeout=validation_timeout,
        invariant_iterations=invariant_iterations,
        keep_temp_files=keep_temp_files
    )