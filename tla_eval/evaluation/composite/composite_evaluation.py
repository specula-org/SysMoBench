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
                spec_module: str = None,
                task=None,
                method=None) -> CompositeEvaluationResult:
        """
        Perform comprehensive evaluation using multiple metrics with iterative improvement.
        
        Process:
        1. Use provided initial TLA+ specification
        2. Action decomposition: evaluate only (no correction)
        3. Compilation check: evaluate → fix if failed (global max 3 corrections)
        4. Invariant verification: evaluate → fix if failed (global max 3 corrections)
        
        Args:
            generation_result: Initial generation result to use as starting point
            task_name: Name of the task
            method_name: Name of the generation method
            model_name: Name of the model used
            spec_module: Optional TLA+ module name for the specification
            task: Task object (for potential corrections)
            method: Method object (for potential corrections)
            
        Returns:
            CompositeEvaluationResult with iterative evaluation results
        """
        logger.info(f"Starting composite evaluation: {task_name}/{method_name}/{model_name}")
        logger.info(f"Process: Initial specification → Action decomposition (eval only) → Compilation check (eval+fix) → Invariant verification (eval+fix)")
        start_time = time.time()
        
        # Create composite result
        composite_result = CompositeEvaluationResult(task_name, method_name, model_name)
        
        # Use the provided generation result as starting point
        if not generation_result.success:
            logger.warning(f"Initial generation failed, cannot proceed with composite evaluation: {generation_result.error_message}")
            composite_result.overall_success = False
            composite_result.generation_error = generation_result.error_message
            return composite_result
        
        # Set initial generation results
        composite_result.generation_successful = generation_result.success
        composite_result.generation_time = generation_result.metadata.get('latency_seconds', 0.0)
        composite_result.generated_specification = generation_result.generated_text
        
        # Current working specification (will be iteratively improved)
        current_spec = generation_result.generated_text
        current_generation_result = generation_result
        
        # Global correction counter (shared across all phases)
        global_correction_attempts = 0
        max_global_corrections = 3
        
        # Track results for each iteration
        action_results_history = []
        compilation_success_round = None
        invariant_success_round = None
        
        try:
            # Step 1: Initial Action Decomposition evaluation
            logger.info(f"Step 1/3: Action decomposition evaluation (Round 1/3)")
            
            try:
                # Evaluate initial specification
                action_result = self.action_evaluator.evaluate(
                    current_generation_result, task_name, method_name, model_name, spec_module
                )
                
                action_results_history.append({
                    'round': 1,
                    'result': action_result,
                    'successful_actions': getattr(action_result, 'successful_actions', 0),
                    'total_actions': getattr(action_result, 'total_actions', 0),
                    'success_rate': getattr(action_result, 'action_success_rate', 0.0)
                })
                
                success_rate = getattr(action_result, 'action_success_rate', 0.0) * 100
                success_status = "✓ PASS" if action_result.overall_success else "✗ FAIL"
                logger.info(f"Action decomposition Round 1: {success_status} ({success_rate:.1f}% actions successful)")
                
                # Store initial action decomposition result
                composite_result.action_decomposition_result = action_result
                
            except Exception as e:
                logger.error(f"Action decomposition evaluation failed: {e}")
                # Create a failed result to continue with next phases
                from ..base.result_types import SyntaxEvaluationResult
                action_result = SyntaxEvaluationResult(task_name, method_name, model_name)
                action_result.overall_success = False
                action_result.generation_error = str(e)
                action_results_history.append({
                    'round': 1,
                    'result': action_result,
                    'successful_actions': 0,
                    'total_actions': 0,
                    'success_rate': 0.0,
                    'error': str(e)
                })
                composite_result.action_decomposition_result = action_result
            
            # Step 2: Compilation Check - Evaluation and correction if needed
            logger.info(f"Step 2/3: Compilation check evaluation")
            
            # Check for consistency: if Action failed but need to check Compilation
            action_failed = not composite_result.action_decomposition_result.overall_success
            
            try:
                # Evaluate current specification
                compilation_result = self.compilation_evaluator.evaluate(
                    current_generation_result, task_name, method_name, model_name, spec_module
                )
                
                success_status = "✓ PASS" if compilation_result.overall_success else "✗ FAIL"
                logger.info(f"Compilation check result: {success_status}")
                
                # Check for unexpected success case
                if action_failed and compilation_result.overall_success:
                    logger.error("✗ CONSISTENCY ERROR: Action decomposition failed but compilation succeeded - this violates expectations!")
                    composite_result.overall_success = False
                    composite_result.compilation_check_result = compilation_result
                    return composite_result
                
                # Store initial compilation result
                composite_result.compilation_check_result = compilation_result
                
                # If compilation failed, attempt corrections
                if not compilation_result.overall_success:
                    logger.info(f"Compilation failed, attempting corrections (max {max_global_corrections - global_correction_attempts} remaining)")
                    
                    while global_correction_attempts < max_global_corrections:
                        if task is not None and method is not None and hasattr(method, '_generate_correction'):
                            logger.info(f"Correction attempt {global_correction_attempts + 1}/{max_global_corrections}")
                            all_errors = compilation_result.syntax_errors + compilation_result.semantic_errors
                            logger.info(f"Errors to fix: {len(all_errors)}")
                            
                            try:
                                # Get the model for correction
                                from ...config import get_configured_model
                                model_obj = get_configured_model(model_name)
                                
                                # Use agent_based's correction method
                                correction_result = method._generate_correction(task, current_spec, all_errors, model_obj)
                                global_correction_attempts += 1
                                
                                if correction_result.success:
                                    current_spec = correction_result.generated_text
                                    current_generation_result = GenerationResult(
                                        generated_text=current_spec,
                                        metadata=correction_result.metadata,
                                        timestamp=time.time(),
                                        success=True
                                    )
                                    logger.info(f"✓ Specification corrected (attempt {global_correction_attempts})")
                                    
                                    # Re-evaluate Action Decomposition with corrected spec
                                    if global_correction_attempts <= 2:  # Only do this for first 2 corrections to get 3 rounds total
                                        round_num = global_correction_attempts + 1
                                        logger.info(f"Re-evaluating Action Decomposition (Round {round_num}/3)")
                                        
                                        try:
                                            action_result_new = self.action_evaluator.evaluate(
                                                current_generation_result, task_name, method_name, model_name, spec_module
                                            )
                                            
                                            action_results_history.append({
                                                'round': round_num,
                                                'result': action_result_new,
                                                'successful_actions': getattr(action_result_new, 'successful_actions', 0),
                                                'total_actions': getattr(action_result_new, 'total_actions', 0),
                                                'success_rate': getattr(action_result_new, 'action_success_rate', 0.0)
                                            })
                                            
                                            success_rate = getattr(action_result_new, 'action_success_rate', 0.0) * 100
                                            success_status = "✓ PASS" if action_result_new.overall_success else "✗ FAIL"
                                            logger.info(f"Action decomposition Round {round_num}: {success_status} ({success_rate:.1f}% actions successful)")
                                            
                                        except Exception as action_error:
                                            logger.error(f"Action decomposition re-evaluation failed: {action_error}")
                                            action_results_history.append({
                                                'round': round_num,
                                                'result': None,
                                                'successful_actions': 0,
                                                'total_actions': 0,
                                                'success_rate': 0.0,
                                                'error': str(action_error)
                                            })
                                    
                                    # Re-evaluate with corrected spec
                                    compilation_result = self.compilation_evaluator.evaluate(
                                        current_generation_result, task_name, method_name, model_name, spec_module
                                    )
                                    composite_result.compilation_check_result = compilation_result
                                    
                                    if compilation_result.overall_success:
                                        compilation_success_round = global_correction_attempts
                                        logger.info(f"✓ Compilation check passed after correction (Round {compilation_success_round})")
                                        break
                                else:
                                    logger.warning(f"✗ Correction attempt {global_correction_attempts} failed: {correction_result.error_message}")
                                    break
                                    
                            except Exception as fix_error:
                                logger.error(f"Error during correction attempt {global_correction_attempts + 1}: {fix_error}")
                                global_correction_attempts += 1
                                break
                        else:
                            logger.warning(f"Cannot perform corrections - agent_based method not available")
                            break
                
            except Exception as e:
                logger.error(f"Compilation check evaluation failed: {e}")
                # Create a failed result to continue with next phases
                from ..base.result_types import SyntaxEvaluationResult
                compilation_result = SyntaxEvaluationResult(task_name, method_name, model_name)
                compilation_result.overall_success = False
                compilation_result.generation_error = str(e)
                composite_result.compilation_check_result = compilation_result
            
            # Step 3: Invariant Verification - Only if both previous phases passed
            action_passed = composite_result.action_decomposition_result.overall_success
            compilation_passed = composite_result.compilation_check_result.overall_success
            
            if action_passed and compilation_passed:
                logger.info(f"Step 3/3: Invariant verification (both previous phases passed)")
                
                try:
                    # Evaluate current specification
                    inv_result = self.invariant_evaluator.evaluate(
                        current_generation_result, task_name, method_name, model_name, spec_module
                    )
                    composite_result.invariant_verification_results.append(inv_result)
                    
                    success_status = "✓ PASS" if inv_result.overall_success else "✗ FAIL"
                    logger.info(f"Invariant verification result: {success_status}")
                    
                    # Record success if first attempt succeeds
                    if inv_result.overall_success:
                        invariant_success_round = 0  # Success on first attempt (before any corrections)
                    
                    # If failed, attempt corrections using remaining global attempts
                    if not inv_result.overall_success:
                        logger.info(f"Invariant verification failed, attempting corrections (max {max_global_corrections - global_correction_attempts} remaining)")
                        
                        while global_correction_attempts < max_global_corrections:
                            if task is not None and method is not None and hasattr(method, '_generate_correction'):
                                logger.info(f"Correction attempt {global_correction_attempts + 1}/{max_global_corrections}")
                                
                                # Collect all errors from invariant verification
                                all_errors = []
                                if hasattr(inv_result, 'invariant_generation_error') and inv_result.invariant_generation_error:
                                    all_errors.append(f"Invariant generation: {inv_result.invariant_generation_error}")
                                if hasattr(inv_result, 'config_generation_error') and inv_result.config_generation_error:
                                    all_errors.append(f"Config generation: {inv_result.config_generation_error}")
                                if hasattr(inv_result, 'model_checking_error') and inv_result.model_checking_error:
                                    all_errors.append(f"Model checking: {inv_result.model_checking_error}")
                                if hasattr(inv_result, 'invariant_violations') and inv_result.invariant_violations:
                                    all_errors.extend([f"Violation: {v}" for v in inv_result.invariant_violations])
                                
                                logger.info(f"Errors to fix: {len(all_errors)}")
                                
                                try:
                                    # Get the model for correction
                                    from ...config import get_configured_model
                                    model_obj = get_configured_model(model_name)
                                    
                                    # Use agent_based's correction method
                                    correction_result = method._generate_correction(task, current_spec, all_errors, model_obj)
                                    global_correction_attempts += 1
                                    
                                    if correction_result.success:
                                        current_spec = correction_result.generated_text
                                        current_generation_result = GenerationResult(
                                            generated_text=current_spec,
                                            metadata=correction_result.metadata,
                                            timestamp=time.time(),
                                            success=True
                                        )
                                        logger.info(f"✓ Specification corrected (attempt {global_correction_attempts})")
                                        
                                        # Re-evaluate with corrected spec
                                        inv_result = self.invariant_evaluator.evaluate(
                                            current_generation_result, task_name, method_name, model_name, spec_module
                                        )
                                        composite_result.invariant_verification_results[-1] = inv_result
                                        
                                        if inv_result.overall_success:
                                            invariant_success_round = global_correction_attempts
                                            logger.info(f"✓ Invariant verification passed after correction (Round {invariant_success_round})")
                                            break
                                    else:
                                        logger.warning(f"✗ Correction attempt {global_correction_attempts} failed: {correction_result.error_message}")
                                        break
                                        
                                except Exception as fix_error:
                                    logger.error(f"Error during correction attempt {global_correction_attempts + 1}: {fix_error}")
                                    global_correction_attempts += 1
                                    break
                            else:
                                logger.warning(f"Cannot perform corrections - agent_based method not available")
                                break
                    
                except Exception as e:
                    logger.error(f"Invariant verification evaluation failed: {e}")
                    # Create a failed result
                    from ..base.result_types import SemanticEvaluationResult
                    inv_result = SemanticEvaluationResult(task_name, method_name, model_name)
                    inv_result.overall_success = False
                    inv_result.error_message = str(e)
                    composite_result.invariant_verification_results.append(inv_result)
            else:
                logger.info(f"Step 3/3: Skipping invariant verification (prerequisites not met: action={action_passed}, compilation={compilation_passed})")
                
            # Update the final specification in the composite result
            composite_result.generated_specification = current_spec
            
            logger.info(f"Global correction attempts used: {global_correction_attempts}/{max_global_corrections}")
            
            # Print detailed summary report
            self._print_evaluation_summary(action_results_history, compilation_success_round, invariant_success_round, global_correction_attempts)
            
            # Calculate overall generation statistics across all iterations
            self._calculate_composite_generation_stats(composite_result)
            
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
    
    def _print_evaluation_summary(self, action_results_history, compilation_success_round, invariant_success_round, global_correction_attempts):
        """Print detailed evaluation summary with results from all rounds."""
        logger.info("=" * 60)
        logger.info("COMPOSITE EVALUATION DETAILED SUMMARY")
        logger.info("=" * 60)
        
        # Action Decomposition Results
        logger.info("Action Decomposition Results (3 rounds):")
        for i, action_result in enumerate(action_results_history):
            round_num = action_result['round']
            if 'error' in action_result:
                logger.info(f"  Round {round_num}: ✗ ERROR - {action_result['error']}")
            else:
                success_rate = action_result['success_rate'] * 100
                successful = action_result['successful_actions']
                total = action_result['total_actions']
                logger.info(f"  Round {round_num}: {successful}/{total} actions successful ({success_rate:.1f}%)")
        
        # Fill missing rounds
        while len(action_results_history) < 3:
            missing_round = len(action_results_history) + 1
            logger.info(f"  Round {missing_round}: SKIPPED (insufficient corrections)")
            action_results_history.append({'round': missing_round, 'skipped': True})
        
        # Compilation Check Results  
        logger.info("Compilation Check Results:")
        if compilation_success_round is not None:
            logger.info(f"  ✓ SUCCESS in Round {compilation_success_round + 1} (after {compilation_success_round} corrections)")
        else:
            logger.info(f"  ✗ FAILED after {global_correction_attempts} correction attempts")
        
        # Invariant Verification Results
        logger.info("Invariant Verification Results:")
        if invariant_success_round is not None:
            if invariant_success_round == 0:
                logger.info(f"  ✓ SUCCESS in initial attempt (no corrections needed)")
            else:
                logger.info(f"  ✓ SUCCESS in Round {invariant_success_round + 1} (after {invariant_success_round} corrections)")
        else:
            logger.info(f"  ✗ FAILED or SKIPPED")
        
        logger.info("=" * 60)
    
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
    
    def _calculate_composite_generation_stats(self, composite_result: CompositeEvaluationResult):
        """Calculate overall generation statistics from composite evaluation process"""
        # Count total evaluations performed
        total_evaluations = 1  # Action decomposition
        total_evaluations += 1  # Compilation check
        if composite_result.invariant_verification_results:
            total_evaluations += 1  # Invariant verification (if executed)
        
        # Count successful evaluations
        successful_evaluations = 0
        if composite_result.action_decomposition_result and composite_result.action_decomposition_result.overall_success:
            successful_evaluations += 1
        if composite_result.compilation_check_result and composite_result.compilation_check_result.overall_success:
            successful_evaluations += 1
        if composite_result.invariant_verification_results:
            if any(r.overall_success for r in composite_result.invariant_verification_results):
                successful_evaluations += 1
        
        logger.info(f"Composite evaluation summary: {successful_evaluations}/{total_evaluations} phases successful")
        logger.info(f"Initial generation time: {composite_result.generation_time:.2f}s")
        
        # The generated_specification contains the final (possibly corrected) version
    
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