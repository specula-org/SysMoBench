"""
Composite Evaluator: Integrated evaluation combining multiple metrics.

This evaluator implements a comprehensive evaluation pipeline that:
1. Generates TLA+ specification once using agent-based method
2. Performs action decomposition evaluation
3. Performs compilation check evaluation 
4. If compilation succeeds, performs runtime check (3 iterations)
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
from ..semantics.runtime_check import RuntimeCheckEvaluator
from ..semantics.manual_invariant_evaluator import ManualInvariantEvaluator

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
                 keep_temp_files: bool = False,
                 max_correction_attempts: int = 3):
        """
        Initialize composite evaluator.
        
        Args:
            validation_timeout: Timeout for TLA+ validation in seconds
            invariant_iterations: Number of invariant verification iterations
            keep_temp_files: Whether to keep temporary files for debugging
            max_correction_attempts: Maximum number of global correction attempts
        """
        super().__init__(timeout=validation_timeout)
        self.invariant_iterations = invariant_iterations
        self.keep_temp_files = keep_temp_files
        self.max_correction_attempts = max_correction_attempts
        
        # Initialize sub-evaluators
        self.action_evaluator = ActionDecompositionEvaluator(
            validation_timeout=validation_timeout,
            keep_temp_files=keep_temp_files
        )
        self.compilation_evaluator = CompilationCheckEvaluator(
            validation_timeout=validation_timeout
        )
        self.runtime_check_evaluator = RuntimeCheckEvaluator(
            tlc_timeout=validation_timeout
        )
        self.manual_invariant_evaluator = ManualInvariantEvaluator(
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
        max_global_corrections = self.max_correction_attempts
        
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
            
            # Step 3: Runtime Check - Only if both previous phases passed
            action_passed = composite_result.action_decomposition_result.overall_success
            compilation_passed = composite_result.compilation_check_result.overall_success
            
            if action_passed and compilation_passed:
                logger.info(f"Step 3/4: Runtime check (both previous phases passed)")
                
                try:
                    # Evaluate current specification
                    inv_result = self.runtime_check_evaluator.evaluate(
                        current_generation_result, task_name, method_name, model_name, spec_module
                    )
                    composite_result.runtime_check_results.append(inv_result)
                    
                    success_status = "✓ PASS" if inv_result.overall_success else "✗ FAIL"
                    logger.info(f"Runtime check result: {success_status}")
                    
                    # Record success if first attempt succeeds
                    if inv_result.overall_success:
                        invariant_success_round = 0  # Success on first attempt (before any corrections)
                    
                    # If failed, attempt corrections using remaining global attempts
                    if not inv_result.overall_success:
                        logger.info(f"Runtime check failed, attempting corrections (max {max_global_corrections - global_correction_attempts} remaining)")
                        
                        while global_correction_attempts < max_global_corrections:
                            if task is not None and method is not None and hasattr(method, '_generate_correction'):
                                logger.info(f"Correction attempt {global_correction_attempts + 1}/{max_global_corrections}")
                                
                                # Collect all errors from runtime check
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
                                        inv_result = self.runtime_check_evaluator.evaluate(
                                            current_generation_result, task_name, method_name, model_name, spec_module
                                        )
                                        composite_result.runtime_check_results[-1] = inv_result
                                        
                                        if inv_result.overall_success:
                                            invariant_success_round = global_correction_attempts
                                            logger.info(f"✓ Runtime check passed after correction (Round {invariant_success_round})")
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
                    logger.error(f"Runtime check evaluation failed: {e}")
                    # Create a failed result
                    from ..base.result_types import SemanticEvaluationResult
                    inv_result = SemanticEvaluationResult(task_name, method_name, model_name)
                    inv_result.overall_success = False
                    inv_result.error_message = str(e)
                    composite_result.runtime_check_results.append(inv_result)
            else:
                logger.info(f"Step 3/4: Skipping runtime check (prerequisites not met: action={action_passed}, compilation={compilation_passed})")
            
            # Step 4: Manual Invariant Verification - Only if all previous phases passed
            action_passed = composite_result.action_decomposition_result.overall_success
            compilation_passed = composite_result.compilation_check_result.overall_success
            runtime_passed = any(r.overall_success for r in composite_result.runtime_check_results)
            
            if action_passed and compilation_passed and runtime_passed:
                logger.info(f"Step 4/4: Manual invariant verification (all previous phases passed)")
                
                try:
                    # Evaluate current specification with manual invariants
                    manual_inv_result = self.manual_invariant_evaluator.evaluate(
                        current_generation_result, task_name, method_name, model_name, spec_module
                    )
                    composite_result.manual_invariant_result = manual_inv_result
                    
                    success_status = "✓ PASS" if manual_inv_result.overall_success else "✗ FAIL"
                    total_invariants = manual_inv_result.custom_data.get('total_invariants', 0) if manual_inv_result.custom_data else 0
                    passed_invariants = manual_inv_result.custom_data.get('passed_invariants', 0) if manual_inv_result.custom_data else 0
                    logger.info(f"Manual invariant verification: {success_status} ({passed_invariants}/{total_invariants} invariants passed)")
                    
                except Exception as e:
                    logger.error(f"Manual invariant verification failed: {e}")
                    from ..base.result_types import SemanticEvaluationResult
                    manual_inv_result = SemanticEvaluationResult(task_name, method_name, model_name)
                    manual_inv_result.overall_success = False
                    manual_inv_result.error_message = str(e)
                    composite_result.manual_invariant_result = manual_inv_result
            else:
                logger.info(f"Step 4/4: Skipping manual invariant verification (prerequisites not met: action={action_passed}, compilation={compilation_passed}, runtime={runtime_passed})")
                
            # Update the final specification in the composite result
            composite_result.generated_specification = current_spec
            
            logger.info(f"Global correction attempts used: {global_correction_attempts}/{max_global_corrections}")
            
            # Calculate overall generation statistics across all iterations
            self._calculate_composite_generation_stats(composite_result)
            
            # Calculate overall success
            composite_result.overall_success = self._calculate_overall_success(composite_result)
            
            # Create output directory and save results first
            output_dir = self._create_output_directory(task_name, method_name, model_name)
            
            # Print detailed summary report with output_dir for JSON export
            self._print_evaluation_summary(action_results_history, compilation_success_round, invariant_success_round, global_correction_attempts, composite_result, output_dir=output_dir)
            
            # Save composite results to the already created directory
            self._save_composite_results(composite_result, output_dir)
            
        except Exception as e:
            logger.error(f"Composite evaluation error: {e}")
            composite_result.overall_success = False
        
        finally:
            # Results already saved in try block
            pass
        
        end_time = time.time()
        total_time = end_time - start_time
        logger.info(f"Composite evaluation complete: success={composite_result.overall_success}, total_time={total_time:.2f}s")
        
        return composite_result
    
    def _print_evaluation_summary(self, action_results_history, compilation_success_round, invariant_success_round, global_correction_attempts, composite_result, output_dir=None):
        """Print detailed evaluation summary with results from all rounds."""
        # Generate experiment data for JSON export
        experiment_data = self._generate_experiment_data(
            action_results_history, compilation_success_round, invariant_success_round, 
            global_correction_attempts, composite_result
        )
        
        # Save experiment data as JSON
        if output_dir:
            self._save_experiment_data(experiment_data, output_dir)
        
        # Print improved summary
        self._print_improved_summary(experiment_data, action_results_history, 
                                   compilation_success_round, invariant_success_round,
                                   global_correction_attempts, composite_result)
    
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
        if composite_result.runtime_check_results:
            inv_success = any(r.overall_success for r in composite_result.runtime_check_results)
            
            # If manual invariant also ran, consider it in success calculation
            if composite_result.manual_invariant_result:
                manual_inv_success = composite_result.manual_invariant_result.overall_success
                return syntax_success and inv_success and manual_inv_success
            
            return syntax_success and inv_success
        
        return syntax_success
    
    def _save_composite_results(self, composite_result: CompositeEvaluationResult, output_dir):
        """Save composite evaluation results to output directory"""
        
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
            "runtime_check_iterations": len(composite_result.runtime_check_results),
            "runtime_check_successful_iterations": sum(
                1 for r in composite_result.runtime_check_results if r.overall_success
            ),
            "overall_successful": composite_result.overall_success
        }
        
        metadata = {
            "task_name": composite_result.task_name,
            "method_name": composite_result.method_name,
            "model_name": composite_result.model_name,
            "metric": "composite",
            "evaluation_timestamp": time.time(),
            "validation_timeout": self.timeout,
            "invariant_iterations": self.invariant_iterations,
            "keep_temp_files": self.keep_temp_files
        }
        
        # Save specification to output directory
        if composite_result.generated_specification:
            spec_file_path = output_dir / f"{composite_result.task_name}.tla"
            with open(spec_file_path, 'w', encoding='utf-8') as f:
                f.write(composite_result.generated_specification)
            metadata["specification_file"] = str(spec_file_path)
        
        output_manager = get_output_manager()
        output_manager.save_result(output_dir, result_data, metadata)
        logger.info(f"Composite results saved to: {output_dir}")
        
        # Store output directory path in result for display
        composite_result.output_directory = str(output_dir)
    
    def _calculate_composite_generation_stats(self, composite_result: CompositeEvaluationResult):
        """Calculate overall generation statistics from composite evaluation process"""
        # Count total evaluations performed
        total_evaluations = 1  # Action decomposition
        total_evaluations += 1  # Compilation check
        if composite_result.runtime_check_results:
            total_evaluations += 1  # Invariant verification (if executed)
        
        # Count successful evaluations
        successful_evaluations = 0
        if composite_result.action_decomposition_result and composite_result.action_decomposition_result.overall_success:
            successful_evaluations += 1
        if composite_result.compilation_check_result and composite_result.compilation_check_result.overall_success:
            successful_evaluations += 1
        if composite_result.runtime_check_results:
            if any(r.overall_success for r in composite_result.runtime_check_results):
                successful_evaluations += 1
        
        logger.info(f"Composite evaluation summary: {successful_evaluations}/{total_evaluations} phases successful")
        logger.info(f"Initial generation time: {composite_result.generation_time:.2f}s")
        
        # The generated_specification contains the final (possibly corrected) version
    
    def _get_evaluation_type(self) -> str:
        """Return the evaluation type identifier"""
        return "composite"
    
    def _create_output_directory(self, task_name: str, method_name: str, model_name: str):
        """Create output directory for composite evaluation results."""
        output_manager = get_output_manager()
        output_dir = output_manager.create_experiment_dir(
            metric="composite",
            task=task_name,
            method=method_name, 
            model=model_name
        )
        return output_dir
    
    def _generate_experiment_data(self, action_results_history, compilation_success_round, 
                                invariant_success_round, global_correction_attempts, composite_result):
        """Generate structured experiment data for JSON export."""
        
        # Calculate iteration statistics - handle any number of iterations
        total_iterations = len(action_results_history)
        successful_iteration = None
        if compilation_success_round is not None and invariant_success_round is not None:
            successful_iteration = max(compilation_success_round, invariant_success_round) + 1
        
        # Phase results for each iteration
        phase_results = []
        phase4_passed = 0
        phase4_total = 0
        phase4_failed_invariants = []
        
        for i, action_result in enumerate(action_results_history):
            iteration = i + 1
            
            # Phase 1: Action Decomposition
            if 'error' in action_result:
                phase1_success = False
                phase1_ratio = 0.0
            else:
                phase1_success = action_result['success_rate'] >= 1.0
                phase1_ratio = action_result['success_rate']
            
            # Phase 2: Compilation Check 
            phase2_success = compilation_success_round is not None and compilation_success_round < iteration
            
            # Phase 3: Runtime Check
            phase3_success = invariant_success_round is not None and invariant_success_round < iteration
            
            # Phase 4: Manual Invariant Verification (only if phases 1-3 passed)
            phase4_success = False
            iter_phase4_passed = 0
            iter_phase4_total = 0
            iter_phase4_failed_invariants = []
            
            if phase1_success and phase2_success and phase3_success and composite_result.manual_invariant_result:
                phase4_success = composite_result.manual_invariant_result.overall_success
                if composite_result.manual_invariant_result.custom_data:
                    iter_phase4_passed = composite_result.manual_invariant_result.custom_data.get('passed_invariants', 0)
                    iter_phase4_total = composite_result.manual_invariant_result.custom_data.get('total_invariants', 0)
                    iter_phase4_failed_invariants = composite_result.manual_invariant_result.custom_data.get('failed_invariants', [])
                    
                    # Store for final statistics
                    phase4_passed = iter_phase4_passed
                    phase4_total = iter_phase4_total
                    phase4_failed_invariants = iter_phase4_failed_invariants
            
            phase_results.append({
                "iteration": iteration,
                "phase1_action_decomposition": {
                    "success": phase1_success,
                    "success_ratio": phase1_ratio
                },
                "phase2_compilation": {
                    "success": phase2_success
                },
                "phase3_runtime": {
                    "success": phase3_success
                },
                "phase4_manual_invariant": {
                    "success": phase4_success,
                    "passed_invariants": iter_phase4_passed,
                    "total_invariants": iter_phase4_total,
                    "failed_invariants": iter_phase4_failed_invariants
                }
            })
        
        # Get final phase success states
        final_phase1_success = phase_results[-1]["phase1_action_decomposition"]["success"] if phase_results else False
        final_phase1_ratio = phase_results[-1]["phase1_action_decomposition"]["success_ratio"] if phase_results else 0.0
        
        return {
            "summary": {
                "total_iterations": total_iterations,
                "successful_iteration": successful_iteration,
                "final_success": composite_result.overall_success,
                "generation_time_seconds": composite_result.generation_time,
                "total_evaluation_time_seconds": getattr(composite_result, 'total_time', 0.0)
            },
            "phase_statistics": {
                "phase1_action_decomposition": {
                    "final_success": final_phase1_success,
                    "final_success_ratio": final_phase1_ratio
                },
                "phase2_compilation": {
                    "success": compilation_success_round is not None
                },
                "phase3_runtime": {
                    "success": invariant_success_round is not None  
                },
                "phase4_manual_invariant": {
                    "success": composite_result.manual_invariant_result.overall_success if composite_result.manual_invariant_result else False,
                    "passed_invariants": phase4_passed,
                    "total_invariants": phase4_total,
                    "success_ratio": phase4_passed / phase4_total if phase4_total > 0 else 0.0,
                    "failed_invariants": phase4_failed_invariants
                }
            },
            "iteration_details": phase_results
        }

    def _save_experiment_data(self, experiment_data, output_dir):
        """Save experiment data as JSON for automated analysis."""
        import json
        from pathlib import Path
        
        if not output_dir:
            return
        
        output_path = Path(output_dir)
        json_file = output_path / "experiment_data.json"
        
        try:
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(experiment_data, f, indent=2, ensure_ascii=False)
            logger.info(f"Experiment data saved to: {json_file}")
        except Exception as e:
            logger.warning(f"Failed to save experiment data: {e}")

    def _print_improved_summary(self, experiment_data, action_results_history, 
                              compilation_success_round, invariant_success_round,
                              global_correction_attempts, composite_result):
        """Print improved, clear summary of composite evaluation results."""
        
        logger.info("=" * 70)
        logger.info("COMPOSITE EVALUATION SUMMARY")
        logger.info("=" * 70)
        
        # Overall Results
        summary = experiment_data["summary"]
        logger.info(f"📊 OVERALL RESULT: {'✓ SUCCESS' if summary['final_success'] else '✗ FAILURE'}")
        logger.info(f"📈 Total Iterations: {summary['total_iterations']}")
        if summary['successful_iteration']:
            logger.info(f"🎯 Success achieved in: Iteration {summary['successful_iteration']}")
        logger.info(f"⏱️  Generation Time: {summary['generation_time_seconds']:.1f}s")
        logger.info(f"⏱️  Total Time: {summary['total_evaluation_time_seconds']:.1f}s")
        
        logger.info("")
        logger.info("🔄 ITERATION BREAKDOWN:")
        
        # Iteration Details - handle any number of iterations
        for iteration_data in experiment_data["iteration_details"]:
            iter_num = iteration_data["iteration"]
            logger.info(f"  Iteration {iter_num}:")
            
            # Phase 1
            p1 = iteration_data["phase1_action_decomposition"]
            status1 = "✓ PASS" if p1["success"] else "✗ FAIL"
            logger.info(f"    Phase 1 (Actions): {status1} ({p1['success_ratio']:.1%})")
            
            if p1["success"]:
                # Phase 2
                p2 = iteration_data["phase2_compilation"]  
                status2 = "✓ PASS" if p2["success"] else "✗ FAIL"
                logger.info(f"    Phase 2 (Compile): {status2}")
                
                if p2["success"]:
                    # Phase 3  
                    p3 = iteration_data["phase3_runtime"]
                    status3 = "✓ PASS" if p3["success"] else "✗ FAIL"
                    logger.info(f"    Phase 3 (Runtime): {status3}")
                    
                    if p3["success"]:
                        # Phase 4
                        p4 = iteration_data["phase4_manual_invariant"]
                        if p4["total_invariants"] > 0:
                            status4 = "✓ PASS" if p4["success"] else "✗ FAIL"
                            logger.info(f"    Phase 4 (Invariants): {status4} ({p4['passed_invariants']}/{p4['total_invariants']})")
                            if p4["failed_invariants"]:
                                logger.info(f"      Failed: {', '.join(p4['failed_invariants'])}")
                        else:
                            logger.info(f"    Phase 4 (Invariants): ⚠ SKIPPED")
                    else:
                        logger.info(f"    Phase 4 (Invariants): ⚠ SKIPPED (Phase 3 failed)")
                else:
                    logger.info(f"    Phase 3 (Runtime): ⚠ SKIPPED (Phase 2 failed)")
                    logger.info(f"    Phase 4 (Invariants): ⚠ SKIPPED (Phase 2 failed)")
            else:
                logger.info(f"    Phase 2 (Compile): ⚠ SKIPPED (Phase 1 failed)")
                logger.info(f"    Phase 3 (Runtime): ⚠ SKIPPED (Phase 1 failed)")  
                logger.info(f"    Phase 4 (Invariants): ⚠ SKIPPED (Phase 1 failed)")
        
        logger.info("")
        logger.info("📈 FINAL PHASE STATISTICS:")
        stats = experiment_data["phase_statistics"]
        logger.info(f"  Phase 1 (Action Decomposition): {'✓' if stats['phase1_action_decomposition']['final_success'] else '✗'} ({stats['phase1_action_decomposition']['final_success_ratio']:.1%})")
        logger.info(f"  Phase 2 (Compilation Check): {'✓' if stats['phase2_compilation']['success'] else '✗'}")
        logger.info(f"  Phase 3 (Runtime Check): {'✓' if stats['phase3_runtime']['success'] else '✗'}")
        
        p4_stats = stats['phase4_manual_invariant']
        if p4_stats['total_invariants'] > 0:
            logger.info(f"  Phase 4 (Manual Invariants): {'✓' if p4_stats['success'] else '✗'} ({p4_stats['passed_invariants']}/{p4_stats['total_invariants']}, {p4_stats['success_ratio']:.1%})")
            if p4_stats['failed_invariants']:
                logger.info(f"    Failed Invariants: {', '.join(p4_stats['failed_invariants'])}")
        else:
            logger.info(f"  Phase 4 (Manual Invariants): ⚠ NOT EXECUTED")
        
        logger.info("=" * 70)


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

