"""
Compilation Check Evaluator: Syntax-level evaluation for TLA+ specifications.

This evaluator checks whether generated TLA+ specifications can be compiled
successfully using the TLA tools (SANY parser).
"""

import logging
import re
import time
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional


_TLA_FENCE = re.compile(r"```\s*tla\s*\n(.*?)\n```", re.DOTALL | re.IGNORECASE)
_CFG_FENCE = re.compile(r"```\s*cfg\s*\n(.*?)\n```", re.DOTALL | re.IGNORECASE)


def split_tla_and_cfg(raw: str) -> Tuple[str, Optional[str]]:
    """
    Parse model output expected to contain ```tla and ```cfg fenced blocks.
    Falls back to treating the whole text as raw TLA when no fences are present
    (backwards compat with pre-fenced prompts).
    """
    tla_match = _TLA_FENCE.search(raw)
    cfg_match = _CFG_FENCE.search(raw)
    tla = tla_match.group(1) if tla_match else raw
    cfg = cfg_match.group(1) if cfg_match else None
    return tla, cfg

from ...core.verification.validators import TLAValidator, ValidationResult
from ...models.base import GenerationResult
from ...utils.output_manager import get_output_manager
from ..base.evaluator import BaseEvaluator
from ..base.result_types import SyntaxEvaluationResult
from ...core.verification.error_statistics_manager import classify_and_record_tlc_result, TLCErrorCategory, get_experiment_error_statistics_manager

logger = logging.getLogger(__name__)


class CompilationCheckEvaluator(BaseEvaluator):
    """
    Evaluator for TLA+ specification compilation checking.
    
    This evaluator checks whether generated TLA+ specifications can be
    compiled successfully using the TLA tools (SANY parser).
    """
    
    def __init__(self, validation_timeout: int = 30):
        """
        Initialize compilation check evaluator.
        
        Args:
            validation_timeout: Timeout for TLA+ validation in seconds
        """
        super().__init__(timeout=validation_timeout)
        # Create separate error statistics manager for this evaluator
        self.error_stats_manager = get_experiment_error_statistics_manager()
        # Create validator with custom error statistics manager
        self.validator = TLAValidator(timeout=validation_timeout, error_stats_manager=self.error_stats_manager)
    
    def evaluate(self, 
                generation_result: GenerationResult,
                task_name: str,
                method_name: str,
                model_name: str,
                spec_module: str = None,
                spec_file_path: Optional[str] = None,
                config_file_path: Optional[str] = None) -> SyntaxEvaluationResult:
        """
        Evaluate a single generation result for compilation success.
        
        Args:
            generation_result: Result from TLA+ generation
            task_name: Name of the task
            method_name: Name of the generation method
            model_name: Name of the model used
            spec_module: Optional TLA+ module name for the specification
            spec_file_path: Optional path to existing .tla file (use instead of generation_result)
            config_file_path: Optional path to existing .cfg file (unused but kept for interface consistency)
            
        Returns:
            SyntaxEvaluationResult with evaluation metrics
        """
        logger.info(f"Evaluating compilation: {task_name}/{method_name}/{model_name}")
        
        # Create structured output directory
        output_manager = get_output_manager()
        output_dir = output_manager.create_experiment_dir(
            metric="compilation_check",
            task=task_name,
            method=method_name,
            model=model_name
        )
        logger.info(f"Using output directory: {output_dir}")
        
        # Create evaluation result
        eval_result = SyntaxEvaluationResult(task_name, method_name, model_name)
        self._set_generation_result(eval_result, generation_result)
        
        # Determine input source and get TLA+ content
        if spec_file_path and Path(spec_file_path).exists():
            # Use existing spec file
            logger.info(f"Using existing spec file: {spec_file_path}")
            try:
                with open(spec_file_path, 'r', encoding='utf-8') as f:
                    tla_content = f.read()
            except Exception as e:
                logger.error(f"Failed to read spec file: {e}")
                validation_result = ValidationResult(
                    success=False,
                    output=f"Failed to read spec file: {e}",
                    syntax_errors=[f"Cannot read spec file: {e}"],
                    semantic_errors=[],
                    compilation_time=0.0
                )
                self._set_validation_result(eval_result, validation_result)
                return eval_result
        else:
            # Use generated content
            if not generation_result.success:
                logger.warning(f"Generation failed, cannot proceed: {generation_result.error_message}")
                validation_result = ValidationResult(
                    success=False,
                    output="Generation failed - no specification to validate",
                    syntax_errors=[],
                    semantic_errors=["Generation failed"],
                    compilation_time=0.0
                )
                self._set_validation_result(eval_result, validation_result)
                return eval_result
            tla_content, cfg_content = split_tla_and_cfg(generation_result.generated_text)

        # For the spec_file_path branch, there is no separate cfg extraction —
        # callers that pass a pre-written .tla are expected to also have its
        # .cfg sibling in place. Only the generated-text branch produces cfg.
        if 'cfg_content' not in locals():
            cfg_content = None

        # Set the actual content being evaluated (may be from file or generation)
        eval_result.generated_specification = tla_content
        
        # Validate the generated specification
        try:
            logger.debug("Starting TLA+ specification validation...")
            # Save specification to structured output directory
            spec_file_path = output_dir / f"{spec_module or 'UnnamedModule'}.tla"
            with open(spec_file_path, 'w', encoding='utf-8') as spec_file:
                spec_file.write(tla_content)
            
            # Validate using the saved file path
            validation_result = self.validator.validate_file(str(spec_file_path))
            
            # DEBUG: Print detailed validation info
            logger.info(f"DEBUG: validation_result.success = {validation_result.success}")
            logger.info(f"DEBUG: validation_result.errors count = {len(validation_result.errors)}")
            logger.info(f"DEBUG: validation_result.output = {validation_result.output[:500]}...")
            
            self._set_validation_result(eval_result, validation_result)
            
            if validation_result.success:
                logger.info("✓ Specification compiled successfully")
            else:
                error_count = len(validation_result.errors)
                if error_count == 0:
                    logger.warning(f"Compilation success BUT validation_result.success={validation_result.success}")
                else:
                    logger.warning(f"Compilation failed with {error_count} errors")
                
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
            self._set_validation_result(eval_result, validation_result)
        
        # Save TLA specification to output directory
        if tla_content:
            module_name = spec_module or task_name
            spec_file_path = output_dir / f"{module_name}.tla"
            with open(spec_file_path, 'w', encoding='utf-8') as f:
                f.write(tla_content)
            logger.info(f"Saved specification to: {spec_file_path}")

            # Persist .cfg alongside .tla so Phase 2 (TLC) and Phase 3b
            # (invariant) can run. If the model didn't emit a ```cfg fenced
            # block, do NOT synthesize fake content — a missing cfg is a
            # real signal that the model didn't follow the output contract.
            # Downstream phases will fail on cfg=None and that's correct.
            if cfg_content:
                cfg_path = output_dir / f"{module_name}.cfg"
                with open(cfg_path, 'w', encoding='utf-8') as f:
                    f.write(cfg_content)
                logger.info(f"Saved config to: {cfg_path}")
            else:
                logger.warning(
                    f"Model did not emit a ```cfg fenced block — "
                    f"no cfg written, downstream Phase 2/3b will fail"
                )
        
        # Persist adapter's usage/cost info for Phase 0 accounting.
        # The adapter returns a GenerationResult whose metadata carries
        # token counts under keys like model/litellm_model/usage/finish_reason.
        # run_benchmark.py re-wraps that result, nesting the adapter metadata
        # under a "generation_metadata" key (see direct_call/method.py). So we
        # look there first and fall back to the top level for direct evaluator
        # calls that bypass the benchmark wrapper.
        gen_meta = getattr(generation_result, 'metadata', {}) or {}
        nested = gen_meta.get("generation_metadata") or {}
        def _pick(key):
            return nested.get(key) if nested.get(key) is not None else gen_meta.get(key)
        generation_usage = {
            "model": _pick("model"),
            "litellm_model": _pick("litellm_model"),
            "provider": _pick("provider"),
            "finish_reason": _pick("finish_reason"),
            "latency_seconds": _pick("latency_seconds"),
            "usage": _pick("usage"),
            "response_id": _pick("response_id"),
            # SANY-feedback retry loop (direct_call). These live at the top
            # of gen_meta, not in the nested adapter metadata.
            "attempts_used": gen_meta.get("attempts_used"),
            "final_sany_success": gen_meta.get("final_sany_success"),
            "correction_attempts": gen_meta.get("correction_attempts"),
        }
        try:
            import json as _json
            with open(output_dir / "generation_usage.json", 'w', encoding='utf-8') as f:
                _json.dump(generation_usage, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.warning(f"Failed to save generation_usage.json: {e}")

        # Save results and metadata
        result_data = {
            "overall_success": eval_result.overall_success,
            "generation_successful": eval_result.generation_successful,
            "compilation_successful": eval_result.compilation_successful,
            "generation_time": eval_result.generation_time,
            "compilation_time": eval_result.compilation_time,
            "syntax_errors": eval_result.syntax_errors,
            "semantic_errors": eval_result.semantic_errors,
            "generation_error": eval_result.generation_error
        }
        
        metadata = {
            "task_name": task_name,
            "method_name": method_name,
            "model_name": model_name,
            "metric": "compilation_check",
            "spec_module": spec_module,
            "validation_timeout": self.timeout,
            "evaluation_timestamp": time.time()
        }
        
        output_manager.save_result(output_dir, result_data, metadata)
        
        # Save error statistics for this compilation_check run
        try:
            stats_file_path = self.error_stats_manager.save_experiment_statistics(
                output_dir=output_dir,
                task_name=task_name,
                method_name=method_name,
                model_name=model_name
            )
            logger.info(f"Error statistics saved to: {stats_file_path}")
        except Exception as stats_error:
            logger.error(f"Failed to save error statistics: {stats_error}")
        
        logger.info(f"Evaluation complete: success={eval_result.overall_success}")
        return eval_result
    
    def evaluate_batch(self, 
                      results: List[Tuple[GenerationResult, str, str, str]]) -> List[SyntaxEvaluationResult]:
        """
        Evaluate multiple generation results in batch.
        
        Args:
            results: List of tuples (generation_result, task_name, method_name, model_name)
            
        Returns:
            List of SyntaxEvaluationResult
        """
        logger.info(f"Starting batch evaluation of {len(results)} results")
        
        evaluation_results = []
        
        for i, (generation_result, task_name, method_name, model_name) in enumerate(results):
            logger.info(f"Processing batch item {i+1}/{len(results)}")
            
            try:
                eval_result = self.evaluate(
                    generation_result, task_name, method_name, model_name
                )
                evaluation_results.append(eval_result)
                
            except Exception as e:
                logger.error(f"Batch evaluation error for item {i+1}: {e}")
                # Create error result
                eval_result = SyntaxEvaluationResult(task_name, method_name, model_name)
                eval_result.generation_error = f"Batch evaluation error: {e}"
                evaluation_results.append(eval_result)
        
        logger.info(f"Batch evaluation complete: {len(evaluation_results)} results")
        return evaluation_results
    
    def _set_generation_result(self, eval_result: SyntaxEvaluationResult, generation_result: GenerationResult):
        """Set generation results on evaluation result"""
        eval_result.generation_successful = generation_result.success
        eval_result.generation_time = generation_result.metadata.get('latency_seconds', 0.0)
        eval_result.generated_specification = generation_result.generated_text
        
        if not generation_result.success:
            eval_result.generation_error = generation_result.error_message
    
    def _set_validation_result(self, eval_result: SyntaxEvaluationResult, validation_result: ValidationResult):
        """Set validation results on evaluation result"""
        eval_result.compilation_successful = validation_result.success
        eval_result.compilation_time = validation_result.compilation_time
        eval_result.syntax_errors = validation_result.syntax_errors
        eval_result.semantic_errors = validation_result.semantic_errors
        eval_result.compilation_output = validation_result.output
        
        # Legacy compatibility
        eval_result.compilation_errors = validation_result.errors
        
        # Overall success requires both generation and compilation to succeed
        eval_result.overall_success = eval_result.generation_successful and eval_result.compilation_successful
    
    def _calculate_summary(self, results: List[SyntaxEvaluationResult]) -> Dict[str, Any]:
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
    
    def _get_evaluation_type(self) -> str:
        """Return the evaluation type identifier"""
        return "syntax_compilation_check"


# Convenience function for backward compatibility
def create_compilation_check_evaluator(validation_timeout: int = 30) -> CompilationCheckEvaluator:
    """
    Factory function to create a compilation check evaluator.
    
    Args:
        validation_timeout: Timeout for TLA+ validation in seconds
        
    Returns:
        CompilationCheckEvaluator instance
    """
    return CompilationCheckEvaluator(validation_timeout=validation_timeout)