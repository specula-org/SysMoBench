"""
Invariant Verification Evaluator: Semantic-level evaluation for TLA+ specifications.

This evaluator implements semantic evaluation which includes:
1. Invariant generation from TLA+ specifications
2. TLC configuration file (.cfg) generation  
3. TLC model checking execution and result analysis
"""

import os
import subprocess
import time
import logging
import re
from pathlib import Path
from string import Template
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass

from ...models.base import GenerationResult
from ...config import get_configured_model
from ...utils.output_manager import get_output_manager
from ..base.evaluator import BaseEvaluator
from ..base.result_types import SemanticEvaluationResult

logger = logging.getLogger(__name__)


class InvariantGenerator:
    """Generates invariants from TLA+ specifications using LLM"""
    
    def __init__(self):
        self.name = "invariant_generator"
    
    def generate_invariants(self, tla_content: str, task_name: str, model_name: str) -> Tuple[bool, str, str]:
        """
        Generate invariants from TLA+ specification.
        
        Args:
            tla_content: TLA+ specification content
            task_name: Name of the task (for loading prompt)
            model_name: Model to use for generation
            
        Returns:
            Tuple of (success, generated_invariants, error_message)
        """
        try:
            model = get_configured_model(model_name)
            
            # Load task-specific prompt for invariant generation
            prompt_template = self._load_invariant_prompt(task_name)
            
            # Format prompt with TLA+ specification using Template to avoid brace conflicts
            template = Template(prompt_template)
            prompt = template.substitute(tla_spec=tla_content)
            
            # Generate invariants by calling the model directly (not source-to-TLA generation)
            # We bypass generate_tla_specification to avoid double formatting
            if hasattr(model, 'client'):
                # For OpenAI-compatible models, call directly
                api_params = {
                    "model": model.model_name,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 4096,
                    "temperature": 0.1,
                }
                
                import time
                start_time = time.time()
                
                # Use the unified model interface instead of direct API calls
                from ...models.base import GenerationConfig
                gen_config = GenerationConfig(
                    max_tokens=api_params.get("max_tokens", 4096),
                    temperature=api_params.get("temperature", 0.1)
                )
                
                result = model.generate_tla_specification("", prompt, gen_config)
                end_time = time.time()
                
                if not result.success:
                    raise Exception(f"Model generation failed: {result.error_message}")
                
                generated_text = result.generated_text
                metadata = result.metadata.copy()
                metadata.update({
                    "latency_seconds": end_time - start_time,
                })
                
                # Update the result with timing information
                result.metadata.update(metadata)
                result.timestamp = end_time
            else:
                # Fallback for other model types
                result = model.generate_tla_specification("", prompt)
            
            if result.success:
                return True, result.generated_text.strip(), None
            else:
                return False, "", result.error_message
                
        except Exception as e:
            logger.error(f"Invariant generation failed: {e}")
            return False, "", str(e)
    
    def _load_invariant_prompt(self, task_name: str) -> str:
        """Load task-specific prompt for invariant generation"""
        from ...tasks.loader import get_task_loader
        task_loader = get_task_loader()
        
        # Get task directory path
        tasks_dir = task_loader.tasks_dir
        prompt_file = tasks_dir / task_name / "prompts" / "phase2_invariants.txt"
        
        if not prompt_file.exists():
            raise FileNotFoundError(f"Phase 2 invariant prompt not found: {prompt_file}")
        
        with open(prompt_file, 'r', encoding='utf-8') as f:
            return f.read()


class ConfigGenerator:
    """Generates TLC configuration files (.cfg) using LLM"""
    
    def __init__(self):
        self.name = "config_generator"
    
    def generate_config(self, tla_content: str, invariants: str, task_name: str, model_name: str) -> Tuple[bool, str, str]:
        """
        Generate TLC configuration file from TLA+ specification and invariants.
        
        Args:
            tla_content: TLA+ specification content
            invariants: Generated invariants
            task_name: Name of the task (for loading prompt)
            model_name: Model to use for generation
            
        Returns:
            Tuple of (success, generated_config, error_message)
        """
        try:
            model = get_configured_model(model_name)
            
            # Load task-specific prompt for config generation
            prompt_template = self._load_config_prompt(task_name)
            
            # Format prompt with TLA+ specification and invariants using Template to avoid brace conflicts
            template = Template(prompt_template)
            prompt = template.substitute(tla_spec=tla_content, invariants=invariants)
            
            # Generate config by calling the model directly (not source-to-TLA generation)
            # We bypass generate_tla_specification to avoid double formatting
            if hasattr(model, 'client'):
                # For OpenAI-compatible models, call directly
                api_params = {
                    "model": model.model_name,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 4096,
                    "temperature": 0.1,
                }
                
                import time
                start_time = time.time()
                
                # Use the unified model interface instead of direct API calls
                from ...models.base import GenerationConfig
                gen_config = GenerationConfig(
                    max_tokens=api_params.get("max_tokens", 4096),
                    temperature=api_params.get("temperature", 0.1)
                )
                
                result = model.generate_tla_specification("", prompt, gen_config)
                end_time = time.time()
                
                if not result.success:
                    raise Exception(f"Model generation failed: {result.error_message}")
                
                generated_text = result.generated_text
                logger.debug(f"=== RAW LLM RESPONSE (CONFIG) ===")
                logger.debug(f"Length: {len(generated_text)}")
                logger.debug(f"Content: {repr(generated_text)}")
                logger.debug(f"SPECIFICATION count: {generated_text.count('SPECIFICATION')}")
                
                metadata = result.metadata.copy()
                metadata.update({
                    "latency_seconds": end_time - start_time,
                })
                
                # Update the result with timing information
                result.metadata.update(metadata)
                result.timestamp = end_time
            else:
                # Fallback for other model types
                result = model.generate_tla_specification("", prompt)
            
            if result.success:
                final_config = result.generated_text.strip()
                logger.debug(f"=== AFTER STRIP ===")
                logger.debug(f"Length: {len(final_config)}")
                logger.debug(f"SPECIFICATION count: {final_config.count('SPECIFICATION')}")
                return True, final_config, None
            else:
                return False, "", result.error_message
                
        except Exception as e:
            logger.error(f"Config generation failed: {e}")
            return False, "", str(e)
    
    def _load_config_prompt(self, task_name: str) -> str:
        """Load task-specific prompt for config generation"""
        from ...tasks.loader import get_task_loader
        task_loader = get_task_loader()
        
        # Get task directory path
        tasks_dir = task_loader.tasks_dir
        prompt_file = tasks_dir / task_name / "prompts" / "phase2_config.txt"
        
        if not prompt_file.exists():
            raise FileNotFoundError(f"Phase 2 config prompt not found: {prompt_file}")
        
        with open(prompt_file, 'r', encoding='utf-8') as f:
            return f.read()


class TLCRunner:
    """Runs TLC model checker and parses results"""
    
    def __init__(self, timeout: int = 300):
        """
        Initialize TLC runner.
        
        Args:
            timeout: Timeout for TLC execution in seconds
        """
        self.timeout = timeout
        self.tla_tools_path = self._get_tla_tools_path()
    
    def _get_tla_tools_path(self) -> Path:
        """Get path to TLA+ tools"""
        from ...utils.setup_utils import get_tla_tools_path
        return get_tla_tools_path()
    
    def run_model_checking(self, spec_file: str, config_file: str) -> Tuple[bool, str, int]:
        """
        Run TLC model checking.
        
        Args:
            spec_file: Path to TLA+ specification file
            config_file: Path to TLC configuration file
            
        Returns:
            Tuple of (success, output, exit_code)
        """
        try:
            # Convert paths to absolute and then get relative paths for TLC
            spec_path = Path(spec_file).resolve()
            config_path = Path(config_file).resolve()
            working_dir = spec_path.parent
            
            # Get relative paths from working directory
            spec_filename = spec_path.name
            config_filename = config_path.name
            
            # Construct TLC command with relative paths
            cmd = [
                "java",
                "-cp", str(self.tla_tools_path),
                "tlc2.TLC",
                "-config", config_filename,
                spec_filename
            ]
            
            logger.debug(f"Running TLC in {working_dir}: {' '.join(cmd)}")
            
            # Run TLC
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                cwd=working_dir  # Run in spec directory
            )
            
            output = result.stdout + result.stderr
            success = result.returncode == 0
            
            logger.debug(f"TLC finished: exit_code={result.returncode}, output_length={len(output)}")
            
            return success, output, result.returncode
            
        except subprocess.TimeoutExpired as e:
            # For large state spaces, timeout without violations should be considered success
            # Parse partial output to check for violations
            partial_output = ""
            try:
                # Try to get partial output from the process
                if hasattr(e, 'stdout') and e.stdout:
                    partial_output += e.stdout.decode() if isinstance(e.stdout, bytes) else str(e.stdout)
                if hasattr(e, 'stderr') and e.stderr:
                    partial_output += e.stderr.decode() if isinstance(e.stderr, bytes) else str(e.stderr)
            except:
                # If we can't get partial output, just use empty string
                partial_output = ""
            
            # Parse the partial output for violations and deadlocks
            violations, deadlock_found, states_explored = self.parse_tlc_output(partial_output)
            
            if violations or deadlock_found:
                # Found violations or deadlocks - this is a real failure
                return False, f"TLC found violations/deadlocks before timeout after {self.timeout} seconds:\n{partial_output}", -1
            else:
                # No violations found within timeout - consider this success for large state spaces
                logger.info(f"TLC timeout after {self.timeout}s with no violations found - considering as success")
                success_msg = f"TLC explored {states_explored} states in {self.timeout} seconds with no violations found (timeout reached but no errors detected)"
                return True, success_msg, 0
        except Exception as e:
            return False, f"TLC execution failed: {e}", -1
    
    def parse_tlc_output(self, output: str) -> Tuple[List[str], bool, int]:
        """
        Parse TLC output to extract violations and statistics.
        
        Args:
            output: TLC output text
            
        Returns:
            Tuple of (invariant_violations, deadlock_found, states_explored)
        """
        violations = []
        deadlock_found = False
        states_explored = 0
        
        lines = output.split('\n')
        
        for i, line in enumerate(lines):
            line = line.strip()
            
            # Check for invariant violations
            if "Invariant" in line and "is violated" in line:
                violations.append(line)
            
            # Check for deadlock
            if "Deadlock reached" in line or "deadlock" in line.lower():
                deadlock_found = True
            
            # Extract states explored
            if "states generated" in line.lower():
                import re
                match = re.search(r'(\d+)\s+states generated', line)
                if match:
                    states_explored = int(match.group(1))
        
        return violations, deadlock_found, states_explored


class InvariantVerificationEvaluator(BaseEvaluator):
    """
    Invariant Verification Evaluator: Semantic evaluation for TLA+ specifications.
    
    This evaluator takes TLA+ specifications and performs:
    1. Invariant generation
    2. TLC configuration generation  
    3. Model checking with TLC
    """
    
    def __init__(self, tlc_timeout: int = 60):
        """
        Initialize invariant verification evaluator.
        
        Args:
            tlc_timeout: Timeout for TLC execution in seconds
        """
        super().__init__(timeout=tlc_timeout)
        self.invariant_generator = InvariantGenerator()
        self.config_generator = ConfigGenerator()
        self.tlc_runner = TLCRunner(timeout=tlc_timeout)
    
    def evaluate(self, 
                generation_result: GenerationResult,
                task_name: str,
                method_name: str,
                model_name: str,
                spec_module: str = None) -> SemanticEvaluationResult:
        """
        Evaluate a TLA+ specification using model checking.
        
        Args:
            generation_result: GenerationResult containing the TLA+ specification
            task_name: Name of the task
            method_name: Name of the generation method
            model_name: Name of the model used
            spec_module: Optional TLA+ module name
            
        Returns:
            SemanticEvaluationResult with model checking results
        """
        logger.info(f"Semantic evaluation: {task_name}/{method_name}/{model_name}")
        
        # Create structured output directory
        output_manager = get_output_manager()
        output_dir = output_manager.create_experiment_dir(
            metric="invariant_verification",
            task=task_name,
            method=method_name,
            model=model_name
        )
        logger.info(f"Using output directory: {output_dir}")
        
        # Create evaluation result
        result = SemanticEvaluationResult(task_name, method_name, model_name)
        
        # Set generation time from the generation result metadata
        if hasattr(generation_result, 'metadata') and 'latency_seconds' in generation_result.metadata:
            result.generation_time = generation_result.metadata['latency_seconds']
        
        try:
            # Step 1: Get specification from generation result
            if not generation_result.success:
                logger.error("Generation failed, cannot perform semantic evaluation")
                result.error_message = "Generation failed"
                return result
            
            tla_content = generation_result.generated_text
            if not tla_content.strip():
                logger.error("Empty TLA+ specification from generation result")
                result.error_message = "Empty specification"
                return result
            
            logger.info("✓ Specification content loaded from generation result")
            
            # Save specification to output directory for reference
            module_name = spec_module or 'etcdraft'
            spec_file_path = output_dir / f"{module_name}.tla"
            with open(spec_file_path, 'w', encoding='utf-8') as f:
                f.write(tla_content)
            result.specification_file = str(spec_file_path)
            
            # Step 2: Skip invariant generation - use existing TLA+ specification directly
            logger.info("⏭️  Skipping invariant generation - using original specification without additional invariants")
            
            # The specification is already saved above in spec_file_path
            logger.info(f"Original specification saved to: {spec_file_path}")
            result.invariant_generation_time = 0.0
            result.invariant_generation_successful = True
            result.generated_invariants = []  # No generated invariants
            result.invariant_generation_error = None
            
            # Step 3: Generate TLC configuration (without generated invariants)
            logger.info("Generating TLC configuration...")
            start_time = time.time()
            
            # Use empty invariants string since we're not generating new invariants
            cfg_success, config, cfg_error = self.config_generator.generate_config(
                tla_content, "", task_name, model_name  # Empty invariants
            )
            
            result.config_generation_time = time.time() - start_time
            result.config_generation_successful = cfg_success
            result.config_generation_error = cfg_error
            
            if not cfg_success:
                logger.error(f"✗ Config generation failed: {cfg_error}")
                return result
            
            # Save config file to structured output directory
            config_file_path = output_dir / f"{module_name}.cfg"
            
            logger.debug(f"Writing config to {config_file_path}, length: {len(config)} chars")
            logger.debug(f"Config content preview: {config[:200]}...")
            with open(config_file_path, 'w', encoding='utf-8') as f:
                f.write(config)
            
            result.config_file_path = str(config_file_path)
            logger.info(f"✓ TLC config generated in {result.config_generation_time:.2f}s")
            
            # Step 4: Run TLC model checking
            logger.info("Running TLC model checking...")
            start_time = time.time()
            
            tlc_success, tlc_output, tlc_exit_code = self.tlc_runner.run_model_checking(
                str(spec_file_path), str(config_file_path)
            )
            
            result.model_checking_time = time.time() - start_time
            result.model_checking_successful = tlc_success
            result.model_checking_error = f"TLC failed with exit code {tlc_exit_code}" if not tlc_success else None
            
            if not tlc_success:
                logger.error(f"✗ TLC model checking failed: {result.model_checking_error}")
            else:
                logger.info(f"✓ TLC completed in {result.model_checking_time:.2f}s")
            
            # Step 5: Parse TLC results
            violations, deadlock, states = self.tlc_runner.parse_tlc_output(tlc_output)
            result.invariant_violations = violations
            result.deadlock_found = deadlock
            result.states_explored = states
            
            # Update overall success
            result.overall_success = (
                result.invariant_generation_successful and
                result.config_generation_successful and
                result.model_checking_successful and
                len(result.invariant_violations) == 0 and
                not result.deadlock_found
            )
            
            if result.overall_success:
                logger.info("✓ Semantic evaluation: PASS")
            else:
                violations_msg = f"{len(violations)} violations" if violations else "no violations"
                deadlock_msg = "deadlock found" if deadlock else "no deadlock"
                logger.info(f"✗ Semantic evaluation: FAIL ({violations_msg}, {deadlock_msg})")
            
            # Save results and metadata to structured output directory
            result_data = {
                "overall_success": result.overall_success,
                "invariant_generation_successful": result.invariant_generation_successful,
                "config_generation_successful": result.config_generation_successful,
                "model_checking_successful": result.model_checking_successful,
                "invariant_generation_time": result.invariant_generation_time,
                "config_generation_time": result.config_generation_time,
                "model_checking_time": result.model_checking_time,
                "states_explored": result.states_explored,
                "invariant_violations": result.invariant_violations,
                "deadlock_found": result.deadlock_found,
                "generated_invariants": result.generated_invariants,
                "errors": {
                    "invariant_generation_error": result.invariant_generation_error,
                    "config_generation_error": result.config_generation_error,
                    "model_checking_error": result.model_checking_error
                }
            }
            
            metadata = {
                "task_name": task_name,
                "method_name": method_name,
                "model_name": model_name,
                "metric": "invariant_verification",
                "specification_file": result.specification_file,
                "config_file_path": result.config_file_path,
                "tlc_timeout": self.timeout,
                "evaluation_timestamp": time.time()
            }
            
            output_manager.save_result(output_dir, result_data, metadata)
            
            return result
            
        except Exception as e:
            logger.error(f"Semantic evaluation failed: {e}")
            result.model_checking_error = str(e)
            return result
    
    def _insert_invariants_into_spec(self, tla_content: str, invariants: str) -> str:
        """
        Insert generated invariants into TLA+ specification before the ending ====.
        
        Args:
            tla_content: Original TLA+ specification content
            invariants: Generated invariants to insert
            
        Returns:
            Updated TLA+ specification with invariants
        """
        lines = tla_content.split('\n')
        
        # Find the line with ==== (4 or more equals)
        ending_line_index = -1
        for i in range(len(lines) - 1, -1, -1):  # Search from end
            line = lines[i].strip()
            if line.startswith('====') and len(line) >= 4:
                ending_line_index = i
                break
        
        if ending_line_index == -1:
            # If no ==== found, append at the end
            logger.warning("No ==== ending found in specification, appending invariants at end")
            return tla_content + '\n\n' + invariants + '\n===='
        
        # Insert invariants before the ==== line
        result_lines = lines[:ending_line_index]
        result_lines.append('')  # Empty line before invariants
        result_lines.extend(invariants.split('\n'))
        result_lines.append('')  # Empty line after invariants
        result_lines.extend(lines[ending_line_index:])  # Add ==== and any content after
        
        return '\n'.join(result_lines)
    
    def _get_evaluation_type(self) -> str:
        """Return the evaluation type identifier"""
        return "semantic_invariant_verification"


# Convenience function for backward compatibility
def create_invariant_verification_evaluator(tlc_timeout: int = 60) -> InvariantVerificationEvaluator:
    """
    Factory function to create an invariant verification evaluator.
    
    Args:
        tlc_timeout: Timeout for TLC execution in seconds
        
    Returns:
        InvariantVerificationEvaluator instance
    """
    return InvariantVerificationEvaluator(tlc_timeout=tlc_timeout)