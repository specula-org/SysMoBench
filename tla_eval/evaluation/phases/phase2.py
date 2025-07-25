"""
Phase 2 Evaluation: Model Checking with TLC

This module implements Phase 2 evaluation which includes:
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

logger = logging.getLogger(__name__)


@dataclass
class Phase2EvaluationResult:
    """Result of Phase 2 evaluation (model checking)"""
    task_name: str
    method_name: str
    model_name: str
    timestamp: float
    
    # Input specification info
    specification_file: str = None
    specification_valid: bool = False
    
    # Invariant generation results
    invariant_generation_successful: bool = False
    invariant_generation_time: float = 0.0
    generated_invariants: str = ""
    invariant_generation_error: str = None
    
    # Config generation results
    config_generation_successful: bool = False
    config_generation_time: float = 0.0
    generated_config: str = ""
    config_file_path: str = None
    config_generation_error: str = None
    
    # TLC model checking results
    model_checking_successful: bool = False
    model_checking_time: float = 0.0
    tlc_output: str = ""
    tlc_exit_code: int = None
    model_checking_error: str = None
    
    # Analysis results
    invariant_violations: List[str] = None
    deadlock_found: bool = False
    states_explored: int = 0
    
    # Overall success
    overall_success: bool = False
    
    def __post_init__(self):
        if self.invariant_violations is None:
            self.invariant_violations = []
        if self.timestamp == 0:
            self.timestamp = time.time()
            
        # Overall success requires all steps to succeed and no violations
        self.overall_success = (
            self.specification_valid and
            self.invariant_generation_successful and
            self.config_generation_successful and
            self.model_checking_successful and
            len(self.invariant_violations) == 0 and
            not self.deadlock_found
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary for serialization"""
        return {
            "task_name": self.task_name,
            "method_name": self.method_name,
            "model_name": self.model_name,
            "timestamp": self.timestamp,
            "specification": {
                "file": self.specification_file,
                "valid": self.specification_valid
            },
            "invariant_generation": {
                "successful": self.invariant_generation_successful,
                "time": self.invariant_generation_time,
                "invariants": self.generated_invariants,
                "error": self.invariant_generation_error
            },
            "config_generation": {
                "successful": self.config_generation_successful,
                "time": self.config_generation_time,
                "config": self.generated_config,
                "config_file": self.config_file_path,
                "error": self.config_generation_error
            },
            "model_checking": {
                "successful": self.model_checking_successful,
                "time": self.model_checking_time,
                "output": self.tlc_output,
                "exit_code": self.tlc_exit_code,
                "error": self.model_checking_error
            },
            "analysis": {
                "invariant_violations": self.invariant_violations,
                "deadlock_found": self.deadlock_found,
                "states_explored": self.states_explored
            },
            "overall_success": self.overall_success
        }


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
                response = model.client.chat.completions.create(**api_params)
                end_time = time.time()
                
                generated_text = response.choices[0].message.content
                metadata = {
                    "model": model.model_name,
                    "latency_seconds": end_time - start_time,
                    "finish_reason": response.choices[0].finish_reason,
                }
                
                from ...models.base import GenerationResult
                result = GenerationResult(
                    generated_text=generated_text,
                    metadata=metadata,
                    timestamp=end_time,
                    success=True
                )
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
                response = model.client.chat.completions.create(**api_params)
                end_time = time.time()
                
                generated_text = response.choices[0].message.content
                logger.debug(f"=== RAW LLM RESPONSE (CONFIG) ===")
                logger.debug(f"Length: {len(generated_text)}")
                logger.debug(f"Content: {repr(generated_text)}")
                logger.debug(f"SPECIFICATION count: {generated_text.count('SPECIFICATION')}")
                
                metadata = {
                    "model": model.model_name,
                    "latency_seconds": end_time - start_time,
                    "finish_reason": response.choices[0].finish_reason,
                }
                
                from ...models.base import GenerationResult
                result = GenerationResult(
                    generated_text=generated_text,
                    metadata=metadata,
                    timestamp=end_time,
                    success=True
                )
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
            
        except subprocess.TimeoutExpired:
            return False, f"TLC execution timed out after {self.timeout} seconds", -1
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


class Phase2Evaluator:
    """
    Phase 2 Evaluator: Model Checking with TLC
    
    This evaluator takes TLA+ specifications (from Phase 1) and performs:
    1. Invariant generation
    2. TLC configuration generation  
    3. Model checking with TLC
    """
    
    def __init__(self, tlc_timeout: int = 300):
        """
        Initialize Phase 2 evaluator.
        
        Args:
            tlc_timeout: Timeout for TLC execution in seconds
        """
        self.invariant_generator = InvariantGenerator()
        self.config_generator = ConfigGenerator()
        self.tlc_runner = TLCRunner(timeout=tlc_timeout)
        
        logger.info(f"Phase 2 Evaluator initialized with {tlc_timeout}s TLC timeout")
    
    def evaluate_specification(self, 
                             spec_file_path: str,
                             task_name: str,
                             method_name: str,
                             model_name: str,
                             spec_module: str = None) -> Phase2EvaluationResult:
        """
        Evaluate a TLA+ specification using model checking.
        
        Args:
            spec_file_path: Path to TLA+ specification file
            task_name: Name of the task
            method_name: Name of the generation method
            model_name: Name of the model used
            spec_module: Optional TLA+ module name
            
        Returns:
            Phase2EvaluationResult with model checking results
        """
        logger.info(f"Phase 2 evaluation: {task_name}/{method_name}/{model_name}")
        
        # Create evaluation result
        result = Phase2EvaluationResult(task_name, method_name, model_name, time.time())
        result.specification_file = spec_file_path
        
        try:
            # Step 1: Read and validate specification
            if not Path(spec_file_path).exists():
                result.specification_valid = False
                logger.error(f"Specification file not found: {spec_file_path}")
                return result
            
            with open(spec_file_path, 'r', encoding='utf-8') as f:
                tla_content = f.read()
            
            result.specification_valid = True
            logger.info("✓ Specification file loaded")
            
            # Step 2: Generate invariants
            logger.info("Generating invariants...")
            start_time = time.time()
            
            inv_success, invariants, inv_error = self.invariant_generator.generate_invariants(
                tla_content, task_name, model_name
            )
            
            result.invariant_generation_time = time.time() - start_time
            result.invariant_generation_successful = inv_success
            result.generated_invariants = invariants
            result.invariant_generation_error = inv_error
            
            if not inv_success:
                logger.error(f"✗ Invariant generation failed: {inv_error}")
                return result
            
            logger.info(f"✓ Invariants generated in {result.invariant_generation_time:.2f}s")
            
            # Insert invariants into TLA+ specification before ====
            updated_tla_content = self._insert_invariants_into_spec(tla_content, invariants)
            
            # Save updated specification with invariants
            with open(spec_file_path, 'w', encoding='utf-8') as f:
                f.write(updated_tla_content)
            logger.info(f"Invariants inserted into specification: {spec_file_path}")
            
            # Step 3: Generate TLC configuration
            # Use original TLA content (without invariants) to avoid duplication in prompt
            logger.info("Generating TLC configuration...")
            start_time = time.time()
            
            cfg_success, config, cfg_error = self.config_generator.generate_config(
                tla_content, invariants, task_name, model_name
            )
            
            result.config_generation_time = time.time() - start_time
            result.config_generation_successful = cfg_success
            result.generated_config = config
            result.config_generation_error = cfg_error
            
            if not cfg_success:
                logger.error(f"✗ Config generation failed: {cfg_error}")
                return result
            
            # Save config file
            spec_dir = Path(spec_file_path).parent
            module_name = spec_module or Path(spec_file_path).stem
            config_file_path = spec_dir / f"{module_name}.cfg"
            
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
                spec_file_path, str(config_file_path)
            )
            
            result.model_checking_time = time.time() - start_time
            result.model_checking_successful = tlc_success
            result.tlc_output = tlc_output
            result.tlc_exit_code = tlc_exit_code
            
            if not tlc_success:
                result.model_checking_error = f"TLC failed with exit code {tlc_exit_code}"
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
                result.specification_valid and
                result.invariant_generation_successful and
                result.config_generation_successful and
                result.model_checking_successful and
                len(result.invariant_violations) == 0 and
                not result.deadlock_found
            )
            
            if result.overall_success:
                logger.info("✓ Phase 2 evaluation: PASS")
            else:
                violations_msg = f"{len(violations)} violations" if violations else "no violations"
                deadlock_msg = "deadlock found" if deadlock else "no deadlock"
                logger.info(f"✗ Phase 2 evaluation: FAIL ({violations_msg}, {deadlock_msg})")
            
            return result
            
        except Exception as e:
            logger.error(f"Phase 2 evaluation failed: {e}")
            result.model_checking_error = str(e)
            return result
    
    def save_results(self, results: List[Phase2EvaluationResult], output_file: str):
        """Save Phase 2 evaluation results to file"""
        import json
        
        output_data = {
            "evaluation_type": "phase2_model_checking",
            "total_evaluations": len(results),
            "timestamp": time.time(),
            "results": [result.to_dict() for result in results]
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Phase 2 results saved to: {output_file}")
    
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