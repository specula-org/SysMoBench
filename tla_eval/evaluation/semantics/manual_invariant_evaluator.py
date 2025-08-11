"""
Manual Invariant Evaluator: Phase 3 evaluation for TLA+ specifications.

This evaluator implements the third phase of evaluation which includes:
1. Loading expert-written invariant templates for the task
2. Using LLM to translate generic invariants to the specific TLA+ specification  
3. Running TLC model checking with the translated invariants
4. Reporting detailed results for each invariant test
"""

import os
import logging
import time
import yaml
from pathlib import Path
from string import Template
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass

from ...models.base import GenerationResult, GenerationConfig
from ...config import get_configured_model
from ...utils.output_manager import get_output_manager
from ..base.evaluator import BaseEvaluator
from ..base.result_types import SemanticEvaluationResult

# Import TLC runner from runtime_check
from .runtime_check import TLCRunner, ConfigGenerator

logger = logging.getLogger(__name__)


@dataclass
class InvariantTemplate:
    """Represents a single invariant template from the YAML file"""
    name: str
    type: str  # "safety" or "liveness" 
    natural_language: str
    formal_description: str
    tla_example: str


@dataclass  
class InvariantTestResult:
    """Result of testing a single invariant"""
    name: str
    success: bool
    translated_invariant: str
    error_message: Optional[str] = None
    states_explored: int = 0
    verification_time: float = 0.0
    tlc_output: str = ""


class InvariantTemplateLoader:
    """Loads invariant templates from YAML files"""
    
    def __init__(self, templates_dir: str = "data/invariant_templates"):
        self.templates_dir = Path(templates_dir)
    
    def load_task_invariants(self, task_name: str) -> List[InvariantTemplate]:
        """
        Load invariant templates for a specific task.
        
        Args:
            task_name: Name of the task (e.g., "etcd", "spin")
            
        Returns:
            List of InvariantTemplate objects
        """
        invariants_file = self.templates_dir / task_name / "invariants.yaml"
        
        if not invariants_file.exists():
            raise FileNotFoundError(f"Invariants file not found: {invariants_file}")
        
        with open(invariants_file, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        
        templates = []
        for inv_data in data.get('invariants', []):
            template = InvariantTemplate(
                name=inv_data['name'],
                type=inv_data['type'],
                natural_language=inv_data['natural_language'],
                formal_description=inv_data['formal_description'],
                tla_example=inv_data['tla_example']
            )
            templates.append(template)
        
        logger.info(f"Loaded {len(templates)} invariant templates for task: {task_name}")
        return templates


class InvariantTranslator:
    """Translates generic invariant templates to specific TLA+ specifications"""
    
    def __init__(self):
        self.name = "invariant_translator"
    
    def translate_all_invariants(self, 
                                templates: List[InvariantTemplate],
                                tla_content: str, 
                                task_name: str, 
                                model_name: str) -> Tuple[bool, Dict[str, str], str]:
        """
        Translate all invariant templates to the specific TLA+ specification in one call.
        
        Args:
            templates: List of invariant templates to translate
            tla_content: Target TLA+ specification content
            task_name: Name of the task (for loading prompt)
            model_name: Model to use for translation
            
        Returns:
            Tuple of (success, {invariant_name: translated_invariant}, error_message)
        """
        try:
            model = get_configured_model(model_name)
            
            # Load task-specific prompt template
            prompt_template = self._load_translation_prompt(task_name)
            
            # Format invariant templates for the prompt
            invariant_templates_text = self._format_templates_for_prompt(templates)
            
            # Format prompt with TLA+ specification and templates
            template = Template(prompt_template)
            prompt = template.substitute(
                tla_specification=tla_content,
                invariant_templates=invariant_templates_text
            )
            
            # Generate invariant implementations - use model's configured values
            # Don't override the model's configuration, let it use configured temperature and max_tokens
            gen_config = GenerationConfig(
                use_json_mode=True  # Enable JSON mode for structured output
                # Note: temperature and max_tokens not set - will use model's configured values
            )
            
            start_time = time.time()
            result = model.generate_direct(prompt, gen_config)
            end_time = time.time()
            
            if not result.success:
                return False, {}, result.error_message
            
            logger.info(f"Generated text length: {len(result.generated_text)} characters")
            
            # Parse the generated invariants
            translated_invariants = self._parse_generated_invariants(
                result.generated_text, templates
            )
            
            logger.info(f"Successfully translated {len(translated_invariants)} invariants in {end_time - start_time:.2f}s")
            return True, translated_invariants, None
            
        except Exception as e:
            logger.error(f"Invariant translation failed: {e}")
            return False, {}, str(e)
    
    def _load_translation_prompt(self, task_name: str) -> str:
        """Load task-specific prompt for invariant translation"""
        from ...tasks.loader import get_task_loader
        task_loader = get_task_loader()
        
        # Get task directory path
        tasks_dir = task_loader.tasks_dir
        prompt_file = tasks_dir / task_name / "prompts" / "phase3_invariant_implementation.txt"
        
        if not prompt_file.exists():
            raise FileNotFoundError(f"Phase 3 invariant prompt not found: {prompt_file}")
        
        with open(prompt_file, 'r', encoding='utf-8') as f:
            return f.read()
    
    def _format_templates_for_prompt(self, templates: List[InvariantTemplate]) -> str:
        """Format invariant templates for inclusion in the prompt"""
        formatted_templates = []
        
        for template in templates:
            formatted = f"""
### {template.name} ({template.type.upper()})
**Description**: {template.natural_language}

**Formal**: {template.formal_description}

**TLA+ Example**:
```
{template.tla_example.strip()}
```
"""
            formatted_templates.append(formatted)
        
        return '\n'.join(formatted_templates)
    
    def _parse_generated_invariants(self, 
                                  generated_text: str, 
                                  templates: List[InvariantTemplate]) -> Dict[str, str]:
        """Parse the generated text to extract individual invariant definitions"""
        translated_invariants = {}
        
        try:
            # Try to parse as JSON first
            import json
            
            # Clean the text: remove markdown code blocks if present
            clean_text = generated_text.strip()
            if clean_text.startswith('```json'):
                # Remove ```json from start and ``` from end
                lines = clean_text.split('\n')
                if lines[0].strip() == '```json' and lines[-1].strip() == '```':
                    clean_text = '\n'.join(lines[1:-1])
            elif clean_text.startswith('```'):
                # Remove generic ``` blocks
                lines = clean_text.split('\n')
                if lines[0].strip() == '```' and lines[-1].strip() == '```':
                    clean_text = '\n'.join(lines[1:-1])
            
            data = json.loads(clean_text)
            
            # Expect format: {"invariants": ["Name == Expression", ...]}
            if isinstance(data, dict) and "invariants" in data:
                invariant_list = data["invariants"]
                if isinstance(invariant_list, list):
                    logger.info("Parsing JSON format invariants")
                    
                    for i, invariant_definition in enumerate(invariant_list):
                        logger.info(f"Processing invariant {i+1}: {len(invariant_definition)} chars")
                        
                        if isinstance(invariant_definition, str) and invariant_definition.strip():
                            # Extract invariant name from the definition (everything before '==')
                            invariant_name = invariant_definition.split('==')[0].strip()
                            
                            # Find matching template by name
                            for template in templates:
                                if template.name.lower() == invariant_name.lower():
                                    translated_invariants[template.name] = invariant_definition
                                    logger.info(f"✓ Stored invariant: {template.name}")
                                    break
                            else:
                                logger.warning(f"No matching template for: {invariant_name}")
                        else:
                            logger.warning(f"Skipped empty or invalid invariant: {repr(invariant_definition[:50])}")
                    
                    return translated_invariants
            
        except json.JSONDecodeError:
            logger.info("JSON parsing failed, falling back to line-based parsing")
        
        # Fallback to original line-based parsing for backward compatibility
        lines = generated_text.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            if not line or line.startswith('```') or line.startswith('#'):
                continue
            
            # Look for pattern: InvariantName == <expression>
            if ' == ' in line:
                parts = line.split(' == ', 1)
                if len(parts) == 2:
                    invariant_name = parts[0].strip()
                    invariant_definition = line  # Keep the full definition
                    
                    # Match to template names (case-insensitive)
                    for template in templates:
                        if template.name.lower() == invariant_name.lower():
                            translated_invariants[template.name] = invariant_definition
                            logger.debug(f"Parsed line-based invariant: {template.name}")
                            break
        
        return translated_invariants


class ManualInvariantEvaluator(BaseEvaluator):
    """
    Manual Invariant Evaluator: Phase 3 evaluation for TLA+ specifications.
    
    This evaluator implements the third phase of evaluation:
    1. Load expert-written invariant templates
    2. Translate templates to specific TLA+ specification
    3. Run TLC model checking for each invariant
    4. Report detailed results
    """
    
    def __init__(self, tlc_timeout: int = 60, templates_dir: str = "data/invariant_templates"):
        """
        Initialize manual invariant evaluator.
        
        Args:
            tlc_timeout: Timeout for TLC execution in seconds
            templates_dir: Directory containing invariant templates
        """
        super().__init__(timeout=tlc_timeout)
        self.template_loader = InvariantTemplateLoader(templates_dir)
        self.translator = InvariantTranslator()
        self.config_generator = ConfigGenerator()
        self.tlc_runner = TLCRunner(timeout=tlc_timeout)
    
    def evaluate(self, 
                generation_result: GenerationResult,
                task_name: str,
                method_name: str,
                model_name: str,
                spec_module: Optional[str] = None) -> SemanticEvaluationResult:
        """
        Evaluate a TLA+ specification using manual invariant testing.
        
        Args:
            generation_result: GenerationResult containing the TLA+ specification
            task_name: Name of the task
            method_name: Name of the generation method
            model_name: Name of the model used
            spec_module: Optional TLA+ module name
            
        Returns:
            SemanticEvaluationResult with manual invariant testing results
        """
        logger.info(f"Manual invariant evaluation: {task_name}/{method_name}/{model_name}")
        
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
        
        if not generation_result.success:
            result.invariant_generation_error = "Generation failed"
            result.overall_success = False
            return result
        
        # Save the TLA+ specification
        spec_file = output_dir / f"{spec_module or task_name}.tla"
        with open(spec_file, 'w', encoding='utf-8') as f:
            f.write(generation_result.generated_text)
        result.specification_file = str(spec_file)
        
        try:
            # Step 1: Load invariant templates
            logger.info("Step 1: Loading invariant templates...")
            templates = self.template_loader.load_task_invariants(task_name)
            
            # Step 2: Translate all invariants in one LLM call
            logger.info("Step 2: Translating invariants to specification...")
            translation_start = time.time()
            success, translated_invariants, error = self.translator.translate_all_invariants(
                templates, generation_result.generated_text, task_name, model_name
            )
            
            result.invariant_generation_time = time.time() - translation_start
            result.invariant_generation_successful = success
            
            if not success:
                result.invariant_generation_error = error
                result.overall_success = False
                return result
            
            logger.info(f"Successfully translated {len(translated_invariants)} invariants")
            
            # Step 3: Test each invariant individually  
            logger.info("Step 3: Testing invariants with TLC...")
            invariant_results = []
            
            for template in templates:
                if template.name not in translated_invariants:
                    logger.warning(f"Invariant {template.name} was not translated, skipping")
                    continue
                
                invariant_test_result = self._test_single_invariant(
                    template, translated_invariants[template.name],
                    generation_result.generated_text, output_dir, spec_module or task_name,
                    task_name, model_name
                )
                invariant_results.append(invariant_test_result)
            
            # Step 4: Aggregate results
            result.model_checking_successful = any(r.success for r in invariant_results)
            result.model_checking_time = sum(r.verification_time for r in invariant_results)
            
            # Set overall success
            result.overall_success = (
                result.invariant_generation_successful and
                result.model_checking_successful and
                len(invariant_results) > 0
            )
            
            # Store detailed results
            result.custom_data = {
                "invariant_results": [
                    {
                        "name": r.name,
                        "success": r.success,
                        "states_explored": r.states_explored,
                        "verification_time": r.verification_time,
                        "error_message": r.error_message
                    }
                    for r in invariant_results
                ],
                "total_invariants": len(templates),
                "translated_invariants": len(translated_invariants),
                "passed_invariants": sum(1 for r in invariant_results if r.success),
                "failed_invariants": sum(1 for r in invariant_results if not r.success)
            }
            
            # Log summary
            passed = result.custom_data["passed_invariants"] 
            total = len(invariant_results)
            logger.info(f"Manual invariant testing: {passed}/{total} invariants passed")
            
            return result
            
        except Exception as e:
            logger.error(f"Manual invariant evaluation failed: {e}")
            result.invariant_generation_error = str(e)
            result.overall_success = False
            return result
    
    def _test_single_invariant(self, 
                              template: InvariantTemplate,
                              translated_invariant: str, 
                              tla_content: str,
                              output_dir: Path,
                              spec_module: str,
                              task_name: str,
                              model_name: str) -> InvariantTestResult:
        """Test a single invariant using TLC"""
        
        logger.debug(f"Testing invariant: {template.name}")
        
        try:
            # Create directory for this invariant
            invariant_dir = output_dir / template.name
            invariant_dir.mkdir(exist_ok=True)
            
            # Create modified TLA+ spec with the invariant
            modified_spec = self._add_invariant_to_spec(
                tla_content, translated_invariant, template.name
            )
            
            # Save modified spec with correct module name
            modified_spec_file = invariant_dir / f"{spec_module}.tla"
            with open(modified_spec_file, 'w', encoding='utf-8') as f:
                f.write(modified_spec)
            
            # Generate config file for this invariant
            invariant_list = template.name  # Single invariant
            config_success, config_content, config_error = self.config_generator.generate_config(
                modified_spec, invariant_list, task_name, model_name  # Use the same model
            )
            
            if not config_success:
                return InvariantTestResult(
                    name=template.name,
                    success=False,
                    translated_invariant=translated_invariant,
                    error_message=f"Config generation failed: {config_error}"
                )
            
            # Save config file with correct module name
            config_file = invariant_dir / f"{spec_module}.cfg"
            with open(config_file, 'w', encoding='utf-8') as f:
                f.write(config_content)
            
            # Run TLC
            start_time = time.time()
            tlc_success, tlc_output, tlc_exit_code = self.tlc_runner.run_model_checking(
                str(modified_spec_file), str(config_file)
            )
            verification_time = time.time() - start_time
            
            # Parse TLC results
            violations, deadlock_found, states_explored = self.tlc_runner.parse_tlc_output(tlc_output)
            
            success = tlc_success and len(violations) == 0 and not deadlock_found
            
            return InvariantTestResult(
                name=template.name,
                success=success,
                translated_invariant=translated_invariant,
                states_explored=states_explored,
                verification_time=verification_time,
                tlc_output=tlc_output,
                error_message=None if success else f"TLC failed: {len(violations)} violations, deadlock: {deadlock_found}"
            )
            
        except Exception as e:
            return InvariantTestResult(
                name=template.name,
                success=False,
                translated_invariant=translated_invariant,
                error_message=f"Testing failed: {str(e)}"
            )
    
    def _add_invariant_to_spec(self, tla_content: str, invariant_definition: str, invariant_name: str) -> str:
        """Add a single invariant definition to the TLA+ specification"""
        
        lines = tla_content.split('\n')
        result_lines = []
        
        # Find the insertion point (before the closing ====)
        invariant_inserted = False
        
        for i, line in enumerate(lines):
            # Check if this is the final separator line (could be ==== or longer ========...)  
            if line.strip().startswith('====') and i == len(lines) - 1:
                # Insert invariant before final separator
                if not invariant_inserted:
                    result_lines.append('')
                    result_lines.append(f'\\* Manual invariant: {invariant_name}')
                    result_lines.append(invariant_definition)
                    result_lines.append('')
                    invariant_inserted = True
            
            result_lines.append(line)
        
        # If no ==== found, append at the end
        if not invariant_inserted:
            result_lines.append('')
            result_lines.append(f'\\* Manual invariant: {invariant_name}')
            result_lines.append(invariant_definition)
        
        return '\n'.join(result_lines)
    
    def _get_evaluation_type(self) -> str:
        """Return the evaluation type identifier"""
        return "semantic_invariant_verification"


# Convenience function for backward compatibility
def create_manual_invariant_evaluator(tlc_timeout: int = 60, 
                                     templates_dir: str = "data/invariant_templates") -> ManualInvariantEvaluator:
    """
    Factory function to create a manual invariant evaluator.
    
    Args:
        tlc_timeout: Timeout for TLC execution in seconds
        templates_dir: Directory containing invariant templates
        
    Returns:
        ManualInvariantEvaluator instance
    """
    return ManualInvariantEvaluator(tlc_timeout=tlc_timeout, templates_dir=templates_dir)