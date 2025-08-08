#!/usr/bin/env python3

from tla_eval.config import get_configured_model
from tla_eval.models.base import GenerationConfig
from tla_eval.evaluation.semantics.manual_invariant_evaluator import InvariantTemplateLoader, InvariantTranslator
import os

# Set API key
os.environ['GEMINI_API_KEY'] = "AIzaSyD0nLf7OAaWWT9sGb5SWbCFf5Ryq7m7KEE"

# Load the latest generated spin spec
latest_spec_path = "/home/ubuntu/LLM_Gen_TLA_benchmark_framework/output/invariant_verification/spin/direct_call_gemini/20250808133240/spin.tla"
with open(latest_spec_path, 'r') as f:
    tla_content = f.read()

print("=== TLA+ Specification ===")
print(tla_content[:500] + "..." if len(tla_content) > 500 else tla_content)
print()

# Load invariant templates
loader = InvariantTemplateLoader()
templates = loader.load_task_invariants("spin")

print(f"=== Loaded {len(templates)} Templates ===")
for template in templates:
    print(f"- {template.name} ({template.type})")
print()

# Test translation
translator = InvariantTranslator()
success, translated_invariants, error = translator.translate_all_invariants(
    templates, tla_content, "spin", "gemini"
)

print("=== Translation Result ===")
print(f"Success: {success}")
print(f"Error: {error}")
print(f"Translated: {len(translated_invariants)} invariants")
print()

if success:
    print("=== Generated Text for Analysis ===")
    # We need to get the raw generated text for debugging
    # Let's manually call the LLM to see the exact output
    
    from tla_eval.config import get_configured_model
    from string import Template
    
    model = get_configured_model("gemini")
    prompt_template = translator._load_translation_prompt("spin")
    invariant_templates_text = translator._format_templates_for_prompt(templates)
    
    template = Template(prompt_template)
    prompt = template.substitute(
        tla_specification=tla_content,
        invariant_templates=invariant_templates_text
    )
    
    gen_config = GenerationConfig(max_tokens=8192, temperature=0.1)
    result = model.generate_direct(prompt, gen_config)
    
    print("RAW LLM OUTPUT:")
    print("=" * 50)
    print(result.generated_text)
    print("=" * 50)
    print()
    
    print("=== Parsing Analysis ===")
    lines = result.generated_text.strip().split('\n')
    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
        print(f"Line {i+1}: {repr(line)}")
        if ' == ' in line:
            parts = line.split(' == ', 1)
            invariant_name = parts[0].strip()
            print(f"  -> Found invariant: {repr(invariant_name)}")
            
            # Check against template names
            for template in templates:
                if template.name.lower() == invariant_name.lower():
                    print(f"  -> ✓ Matches template: {template.name}")
                    break
            else:
                print(f"  -> ✗ No matching template found")
        print()

print("=== Final Translated Invariants ===")
for name, definition in translated_invariants.items():
    print(f"{name}: {definition}")