#!/usr/bin/env python3
"""
Basic usage example of the TLA+ evaluation framework.

This example demonstrates how to:
1. Initialize a model adapter
2. Load a benchmark task  
3. Generate TLA+ specifications
4. Evaluate the results
"""

import os
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from tla_eval.models import get_model_adapter, ModelFactory
from tla_eval.utils import get_default_prompt


def main():
    """Demonstrate basic usage of the framework."""
    
    # Example 1: List available models
    print("=== Example 1: Available Models ===")
    available = ModelFactory.list_available_models()
    print(f"Predefined models: {available['predefined_models']}")
    print(f"Available providers: {available['providers']}")
    
    # Example 2: Using OpenAI GPT-4 (if available)
    print("\n=== Example 2: OpenAI GPT-4 ===")
    
    if not os.getenv("OPENAI_API_KEY"):
        print("Warning: OPENAI_API_KEY not found. Skipping OpenAI example.")
    else:
        try:
            # Initialize model adapter using predefined config
            model = get_model_adapter("openai_gpt4")
            print(f"Initialized model: {model.get_model_info()}")
            
            if model.is_available():
                print("Model is available and ready to use!")
            else:
                print("Model is not available.")
                
        except Exception as e:
            print(f"Could not initialize OpenAI model: {e}")
    
    # Example 3: Using Anthropic Claude (if available)
    print("\n=== Example 3: Anthropic Claude ===")
    
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("Warning: ANTHROPIC_API_KEY not found. Skipping Anthropic example.")
    else:
        try:
            # Initialize model adapter using predefined config
            model = get_model_adapter("anthropic_claude3_sonnet")
            print(f"Initialized model: {model.get_model_info()}")
            
            if model.is_available():
                print("Model is available and ready to use!")
            else:
                print("Model is not available.")
                
        except Exception as e:
            print(f"Could not initialize Anthropic model: {e}")
    
    # Example 4: Generate TLA+ specification (demo)
    print("\n=== Example 4: Generate TLA+ specification ===")
    
    sample_code = """
// Simple mutex implementation
type Mutex struct {
    locked bool
}

func (m *Mutex) Lock() {
    for m.locked {
        // spin wait
    }
    m.locked = true
}

func (m *Mutex) Unlock() {
    m.locked = false
}
"""
    
    print(f"Sample source code:\n{sample_code}")
    
    # Try to generate with any available model
    model = None
    if os.getenv("OPENAI_API_KEY"):
        try:
            model = get_model_adapter("openai_gpt4")
        except:
            pass
    elif os.getenv("ANTHROPIC_API_KEY"):
        try:
            model = get_model_adapter("anthropic_claude3_sonnet")
        except:
            pass
    
    if model and model.is_available():
        try:
            prompt_template = get_default_prompt()
            print("\nGenerating TLA+ specification...")
            result = model.generate_tla_specification(sample_code, prompt_template.template)
            
            if result.success:
                print(f"\nGenerated TLA+ specification:\n{result.generated_text}")
                print(f"\nGeneration metadata: {result.metadata}")
            else:
                print(f"Generation failed: {result.error_message}")
                
        except Exception as e:
            print(f"Error during generation: {e}")
    else:
        print("No API keys available for demonstration.")
    
    print("\n=== Framework is ready! ===")


if __name__ == "__main__":
    main()