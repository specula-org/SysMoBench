# Model Usage Guide

This guide explains how to use different models with the TLA+ benchmark framework.

## Flexible Model Specification

The framework supports flexible model specification using the format `provider:model_name`.

### OpenAI Models

```bash
# GPT-4
python scripts/run_benchmark.py --model openai:gpt-4
```

### Anthropic Models

```bash
# Claude 3 Opus
python scripts/run_benchmark.py --model anthropic:claude-3-opus-20240229
```

## Model Parameters

You can customize model behavior using `--model-args`:

```bash
--model-args "temperature=0.1,max_tokens=4096,top_p=0.9"
```

### Supported Parameters

- `temperature`: Sampling temperature (0.0 to 2.0)
- `max_tokens`: Maximum tokens to generate
- `top_p`: Nucleus sampling parameter
- `timeout`: API request timeout in seconds
- `max_retries`: Maximum retry attempts

## API Keys

Set your API keys as environment variables:

```bash
# OpenAI
export OPENAI_API_KEY=your_openai_key_here

# Anthropic
export ANTHROPIC_API_KEY=your_anthropic_key_here
```

## Testing Your Setup

```bash
# Check available providers
python scripts/run_benchmark.py --dry-run

# Test with a specific model
python scripts/run_benchmark.py --model openai:gpt-4 --dry-run
```

## Adding Custom Models

You can easily add support for new model providers by extending the `ModelAdapter` class. See the existing adapters in `tla_eval/models/` for examples.
