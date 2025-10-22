# LLM Abstraction Layer

This module provides a unified interface for different Large Language Model (LLM) providers through the gen_ai_hub proxy, enabling seamless integration with OpenAI, AWS Bedrock, and Google Vertex AI models.

## Overview

The LLM abstraction layer implements a factory pattern to create model instances with consistent interfaces across different providers. It supports both LangChain-based and native API implementations for optimal performance and flexibility.

## Supported Models

### OpenAI Models
- **gpt-4.1** (Default) - Enhanced reasoning and analysis with 800K token context window
  - Temperature support: Yes
  - JSON mode support: Yes
  
- **o3** - Advanced reasoning model with 128K token context window
  - Temperature support: No
  - JSON mode support: Yes

### Google Vertex AI Models
- **gemini-2.5-pro** - Large context window (800K tokens) with multimodal capabilities
  - Temperature support: Yes
  - JSON mode support: Yes
  
- **gemini-2.5-flash** - Fast and efficient model with 800K token context window
  - Temperature support: Yes
  - JSON mode support: Yes

### AWS Bedrock Models
- **anthropic--claude-4-sonnet** / **claude-4-sonnet** - Balanced performance and reasoning with 200K token context window
  - Temperature support: Yes
  - JSON mode support: Yes

## Architecture

### Factory Pattern

The system provides two factory implementations:

1. **LangChain Factory** (`factory.py`) - Uses LangChain wrappers for model integration
2. **Native Factory** (`factory_native.py`) - Direct API integration for better performance

Both factories expose the same interface:

```python
def create_llm(
    model_name: Optional[str] = None,
    temperature: float = 0.0,
    **kwargs
) -> Any
```

### Configuration System

Model configurations are centralized in `config.py`:

```python
MODEL_CONFIGS = {
    "model-name": {
        "provider": "provider-name",
        "temperature": 0.0,
        "description": "Model description",
        "supports_json": True,
        "supports_temperature": True,
        "max_tokens": 800000
    }
}
```

Provider-specific defaults ensure optimal settings for each platform:

```python
PROVIDER_DEFAULTS = {
    "openai": {"request_timeout": 600, "max_retries": 3},
    "bedrock": {"region_name": "us-east-1", "request_timeout": 600},
    "vertexai": {"location": "us-central1", "request_timeout": 600}
}
```

## Prompt System and TQDCS Framework

### TQDCS Categories

The system implements the TQDCS (Technology, Quality, Delivery, Cost, Sustainability) framework for structured knowledge extraction:

- **T (Technology)** - Technical specifications, capabilities, features
- **Q (Quality)** - Standards, certifications, compliance
- **D (Delivery)** - Lead times, logistics, capacity
- **C (Cost)** - Pricing, payment terms, financial aspects
- **S (Sustainability)** - Environmental impact, green initiatives

### Prompt Templates

The `prompts/` directory contains structured templates:

- `system_prompt.txt` - Main system prompt with TQDCS rules
- `human_prompt.txt` - User-facing prompt template
- `discovery_prompt.txt` - Pattern discovery prompt
- `validation_system_prompt.txt` - Validation system prompt
- `validation_human_prompt.txt` - Validation user prompt
- `tqdcs_categories.json` - TQDCS category definitions

## Usage Examples

### Basic Usage

```python
from resources.kg_creation.llm import create_llm

# Create default model (gpt-4.1)
llm = create_llm()

# Create specific model
llm = create_llm(model_name="gemini-2.5-pro", temperature=0.0)

# Invoke the model
response = llm.invoke("Your prompt here")
print(response.content)
```

### With Environment Variable

```python
import os
os.environ["LLM_MODEL"] = "claude-4-sonnet"

from resources.kg_creation.llm import create_llm

# Will use claude-4-sonnet
llm = create_llm()
```

### Advanced Configuration

```python
from resources.kg_creation.llm import create_llm, get_model_info

# Get model information
info = get_model_info("gpt-4.1")
print(f"Max tokens: {info['max_tokens']}")

# Create with custom parameters
llm = create_llm(
    model_name="gpt-4.1",
    temperature=0.2,
    max_tokens=4000,
    top_p=0.95
)
```

### Multimodal Support (Images)

```python
from langchain_core.messages import HumanMessage

# Create message with image
message = HumanMessage(
    content=[
        {"type": "text", "text": "Analyze this image"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}
    ]
)

# Invoke with multimodal content
response = llm.invoke([message])
```

## Adding New Models

To add a new model:

1. **Update Configuration** in `config.py`:
```python
MODEL_CONFIGS["new-model"] = {
    "provider": "provider-name",
    "temperature": 0.0,
    "description": "Model description",
    "supports_json": True,
    "supports_temperature": True,
    "max_tokens": 100000
}
```

2. **Handle Provider Logic** (if new provider):
   - Add provider defaults to `PROVIDER_DEFAULTS`
   - Update factory methods in both `factory.py` and `factory_native.py`
   - Implement wrapper class in `factory_native.py` if using native implementation

3. **Test Integration**:
```python
from resources.kg_creation.llm import create_llm, SUPPORTED_MODELS

# Verify model is listed
assert "new-model" in SUPPORTED_MODELS

# Test creation
llm = create_llm(model_name="new-model")
response = llm.invoke("Test prompt")
```

## Dependencies

- `gen_ai_hub` - SAP's unified AI proxy
- `langchain` - For LangChain factory implementation
- Standard Python libraries: `os`, `logging`, `typing`, `dataclasses`

## Logging

The module uses Python's standard logging with logger name `__name__`. Enable debug logging to see detailed model creation and invocation information:

```python
import logging
logging.getLogger("resources.kg_creation.llm").setLevel(logging.DEBUG)
```