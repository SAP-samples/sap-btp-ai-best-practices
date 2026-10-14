---
name: access-to-generative-ai-models
description: Implement Python code and notebooks that call generative AI models through SAP Gen AI Hub using native SDK clients, OpenAI-compatible Responses and Chat Completions APIs, LangChain wrappers, Amazon Bedrock/Anthropic, Google Gemini, multimodal image/video/PDF inputs, structured outputs, reasoning controls, token usage handling, and Gen AI Hub orchestration.
---

# Access Models via SAP Gen AI Hub

Use this skill to implement direct model calls through SAP Gen AI Hub.

## Set Up the Environment

Load SAP AI Core credentials from `.env` before creating clients:

```python
from dotenv import load_dotenv

load_dotenv()
```

Expected variables:

```bash
AICORE_AUTH_URL=""
AICORE_CLIENT_ID=""
AICORE_CLIENT_SECRET=""
AICORE_BASE_URL=""
AICORE_RESOURCE_GROUP=""
```

Use `sap-ai-sdk-gen[all]`, `python-dotenv`, `boto3`, `langchain`, `langchain-openai`, `langchain_aws`, and `langchain-google-vertexai` when matching the Python examples.

## Choose the Integration Pattern

- Use native OpenAI `responses` for new OpenAI-shaped work: text, structured output, image input, PDF input, reasoning controls, and stateful follow-up calls.
- Use native OpenAI `chat.completions` only for legacy chat-completion code, simple compatibility examples, or OpenAI-compatible models that are exposed through that path such as Perplexity and Meta deployments.
- Use native Amazon Bedrock clients for direct Anthropic/Amazon `converse` calls, especially when following Bedrock message formats.
- Use native Google GenAI clients for Gemini image, video, audio, and PDF calls.
- Use LangChain wrappers when the model is part of a chain, retriever, tool/agent flow, memory flow, or other LangChain component.
- Use Gen AI Hub orchestration when combining LLM calls with orchestration modules such as filtering, masking, grounding, translation, or templating.

Keep model names in variables and verify that the selected deployment supports the requested modality and parameters in the target SAP AI Core resource group.

## Native OpenAI Responses API

Use `responses` as the default native OpenAI interface for new code:

```python
from gen_ai_hub.proxy.native.openai import responses

model = "gpt-5.4"

response = responses.create(
    model=model,
    instructions=(
        "You are a concise SAP BTP assistant. "
        "Answer in one short paragraph and avoid unsupported assumptions."
    ),
    input="In one sentence, explain what SAP AI Core provides inside SAP BTP.",
    reasoning={"effort": "none"},
)

print(response.output_text)

usage = getattr(response, "usage", None)
if usage:
    print(f"Input tokens: {usage.input_tokens}")
    print(f"Output tokens: {usage.output_tokens}")
    print(f"Total tokens: {usage.total_tokens}")
```

Responses uses `instructions` for system guidance, `input` for plain text or typed content blocks, typed `output` items for results, and `output_text` for common text answers.

## Structured Output

Use `responses.parse` with a Pydantic model when callers need schema-stable output:

```python
from pydantic import BaseModel, Field
from gen_ai_hub.proxy.native.openai import responses


class CapabilitySummary(BaseModel):
    """Structured summary returned by the model.

    Attributes:
        capability: Name of the SAP BTP capability.
        use_case: Practical scenario where the capability is useful.
        integration_notes: Developer-facing implementation notes.
    """

    capability: str = Field(description="Name of the SAP BTP capability.")
    use_case: str = Field(description="Concrete use case for the capability.")
    integration_notes: list[str] = Field(description="Developer-facing notes.")


structured_response = responses.parse(
    model="gpt-5.4",
    instructions="Return only the structured object requested by the schema.",
    input="Summarize SAP AI Core for a developer evaluating SAP BTP.",
    text_format=CapabilitySummary,
    reasoning={"effort": "none"},
)

summary = structured_response.output_parsed
print(summary.model_dump())
```

## Images with OpenAI Responses

Encode local images as data URLs and pass `input_image` content blocks:

```python
import base64
import mimetypes
from pathlib import Path

from gen_ai_hub.proxy.native.openai import responses


def load_image_as_base64(path: str) -> tuple[str, str]:
    """Return base64 data and MIME type for an image file.

    Args:
        path: Local path to an image file.

    Returns:
        A tuple containing the base64-encoded image and detected MIME type.
    """

    image_bytes = Path(path).read_bytes()
    mime_type = mimetypes.guess_type(path)[0] or "image/png"
    return base64.b64encode(image_bytes).decode("utf-8"), mime_type


base64_data, mime_type = load_image_as_base64("SAP_logo.png")

response = responses.create(
    model="gpt-5.4",
    instructions="Describe the image briefly and call out visible brand text.",
    input=[
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": "What is in this image?"},
                {
                    "type": "input_image",
                    "image_url": f"data:{mime_type};base64,{base64_data}",
                },
            ],
        }
    ],
    reasoning={"effort": "none"},
)

print(response.output_text)
```

## PDF Files with OpenAI Responses

For PDF-capable OpenAI deployments, send the file as an `input_file` block. Do not manually extract PDF text unless the deployment lacks file support or the task requires custom preprocessing.

```python
import base64
from pathlib import Path

from gen_ai_hub.proxy.native.openai import responses

pdf_path = Path("sample_files/Paper ConTextTab.pdf")
pdf_base64 = base64.b64encode(pdf_path.read_bytes()).decode("utf-8")

response = responses.create(
    model="gpt-5.4",
    instructions="Summarize technical PDFs for developers in concise bullets.",
    input=[
        {
            "role": "user",
            "content": [
                {
                    "type": "input_file",
                    "filename": pdf_path.name,
                    "file_data": f"data:application/pdf;base64,{pdf_base64}",
                },
                {
                    "type": "input_text",
                    "text": "Summarize the problem, approach, key techniques, and main findings.",
                },
            ],
        }
    ],
    reasoning={"effort": "low"},
)

print(response.output_text)
```

## Reasoning Models

For `gpt-5` style models, use `reasoning` with Responses API and avoid chat-style `temperature`/`max_tokens` parameters unless the selected deployment explicitly supports them:

```python
from gen_ai_hub.proxy.native.openai import responses

reasoning_response = responses.create(
    model="gpt-5.4",
    input=(
        "A team wants to evaluate whether to move a chat-completions based "
        "SAP BTP assistant to the Responses API. Give a short migration checklist."
    ),
    reasoning={"effort": "low", "summary": "auto"},
)

print(reasoning_response.output_text)

for item in reasoning_response.output:
    if getattr(item, "type", None) == "reasoning":
        for summary in getattr(item, "summary", []) or []:
            print(getattr(summary, "text", summary))

follow_up = responses.create(
    model="gpt-5.4",
    previous_response_id=reasoning_response.id,
    input="Now turn the checklist into three concrete validation tests.",
    reasoning={"effort": "low", "summary": "auto"},
)

print(follow_up.output_text)
```

Use `previous_response_id` for supported stateful follow-ups. For stateless or zero-data-retention setups, keep the prior output items yourself and only rely on encrypted reasoning content when the selected deployment supports it.

## Native Chat Completions

Use chat completions for older OpenAI-shaped examples or compatible models that still use `messages` and return `choices[0].message.content`:

```python
from gen_ai_hub.proxy.native.openai import chat

model = "gpt-4o"
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is the capital of France?"},
]

response = chat.completions.create(
    messages=messages,
    model=model,
    temperature=0.6,
    max_tokens=1000,
)

print(response.choices[0].message.content)
```

For Perplexity deployments such as `sonar-pro`, use the same chat-completions shape and inspect returned `citations` or `search_results` when present.

## Native Amazon Bedrock and Anthropic

Use `Session().client(model_name=...)` and Bedrock `converse` message blocks:

```python
from gen_ai_hub.proxy.native.amazon.clients import Session

model = "anthropic--claude-4.5-sonnet"
bedrock = Session().client(model_name=model)

response = bedrock.converse(
    messages=[
        {
            "role": "user",
            "content": [{"text": "What is the capital of France?"}],
        }
    ],
    inferenceConfig={"maxTokens": 1000, "temperature": 0.6},
)

print(response["output"]["message"]["content"][0]["text"])
```

For Bedrock image input, include an image block with bytes:

```python
image_data = Path("SAP_logo.png").read_bytes()
messages = [
    {
        "role": "user",
        "content": [
            {"text": "What is the content of the image?"},
            {"image": {"format": "png", "source": {"bytes": image_data}}},
        ],
    }
]
```

## Native Gemini

Use the Google GenAI client through the Gen AI Hub proxy:

```python
from gen_ai_hub.proxy.core.proxy_clients import get_proxy_client
from gen_ai_hub.proxy.native.google_genai.clients import Client

proxy_client = get_proxy_client("gen-ai-hub")
client = Client(proxy_client=proxy_client)

response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents=[
        {
            "role": "user",
            "parts": [
                {"text": "What is in this image?"},
                {"inline_data": {"mime_type": "image/png", "data": base64_data}},
            ],
        }
    ],
)

print(response.text)
```

For Gemini PDFs, pass inline `application/pdf` bytes:

```python
from pathlib import Path

from google.genai import types


def build_gemini_pdf_contents(pdf_path: str, prompt: str) -> list[types.Content]:
    """Return Gemini contents with inline PDF bytes and a text prompt.

    Args:
        pdf_path: Local path to the PDF file.
        prompt: Instruction to apply to the PDF.

    Returns:
        A list containing one user content item with PDF and text parts.
    """

    path = Path(pdf_path)
    pdf_part = types.Part.from_bytes(
        data=path.read_bytes(),
        mime_type="application/pdf",
    )
    prompt_part = types.Part.from_text(text=prompt)
    return [types.Content(role="user", parts=[pdf_part, prompt_part])]


contents = build_gemini_pdf_contents(
    "sample_files/Paper ConTextTab.pdf",
    "Summarize the attached PDF in concise bullet points.",
)
response = client.models.generate_content(model="gemini-2.5-pro", contents=contents)
print(response.text)
```

Read Gemini token usage from `response.usage_metadata` with `prompt_token_count`, `candidates_token_count`, `total_token_count`, optional `cached_content_token_count`, and optional `thoughts_token_count`.

## LangChain Wrappers

Use LangChain wrappers only when the surrounding implementation needs LangChain abstractions:

```python
from gen_ai_hub.proxy.langchain.openai import ChatOpenAI

chat = ChatOpenAI(
    proxy_model_name="gpt-4o",
    max_tokens=1000,
    temperature=0.6,
)
response = chat.invoke("What is the capital of France?")
print(response.content)
```

For OpenAI reasoning models in LangChain, use `verbosity` and `reasoning_effort`:

```python
chat = ChatOpenAI(
    proxy_model_name="gpt-5",
    verbosity="medium",
    reasoning_effort="low",
)
response = chat.invoke("Explain Green's theorem briefly.")
print(response.content)
```

For Anthropic through Bedrock:

```python
from gen_ai_hub.proxy.langchain.amazon import ChatBedrock

chat = ChatBedrock(
    model_name="anthropic--claude-4.5-sonnet",
    temperature=0.6,
    model_kwargs={"max_tokens": 1000},
)
```

For Gemini:

```python
from gen_ai_hub.proxy.langchain.google_genai import ChatGoogleGenerativeAI

chat = ChatGoogleGenerativeAI(
    model="gemini-2.5-pro",
    temperature=0.6,
    max_output_tokens=1000,
)
```

LangChain responses usually expose text as `response.content` and token details under `response.usage_metadata` or `response.response_metadata`.

## Orchestration Service

Use orchestration for Gen AI Hub module composition. Keep the LLM config minimal when the selected model rejects classic generation parameters:

```python
from gen_ai_hub.orchestration.models.config import OrchestrationConfig
from gen_ai_hub.orchestration.models.llm import LLM
from gen_ai_hub.orchestration.models.message import SystemMessage, UserMessage
from gen_ai_hub.orchestration.models.template import Template
from gen_ai_hub.orchestration.service import OrchestrationService

template = Template(
    messages=[
        SystemMessage("You are a helpful assistant."),
        UserMessage("What is the capital of France?"),
    ]
)

config = OrchestrationConfig(
    template=template,
    llm=LLM(name="gpt-4o", version="latest"),
)

result = OrchestrationService(config=config).run()
print(result.orchestration_result.choices[0].message.content)
```

## Validation Checklist

- Confirm the environment loads before constructing clients.
- Run a minimal text call before adding images, PDFs, tools, chains, or orchestration modules.
- Check output access by API family: `output_text` for Responses, `output_parsed` for structured Responses, `choices[0].message.content` for chat completions and orchestration, `response.text` for Gemini, and `response.content` for LangChain.
- Print token usage when available because field names differ by API family.
- For PDF and image examples, assert that the response references the attached artifact, not just the prompt text.
- For reasoning examples, inspect reasoning summaries and reasoning token counts when returned.
- Keep functions and classes documented with docstrings. When creating CLI scripts, start the file with a module docstring that includes example commands.

## Related Skills

- `token-logger` — required token-usage logging for production APIs.
- `langgraph-genai-hub-setup` — use instead when the LLM lives inside a LangGraph app.
- `sap-btp-ai` — routing and shared environment conventions.
