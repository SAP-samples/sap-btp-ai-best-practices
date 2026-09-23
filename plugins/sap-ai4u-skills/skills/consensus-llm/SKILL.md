---
name: consensus-llm
description: Install, integrate, test, or troubleshoot the reusable consensus_llm Python module for configurable multi-model fan-out and judge synthesis through SAP Generative AI Hub. Use for normal text, image, or PDF requests that should run across arbitrary GPT Responses, Gemini, Claude Bedrock, or explicit chat-compatible model mixtures and return one final answer with quorum, usage, and failure diagnostics.
---

# Consensus LLM

Use the bundled `consensus_llm` wheel to add a general multi-model call without LangGraph or SAP Orchestration.

## Integrate the package

1. Inspect the target Python environment and reuse an installed `consensus-llm` version when suitable.
2. Install `pydantic>=2.0`, `sap-ai-sdk-gen[all]>=7.2.0`, and `python-dotenv` if missing.
3. Resolve this skill's directory and install `assets/consensus_llm-0.2.0-py3-none-any.whl`; never hardcode the source repository path.
4. Load the standard `AICORE_*` credentials before the first client is constructed.
5. Smoke-test one short text request for every selected native backend before adding attachments.

## Make a consensus call

```python
from consensus_llm import ModelConfig, responses

response = await responses.create(
    instructions="Review the document and identify contractual risks.",
    input=[
        {
            "role": "user",
            "content": [
                {"type": "input_file", "path": "contract.pdf"},
                {"type": "input_text", "text": "What should we negotiate?"},
            ],
        }
    ],
    models=[
        ModelConfig(model="gemini-2.5-pro", copies=5),
        ModelConfig(
            model="gpt-5.4",
            copies=3,
            reasoning={"effort": "low"},
        ),
        ModelConfig(model="anthropic--claude-4.6-sonnet"),
    ],
    judge=ModelConfig(
        model="gpt-5.5",
        reasoning={"effort": "medium"},
    ),
    max_concurrency=5,
    min_successes=2,
)

print(response.output_text)
```

## Use a shared structured-output model

Pass a Pydantic `BaseModel` subclass as `output_model` when every panel and the
judge must return the same JSON contract:

```python
from typing import Literal

from pydantic import BaseModel, ConfigDict
from consensus_llm import ModelConfig, responses


class ReviewResult(BaseModel):
    """Portable result contract returned by panels and the judge."""

    model_config = ConfigDict(extra="forbid")

    decision: Literal["approve", "reject", "needs_review"]
    rationale: str
    risks: list[str]


response = await responses.create(
    input="Review the proposal and identify the main risks.",
    models=[
        ModelConfig(model="gpt-5.4", reasoning={"effort": "low"}),
        ModelConfig(model="gemini-2.5-pro"),
        ModelConfig(model="anthropic--claude-4.6-sonnet"),
    ],
    judge=ModelConfig(model="gpt-5.5", reasoning={"effort": "medium"}),
    min_successes=2,
    output_model=ReviewResult,
)

final_result: ReviewResult = response.output_parsed
print(final_result.model_dump())
```

Structured mode uses native provider formatting where available: OpenAI
Responses `responses.parse`, OpenAI chat `chat.completions.parse`, Gemini JSON
`response_schema`, and Bedrock Converse `outputConfig.textFormat`. The library
also validates every response with `output_model.model_validate_json`; invalid
JSON or schema-invalid output is recorded as a failed candidate and does not
count toward `min_successes`. `response.output_text` remains available as
canonical JSON, while `response.output_parsed` and each candidate's
`output_parsed` contain typed Pydantic objects.

Keep schemas portable across providers: prefer simple objects, arrays, scalar
types, enums, and nullable fields. Structured output is provider/deployment
dependent, and native JSON Schema support is a subset that can reject complex
recursive schemas or unsupported constraints.

Keep the caller's requested output format unchanged. Do not impose an extraction schema unless the caller requested one.

## Select native backends

- Route `gpt-*`, `o1*`, `o3*`, and `o4*` through OpenAI Responses. Always keep GPT-5.4 and GPT-5.5 on Responses.
- Route `gemini-*` through the native Google GenAI client.
- Route `anthropic--*` and Claude names through Amazon Bedrock Converse.
- Set `backend="openai_chat"` explicitly for compatible models such as `sonar-pro`.
- Pass provider-native generation settings through `parameters`; use `reasoning` only with OpenAI Responses.

Preserve repeated model entries or `copies` as real independent calls. Do not reinterpret them as weights or silently diversify their prompts.

## Handle files and safety

Accept Responses-style `input_text`, local `input_image`, and local PDF `input_file` blocks. The module converts local paths to each provider's native representation.

Before a live call, explicitly verify that the user authorizes sending every attachment to every configured panel deployment and the judge deployment. Do not retry a rejected private-document disclosure through another path.

## Verify results

- Require at least two expanded panel calls.
- Choose `min_successes` deliberately; inspect `response.partial` and `response.candidates`.
- Catch `ConsensusError` and retain its candidate diagnostics.
- Record `response.usage` because the request consumes every panel call plus the judge.
- Compare individual candidates and the final answer against independently verified facts before claiming improved accuracy.
- Apply the `access-to-generative-ai-models` skill when changing SAP SDK provider payloads or adding a new model family.

The bundled version deliberately excludes persistence, UI, LangGraph, automatic retries, and model voting. Add them only when a concrete project requires them.
