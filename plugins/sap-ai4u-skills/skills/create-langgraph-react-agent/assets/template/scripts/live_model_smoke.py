"""Smoke-test configured SAP Gen AI Hub providers and optional attachments.

Examples:
    .venv/bin/python -m scripts.live_model_smoke --model openai=gpt-5.4
    .venv/bin/python -m scripts.live_model_smoke \
        --model openai=gpt-5.4 \
        --model gemini=gemini-2.5-pro \
        --model claude=anthropic--claude-4.6-sonnet \
        --image sample.png --pdf sample.pdf
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from pydantic import BaseModel
from tqdm import tqdm

from template_agent.config import ModelSettings
from template_agent.models import Attachment
from template_agent.providers import create_chat_model


class SmokeResult(BaseModel):
    """Represent the minimal provider-native structured-output contract."""

    status: Literal["ok"]
    provider: str


@tool
def add(left: int, right: int) -> int:
    """Add two integers and return their sum."""

    return left + right


def build_parser() -> argparse.ArgumentParser:
    """Return CLI arguments for one or more provider/model pairs."""

    parser = argparse.ArgumentParser(description="Live SAP model capability smoke test")
    parser.add_argument(
        "--model",
        action="append",
        required=True,
        metavar="PROVIDER=MODEL",
        help="Repeat for openai, gemini, and/or claude",
    )
    parser.add_argument("--image", type=Path)
    parser.add_argument("--pdf", type=Path)
    return parser


def parse_model(value: str) -> tuple[str, str]:
    """Split and validate one provider=model argument."""

    provider, separator, model = value.partition("=")
    if not separator or provider not in {"openai", "gemini", "claude"} or not model:
        raise ValueError(f"Invalid --model value: {value!r}")
    return provider, model


async def tool_round_trip(provider: str, model_name: str) -> str:
    """Call a tool, send its result back and return the model's final answer.

    The second request is the one that carries the first answer's history (for
    GPT on the Responses API: function call plus reasoning item). It is where SAP
    AI Core used to answer ``404 Resource not found`` for replayed reasoning, so a
    high reasoning effort is used to make sure a reasoning item is produced.
    """

    settings = ModelSettings(
        provider=provider,
        name=model_name,
        reasoning_effort="high" if provider == "openai" else None,
        use_responses_api=True,
    )
    model = create_chat_model(settings).bind_tools([add])
    messages: list[BaseMessage] = [
        HumanMessage(
            # A task that needs planning: trivial prompts produce no reasoning item
            # and would not exercise the replay path.
            content="A warehouse ships 3 pallets of 48 cans and 5 pallets of 36 cans; 7 cans "
            "arrive damaged. Plan the calculation first, then call the add tool exactly once "
            "with left = all cans shipped and right = minus the damaged cans, and answer with "
            "the number of usable cans."
        )
    ]
    first = await model.ainvoke(messages)
    if not first.tool_calls:
        raise RuntimeError("Round trip: model did not call the add tool")
    call = first.tool_calls[0]
    messages += [first, ToolMessage(content=str(add.invoke(call["args"])), tool_call_id=call["id"])]
    final = await model.ainvoke(messages)
    if final.tool_calls or not final.text:
        raise RuntimeError("Round trip: no final answer after the tool result")
    return final.text


async def smoke_one(
    provider: str,
    model_name: str,
    attachments: list[Attachment],
) -> dict[str, object]:
    """Run text, tool, tool round-trip, structured-output, and optional attachment checks."""

    settings = ModelSettings(
        provider=provider,
        name=model_name,
        reasoning_effort="low" if provider == "openai" else None,
        use_responses_api=True,
    )
    model = create_chat_model(settings)
    text = await model.ainvoke("Reply with exactly: text-ok")
    tool_response = await model.bind_tools([add]).ainvoke(
        "Call the add tool with left=2 and right=3."
    )
    if not tool_response.tool_calls:
        raise RuntimeError("Model did not return the requested tool call")
    parsed = await model.with_structured_output(SmokeResult).ainvoke(
        f"Return status='ok' and provider='{provider}'."
    )
    attachment_text: str | None = None
    if attachments:
        content = [{"type": "text", "text": "Identify the supplied attachment types."}]
        content.extend(item.to_content_block(provider) for item in attachments)
        response = await model.ainvoke([HumanMessage(content=content)])
        attachment_text = str(response.content)
    round_trip = await tool_round_trip(provider, model_name)
    if provider == "openai" and getattr(model, "use_responses_api", None) is not True:
        raise RuntimeError("OpenAI model was not configured for Responses API")
    return {
        "provider": provider,
        "model": model_name,
        "text": str(text.content),
        "tool": tool_response.tool_calls[0]["name"],
        "tool_round_trip": round_trip,
        "structured": parsed.model_dump(mode="json") if hasattr(parsed, "model_dump") else parsed,
        "attachments": attachment_text,
    }


async def run(args: argparse.Namespace) -> int:
    """Run every requested model sequentially and print safe JSON diagnostics."""

    load_dotenv(override=False)
    attachments = [
        Attachment(path=path)
        for path in (args.image, args.pdf)
        if path is not None
    ]
    results: list[dict[str, object]] = []
    failures: list[dict[str, str]] = []
    models = [parse_model(value) for value in args.model]
    for provider, model_name in tqdm(models, desc="Provider smoke tests", unit="model"):
        try:
            results.append(await smoke_one(provider, model_name, attachments))
        except Exception as exc:
            failures.append(
                {
                    "provider": provider,
                    "model": model_name,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
    print(json.dumps({"results": results, "failures": failures}, indent=2, ensure_ascii=False))
    return 1 if failures else 0


def main() -> int:
    """Run the async smoke test from a synchronous shell entry point."""

    return asyncio.run(run(build_parser().parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
