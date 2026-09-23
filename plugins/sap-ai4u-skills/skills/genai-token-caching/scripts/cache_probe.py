"""Turn-by-turn prompt-cache probe for SAP Gen AI Hub deployments.

Sends an append-only, growing conversation to one provider and prints
normalized per-turn usage (raw input, uncached input, cache read, cache write,
and output) so you can see what the active model/API/route reports before
relying on it in an agent. Reported and derived cache writes are labelled
separately. Platform behavior drifts, so rerun this after model, SDK, route,
or tenant changes instead of trusting a provider-wide assumption.

Requires a `.env` with SAP AI Core credentials (AICORE_AUTH_URL,
AICORE_CLIENT_ID, AICORE_CLIENT_SECRET, AICORE_BASE_URL,
AICORE_RESOURCE_GROUP) and `sap-ai-sdk-gen[all]` + `python-dotenv` installed.

Examples:
    python cache_probe.py --provider openai --model gpt-5.6-luna --cache-mode implicit --turns 4
    python cache_probe.py --provider openai --model gpt-5.6-luna --cache-mode explicit --turns 4
    python cache_probe.py --provider bedrock --model anthropic--claude-4.6-sonnet --cache-mode explicit --turns 4
    python cache_probe.py --provider gemini --model gemini-3.5-flash --cache-mode implicit --turns 4 --filler-facts 1200
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any
from uuid import uuid4

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - tqdm is optional for copied skills.
    def tqdm(items, **_: Any):
        """Return items unchanged when tqdm is unavailable.

        Args:
            items: Iterable to return.
            **_: Ignored tqdm-compatible keyword arguments.

        Returns:
            The original iterable.
        """

        return items

sys.path.insert(0, str(Path(__file__).resolve().parent))
from normalize_usage import add_derived_cache_writes, normalize_usage  # noqa: E402

Message = dict[str, str]

FILLER_PREFIX = "Reference material follows, unchanged across every turn so it can be cached:"


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the cache probe.

    Returns:
        Parsed command-line arguments.
    """

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--provider", required=True, choices=["openai", "bedrock", "gemini"])
    parser.add_argument("--model", required=True, help="SAP AI Core deployment/model name.")
    parser.add_argument("--turns", type=int, default=6)
    parser.add_argument(
        "--cache-mode",
        choices=["implicit", "explicit"],
        default="implicit",
        help="Cache control to request and measure; support is model/API/route specific.",
    )
    parser.add_argument(
        "--no-nonce",
        action="store_true",
        help="Keep the system prompt byte-identical across probe runs instead of "
        "salting it, so you can observe a cache warmed by an earlier run.",
    )
    parser.add_argument(
        "--filler-facts",
        type=int,
        default=400,
        help="Stable fact count used to make repeated-prefix behavior measurable.",
    )
    return parser.parse_args()


def build_filler(filler_facts: int) -> str:
    """Return stable prompt filler used to cross cache thresholds.

    Args:
        filler_facts: Number of stable facts to include.

    Returns:
        Stable text appended to the system prompt.
    """

    return f"{FILLER_PREFIX} " + " ".join(
        f"fact-{index}: the value is stable." for index in range(filler_facts)
    )


def build_system_prompt(no_nonce: bool, filler_facts: int) -> str:
    """Return a stable system prompt, optionally salted per run.

    Args:
        no_nonce: Whether to omit the per-run nonce.
        filler_facts: Number of stable filler facts.

    Returns:
        System prompt text for all turns in this probe.
    """

    nonce = "" if no_nonce else f"session {uuid4()}. "
    return f"You are a terse test assistant. {nonce}{build_filler(filler_facts)}"


def run(args: argparse.Namespace) -> None:
    """Run the selected provider probe and print a compact usage table.

    Args:
        args: Parsed CLI arguments.

    Returns:
        None.
    """

    from dotenv import load_dotenv

    load_dotenv(dotenv_path=Path(".env"))
    system_prompt = build_system_prompt(args.no_nonce, args.filler_facts)
    messages: list[Message] = []
    state: dict[str, Any] = {}

    caller = {"openai": call_openai, "bedrock": call_bedrock, "gemini": call_gemini}[args.provider]

    rows: list[dict[str, Any]] = []
    for turn in tqdm(range(1, args.turns + 1), desc="cache probe"):
        messages.append(
            {"role": "user", "content": f"Turn {turn}: acknowledge item {uuid4()} in one short sentence."}
        )
        text, response = caller(args.model, messages, system_prompt, args.cache_mode, state)
        messages.append({"role": "assistant", "content": text})
        rows.append(normalize_usage(args.provider, response))

    # Derive a write only when the response shape omitted one. The output keeps
    # reported and derived values visibly separate.
    add_derived_cache_writes(rows)

    print()
    print(
        f"{'turn':>4}  {'raw_in':>7}  {'uncached':>8}  {'cache_read':>10}  "
        f"{'cache_write':>11}  {'write_source':>12}  {'out':>5}"
    )
    for turn, usage in enumerate(rows, start=1):
        write = usage["cache_write_input_tokens"]
        if usage["cache_write_reported"]:
            write_label = write or 0
            write_source = "reported"
        elif usage["cache_write_derived_tokens"] is not None:
            write_label = usage["cache_write_derived_tokens"]
            write_source = "derived"
        else:
            write_label = "—"
            write_source = "missing"
        print(
            f"{turn:>4}  {usage['provider_input_tokens'] or 0:>7}  "
            f"{usage['uncached_input_tokens'] or 0:>8}  "
            f"{usage['cache_read_input_tokens'] or 0:>10}  "
            f"{write_label!s:>11}  "
            f"{write_source:>12}  "
            f"{usage['output_tokens'] or 0:>5}"
        )


def call_openai(
    model: str, messages: list[Message], system_prompt: str, cache_mode: str, state: dict[str, Any]
) -> tuple[str, Any]:
    """Call GPT Responses through SAP Gen AI Hub's LangChain wrapper.

    Args:
        model: SAP AI Core OpenAI deployment name.
        messages: Append-only conversation messages.
        system_prompt: Stable system prompt.
        cache_mode: ``implicit`` or ``explicit``.
        state: Mutable provider state across turns.

    Returns:
        Assistant text and raw response.
    """

    from gen_ai_hub.proxy.langchain.openai import ChatOpenAI

    if "model" not in state:
        client = ChatOpenAI(
            proxy_model_name=model,
            use_responses_api=True,
            prompt_cache_options={"mode": cache_mode},
        )
        # The current SAP wrapper otherwise forwards the Chat Completions
        # default n=1 to Responses, whose client rejects that parameter.
        client.n = None
        state["model"] = client
        state["cache_key"] = f"cache-probe-{uuid4()}"

    breakpoint = {"mode": "explicit"}
    system_content: str | list[dict[str, Any]] = system_prompt
    if cache_mode == "explicit":
        system_content = [
            {
                "type": "text",
                "text": system_prompt,
                "prompt_cache_breakpoint": breakpoint,
            }
        ]

    send: list[dict[str, Any]] = [{"role": "system", "content": system_content}]
    for index, message in enumerate(messages):
        content: str | list[dict[str, Any]] = message["content"]
        if cache_mode == "explicit" and index == len(messages) - 1:
            content = [
                {
                    "type": "text",
                    "text": message["content"],
                    "prompt_cache_breakpoint": breakpoint,
                }
            ]
        send.append({"role": message["role"], "content": content})

    response = state["model"].invoke(
        send,
        prompt_cache_key=state["cache_key"],
    )
    return getattr(response, "text", ""), response


def to_converse_messages(messages: list[Message]) -> list[dict[str, Any]]:
    """Convert append-only messages into Bedrock Converse messages.

    Args:
        messages: Plain ``user``/``assistant`` messages.

    Returns:
        Bedrock Converse message list with consecutive roles merged.
    """

    converse: list[dict[str, Any]] = []
    for message in messages:
        block = {"text": message["content"]}
        if converse and converse[-1]["role"] == message["role"]:
            converse[-1]["content"].append(block)
        else:
            converse.append({"role": message["role"], "content": [block]})
    return converse


def call_bedrock(
    model: str, messages: list[Message], system_prompt: str, cache_mode: str, state: dict[str, Any]
) -> tuple[str, Any]:
    """Call Claude on Bedrock through SAP Gen AI Hub.

    Args:
        model: SAP AI Core Anthropic deployment name.
        messages: Append-only conversation messages.
        system_prompt: Stable system prompt.
        cache_mode: ``implicit`` or ``explicit``.
        state: Mutable provider state across turns.

    Returns:
        Assistant text and raw Bedrock response.
    """

    from gen_ai_hub.proxy.native.amazon.clients import Session

    client = Session().client(model_name=model)
    system: list[dict[str, Any]] = [{"text": system_prompt}]
    converse_messages = to_converse_messages(messages)
    if cache_mode == "explicit":
        system.append({"cachePoint": {"type": "default"}})
        if converse_messages:
            converse_messages[-1]["content"].append({"cachePoint": {"type": "default"}})
    response = client.converse(
        system=system, messages=converse_messages, inferenceConfig={"maxTokens": 200}
    )
    content = response.get("output", {}).get("message", {}).get("content", [])
    text = content[0].get("text", "") if content else ""
    return text, response


def call_gemini(
    model: str, messages: list[Message], system_prompt: str, cache_mode: str, state: dict[str, Any]
) -> tuple[str, Any]:
    """Call Gemini through SAP Gen AI Hub.

    Args:
        model: SAP AI Core Gemini deployment name.
        messages: Append-only conversation messages.
        system_prompt: Stable system prompt.
        cache_mode: ``implicit`` or ``explicit``.
        state: Mutable provider state across turns.

    Returns:
        Assistant text and Gemini usage metadata.
    """

    from gen_ai_hub.proxy.core.proxy_clients import get_proxy_client
    from gen_ai_hub.proxy.native.google_genai.clients import Client
    from google.genai import types

    client = Client(proxy_client=get_proxy_client("gen-ai-hub"))
    config = types.GenerateContentConfig(system_instruction=system_prompt)
    if cache_mode == "explicit":
        if "cached_content" not in state:
            try:
                cached = client.caches.create(
                    model=model,
                    config=types.CreateCachedContentConfig(
                        system_instruction=system_prompt, ttl="3600s"
                    ),
                )
                state["cached_content"] = cached.name
            except Exception as exc:
                raise RuntimeError(
                    "Explicit Gemini cache creation failed on this model/API/route; "
                    "run the implicit control separately. "
                    f"Error type: {type(exc).__name__}."
                ) from None
        config = types.GenerateContentConfig(cached_content=state["cached_content"])
    contents = [
        {"role": "user" if m["role"] == "user" else "model", "parts": [{"text": m["content"]}]}
        for m in messages
    ]
    response = client.models.generate_content(model=model, contents=contents, config=config)
    return getattr(response, "text", ""), getattr(response, "usage_metadata", None)


if __name__ == "__main__":
    run(parse_args())
