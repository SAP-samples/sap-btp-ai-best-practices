"""Context-budget regressions for the browser-session optimizer assistant."""

from pathlib import Path

import asyncio

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from app.agent.config import ContextBudgetSettings, load_config


def test_context_budget_defaults_below_the_provider_ceiling():
    """The production configuration reserves headroom before the one-million-token limit."""
    config = load_config(Path(__file__).parents[1] / "app/agent/config/agent.yaml")

    assert config.context_budget.hard_request_bytes < 4_000_000
    assert config.context_budget.summarize_at_bytes < config.context_budget.hard_request_bytes
    assert config.context_budget.max_tool_result_bytes < config.context_budget.summarize_at_bytes
    assert "get_run_failure_diagnostics" in config.base_prompt


def test_context_budget_rejects_thresholds_that_cannot_compact_before_hard_cap():
    """Misconfiguration cannot silently postpone compaction beyond the hard limit."""
    import pytest

    with pytest.raises(ValueError, match="summarize_at_bytes"):
        ContextBudgetSettings(hard_request_bytes=100_000, summarize_at_bytes=100_000)


def test_oversized_tool_output_is_compacted_before_a_provider_request():
    """A completed tool pair remains intact while its oversized payload is replaced."""
    from app.agent.context_budget import ContextBudgetMiddleware

    budget = ContextBudgetSettings(
        hard_request_bytes=100_000,
        summarize_at_bytes=50_000,
        max_tool_result_bytes=1_000,
        max_summary_input_bytes=10_000,
        max_summary_characters=1_000,
        summary_max_tokens=100,
    )
    tool_call = AIMessage(
        content="",
        tool_calls=[{"id": "call-1", "name": "list_runs", "args": {}}],
    )
    messages = [
        HumanMessage(content="Please inspect the failed run."),
        tool_call,
        ToolMessage(content="x" * 8_000, tool_call_id="call-1", name="list_runs"),
    ]

    prepared = asyncio.run(
        ContextBudgetMiddleware(budget, tools=[]).prepare("system", messages, "turn-1")
    )

    compact_tool = prepared.messages[-1]
    assert isinstance(compact_tool, ToolMessage)
    assert compact_tool.tool_call_id == "call-1"
    assert "tool_output_omitted" in str(compact_tool.content)
    assert prepared.final_request_bytes <= budget.summarize_at_bytes
    assert prepared.actions == ("compact_tool_outputs",)


def test_model_summary_keeps_recent_exchange_and_complete_tool_pairs():
    """Older turns compact into browser context without splitting current tool work."""
    from app.agent.context_budget import ContextBudgetMiddleware

    class Summarizer:
        """Return a fixed safe summary and capture the no-tool prompt."""

        def __init__(self):
            """Start with no invocation payload."""
            self.messages = None

        async def ainvoke(self, messages):
            """Record summary input and return a bounded assistant response."""
            self.messages = messages
            return AIMessage(content="Older run r failed before publication.")

    budget = ContextBudgetSettings(
        hard_request_bytes=100_000,
        summarize_at_bytes=50_000,
        max_tool_result_bytes=1_000,
        max_summary_input_bytes=10_000,
        max_summary_characters=1_000,
        summary_max_tokens=100,
    )
    summarizer = Summarizer()
    tool_call = AIMessage(
        content="",
        tool_calls=[{"id": "call-2", "name": "get_run_status", "args": {"run_id": "r"}}],
    )
    current = HumanMessage(content="Explain the failure now.")
    prepared = asyncio.run(
        ContextBudgetMiddleware(budget, tools=[], summarizer=summarizer).prepare(
            "system",
            [
                HumanMessage(content="old user " * 10_000),
                AIMessage(content="old assistant " * 10_000),
                HumanMessage(content="recent question"),
                tool_call,
                ToolMessage(content="small result", tool_call_id="call-2", name="get_run_status"),
                current,
            ],
            "turn-2",
        )
    )

    assert summarizer.messages is not None
    assert "Older run r failed" in prepared.context_summary
    assert any(message is current for message in prepared.messages)
    assert any(
        isinstance(message, AIMessage) and message.tool_calls and message.tool_calls[0]["id"] == "call-2"
        for message in prepared.messages
    )
    assert any(
        isinstance(message, ToolMessage) and message.tool_call_id == "call-2"
        for message in prepared.messages
    )


def test_summary_failure_falls_back_to_a_deterministic_under_budget_omission():
    """A failed summary call never permits the main provider request to overflow."""
    from app.agent.context_budget import ContextBudgetMiddleware

    class FailingSummarizer:
        """Simulate an unavailable summary-model request."""

        async def ainvoke(self, messages):
            """Fail without returning generated content."""
            raise RuntimeError("summary provider unavailable")

    budget = ContextBudgetSettings(
        hard_request_bytes=100_000,
        summarize_at_bytes=50_000,
        max_tool_result_bytes=1_000,
        max_summary_input_bytes=10_000,
        max_summary_characters=1_000,
        summary_max_tokens=100,
    )
    current = HumanMessage(content="current request")
    prepared = asyncio.run(
        ContextBudgetMiddleware(budget, tools=[], summarizer=FailingSummarizer()).prepare(
            "system", [HumanMessage(content="old " * 20_000), current], "turn-3"
        )
    )

    assert prepared.final_request_bytes <= budget.summarize_at_bytes
    assert "deterministically omitted" in prepared.context_summary
    assert prepared.actions == ("summarization_failed", "deterministic_omission")
