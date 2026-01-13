"""
Verbose callback handler for real-time agent execution logging.

Provides streaming output of tool calls, arguments, results, and LLM reasoning
during agent execution.
"""

from __future__ import annotations

import json
import time
from typing import Any, Dict, List, Optional
from uuid import UUID

from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_core.messages import BaseMessage
from langchain_core.outputs import LLMResult


class VerboseCallbackHandler(BaseCallbackHandler):
    """
    Callback handler that prints real-time verbose output during agent execution.

    Displays:
    - Tool names and arguments when tools are called
    - Tool results (full or summarized based on truncate_results setting)
    - LLM reasoning/thought process
    - Execution timing for each operation
    """

    INDENT = "  "
    SEPARATOR = "-" * 50

    def __init__(
        self,
        show_llm_reasoning: bool = True,
        truncate_results: bool = False,
        result_max_chars: int = 500,
    ):
        """
        Initialize the verbose callback handler.

        Args:
            show_llm_reasoning: If True, display LLM thought process
            truncate_results: If True, truncate large results. Default False for debugging.
            result_max_chars: Max characters before truncating (only if truncate_results=True)
        """
        super().__init__()
        self.show_llm_reasoning = show_llm_reasoning
        self.truncate_results = truncate_results
        self.result_max_chars = result_max_chars
        self._tool_start_times: Dict[UUID, float] = {}
        self._llm_start_times: Dict[UUID, float] = {}
        self._tool_count = 0

    # =========================================================================
    # Tool Callbacks
    # =========================================================================

    def on_tool_start(
        self,
        serialized: Dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        inputs: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """Called when a tool starts executing."""
        self._tool_count += 1
        self._tool_start_times[run_id] = time.time()

        tool_name = serialized.get("name", "Unknown Tool")

        print(f"\n{self.SEPARATOR}")
        print(f"[TOOL {self._tool_count}] {tool_name}")
        print(self.SEPARATOR)

        # Display arguments
        print(f"{self.INDENT}Arguments:")
        if inputs:
            self._print_arguments(inputs)
        elif input_str:
            # Fallback to input_str if inputs not available
            try:
                parsed = json.loads(input_str)
                self._print_arguments(parsed)
            except json.JSONDecodeError:
                print(f"{self.INDENT}{self.INDENT}{input_str[:200]}")

        print(f"{self.INDENT}Status: Running...", flush=True)

    def on_tool_end(
        self,
        output: Any,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        """Called when a tool finishes executing."""
        elapsed = self._calculate_elapsed(run_id, self._tool_start_times)

        print(f"{self.INDENT}Status: Completed ({elapsed:.2f}s)")
        print(f"{self.INDENT}Result:")
        self._print_result(output)

    def on_tool_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        """Called when a tool encounters an error."""
        elapsed = self._calculate_elapsed(run_id, self._tool_start_times)

        print(f"{self.INDENT}Status: FAILED ({elapsed:.2f}s)")
        print(f"{self.INDENT}Error: {type(error).__name__}: {str(error)}")

    # =========================================================================
    # LLM Callbacks
    # =========================================================================

    def on_chat_model_start(
        self,
        serialized: Dict[str, Any],
        messages: List[List[BaseMessage]],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """Called when the chat model starts processing."""
        if not self.show_llm_reasoning:
            return

        self._llm_start_times[run_id] = time.time()
        print(f"\n[LLM] Thinking...", flush=True)

    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        """Called when the LLM finishes generating."""
        if not self.show_llm_reasoning:
            return

        elapsed = self._calculate_elapsed(run_id, self._llm_start_times)

        # Extract reasoning from response if available
        if response.generations:
            for gen_list in response.generations:
                for gen in gen_list:
                    # Check for tool calls in the message
                    if hasattr(gen, "message") and hasattr(gen.message, "tool_calls"):
                        if gen.message.tool_calls:
                            tool_names = [
                                tc.get("name", "unknown")
                                for tc in gen.message.tool_calls
                            ]
                            print(
                                f"[LLM] Decided to call: {', '.join(tool_names)} ({elapsed:.2f}s)"
                            )
                            return

                    # Check for content (final response)
                    if hasattr(gen, "message") and gen.message.content:
                        content = gen.message.content
                        # Handle both string (OpenAI) and list (Gemini) formats
                        has_content = False
                        if isinstance(content, str) and content.strip():
                            has_content = True
                        elif isinstance(content, list) and any(
                            isinstance(b, dict) and b.get("text") for b in content
                        ):
                            has_content = True
                        if has_content:
                            print(f"[LLM] Generated response ({elapsed:.2f}s)")
                            return

        print(f"[LLM] Completed ({elapsed:.2f}s)")

    # =========================================================================
    # Helper Methods
    # =========================================================================

    def _print_arguments(self, args: Dict[str, Any]) -> None:
        """Print tool arguments in a readable format."""
        for key, value in args.items():
            if isinstance(value, str) and len(value) > 100:
                value = value[:100] + "..."
            elif isinstance(value, (list, dict)):
                value_str = json.dumps(value, indent=2)
                if len(value_str) > 200:
                    value_str = value_str[:200] + "..."
                value = value_str
            print(f"{self.INDENT}{self.INDENT}{key}: {value}")

    def _print_result(self, output: Any) -> None:
        """Print tool result, optionally summarizing if large."""
        if isinstance(output, str):
            result_str = output
        else:
            try:
                result_str = json.dumps(output, indent=2, default=str)
            except (TypeError, ValueError):
                result_str = str(output)

        if self.truncate_results and len(result_str) > self.result_max_chars:
            # Summarize large results only if truncation is enabled
            self._print_summarized_result(output, result_str)
        else:
            # Print full result
            for line in result_str.split("\n"):
                print(f"{self.INDENT}{self.INDENT}{line}")

    def _print_summarized_result(self, output: Any, full_str: str) -> None:
        """Print a summarized version of large results."""
        if isinstance(output, dict):
            # Show key structure and status
            keys = list(output.keys())
            status = output.get("status", output.get("message", "N/A"))
            error = output.get("error")

            if error:
                print(f"{self.INDENT}{self.INDENT}Error: {error}")
            else:
                print(f"{self.INDENT}{self.INDENT}Status: {status}")
                print(f"{self.INDENT}{self.INDENT}Keys: {', '.join(keys[:10])}")
                if len(keys) > 10:
                    print(
                        f"{self.INDENT}{self.INDENT}... and {len(keys) - 10} more keys"
                    )
                print(f"{self.INDENT}{self.INDENT}[{len(full_str)} chars total]")
        else:
            # Show truncated output
            print(f"{self.INDENT}{self.INDENT}{full_str[:300]}...")
            print(f"{self.INDENT}{self.INDENT}[{len(full_str)} chars total]")

    def _calculate_elapsed(
        self, run_id: UUID, start_times: Dict[UUID, float]
    ) -> float:
        """Calculate elapsed time for a run."""
        start_time = start_times.pop(run_id, None)
        if start_time:
            return time.time() - start_time
        return 0.0


class TokenTrackingCallback(BaseCallbackHandler):
    """
    Callback handler that tracks token usage across LLM calls.

    Accumulates input/output/total tokens and provides both real-time
    printing and final totals retrieval. Designed for use in interactive
    sessions where per-message and session-level tracking is needed.
    """

    def __init__(self, print_per_call: bool = False):
        """
        Initialize the token tracking callback.

        Args:
            print_per_call: If True, print token usage after each LLM call.
                           If False, only accumulate (for batch printing later).
        """
        super().__init__()
        self.print_per_call = print_per_call
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_tokens = 0
        self.call_count = 0
        # Track per-message deltas for interactive mode
        self._last_checkpoint_input = 0
        self._last_checkpoint_output = 0
        self._last_checkpoint_total = 0
        self._message_count = 0

    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        """Called when the LLM finishes generating. Extract and accumulate token usage."""
        # Extract token usage from response.llm_output
        token_usage = {}
        if response.llm_output:
            token_usage = response.llm_output.get("token_usage", {})

        input_tokens = token_usage.get("prompt_tokens", 0)
        output_tokens = token_usage.get("completion_tokens", 0)
        total = token_usage.get("total_tokens", input_tokens + output_tokens)

        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.total_tokens += total
        self.call_count += 1

        if self.print_per_call and (input_tokens > 0 or output_tokens > 0):
            print(
                f"[TOKEN] Call {self.call_count}: +{input_tokens:,} in / +{output_tokens:,} out "
                f"(cumulative: {self.total_tokens:,} total)"
            )

    def get_totals(self) -> Dict[str, int]:
        """
        Get accumulated token usage totals.

        Returns:
            Dict with input_tokens, output_tokens, total_tokens, llm_calls
        """
        return {
            "input_tokens": self.total_input_tokens,
            "output_tokens": self.total_output_tokens,
            "total_tokens": self.total_tokens,
            "llm_calls": self.call_count,
        }

    def get_message_delta(self) -> Dict[str, int]:
        """
        Get token usage since last checkpoint (for per-message tracking).

        Returns:
            Dict with delta input/output/total tokens since last checkpoint
        """
        delta = {
            "input_tokens": self.total_input_tokens - self._last_checkpoint_input,
            "output_tokens": self.total_output_tokens - self._last_checkpoint_output,
            "total_tokens": self.total_tokens - self._last_checkpoint_total,
        }
        # Update checkpoint
        self._last_checkpoint_input = self.total_input_tokens
        self._last_checkpoint_output = self.total_output_tokens
        self._last_checkpoint_total = self.total_tokens
        self._message_count += 1
        return delta

    def get_message_count(self) -> int:
        """Get number of messages processed (checkpoints taken)."""
        return self._message_count

    def reset(self) -> None:
        """Reset all counters (e.g., on /clear command)."""
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_tokens = 0
        self.call_count = 0
        self._last_checkpoint_input = 0
        self._last_checkpoint_output = 0
        self._last_checkpoint_total = 0
        self._message_count = 0

    def print_session_summary(self) -> None:
        """Print formatted session token usage summary."""
        print("\n" + "=" * 60)
        print("SESSION TOKEN USAGE")
        print("=" * 60)
        print(f"Total messages: {self._message_count}")
        print(f"Total LLM calls: {self.call_count}")
        print(f"Input tokens:  {self.total_input_tokens:,}")
        print(f"Output tokens: {self.total_output_tokens:,}")
        print(f"Total tokens:  {self.total_tokens:,}")
        print("=" * 60)


__all__ = ["VerboseCallbackHandler", "TokenTrackingCallback"]
