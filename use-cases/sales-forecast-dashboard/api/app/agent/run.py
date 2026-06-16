#!/usr/bin/env python3
"""
CLI Entry Point for the What-If Forecasting Agent.

This module provides command-line interfaces for running the agent
in various modes: single query, interactive chat, and batch processing.

Usage:
    # Single query mode
    python -m forecasting.agent.run "What if brand awareness increases 10%?"

    # Interactive mode
    python -m forecasting.agent.run --interactive

    # List available tools
    python -m forecasting.agent.run --list-tools

    # Export agent graph
    python -m forecasting.agent.run --export-graph
"""
from __future__ import annotations
from dotenv import load_dotenv
load_dotenv()

import argparse
import sys
from typing import Optional


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="What-If Forecasting Agent CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Ask a single question
  python -m forecasting.agent.run "What if brand awareness increases 10% in NYC?"

  # Start interactive chat
  python -m forecasting.agent.run --interactive

  # List all available tools
  python -m forecasting.agent.run --list-tools

  # Show agent info
  python -m forecasting.agent.run --info

  # Export agent graph visualization
  python -m forecasting.agent.run --export-graph agent_graph.png
        """,
    )

    # Positional argument for query
    parser.add_argument(
        "query",
        nargs="?",
        help="Question or request for the agent",
    )

    # Mode flags
    parser.add_argument(
        "-i", "--interactive",
        action="store_true",
        help="Start interactive chat mode",
    )

    parser.add_argument(
        "--list-tools",
        action="store_true",
        help="List all available tools",
    )

    parser.add_argument(
        "--info",
        action="store_true",
        help="Show agent information",
    )

    parser.add_argument(
        "--export-graph",
        metavar="PATH",
        help="Export agent graph to PNG file",
    )

    # LLM configuration
    parser.add_argument(
        "--provider",
        default="vertexai",
        choices=["openai", "bedrock", "vertexai"],
        help="LLM provider (default: vertexai)",
    )

    parser.add_argument(
        "--model",
        default="gemini-2.5-flash",
        help="Model name (default: gemini-2.5-flash)",
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature (default: 0.0)",
    )

    # Output options
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Show verbose output including tool calls",
    )

    parser.add_argument(
        "--token",
        action="store_true",
        help="Track and display token usage (per-message and session totals)",
    )

    return parser.parse_args()


def list_tools():
    """Print list of all available tools."""
    from .tools import TOOL_CATEGORIES, get_tool_count

    print("\n" + "=" * 60)
    print("What-If Forecasting Agent - Available Tools")
    print("=" * 60)
    print(f"\nTotal tools: {get_tool_count()}\n")

    for category, tools in TOOL_CATEGORIES.items():
        print(f"\n{category.upper().replace('_', ' ')} ({len(tools)} tools)")
        print("-" * 40)
        for tool in tools:
            # Get tool description from docstring
            doc = tool.description if hasattr(tool, "description") else ""
            first_line = doc.split("\n")[0] if doc else "No description"
            print(f"  {tool.name}")
            print(f"    {first_line[:60]}...")

    print("\n" + "=" * 60)


def show_info():
    """Print agent information."""
    from .agent import get_agent_info

    info = get_agent_info()

    print("\n" + "=" * 60)
    print(f"{info['name']} v{info['version']}")
    print("=" * 60)
    print(f"\nTotal tools: {info['total_tools']}")
    print(f"Supported channels: {', '.join(info['supported_channels'])}")
    print(f"Supported horizons: {info['supported_horizons']}")
    print(f"Default provider: {info['default_provider']}")
    print(f"Default model: {info['default_model']}")

    print("\nTools by category:")
    for cat, count in info["tools_per_category"].items():
        print(f"  {cat}: {count}")

    print("\n" + "=" * 60)


def run_interactive(
    provider: str,
    model_name: str,
    temperature: float,
    verbose: bool = False,
    track_tokens: bool = False,
):
    """Run interactive chat session."""
    from .agent import build_agent_with_checkpointer
    from langchain_core.messages import HumanMessage

    print("\n" + "=" * 60)
    print("What-If Forecasting Agent - Interactive Mode")
    print("=" * 60)
    if verbose:
        print("\n[Verbose mode enabled - showing tool calls and reasoning]")
    if track_tokens:
        print("\n[Token tracking enabled - will show usage after each message]")
    print("\nType your questions below. Commands:")
    print("  /quit or /exit - Exit the chat")
    print("  /clear - Clear conversation history")
    print("  /tools - List available tools")
    print("  /info - Show agent info")
    if track_tokens:
        print("  /tokens - Show current session token usage")
    print("\n" + "-" * 60)

    agent, checkpointer = build_agent_with_checkpointer(provider, model_name, temperature)
    config = {"configurable": {"thread_id": "interactive"}}

    # Build callbacks list
    callbacks = []
    if verbose:
        from .callbacks import VerboseCallbackHandler
        callbacks.append(VerboseCallbackHandler(show_llm_reasoning=True))

    # Create token tracking callback if enabled (persists across messages)
    token_callback = None
    if track_tokens:
        from .callbacks import TokenTrackingCallback
        token_callback = TokenTrackingCallback(print_per_call=False)
        callbacks.append(token_callback)

    if callbacks:
        config["callbacks"] = callbacks

    def print_session_summary():
        """Print session token summary on exit."""
        if token_callback and token_callback.total_tokens > 0:
            token_callback.print_session_summary()

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\nGoodbye!")
            print_session_summary()
            break

        if not user_input:
            continue

        # Handle commands
        if user_input.startswith("/"):
            cmd = user_input[1:].lower()
            if cmd in ["quit", "exit", "q"]:
                print("\nGoodbye!")
                print_session_summary()
                break
            elif cmd == "clear":
                # Reset checkpointer (create new agent)
                agent, checkpointer = build_agent_with_checkpointer(provider, model_name, temperature)
                config = {"configurable": {"thread_id": "interactive"}}
                # Rebuild callbacks
                callbacks = []
                if verbose:
                    from .callbacks import VerboseCallbackHandler
                    callbacks.append(VerboseCallbackHandler(show_llm_reasoning=True))
                if track_tokens:
                    from .callbacks import TokenTrackingCallback
                    # Reset token tracking for new session
                    token_callback = TokenTrackingCallback(print_per_call=False)
                    callbacks.append(token_callback)
                if callbacks:
                    config["callbacks"] = callbacks
                print("\nConversation cleared.")
                if track_tokens:
                    print("[Token counters reset]")
                continue
            elif cmd == "tools":
                list_tools()
                continue
            elif cmd == "info":
                show_info()
                continue
            elif cmd == "tokens" and track_tokens:
                # Show current token usage
                if token_callback:
                    totals = token_callback.get_totals()
                    print(f"\nCurrent session token usage:")
                    print(f"  Messages: {token_callback.get_message_count()}")
                    print(f"  LLM calls: {totals['llm_calls']}")
                    print(f"  Input tokens:  {totals['input_tokens']:,}")
                    print(f"  Output tokens: {totals['output_tokens']:,}")
                    print(f"  Total tokens:  {totals['total_tokens']:,}")
                continue
            else:
                print(f"Unknown command: {cmd}")
                continue

        # Process query
        try:
            if not verbose:
                print("\nAgent: ", end="", flush=True)
            result = agent.invoke(
                {"messages": [HumanMessage(content=user_input)]},
                config=config,
            )
            from .common import normalize_llm_response
            response = normalize_llm_response(result["messages"][-1].content)
            if verbose:
                # In verbose mode, print final response with separator
                print("\n" + "=" * 60)
                print("FINAL RESPONSE:")
                print("=" * 60)
                print(response)
                print("=" * 60)
            else:
                print(response)

            # Print per-message token usage
            if track_tokens and token_callback:
                delta = token_callback.get_message_delta()
                print(f"\n[TOKEN] Query: {delta['input_tokens']:,} in / {delta['output_tokens']:,} out / {delta['total_tokens']:,} total")

        except Exception as e:
            print(f"\nError: {e}")


def run_single_query(
    query: str,
    provider: str,
    model_name: str,
    temperature: float,
    verbose: bool,
    track_tokens: bool = False,
):
    """Run a single query and print the result."""
    from .agent import run_query

    if verbose:
        print(f"\nQuery: {query}")
        print(f"Provider: {provider}, Model: {model_name}, Temperature: {temperature}")
        print("-" * 60)

    try:
        result = run_query(
            query,
            provider=provider,
            model_name=model_name,
            temperature=temperature,
            verbose=verbose,
            track_tokens=track_tokens,
        )

        if not verbose:
            print(result["final_response"])
        else:
            print(f"\nTools called: {len(result['tool_calls'])}")
            for tc in result["tool_calls"]:
                print(f"  - {tc['name']}")

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


def main():
    """Main entry point."""
    args = parse_args()

    # Handle mode flags
    if args.list_tools:
        list_tools()
        return

    if args.info:
        show_info()
        return

    if args.export_graph:
        from .agent import export_agent_graph
        export_agent_graph(
            args.export_graph,
            provider=args.provider,
            model_name=args.model,
        )
        return

    if args.interactive:
        run_interactive(
            provider=args.provider,
            model_name=args.model,
            temperature=args.temperature,
            verbose=args.verbose,
            track_tokens=args.token,
        )
        return

    # Single query mode
    if args.query:
        run_single_query(
            args.query,
            provider=args.provider,
            model_name=args.model,
            temperature=args.temperature,
            verbose=args.verbose,
            track_tokens=args.token,
        )
        return

    # No arguments - show help
    print("No query provided. Use --help for usage information.")
    print("\nQuick start:")
    print('  python -m forecasting.agent.run "What if brand awareness increases 10%?"')
    print("  python -m forecasting.agent.run --interactive")
    print("  python -m forecasting.agent.run --list-tools")


if __name__ == "__main__":
    main()
