# Code-Based Agents

This section provides best-practice examples for building **code-based AI agents** on SAP BTP using the Generative AI Hub. The examples demonstrate the **ReAct (Reasoning and Acting)** pattern, where an LLM iteratively reasons about a problem, decides which tools to call, observes the results, and repeats until it reaches a conclusion.

Two complementary approaches are covered, progressing from foundational understanding to production-ready frameworks:

## Structure

| Folder | Approach | Purpose |
|--------|----------|---------|
| [`native-react/`](./native-react/) | From-scratch Python implementation | Understand the ReAct loop internals: prompt construction, output parsing, tool execution, and the scratchpad mechanism |
| [`langgraph-react/`](./langgraph-react/) | LangGraph framework with LangChain | Build progressively complex agents using state graphs, structured tool calling, conditional routing, and enterprise workflows |

## Learning Path

The recommended order is:

1. **Start with `native-react/`** to understand what happens inside a ReAct agent at the lowest level. This builds intuition for how frameworks like LangGraph work internally.
2. **Continue with `langgraph-react/`** to see how a framework handles the same patterns with less boilerplate, and how to scale from simple tool use to multi-agent routing and enterprise process automation.

## Prerequisites

- An SAP BTP subaccount with an **SAP AI Core** service instance and a **Generative AI Hub** deployment
- A deployed LLM model (the examples use `gpt-4.1`)
- Python 3.10+
- Service key credentials for AI Core (see the `.env.example` files in each subfolder)

## Key Concepts

- **ReAct Pattern**: The agent alternates between *reasoning* (thinking about what to do next) and *acting* (calling a tool), using the observation from each tool call to inform the next step.
- **Tool Calling**: Functions are exposed to the LLM with metadata (name, description, parameter schema) so the model can decide when and how to invoke them.
- **Scratchpad / Message History**: The accumulated history of thoughts, actions, and observations provides context for multi-step problem solving.
- **SAP Generative AI Hub**: All LLM calls are routed through the SAP AI Core proxy, which provides centralized model access, authentication, and governance.
