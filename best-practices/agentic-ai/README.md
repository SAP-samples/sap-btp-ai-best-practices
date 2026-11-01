# SAP BTP AI Best Practices – Agentic AI

This section of the repository contains best practices for building AI agents on SAP Business Technology Platform (BTP) with the generative AI hub in SAP AI Core.

## Contents

| Folder | Description |
|---|---|
| [code-based-agents/native-react](./code-based-agents/native-react/README.md) | A from-scratch Python implementation of the ReAct (Reasoning and Acting) loop: prompt construction, output parsing, tool execution and the scratchpad |
| [code-based-agents/langgraph-react](./code-based-agents/langgraph-react/README.md) | Five progressive LangGraph notebooks: a basic ReAct agent, multi-agent routing, an enterprise procurement workflow, token caching, and checkpointing with session state |

Start with the [code-based agents overview](./code-based-agents/README.md) for the recommended learning path and prerequisites.

## Choosing an approach on SAP BTP

SAP offers two complementary ways to build agents. The samples in this folder cover the **pro-code** path.

| | Low-code | Pro-code (this folder) |
|---|---|---|
| Tooling | Joule Studio (agent builder and skill builder) in SAP Build | SAP Cloud SDK for AI (Python, TypeScript/JavaScript, Java) with an agent framework such as LangGraph |
| Typical fit | Standard business process automation with visual configuration | Complex or specialized workflows that need full programmatic control |
| Runtime | Managed by SAP | Your application on SAP BTP (Cloud Foundry or Kyma) |
| Model access | Generative AI hub | Generative AI hub, through the SDK or the orchestration service |

The [SAP Architecture Center golden path for building AI agents](https://architecture.learning.sap.com/docs/golden-path/ai-golden-path/build-and-deliver/build-ai-agents) and the [reference architecture for agentic AI](https://architecture.learning.sap.com/docs/ref-arch/98efa0) describe both paths in detail.

## Interoperability: A2A and MCP

- **Agent2Agent (A2A)** is the standard SAP recommends for multi-agent collaboration and for connecting code-based agents to Joule. A code-based agent exposes an A2A endpoint, and Joule (or another agent) calls it. The SAP Community post [Integrating Joule with Code-based Agents: Three Paths Forward](https://community.sap.com/t5/technology-blog-posts-by-sap/integrating-joule-with-code-based-agents-three-paths-forward/ba-p/14370520) compares direct A2A calls, an asynchronous REST pattern through SAP Build Process Automation, and an A2A client proxy that streams progress back to Joule.
- **Model Context Protocol (MCP)** gives agents structured access to tools and data. The SAP Architecture Center golden path notes that SAP uses MCP to give Joule agents access to SAP business capabilities, and that A2A is the preferred standard for exposing agents to external parties. If your code-based agent uses MCP servers as tool sources, apply the same controls as for any other tool: least-privilege credentials (for example, through SAP BTP destinations), input validation, and logging of each tool call.

## Practical guidance for code-based agents

These points are demonstrated in the notebooks or follow directly from them:

- **Route all model calls through the generative AI hub.** This keeps authentication, model governance and usage metering in one place, and lets you add orchestration modules such as content filtering and data masking (see [generative-ai](../generative-ai/README.md)).
- **Describe tools precisely.** The model chooses tools from their names, descriptions and parameter schemas, so these are part of your prompt.
- **Bound the loop.** Set a maximum number of reasoning steps and handle tool errors as observations rather than exceptions.
- **Use prompt caching for long, stable prefixes** such as system prompts and tool definitions to cut latency and token cost (see `04_token_caching.ipynb`).
- **Persist state explicitly.** Use checkpointing so that multi-turn sessions and long-running workflows can resume (see `05_checkpointing_and_session_state.ipynb`).
- **Keep a human in the loop for actions with business impact,** such as posting documents or approving purchases.

## Related SAP capabilities

- **Joule Studio** – the agent builder and skill builder for low-code agents, with support for pro-code development in IDEs and Git. See the [Joule Studio product page](https://www.sap.com/products/artificial-intelligence/joule-studio.html).
- **SAP AI Agent Hub** – a central inventory for governing SAP and non-SAP agents, LLMs and MCP servers, announced at SAP Sapphire 2026.

Feature availability changes quickly. Check the [SAP AI release highlights](https://news.sap.com/2026/10/ai-release-highlights-q3-2026/) and the SAP Help Portal for the current status before you design a solution around a specific capability.
