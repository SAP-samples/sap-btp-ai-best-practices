![Pro-Code AI Agents](https://cdn.hubblecontent.osi.office.net/m365content/publish/235d262e-6785-4a03-9953-65b69541c1ab/901627692.jpg)

# Pro-Code AI Agents

This best practice page focuses on building production-ready AI agents using code-based approaches within the SAP ecosystem. For security and governance of AI agents, see the dedicated [**Agentic AI Security and Governance**](/sites/210313/SitePages/Agentic%20AI%20Security%20%26%20Governance.aspx) best practice page. For connecting agents to enterprise tools and data sources via the Model Context Protocol, see the [**Model Context Protocol (MCP)**](/sites/210313/SitePages/Model%20Context%20Protocol%20(MCP).aspx) best practice page.

## Steps

**1** [Overview](/sites/210313/SitePages/Pro-Code%20AI%20Agents.aspx#1.-overview)  
**2** [Pre-requisites](/sites/210313/SitePages/Pro-Code%20AI%20Agents.aspx#2.-pre-requisites)  
**3** [Key Choices and Guidelines](/sites/210313/SitePages/Pro-Code%20AI%20Agents.aspx#3.-key-choices-and-guidelines)  
**4** [Implementation](/sites/210313/SitePages/Pro-Code%20AI%20Agents.aspx#4.-implementation)  

---

## 1. Overview

### Description
AI agents are software systems that go beyond simple prompt-response interactions. They can plan, reason, and act iteratively to complete multi-step tasks by calling external tools, APIs, and data sources. Unlike traditional LLM usage ("single prompt" to "single response"), agents operate in a loop: they reason about what to do next, take an action, observe the result, and decide whether to continue or deliver a final answer.

Within the SAP ecosystem, **code-based agents** represent the most flexible approach to building agentic workflows. They offer full autonomy for advanced intelligence, custom API integrations, and specialized algorithms, in contrast to **content-based agents** (low-code/no-code declarative approach) or **Joule Scenarios** (SAP's built-in conversational AI).

This best practice page covers two complementary approaches:

1. **The ReAct Pattern (from scratch):** Understanding how agents work at a fundamental level, the Thought Action Observation loop that underpins all agentic frameworks. This foundation is essential for debugging, optimizing, and reasoning about agent behavior.
2. **LangGraph:** The recommended production framework for building structured, enterprise-grade agentic workflows within SAP. LangGraph models agent logic as a graph of nodes and edges with shared state, offering the granular control and reliability required for business-critical processes. The content is structured in 3 subsections:
   1. **Basic ReAct Agent**: replaces the manual loop with a structured state graph
   2. **Multi-Agent Routing with Conditional Edges**: LLM-based classifier at the entry point of the graph. The key takeaway is **scoped tool access**: by restricting which tools each agent can call, you reduce cost (fewer tokens describing irrelevant tools), improve reliability (the model is less likely to pick the wrong tool), and enforce domain boundaries
   3. **Enterprise Procurement Workflow**: replaces the toy tools of the first ReAct Agent with six enterprise-grade tools backed by mock data simulating SAP systems.

### Expected Outcome
After following this best practice, developers will be able to:

* Understand how AI agents reason and act through the ReAct paradigm
* Design and implement single-agent and multi-step agentic workflows using LangGraph
* Integrate agents with SAP services, external APIs, and enterprise data sources via the SAP Generative AI Hub SDK
* Make informed decisions about agent architecture, tool design, and workflow structure

### Benefits
* **Autonomous Multi-Step Reasoning:** Agents can decompose complex business tasks into manageable steps, calling the right tools at the right time without requiring the user to orchestrate each step manually.
* **Dynamic Tool Integration:** Agents select and invoke tools (APIs, databases, calculations) based on context, enabling flexible responses to varied business scenarios.
* **Enterprise Workflow Orchestration:** LangGraph's graph-based model allows conditional branching, gating logic, and multi-stage validation, critical for business processes with compliance or approval requirements.
* **Debuggability and Control:** Understanding the underlying ReAct loop, combined with LangGraph's explicit state management, provides full visibility into agent decision-making.

### Key Concepts
* **ReAct (Reason + Act):** A paradigm where the agent alternates between reasoning ("Thought"), executing a tool ("Action"), and processing the result ("Observation") in an iterative loop until it reaches a final answer.
* **Tools:** Python functions that extend the agent's capabilities beyond text generation, e.g., calling APIs, querying databases, performing calculations. Tools are described with a name, description, and input schema so the LLM can select and invoke them.
* **Scratchpad:** The accumulating chain-of-thought record that captures all Thought/Action/Observation steps. This gives the LLM context about what it has already tried and learned.
* **State Graph (LangGraph):** A directed graph where nodes represent processing steps (LLM calls, tool executions, validations) and edges define the flow, including conditional branches. A shared typed state dictionary passes data between nodes.
* **Conditional Gating:** Programmatic checks between workflow steps that route execution based on intermediate results (e.g., "if the summary is too long, route to the refinement node").
* **Orchestration Service:** SAP's orchestration layer within the Generative AI Hub that provides authenticated access to LLMs, prompt templating, and module-based configuration.

---

## 2. Pre-requisites

### Commercial
* SAP AI Core with the “Extended” tier on SAP BTP
* SAP AI Launchpad (Optional but recommended)

You can find pricing details for AI Core “Extended” in the [Discovery Center – AI Core](https://discovery-center.cloud.sap/serviceCatalog/sap-ai-core?tab=service_plan&region=all&service_plan=extended&commercialModel=btpea).

### Technical
* SAP Business Technology Platform (SAP BTP) subaccount ([Setup Guide](/sites/210313/SitePages/SAP%20Business%20Technology%20Platform%20(SAP%20BTP).aspx))
* SAP AI Core ([Setup Guide](/sites/210313/SitePages/Technology%20-%20SAP%20AI%20Core.aspx#setup))

For a deeper understanding of LLM model access, configuration and endpoint settings, refer to the [Access to Generative AI Model](/sites/210313/SitePages/GenAI%20-%20Plain%20-%20Direct%20-%20Inference%20Request.aspx) best practice page

### High-level Reference Architecture
![High-level Reference Architecture](/sites/210313/SiteAssets/SitePages/Pro-Code%20AI%20Agents/1791112200-Access-to-Generative-AI-Models-v2-3.drawio--1-.png)

#### SAP Business Technology Platform (SAP BTP)
SAP Business Technology Platform (BTP) is an integrated suite of cloud services, databases, AI, and development tools that enable businesses to build, extend, and integrate SAP and non-SAP applications efficiently.

#### SAP AI Core
SAP AI Core is a managed AI runtime that enables scalable execution of AI models and pipelines, integrating seamlessly with SAP applications and data on SAP BTP that supports full lifecycle management of AI scenarios.

---

## 3. Key Choices and Guidelines

### Understand the ReAct Pattern Before Using a Framework
**Why this matters:** Every agentic framework (LangGraph, CrewAI, Autogen, etc.) is an abstraction over the same fundamental loop: the agent thinks, acts, observes, and repeats. It is important to understand this pattern and not only the Agentic Framework, to avoid struggling debugging stuck loops and to understand why the LLM chose the wrong tool or optimize agent performance.

### Why LangGraph over other frameworks:

| Criterion | LangGraph | CrewAI | Smolagents | Autogen | Semantic Kernel |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Production Readiness** | **High** (Enterprise standard) | High (Fast time-to-market) | Medium (Needs security layers) | Medium (Latency/cost challenges) | High (Mature SDK) |
| **Workflow Control** | **Full** (Nodes, conditional edges, gating) | Limited (Role-based delegation) | Minimal | Medium (Conversation-based) | Medium (Plugin-based) |
| **SAP Integration** | **Native** (via `gen_ai_hub.proxy.langchain`) | Requires custom LiteLLM adapter + monkey patch | Requires monkey patching | Requires monkey patching | Clean (via OpenAI-compatible proxy client) |
| **Debugging** | **Explicit** (Graph visualization, state inspection) | Opaque (Role delegation) | Minimal logging | Complex (Multi-agent comms) | Medium |
| **Language Support** | Python, Node.js | Python only | Python only | Python only | Python, .NET, Java |

**Recommendation:** Use **LangGraph** as the default framework for SAP code-based agents. It offers the best combination of control, debuggability, and native SAP integration without requiring monkey patching or workarounds.

**When to consider alternatives:**
* **Rapid prototyping / hackathons:** CrewAI or Smolagents for faster setup when production robustness is not a concern.
* **.NET or Java environments:** Semantic Kernel if your team does not work in Python.
* **Pure research / experimentation:** Smolagents for minimal boilerplate.

### Tool Design
* One tool = one clear responsibility. Don't create a "do everything" tool. An agent with 3 focused tools (e.g., `query_orders`, `check_weather`, `send_notification`) will outperform one with a single monolithic tool.
* **Write descriptive docstrings.** The LLM reads the tool's name, description, and parameter schema to decide when and how to call it. Ambiguous descriptions lead to wrong tool selection.
* **Define explicit input schemas.** Use type annotations on all parameters. The ToolInfo class (see Implementation) extracts these automatically.
* **Return structured, concise data.** Tools should return text the LLM can easily parse. Avoid returning raw HTML, massive JSON blobs, or binary data.
* **Handle errors gracefully.** Tools should catch exceptions and return descriptive error messages rather than crashing the agent loop.
* **Limit the number of tools.** Start with 3–5 tools per agent. More tools increase the chance of the LLM selecting the wrong one. If you need more capabilities, consider splitting into multiple specialized agents.

### Use Graph-Based Workflows for Multi-Step Business Logic
* **Model your business process as a graph.** Identify the distinct steps (nodes), the data that flows between them (state), and the conditions that determine routing (conditional edges).
* **Use TypedDict for state.** Define your state schema explicitly using Python's `TypedDict`. This documents what data exists at each point in the workflow and catches bugs early.
* **Add validation gates between steps.** Before proceeding to the next node, check the output quality (e.g., is the summary within word limits? Did the API return valid data?). Route to refinement nodes if checks fail.
* **Keep nodes focused.** Each node should do one thing: one LLM call, one API call, or one validation check. This makes the workflow testable and maintainable.
* **Visualize your graph.** LangGraph can export Mermaid diagrams of the workflow. Use `chain.get_graph().draw_mermaid_png()` during development to verify the flow matches your intent.

### Select the Right LLM for Agent Workloads
* **Use reasoning-capable models for agent loops.** Agent workloads require the LLM to plan, decide on tools, and format structured output (JSON). Models like `gpt-5`, `gpt-5-mini`, `anthropic--claude-4.5--sonnet` or `gemini-2.5-pro` are well suited.
* **Temperature = 0 for deterministic tool calling.** When the agent needs to reliably produce valid JSON tool calls, low temperature reduces parsing errors. Use higher temperature only for creative final answers.
* **Refer to the Access to Generative AI Models best practice** for detailed model selection guidance, benchmarking, and endpoint configuration.

### Plan for Observability from Day One
* **Log every agent step.** Capture each Thought/Action/Observation cycle, including timestamps and token usage. This is essential for debugging, cost tracking, and compliance.
* **Set maximum turn limits.** Always configure a `max_turns` parameter (e.g., 10–20 turns) to prevent runaway agent loops that consume excessive tokens and time.
* **Monitor tool call patterns.** Track which tools are called, how often, and in what sequences. Unusual patterns may indicate the agent is stuck or the prompt needs refinement.
* **Implement human-in-the-loop for high-stakes actions.** For actions that modify data, send communications, or trigger financial transactions, require explicit human approval before execution.

Agentic AI introduces unique security risks including prompt injection, memory poisoning and privilege escalation. These topics are covered in depth in the dedicated [Agentic AI Security and Governance](/sites/210313/SitePages/Agentic%20AI%20Security%20%26%20Governance.aspx) page. At minimum, follow SAP's guidelines on input validation, execution quotas and least-privilege access control for all agent deployments.

---

## 4. Implementation

### Recommendation
**Python with LangGraph** is the recommended stack for building code-based AI agents on SAP BTP. The SAP Generative AI Hub SDK (`sap-ai-sdk-gen`) provides the LLM access layer through LangChain-compatible wrappers, avoiding vendor lock-in while routing all model calls through SAP AI Core for centralized governance. All reference examples below demonstrate this combination.

### SDKs
**SAP Generative AI Hub SDK for Python** (`sap-ai-sdk-gen`) (Recommended)  
Native Python SDK for accessing LLMs via SAP AI Core. Includes LangChain wrappers (ChatOpenAI proxy) so that agents built with LangGraph and LangChain work out of the box with any model deployed in the Generative AI Hub.

**LangGraph + LangChain**  
The recommended agent framework. LangGraph provides state graphs for defining agent workflows with explicit control over routing, branching, and state management. LangChain provides structured tool calling, message handling, and prebuilt components (`ToolNode`, `tools_condition`) that eliminate boilerplate.

### Tutorials and Learning Journeys
* [Getting Started with Agents Using SAP Generative AI Hub](https://community.sap.com/t5/devtoberfest/getting-started-with-agents-using-sap-generative-ai-hub/ev-p/13865119) (Devtoberfest session)
* [SAP AI Core Agent QuickLaunch Series](https://community.sap.com/t5/technology-blog-posts-by-sap/sap-ai-core-agent-quicklaunch-series-part0-prologue/ba-p/14104823) (SAP Community blog series)
* [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)

### Reference Code
The code examples are organized as a learning path in two parts. Start with **Part 1** to understand what happens inside a ReAct agent at the lowest level, then continue with **Part 2** to see how a framework handles the same patterns with less boilerplate and more reliability.

[**SAP BTP AI Best Practices - Code-Based Agents**](https://github.com/SAP-samples/sap-btp-ai-best-practices/tree/main/best-practices/agentic-ai/code-based-agents)

---

## Part 1: The ReAct Pattern from Scratch

Understanding the raw ReAct loop is essential before using any framework. This example implements a complete agent using only the SAP Orchestration Service and plain Python.

The notebook walks through five building blocks: loading SAP AI Core credentials, wrapping the Orchestration Service into a helper function, extracting tool metadata from Python functions automatically, constructing a structured prompt that teaches the LLM the Thought-Action-Observation format, and wiring it all together in an agent loop that feeds observations back into the next LLM call.

The key insight is that an agent is fundamentally just a loop around an LLM call. The entire "intelligence" comes from three things: (1) the prompt structure that teaches the model to reason step by step, (2) the LLM's ability to produce valid JSON tool calls, and (3) the loop that captures each tool's output and feeds it back as context. Every framework abstracts exactly this pattern.

Understanding the raw ReAct loop is essential before using any framework. The following implementation shows how an agent reasons and acts using only the SAP Orchestration Service.

### Reference Code
* [**SAP BTP AI Best Practices - Code-Based Agents - Native ReAct Notebook**](https://github.com/SAP-samples/sap-btp-ai-best-practices/tree/main/best-practices/agentic-ai/code-based-agents/native-react)

---

## Part 2: Langgraph for Production Workflows

LangGraph replaces the manual loop with a structured state graph. Instead of writing your own parsing, scratchpad management, and tool dispatch, you define nodes (processing steps), edges (transitions), and conditional routing and let the framework handle execution. This part contains three progressive notebooks that build from a basic agent to an enterprise workflow.

#### Reference Code
* [**SAP BTP AI Best Practices - Code-Based Agents - LangGraph ReAct Agents**](https://github.com/SAP-samples/sap-btp-ai-best-practices/tree/main/best-practices/agentic-ai/code-based-agents/langgraph-react)

### 2.1 Basic ReAct Agent
The simplest possible Langgraph Agent: two nodes connected in a loop. The **assistant** node calls the LLM with tools bound via `bind_tools()`, and the **tools** node executes whatever the LLM requested. Routing uses `tools_condition`: if the LLM response contains tool calls, go to tools; otherwise, end

![Mermaid Diagram](/sites/210313/SiteAssets/SitePages/Pro-Code%20AI%20Agents/225234594-mermaid-diagram.png)

This notebook demonstrates LLM initialization via the SAP GenAI Hub proxy, the `@tool` decorator for automatic JSON schema generation, `MessagesState` for message history and parallel tool execution (the LLM can request multiple tools in one response). 

#### Reference Code
* [**SAP BTP AI Best Practices - Code-Based Agents - Basic ReAct Notebook**](https://github.com/SAP-samples/sap-btp-ai-best-practices/blob/main/best-practices/agentic-ai/code-based-agents/langgraph-react/01_react_agent.ipynb)

### 2.2 Multi-Agent Routing with Conditional Edges
This notebook adds an LLM-based classifier at the entry point of the graph. The classifier determines the query category (math, weather, or general) and routes to a specialized agent. Each agent has access only to its relevant tools: the math agent can call `add` and `multiply`, the weather agent can call `get_weather`, and the general agent has access to all tools.

![Mermaid Diagram](/sites/210313/SiteAssets/SitePages/Pro-Code%20AI%20Agents/799924742-mermaid-diagram--1-.png)

The key takeaway is **scoped tool access**: by restricting which tools each agent can call, you reduce cost (fewer tokens describing irrelevant tools), improve reliability (the model is less likely to pick the wrong tool), and enforce domain boundaries. The notebook also demonstrates custom state schemas, domain-specific system prompts, and conditional edges with custom routing options

#### Reference Code
* [**SAP BTP AI Best Practices - Code-Based Agents - Multi-Agent Routing with Conditional Edges Notebook**](https://github.com/SAP-samples/sap-btp-ai-best-practices/blob/main/best-practices/agentic-ai/code-based-agents/langgraph-react/02_multi_step_workflow.ipynb)

### 2.3 Enterprise Procurement Workflow
This notebook uses the same simple two-node graph topology as Notebook 1 — but replaces the toy tools with six enterprise-grade tools backed by mock data simulating SAP systems (Material Master, Warehouse Management, Controlling, Vendor Master). The system prompt encodes a detailed six-step procurement procedure:

1. Extract requests details (product, quantity, plant, department)
2. Look up the product in the catalog
3. Check inventory at the requested plant
4. validate the department budget
5. Check supplier constraints (lead time, minimum order quantity)
6. Create a purchase order draft

The key takeaway is that **graph complexity is not always necessary**. A simple graph with well-designed tools and a detailed system prompt handles complex multi-step business workflows effectively. Tool design matters more than graph design: tools that return contextual information (for example, inventory at all plants, not just the one requested) reduce the number of agent iterations and improve response quality

#### Reference Code
* [**SAP BTP AI Best Practices - Code-Based Agents - Enterprise Procurement Workflow Notebook**](https://github.com/SAP-samples/sap-btp-ai-best-practices/blob/main/best-practices/agentic-ai/code-based-agents/langgraph-react/03_procurement_workflow.ipynb)

---

## Related Best Practices

* [Access to Generative AI Models](/sites/210313/SitePages/GenAI%20-%20Plain%20-%20Direct%20-%20Inference%20Request.aspx)
* [Agentic AI Concepts Explained](/sites/210313/SitePages/Agentic-AI-Concepts-Overview.aspx)
* [Model Context Protocol (MCP)](/sites/210313/SitePages/Model%20Context%20Protocol%20(MCP).aspx)
* [Agentic AI Security & Governance](/sites/210313/SitePages/Agentic%20AI%20Security%20%26%20Governance.aspx)

## Related AI Capabilities

* [Agentic Workflows](/sites/210313/SitePages/Functional%20Patterns%20-%20Agentic%20Workflows%20-%20Agentic%20workflows.aspx)

## Contributors

* **Robledo, Francisco** (CSS _BTP Hub_AI)