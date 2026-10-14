---
name: langgraph-agent-patterns
description: "INVOKE THIS SKILL when building LLM agents with tools in LangGraph. Covers ReAct loop (manual and prebuilt), multi-step planning, conditional branching, MessagesState, ToolNode, tools_condition, and MCP integration. Python-only, SAP Gen AI Hub."
---

<overview>
Agent patterns are graph topologies for LLM-driven decision making. Each pattern solves a specific class of problems — pick the simplest one that fits your task.

All patterns use SAP Gen AI Hub proxy wrappers. Co-invoke `langgraph-genai-hub-setup` for LLM initialization details.
</overview>

---

## Pattern Selection

<agent-pattern-selection>

| Pattern | When to Use | Complexity |
|---------|-------------|------------|
| Minimal single-node | Simple LLM call, structured JSON output | Low |
| Multi-step planning | Decompose complex tasks into plan + execute | Low-Medium |
| Conditional branching | Route based on LLM classification | Medium |
| ReAct (prebuilt) | Standard tool-calling agent loop | Medium |
| ReAct (manual) | Custom tool routing logic needed | Medium |
| MCP integration | External tool servers via Model Context Protocol | Medium |

**Default choice**: Use **ReAct (prebuilt)** for tool-calling agents. Only use manual ReAct when you need custom routing logic.

</agent-pattern-selection>

---

## Minimal Single-Node Agent

A single LLM call wrapped in a graph. Useful for structured JSON output without tool calling.

<ex-minimal-agent>
<python>
Single-node graph that enforces JSON output via system prompt.
```python
from typing import Any, Dict, TypedDict
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.output_parsers import JsonOutputParser
from langgraph.graph import StateGraph, END
from gen_ai_hub.proxy.langchain.openai import ChatOpenAI

class State(TypedDict, total=False):
    prompt: str
    result: Dict[str, Any]

llm = ChatOpenAI(proxy_model_name="gpt-4.1", temperature=0.2)
parser = JsonOutputParser()

def call_model(state: Dict[str, Any]) -> Dict[str, Any]:
    system = SystemMessage(content="Respond with STRICT JSON. No backticks.")
    user = HumanMessage(content=state.get("prompt", ""))
    response = llm.invoke([system, user])
    return {"result": parser.parse(response.content)}

graph = StateGraph(State)
graph.add_node("call_model", call_model)
graph.set_entry_point("call_model")
graph.add_edge("call_model", END)
app = graph.compile()

result = app.invoke({"prompt": "Return JSON with 'answer' and 'confidence' about: capital of France?"})
```
</python>
</ex-minimal-agent>

---

## Multi-Step Planning

Decompose complex tasks into a planning step followed by an execution step.

<ex-multi-step-planning>
<python>
Two-node graph: plan decomposes the task, answer executes the plan.
```python
from typing import Any, Dict, TypedDict
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.output_parsers import JsonOutputParser
from langgraph.graph import StateGraph, END
from gen_ai_hub.proxy.langchain.openai import ChatOpenAI

class State(TypedDict, total=False):
    question: str
    plan: Dict[str, Any]
    final: Dict[str, Any]

llm = ChatOpenAI(proxy_model_name="gpt-4.1", temperature=0.2)
parser = JsonOutputParser()

def plan_node(state: Dict[str, Any]) -> Dict[str, Any]:
    system = SystemMessage(content="Decompose the question into steps. Output STRICT JSON: {\"steps\": [...], \"assumptions\": [...]}")
    user = HumanMessage(content=f"Question: {state.get('question', '')}")
    response = llm.invoke([system, user])
    return {"plan": parser.parse(response.content)}

def answer_node(state: Dict[str, Any]) -> Dict[str, Any]:
    system = SystemMessage(content="Execute the plan step by step. Output STRICT JSON: {\"answer\": string, \"used_steps\": [...]}")
    user = HumanMessage(content=f"Question: {state.get('question', '')}. Plan: {state.get('plan', {})}")
    response = llm.invoke([system, user])
    return {"final": parser.parse(response.content)}

graph = StateGraph(State)
graph.add_node("plan", plan_node)
graph.add_node("answer", answer_node)
graph.set_entry_point("plan")
graph.add_edge("plan", "answer")
graph.add_edge("answer", END)
app = graph.compile()
```
</python>
</ex-multi-step-planning>

---

## Conditional Branching

Route execution to different nodes based on LLM classification.

<ex-conditional-branching>
<python>
Classify input intent, then route to specialized nodes.
```python
from typing import Any, Dict, TypedDict
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.output_parsers import JsonOutputParser
from langgraph.graph import StateGraph, END
from gen_ai_hub.proxy.langchain.openai import ChatOpenAI

class State(TypedDict, total=False):
    input: str
    intent: str
    result: Dict[str, Any]

llm = ChatOpenAI(proxy_model_name="gpt-4.1", temperature=0.2)
parser = JsonOutputParser()

def classify(state: Dict[str, Any]) -> Dict[str, Any]:
    system = SystemMessage(content='Classify as "informational" or "creative". Output JSON: {"intent": <value>}')
    user = HumanMessage(content=state.get("input", ""))
    response = llm.invoke([system, user])
    return {"intent": parser.parse(response.content).get("intent")}

def informational(state: Dict[str, Any]) -> Dict[str, Any]:
    system = SystemMessage(content='Provide factual info. Output JSON: {"answer": string, "style": "informational"}')
    user = HumanMessage(content=state.get("input", ""))
    return {"result": parser.parse(llm.invoke([system, user]).content)}

def creative(state: Dict[str, Any]) -> Dict[str, Any]:
    system = SystemMessage(content='Respond creatively. Output JSON: {"answer": string, "style": "creative"}')
    user = HumanMessage(content=state.get("input", ""))
    return {"result": parser.parse(llm.invoke([system, user]).content)}

def route(state: Dict[str, Any]):
    return "informational" if state.get("intent") == "informational" else "creative"

graph = StateGraph(State)
graph.add_node("classify", classify)
graph.add_node("informational", informational)
graph.add_node("creative", creative)
graph.set_entry_point("classify")
graph.add_conditional_edges("classify", route, {"informational": "informational", "creative": "creative"})
graph.add_edge("informational", END)
graph.add_edge("creative", END)
app = graph.compile()
```
</python>
</ex-conditional-branching>

---

## ReAct Pattern

The ReAct (Reasoning and Acting) pattern is the standard approach for building tool-calling agents. The LLM decides which tool to call, the tool executes, and the result feeds back to the LLM until it produces a final answer.

<react-overview>

The graph topology is always the same:
```
START -> assistant -> (tools_condition) -> tools -> assistant -> ... -> END
```

The assistant node calls the LLM (with bound tools). If the LLM returns tool calls, execution flows to the tools node; if not, it flows to END.

</react-overview>

<react-prebuilt-vs-manual>

| Approach | When to Use |
|----------|-------------|
| **Prebuilt** (`ToolNode` + `tools_condition`) | Standard tool-calling loop. Use this by default. |
| **Manual** (custom routing function) | Need custom tool routing, error handling, or pre/post-processing per tool call |

</react-prebuilt-vs-manual>

### ReAct with Prebuilt Components (Recommended)

<ex-react-prebuilt>
<python>
The canonical ReAct pattern using ToolNode and tools_condition.
```python
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import MessagesState, StateGraph, START
from langgraph.prebuilt import ToolNode, tools_condition
from gen_ai_hub.proxy.langchain.openai import ChatOpenAI
from langchain_core.tools import tool

@tool
def calculator(expression: str) -> str:
    """Evaluate a math expression."""
    return str(eval(expression))

@tool
def search(query: str) -> str:
    """Search for information."""
    return f"Search results for: {query}"

# Initialize LLM and bind tools
llm = ChatOpenAI(proxy_model_name="gpt-4.1", temperature=0.2)
tools = [calculator, search]
llm_with_tools = llm.bind_tools(tools)

system_message = SystemMessage(content="You are a helpful assistant. Use tools when needed.")

def assistant(state: MessagesState):
    response = llm_with_tools.invoke([system_message] + state["messages"])
    return {"messages": [response]}

# Build the ReAct graph
graph = StateGraph(MessagesState)
graph.add_node("assistant", assistant)
graph.add_node("tools", ToolNode(tools=tools))
graph.add_edge(START, "assistant")
graph.add_conditional_edges("assistant", tools_condition)
graph.add_edge("tools", "assistant")
app = graph.compile()

# Run
result = app.invoke({"messages": [HumanMessage(content="What is 12 * 7?")]})
```
</python>
</ex-react-prebuilt>

### ReAct with Manual Routing

<ex-react-manual>
<python>
Manual tool routing for custom logic (e.g., logging, filtering, pre-processing).
```python
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import MessagesState, StateGraph, START, END
from gen_ai_hub.proxy.langchain.openai import ChatOpenAI

llm = ChatOpenAI(proxy_model_name="gpt-4.1", temperature=0.2)
tools = [calculator, search]  # @tool-decorated functions
tools_by_name = {t.name: t for t in tools}
llm_with_tools = llm.bind_tools(tools)

system_message = SystemMessage(content="You are a helpful assistant.")

def assistant(state: MessagesState):
    return {"messages": [llm_with_tools.invoke([system_message] + state["messages"])]}

def tool_executor(state: MessagesState):
    """Execute tool calls with custom logic."""
    results = []
    for call in state["messages"][-1].tool_calls:
        tool_fn = tools_by_name[call["name"]]
        result = tool_fn.invoke(call["args"])
        results.append(ToolMessage(content=str(result), tool_call_id=call["id"]))
    return {"messages": results}

def should_continue(state: MessagesState):
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"
    return END

graph = StateGraph(MessagesState)
graph.add_node("assistant", assistant)
graph.add_node("tools", tool_executor)
graph.add_edge(START, "assistant")
graph.add_conditional_edges("assistant", should_continue, {"tools": "tools", END: END})
graph.add_edge("tools", "assistant")
app = graph.compile()
```
</python>
</ex-react-manual>

---

## MCP Integration

The Model Context Protocol (MCP) allows loading tools from external servers. MCP tools are standard LangChain `BaseTool` instances and work with `ToolNode` normally.

<ex-mcp-integration>
<python>
Load tools from an MCP server and use them in a ReAct graph.
```python
import asyncio
import sys
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import MessagesState, StateGraph, START
from langgraph.prebuilt import ToolNode, tools_condition
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_mcp_adapters.tools import load_mcp_tools
from gen_ai_hub.proxy.langchain.openai import ChatOpenAI

async def run_mcp_agent(prompt: str, server_script: str):
    """Run a ReAct agent with MCP-provided tools."""
    server_params = StdioServerParameters(
        command=sys.executable,
        args=[server_script],
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # Load tools from MCP server — returns list[BaseTool]
            tools = await load_mcp_tools(session)

            llm = ChatOpenAI(proxy_model_name="gpt-4.1", temperature=0.2)
            llm_with_tools = llm.bind_tools(tools)
            system_msg = SystemMessage(content="Use tools when appropriate.")

            def assistant(state: MessagesState):
                return {"messages": [llm_with_tools.invoke([system_msg] + state["messages"])]}

            graph = StateGraph(MessagesState)
            graph.add_node("assistant", assistant)
            graph.add_node("tools", ToolNode(tools=tools))
            graph.add_edge(START, "assistant")
            graph.add_conditional_edges("assistant", tools_condition)
            graph.add_edge("tools", "assistant")
            app = graph.compile()

            return await app.ainvoke({"messages": [HumanMessage(content=prompt)]})

# Run with: asyncio.run(run_mcp_agent("Compute 12 + 7", "mcp_server.py"))
```
</python>
</ex-mcp-integration>

---

## Fixes

<fix-bind-tools>
<python>
Must call `bind_tools()` before using the LLM in the assistant node.
```python
# WRONG: LLM without tools — will never produce tool calls
def assistant(state: MessagesState):
    return {"messages": [llm.invoke(state["messages"])]}

# CORRECT: Bind tools first
llm_with_tools = llm.bind_tools(tools)
def assistant(state: MessagesState):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}
```
</python>
</fix-bind-tools>

<fix-tools-condition-import>
<python>
Import `ToolNode` and `tools_condition` from `langgraph.prebuilt`.
```python
# WRONG
from langgraph.graph import tools_condition  # ImportError!

# CORRECT
from langgraph.prebuilt import ToolNode, tools_condition
```
</python>
</fix-tools-condition-import>

<fix-messages-state>
<python>
Use `MessagesState` for tool-calling agents, not custom state.
```python
# WRONG: Custom state loses tool call metadata
class State(TypedDict):
    query: str
    response: str

# CORRECT: MessagesState preserves full message objects including tool calls
from langgraph.graph import MessagesState

graph = StateGraph(MessagesState)
```
</python>
</fix-messages-state>

<fix-tool-node-errors>
<python>
Enable error handling so the LLM can recover from tool failures.
```python
# Without error handling, tool errors crash the graph
tool_node = ToolNode(tools)

# With error handling, errors become ToolMessages the LLM can process
tool_node = ToolNode(tools, handle_tool_errors=True)
```
</python>
</fix-tool-node-errors>

<boundaries>
### What You Should NOT Do

- Build manual tool routing when `tools_condition` from `langgraph.prebuilt` suffices
- Use `create_react_agent` from `langchain.agents` — use the LangGraph pattern shown above instead
- Forget to handle the no-tool-calls exit condition — that is how the ReAct loop terminates
- Use custom state schemas for tool-calling agents — `MessagesState` preserves tool call metadata
- Create tools inside node functions — define them at module level and bind once
</boundaries>
