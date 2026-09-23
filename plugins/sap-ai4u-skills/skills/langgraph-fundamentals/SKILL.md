---
name: langgraph-fundamentals
description: "INVOKE THIS SKILL when writing ANY LangGraph code. Covers StateGraph, state schemas (TypedDict/dataclass/Pydantic), reducers, nodes, edges, Command, Send, parallel execution, map-reduce, message trimming, invoke, streaming, and error handling. Python-only, SAP Gen AI Hub."
---

<overview>
LangGraph models agent workflows as **directed graphs**:

- **StateGraph**: Main class for building stateful graphs
- **Nodes**: Functions that perform work and update state
- **Edges**: Define execution order (static or conditional)
- **START/END**: Special nodes marking entry and exit points
- **State with Reducers**: Control how state updates are merged

Graphs must be `compile()`d before execution.

Co-invoke `langgraph-genai-hub-setup` for LLM initialization details.
</overview>

<design-methodology>

### Designing a LangGraph application

Follow these 5 steps when building a new graph:

1. **Map out discrete steps** — sketch a flowchart of your workflow. Each step becomes a node.
2. **Identify what each step does** — categorize nodes: LLM step, data step, action step, or user input step. For each, determine static context (prompt), dynamic context (from state), retry strategy, and desired outcome.
3. **Design your state** — state is shared memory for all nodes. Store raw data, format prompts on-demand inside nodes.
4. **Build your nodes** — implement each step as a function that takes state and returns partial updates.
5. **Wire it together** — connect nodes with edges, add conditional routing, compile with a checkpointer if needed.

</design-methodology>

<when-to-use-langgraph>

| Use LangGraph When | Use Alternatives When |
|-------------------|----------------------|
| Need fine-grained control over agent orchestration | Quick prototyping → LangChain agents |
| Building complex workflows with branching/loops | Simple stateless workflows → LangChain direct |
| Require human-in-the-loop, persistence | Batteries-included features → Deep Agents |

</when-to-use-langgraph>

---

## State Management

<state-update-strategies>

| Need | Solution | Example |
|------|----------|---------|
| Overwrite value | No reducer (default) | Simple fields like counters |
| Append to list | Reducer (operator.add / concat) | Message history, logs |
| Custom logic | Custom reducer function | Complex merging |

</state-update-strategies>

<ex-state-with-reducer>
<python>
Define state schema with reducers for accumulating lists and summing integers.
```python
from typing_extensions import TypedDict, Annotated
import operator

class State(TypedDict):
    name: str  # Default: overwrites on update
    messages: Annotated[list, operator.add]  # Appends to list
    total: Annotated[int, operator.add]  # Sums integers
```
</python>
</ex-state-with-reducer>

<fix-forgot-reducer-for-list>
<python>
Without a reducer, returning a list overwrites previous values.
```python
# WRONG: List will be OVERWRITTEN
class State(TypedDict):
    messages: list  # No reducer!

# Node 1 returns: {"messages": ["A"]}
# Node 2 returns: {"messages": ["B"]}
# Final: {"messages": ["B"]}  # "A" is LOST!

# CORRECT: Use Annotated with operator.add
from typing import Annotated
import operator

class State(TypedDict):
    messages: Annotated[list, operator.add]
# Final: {"messages": ["A", "B"]}
```
</python>
</fix-forgot-reducer-for-list>

<fix-state-must-return-dict>
<python>
Nodes must return partial updates, not mutate and return full state.
```python
# WRONG: Returning entire state object
def my_node(state: State) -> State:
    state["field"] = "updated"
    return state  # Don't mutate and return!

# CORRECT: Return dict with only the updates
def my_node(state: State) -> dict:
    return {"field": "updated"}
```
</python>
</fix-state-must-return-dict>

---

## State Schema Choices

<state-schema-selection>

| Schema Type | Runtime Validation | Defaults | Best For |
|------------|-------------------|----------|----------|
| `TypedDict` | No | No (use `total=False`) | Lightweight state, most common |
| `dataclass` | No | Yes (`__post_init__`) | Structured data with defaults |
| `Pydantic BaseModel` | Yes (`field_validator`) | Yes | Input validation, strict contracts |

</state-schema-selection>

<ex-state-schema-typeddict>
<python>
TypedDict state — lightest, no runtime validation. Use `total=False` for optional fields.
```python
from typing import Any, Dict, List, Literal, TypedDict

class State(TypedDict, total=False):
    name: str
    age: int
    mood: Literal["happy", "sad", "excited"]
    interests: List[str]
    result: Dict[str, Any]
```
</python>
</ex-state-schema-typeddict>

<ex-state-schema-pydantic>
<python>
Pydantic state — validates at runtime with field validators.
```python
from typing import Any, Dict, List, Literal
from pydantic import BaseModel, field_validator

class State(BaseModel):
    name: str = ""
    age: int = 0
    mood: Literal["happy", "sad", "excited"] = "happy"
    interests: List[str] = []
    result: Dict[str, Any] = {}

    @field_validator("age")
    @classmethod
    def validate_age(cls, value):
        if value < 0 or value > 150:
            raise ValueError("Age must be between 0 and 150")
        return value
```
</python>
</ex-state-schema-pydantic>

<fix-pydantic-arbitrary-types>
<python>
Pydantic state with LangChain message types requires extra config.
```python
from pydantic import ConfigDict

class State(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    messages: List[BaseMessage]  # LangChain types need arbitrary_types_allowed
```
</python>
</fix-pydantic-arbitrary-types>

---

## State Reducers

<ex-custom-reducer>
<python>
Custom reducer that aggregates numeric metrics from parallel nodes.
```python
from typing import Dict, Annotated

def metrics_aggregator(left: Dict[str, int], right: Dict[str, int]) -> Dict[str, int]:
    """Sum matching keys from parallel metric collectors."""
    result = (left or {}).copy()
    for key, value in (right or {}).items():
        result[key] = result.get(key, 0) + value
    return result

class AnalysisState(TypedDict):
    metrics: Annotated[Dict[str, int], metrics_aggregator]
    summary: str
```
</python>
</ex-custom-reducer>

<fix-parallel-no-reducer>
<python>
Parallel nodes writing to the same key without a reducer causes an error.
```python
# WRONG: Two parallel nodes both write to "items" with no reducer
class State(TypedDict):
    items: list  # No reducer!

graph.add_edge(START, "task_a")  # task_a returns {"items": ["A"]}
graph.add_edge(START, "task_b")  # task_b returns {"items": ["B"]}
# ERROR: Cannot resolve conflicting updates!

# CORRECT: Add a reducer
class State(TypedDict):
    items: Annotated[list, operator.add]  # Now parallel writes merge correctly
```
</python>
</fix-parallel-no-reducer>

---

## Nodes

<node-function-signatures>

Node functions accept these arguments:

<python>

| Signature | When to Use |
|-----------|-------------|
| `def node(state: State)` | Simple nodes that only need state |
| `def node(state: State, config: RunnableConfig)` | Need thread_id, tags, or configurable values |
| `def node(state: State, runtime: Runtime[Context])` | Need runtime context, store, or stream_writer |

```python
from langchain_core.runnables import RunnableConfig
from langgraph.runtime import Runtime

def plain_node(state: State):
    return {"results": "done"}

def node_with_config(state: State, config: RunnableConfig):
    thread_id = config["configurable"]["thread_id"]
    return {"results": f"Thread: {thread_id}"}

def node_with_runtime(state: State, runtime: Runtime[Context]):
    user_id = runtime.context.user_id
    return {"results": f"User: {user_id}"}
```
</python>

</node-function-signatures>

---

## Edges

<edge-type-selection>

| Need | Edge Type | When to Use |
|------|-----------|-------------|
| Always go to same node | `add_edge()` | Fixed, deterministic flow |
| Route based on state | `add_conditional_edges()` | Dynamic branching |
| Update state AND route | `Command` | Combine logic in single node |
| Fan-out to multiple nodes | `Send` | Parallel processing with dynamic inputs |

</edge-type-selection>

<ex-basic-graph>
<python>
Simple two-node graph with linear edges.
```python
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict

class State(TypedDict):
    input: str
    output: str

def process_input(state: State) -> dict:
    return {"output": f"Processed: {state['input']}"}

def finalize(state: State) -> dict:
    return {"output": state["output"].upper()}

graph = (
    StateGraph(State)
    .add_node("process", process_input)
    .add_node("finalize", finalize)
    .add_edge(START, "process")
    .add_edge("process", "finalize")
    .add_edge("finalize", END)
    .compile()
)

result = graph.invoke({"input": "hello"})
print(result["output"])  # "PROCESSED: HELLO"
```
</python>
</ex-basic-graph>

<ex-conditional-edges>
<python>
Route to different nodes based on state with conditional edges.
```python
from typing import Literal
from langgraph.graph import StateGraph, START, END

class State(TypedDict):
    query: str
    route: str
    result: str

def classify(state: State) -> dict:
    if "weather" in state["query"].lower():
        return {"route": "weather"}
    return {"route": "general"}

def route_query(state: State) -> Literal["weather", "general"]:
    return state["route"]

graph = (
    StateGraph(State)
    .add_node("classify", classify)
    .add_node("weather", lambda s: {"result": "Sunny, 72F"})
    .add_node("general", lambda s: {"result": "General response"})
    .add_edge(START, "classify")
    .add_conditional_edges("classify", route_query, ["weather", "general"])
    .add_edge("weather", END)
    .add_edge("general", END)
    .compile()
)
```
</python>
</ex-conditional-edges>

---

## Command

Command combines state updates and routing in a single return value. Fields:
- **`update`**: State updates to apply (like returning a dict from a node)
- **`goto`**: Node name(s) to navigate to next
- **`resume`**: Value to resume after `interrupt()` — see human-in-the-loop skill

<ex-command-state-and-routing>
<python>
Command lets you update state AND choose next node in one return.
```python
from langgraph.types import Command
from typing import Literal

class State(TypedDict):
    count: int
    result: str

def node_a(state: State) -> Command[Literal["node_b", "node_c"]]:
    """Update state AND decide next node in one return."""
    new_count = state["count"] + 1
    if new_count > 5:
        return Command(update={"count": new_count}, goto="node_c")
    return Command(update={"count": new_count}, goto="node_b")

graph = (
    StateGraph(State)
    .add_node("node_a", node_a)
    .add_node("node_b", lambda s: {"result": "B"})
    .add_node("node_c", lambda s: {"result": "C"})
    .add_edge(START, "node_a")
    .add_edge("node_b", END)
    .add_edge("node_c", END)
    .compile()
)
```
</python>
</ex-command-state-and-routing>

<command-return-type-annotations>

**Python**: Use `Command[Literal["node_a", "node_b"]]` as the return type annotation to declare valid goto destinations.

</command-return-type-annotations>

<warning-command-static-edges>

**Warning**: `Command` only adds **dynamic** edges — static edges defined with `add_edge` still execute. If `node_a` returns `Command(goto="node_c")` and you also have `graph.add_edge("node_a", "node_b")`, **both** `node_b` and `node_c` will run.

</warning-command-static-edges>

---

## Send API

Fan-out with `Send`: return `[Send("worker", {...})]` from a conditional edge to spawn parallel workers. Requires a reducer on the results field.

<ex-orchestrator-worker>
<python>
Fan out tasks to parallel workers using the Send API and aggregate results.
```python
from langgraph.types import Send
from typing import Annotated
import operator

class OrchestratorState(TypedDict):
    tasks: list[str]
    results: Annotated[list, operator.add]
    summary: str

def orchestrator(state: OrchestratorState):
    """Fan out tasks to workers."""
    return [Send("worker", {"task": task}) for task in state["tasks"]]

def worker(state: dict) -> dict:
    return {"results": [f"Completed: {state['task']}"]}

def synthesize(state: OrchestratorState) -> dict:
    return {"summary": f"Processed {len(state['results'])} tasks"}

graph = (
    StateGraph(OrchestratorState)
    .add_node("worker", worker)
    .add_node("synthesize", synthesize)
    .add_conditional_edges(START, orchestrator, ["worker"])
    .add_edge("worker", "synthesize")
    .add_edge("synthesize", END)
    .compile()
)

result = graph.invoke({"tasks": ["Task A", "Task B", "Task C"]})
```
</python>
</ex-orchestrator-worker>

<fix-send-accumulator>
<python>
Use a reducer to accumulate parallel worker results (otherwise last worker overwrites).
```python
# WRONG: No reducer - last worker overwrites
class State(TypedDict):
    results: list

# CORRECT
class State(TypedDict):
    results: Annotated[list, operator.add]  # Accumulates
```
</python>
</fix-send-accumulator>

---

## Message Trimming

Long conversations can exceed LLM context limits. Trim messages before invoking the LLM inside your node.

<message-trimming-strategies>

| Strategy | How It Works | Best For |
|----------|-------------|----------|
| Count-based | Keep last N messages | Simple limit enforcement |
| Token-based | Estimate tokens, trim to budget | Cost management |
| Smart | Preserve system + early context + recent | Context-sensitive conversations |
| Summary-based | LLM summarizes old messages, keep recent | Long sessions with important early context |

</message-trimming-strategies>

<ex-message-trimming>
<python>
Count-based trimming that always preserves SystemMessages.
```python
from langchain_core.messages import BaseMessage, SystemMessage

def trim_messages_by_count(messages: list[BaseMessage], max_messages: int = 10) -> list[BaseMessage]:
    """Keep system messages and the most recent N non-system messages."""
    if len(messages) <= max_messages:
        return messages
    system_msgs = [m for m in messages if isinstance(m, SystemMessage)]
    other_msgs = [m for m in messages if not isinstance(m, SystemMessage)]
    recent = other_msgs[-(max_messages - len(system_msgs)):]
    return system_msgs + recent

# Use inside a node
def chat_node(state: MessagesState):
    trimmed = trim_messages_by_count(state["messages"], max_messages=8)
    response = llm.invoke(trimmed)
    return {"messages": [response]}
```
</python>
</ex-message-trimming>

<fix-preserve-system-message>
<python>
Always preserve SystemMessage when trimming — it contains critical instructions.
```python
# WRONG: Naive slicing drops the system message
trimmed = messages[-10:]  # System message at index 0 is lost!

# CORRECT: Separate and preserve system messages
system_msgs = [m for m in messages if isinstance(m, SystemMessage)]
other_msgs = [m for m in messages if not isinstance(m, SystemMessage)]
trimmed = system_msgs + other_msgs[-8:]
```
</python>
</fix-preserve-system-message>

---

## Parallel Execution (Fan-out / Fan-in)

Multiple edges from the same source node cause all targets to execute in the same superstep (parallel). All must complete before the next superstep begins.

<ex-parallel-fanout>
<python>
Three parallel research nodes fan out from START, fan in to a synthesizer.
```python
import operator
from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, START, END

class ResearchState(TypedDict):
    query: str
    stock_data: Annotated[list, operator.add]
    news_data: Annotated[list, operator.add]
    info_data: Annotated[list, operator.add]
    report: str

def get_stock(state: ResearchState) -> dict:
    return {"stock_data": [f"Price for {state['query']}: $150.00"]}

def get_news(state: ResearchState) -> dict:
    return {"news_data": [f"Latest news for {state['query']}"]}

def get_info(state: ResearchState) -> dict:
    return {"info_data": [f"Company info for {state['query']}"]}

def synthesize(state: ResearchState) -> dict:
    all_data = state["stock_data"] + state["news_data"] + state["info_data"]
    return {"report": f"Report with {len(all_data)} data points"}

builder = StateGraph(ResearchState)
builder.add_node("stock", get_stock)
builder.add_node("news", get_news)
builder.add_node("info", get_info)
builder.add_node("synthesize", synthesize)

# Fan-out: START to all three (parallel)
builder.add_edge(START, "stock")
builder.add_edge(START, "news")
builder.add_edge(START, "info")

# Fan-in: all three to synthesizer
builder.add_edge("stock", "synthesize")
builder.add_edge("news", "synthesize")
builder.add_edge("info", "synthesize")
builder.add_edge("synthesize", END)

graph = builder.compile()
```
</python>
</ex-parallel-fanout>

---

## Map-Reduce Pattern

Fan-out mappers analyze different aspects in parallel, then a single reducer combines results. Uses static edges for a fixed set of mappers or the Send API for dynamic fan-out.

<ex-map-reduce>
<python>
Four parallel analysis mappers + one reducer, using static edges.
```python
import operator
from typing import Any, Dict, List, Annotated, TypedDict
from langgraph.graph import StateGraph, START, END
from gen_ai_hub.proxy.langchain.openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

class AnalysisState(TypedDict):
    code: str
    security: Annotated[List[Dict[str, Any]], operator.add]
    performance: Annotated[List[Dict[str, Any]], operator.add]
    quality: Annotated[List[Dict[str, Any]], operator.add]
    report: Dict[str, Any]

llm = ChatOpenAI(proxy_model_name="gpt-4.1", temperature=0.2)

def security_mapper(state: AnalysisState) -> dict:
    system = SystemMessage(content="Analyze code for security issues. Return JSON: {\"issues\": [...]}")
    response = llm.invoke([system, HumanMessage(content=state["code"])])
    return {"security": [{"aspect": "security", "raw": response.content}]}

def performance_mapper(state: AnalysisState) -> dict:
    system = SystemMessage(content="Analyze code for performance issues. Return JSON: {\"issues\": [...]}")
    response = llm.invoke([system, HumanMessage(content=state["code"])])
    return {"performance": [{"aspect": "performance", "raw": response.content}]}

def quality_mapper(state: AnalysisState) -> dict:
    system = SystemMessage(content="Analyze code quality (SOLID, readability). Return JSON: {\"issues\": [...]}")
    response = llm.invoke([system, HumanMessage(content=state["code"])])
    return {"quality": [{"aspect": "quality", "raw": response.content}]}

def reducer(state: AnalysisState) -> dict:
    total = len(state["security"]) + len(state["performance"]) + len(state["quality"])
    return {"report": {"total_aspects_analyzed": total}}

builder = StateGraph(AnalysisState)
builder.add_node("security", security_mapper)
builder.add_node("performance", performance_mapper)
builder.add_node("quality", quality_mapper)
builder.add_node("reduce", reducer)

builder.add_edge(START, "security")
builder.add_edge(START, "performance")
builder.add_edge(START, "quality")
builder.add_edge("security", "reduce")
builder.add_edge("performance", "reduce")
builder.add_edge("quality", "reduce")
builder.add_edge("reduce", END)

graph = builder.compile()
```
</python>
</ex-map-reduce>

---

## Running Graphs: Invoke and Stream

<invoke-basics>

Call `graph.invoke(input, config)` to run a graph to completion and return the final state.

<python>
```python
result = graph.invoke({"input": "hello"})
# With config (for persistence, tags, etc.)
result = graph.invoke({"input": "hello"}, {"configurable": {"thread_id": "1"}})
```
</python>

</invoke-basics>

<stream-mode-selection>

| Mode | What it Streams | Use Case |
|------|----------------|----------|
| `values` | Full state after each step | Monitor complete state |
| `updates` | State deltas | Track incremental updates |
| `messages` | LLM tokens + metadata | Chat UIs |
| `custom` | User-defined data | Progress indicators |

</stream-mode-selection>

<ex-stream-llm-tokens>
<python>
Stream LLM tokens in real-time for chat UI display.
```python
from langchain_core.messages import HumanMessage

for chunk in graph.stream(
    {"messages": [HumanMessage("Hello")]},
    stream_mode="messages"
):
    token, metadata = chunk
    if hasattr(token, "content"):
        print(token.content, end="", flush=True)
```
</python>
</ex-stream-llm-tokens>

<ex-stream-custom-data>
<python>
Emit custom progress updates from within nodes using the stream writer.
```python
from langgraph.config import get_stream_writer

def my_node(state):
    writer = get_stream_writer()
    writer("Processing step 1...")
    # Do work
    writer("Complete!")
    return {"result": "done"}

for chunk in graph.stream({"data": "test"}, stream_mode="custom"):
    print(chunk)
```
</python>
</ex-stream-custom-data>

---

## Error Handling

Match the error type to the right handler:

<error-handling-table>

| Error Type | Who Fixes | Strategy | Example |
|---|---|---|---|
| Transient (network, rate limits) | System | `RetryPolicy(max_attempts=3)` | `add_node(..., retry_policy=...)` |
| LLM-recoverable (tool failures) | LLM | `ToolNode(tools, handle_tool_errors=True)` | Error returned as ToolMessage |
| User-fixable (missing info) | Human | `interrupt({"message": ...})` | Collect missing data (see HITL skill) |
| Unexpected | Developer | Let bubble up | `raise` |

</error-handling-table>

<ex-retry-policy>
<python>
Use RetryPolicy for transient errors (network issues, rate limits).
```python
from langgraph.types import RetryPolicy

workflow.add_node(
    "search_documentation",
    search_documentation,
    retry_policy=RetryPolicy(max_attempts=3, initial_interval=1.0)
)
```
</python>
</ex-retry-policy>

<ex-tool-node-error-handling>
<python>
Use ToolNode from langgraph.prebuilt to handle tool execution and errors. When handle_tool_errors=True, errors are returned as ToolMessages so the LLM can recover.
```python
from langgraph.prebuilt import ToolNode

tool_node = ToolNode(tools, handle_tool_errors=True)

workflow.add_node("tools", tool_node)
```
</python>
</ex-tool-node-error-handling>

---

## Common Fixes

<fix-compile-before-execution>
<python>
Must compile() to get executable graph.
```python
# WRONG
builder.invoke({"input": "test"})  # AttributeError!

# CORRECT
graph = builder.compile()
graph.invoke({"input": "test"})
```
</python>
</fix-compile-before-execution>

<fix-infinite-loop-needs-exit>
<python>
Provide conditional path to END to avoid infinite loops.
```python
# WRONG: Loops forever
builder.add_edge("node_a", "node_b")
builder.add_edge("node_b", "node_a")

# CORRECT
def should_continue(state):
    return END if state["count"] > 10 else "node_b"
builder.add_conditional_edges("node_a", should_continue)
```
</python>
</fix-infinite-loop-needs-exit>

<fix-common-mistakes>
Other common mistakes:
```python
# Router must return names of nodes that exist in the graph
builder.add_node("my_node", func)  # Add node BEFORE referencing in edges
builder.add_conditional_edges("node_a", router, ["my_node"])

# Command return type needs Literal for routing destinations (Python)
def node_a(state) -> Command[Literal["node_b", "node_c"]]:
    return Command(goto="node_b")

# START is entry-only — cannot route back to it
builder.add_edge("node_a", START)  # WRONG!
builder.add_edge("node_a", "entry")  # Use a named entry node instead

# Reducer expects matching types
return {"items": ["item"]}  # List for list reducer, not a string
```
</fix-common-mistakes>

<boundaries>
### What You Should NOT Do

- Mutate state directly — always return partial update dicts from nodes
- Route back to START — it's entry-only; use a named node instead
- Forget reducers on list fields — without one, last write wins
- Mix static edges with Command goto without understanding both will execute
- Use parallel fan-out without reducers on fields receiving concurrent updates
</boundaries>

## Related Skills

- `langgraph-genai-hub-setup` — LLM initialization (always co-invoke).
- `langgraph-agent-patterns` — ReAct, tool calling, MCP.
- `langgraph-persistence` — checkpointers, threads, Store, subgraphs.
- `langgraph-human-in-the-loop` — interrupt/resume, breakpoints.
- `token-logger` — token-usage logging when the graph runs in a production API.
- `sap-btp-ai` — routing and shared environment conventions.
