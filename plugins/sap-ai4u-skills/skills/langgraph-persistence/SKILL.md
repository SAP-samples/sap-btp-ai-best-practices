---
name: langgraph-persistence
description: "INVOKE THIS SKILL when your LangGraph needs to persist state, remember conversations, travel through history, use SQLite/Postgres checkpointers, or compose subgraphs. Covers checkpointers, thread_id, time travel, Store, subgraph composition, and subgraph persistence modes. Python-only, SAP Gen AI Hub."
---

<overview>
LangGraph's persistence layer enables durable execution by checkpointing graph state:

- **Checkpointer**: Saves/loads graph state at every super-step
- **Thread ID**: Identifies separate checkpoint sequences (conversations)
- **Store**: Cross-thread memory for user preferences, facts

**Two memory types:**
- **Short-term** (checkpointer): Thread-scoped conversation history
- **Long-term** (store): Cross-thread user preferences, facts

Co-invoke `langgraph-genai-hub-setup` for LLM initialization details.
</overview>

<checkpointer-selection>

| Checkpointer | Use Case | Production Ready |
|--------------|----------|------------------|
| `InMemorySaver` | Testing, development | No |
| `SqliteSaver` | Local development, small production | Partial |
| `PostgresSaver` | Production | Yes |

</checkpointer-selection>

---

## Checkpointer Setup

<ex-basic-persistence>
<python>
Set up a basic graph with in-memory checkpointing and thread-based state persistence.
```python
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict, Annotated
import operator

class State(TypedDict):
    messages: Annotated[list, operator.add]

def add_message(state: State) -> dict:
    return {"messages": ["Bot response"]}

checkpointer = InMemorySaver()

graph = (
    StateGraph(State)
    .add_node("respond", add_message)
    .add_edge(START, "respond")
    .add_edge("respond", END)
    .compile(checkpointer=checkpointer)  # Pass at compile time
)

# ALWAYS provide thread_id
config = {"configurable": {"thread_id": "conversation-1"}}

result1 = graph.invoke({"messages": ["Hello"]}, config)
print(len(result1["messages"]))  # 2

result2 = graph.invoke({"messages": ["How are you?"]}, config)
print(len(result2["messages"]))  # 4 (previous + new)
```
</python>
</ex-basic-persistence>

---

## SQLite Checkpointer

SQLite provides on-disk persistence that survives process restarts. Use the context manager pattern for proper connection lifecycle.

<ex-sqlite-checkpointer>
<python>
SQLite checkpointer with cross-process persistence.
```python
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import MessagesState, StateGraph, START
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import HumanMessage, SystemMessage
from gen_ai_hub.proxy.langchain.openai import ChatOpenAI

def build_agent(checkpointer):
    llm = ChatOpenAI(proxy_model_name="gpt-4.1", temperature=0.2)
    llm_with_tools = llm.bind_tools(tools)
    system_msg = SystemMessage(content="You are a helpful assistant.")

    def assistant(state: MessagesState):
        return {"messages": [llm_with_tools.invoke([system_msg] + state["messages"])]}

    graph = StateGraph(MessagesState)
    graph.add_node("assistant", assistant)
    graph.add_node("tools", ToolNode(tools=tools))
    graph.add_edge(START, "assistant")
    graph.add_conditional_edges("assistant", tools_condition)
    graph.add_edge("tools", "assistant")
    return graph.compile(checkpointer=checkpointer)

# First run: create agent, do work
db_path = "agent_memory.db"
config = {"configurable": {"thread_id": "session-1"}}

with SqliteSaver.from_conn_string(db_path) as checkpointer:
    agent = build_agent(checkpointer)
    agent.invoke({"messages": [HumanMessage(content="Remember: I prefer concise answers.")]}, config)

# Second run (new process): memory persists!
with SqliteSaver.from_conn_string(db_path) as checkpointer:
    agent = build_agent(checkpointer)
    result = agent.invoke({"messages": [HumanMessage(content="What preference did I mention?")]}, config)
    # Agent remembers the preference from the first run
```
</python>
</ex-sqlite-checkpointer>

---

## PostgreSQL Checkpointer (Production)

<ex-production-postgres>
<python>
Configure PostgreSQL-backed checkpointing for production deployments.
```python
from langgraph.checkpoint.postgres import PostgresSaver

with PostgresSaver.from_conn_string(
    "postgresql://user:pass@localhost/db"
) as checkpointer:
    checkpointer.setup()  # only needed on first use to create tables
    graph = builder.compile(checkpointer=checkpointer)
```
</python>
</ex-production-postgres>

---

## Thread Management

<ex-separate-threads>
<python>
Demonstrate isolated state between different thread IDs.
```python
# Different threads maintain separate state
alice_config = {"configurable": {"thread_id": "user-alice"}}
bob_config = {"configurable": {"thread_id": "user-bob"}}

graph.invoke({"messages": ["Hi from Alice"]}, alice_config)
graph.invoke({"messages": ["Hi from Bob"]}, bob_config)

# Alice's state is isolated from Bob's
```
</python>
</ex-separate-threads>

---

## State History & Time Travel

<ex-resume-from-checkpoint>
<python>
Time travel: browse checkpoint history and replay or fork from a past state.
```python
config = {"configurable": {"thread_id": "session-1"}}

result = graph.invoke({"messages": ["start"]}, config)

# Browse checkpoint history
states = list(graph.get_state_history(config))

# Replay from a past checkpoint
past = states[-2]
result = graph.invoke(None, past.config)  # None = resume from checkpoint

# Or fork: update state at a past checkpoint, then resume
fork_config = graph.update_state(past.config, {"messages": ["edited"]})
result = graph.invoke(None, fork_config)
```
</python>
</ex-resume-from-checkpoint>

<ex-update-state>
<python>
Manually update graph state before resuming execution.
```python
config = {"configurable": {"thread_id": "session-1"}}

# Modify state before resuming
graph.update_state(config, {"data": "manually_updated"})

# Resume with updated state
result = graph.invoke(None, config)
```
</python>
</ex-update-state>

---

## Subgraph Composition

Subgraphs are compiled graphs added as nodes to a parent graph. They enable hierarchical organization, parallel processing, and modular design.

<subgraph-composition-overview>

Key concepts:
- **Overlapping keys**: Parent and subgraph communicate through shared state keys
- **Output schemas**: `StateGraph(state_schema=..., output_schema=...)` controls what subgraphs return
- **Parallel subgraphs**: Use `Annotated[list, operator.add]` on parent fields that receive concurrent updates

</subgraph-composition-overview>

<ex-subgraph-composition>
<python>
Parent graph with parallel subgraphs using overlapping keys and output schemas.
```python
from typing import List, Optional, Annotated
from typing_extensions import TypedDict
from operator import add
from langgraph.graph import StateGraph, START, END

# --- Data model ---
class Log(TypedDict):
    id: str
    question: str
    answer: str
    grade: Optional[int]

# --- Subgraph 1: Failure Analysis ---
class FailureAnalysisState(TypedDict):
    cleaned_logs: List[Log]      # Input from parent (overlapping key)
    failures: List[Log]          # Internal state
    fa_summary: str              # Output
    processed_logs: List[str]    # Output

class FailureAnalysisOutput(TypedDict):
    fa_summary: str              # Only these keys return to parent
    processed_logs: List[str]

def build_failure_subgraph():
    def get_failures(state: FailureAnalysisState) -> dict:
        failures = [log for log in state["cleaned_logs"] if log.get("grade") is not None]
        return {"failures": failures}

    def summarize(state: FailureAnalysisState) -> dict:
        return {
            "fa_summary": f"Found {len(state['failures'])} issues",
            "processed_logs": [f"fa-{f['id']}" for f in state["failures"]],
        }

    builder = StateGraph(
        state_schema=FailureAnalysisState,
        output_schema=FailureAnalysisOutput,  # Controls return values
    )
    builder.add_node("get_failures", get_failures)
    builder.add_node("summarize", summarize)
    builder.add_edge(START, "get_failures")
    builder.add_edge("get_failures", "summarize")
    builder.add_edge("summarize", END)
    return builder.compile()

# --- Subgraph 2: Question Summarization ---
class QuestionSummaryState(TypedDict):
    cleaned_logs: List[Log]      # Input from parent (overlapping key)
    report: str                  # Output
    processed_logs: List[str]    # Output

class QuestionSummaryOutput(TypedDict):
    report: str
    processed_logs: List[str]

def build_summary_subgraph():
    def summarize(state: QuestionSummaryState) -> dict:
        questions = [log["question"] for log in state["cleaned_logs"]]
        return {
            "report": f"Analyzed {len(questions)} questions",
            "processed_logs": [f"qs-{log['id']}" for log in state["cleaned_logs"]],
        }

    builder = StateGraph(
        state_schema=QuestionSummaryState,
        output_schema=QuestionSummaryOutput,
    )
    builder.add_node("summarize", summarize)
    builder.add_edge(START, "summarize")
    builder.add_edge("summarize", END)
    return builder.compile()

# --- Parent Graph ---
class ParentState(TypedDict):
    raw_logs: List[Log]
    cleaned_logs: List[Log]
    fa_summary: str
    report: str
    processed_logs: Annotated[List[str], add]  # Reducer for parallel results

def clean_logs(state: ParentState) -> dict:
    return {"cleaned_logs": state["raw_logs"]}

builder = StateGraph(ParentState)
builder.add_node("clean", clean_logs)
builder.add_node("failure_analysis", build_failure_subgraph())      # Subgraph as node
builder.add_node("question_summary", build_summary_subgraph())      # Subgraph as node

builder.add_edge(START, "clean")
builder.add_edge("clean", "failure_analysis")       # Parallel
builder.add_edge("clean", "question_summary")       # Parallel
builder.add_edge("failure_analysis", END)
builder.add_edge("question_summary", END)

graph = builder.compile()
```
</python>
</ex-subgraph-composition>

---

## Subgraph Checkpointer Scoping

When compiling a subgraph, the `checkpointer` parameter controls persistence behavior. This is critical for subgraphs that use interrupts, need multi-turn memory, or run in parallel.

<subgraph-checkpointer-scoping-table>

| Feature | `checkpointer=False` | `None` (default) | `True` |
|---|---|---|---|
| Interrupts (HITL) | No | Yes | Yes |
| Multi-turn memory | No | No | Yes |
| Multiple calls (different subgraphs) | Yes | Yes | Warning (namespace conflicts possible) |
| Multiple calls (same subgraph) | Yes | Yes | No |
| State inspection | No | Warning (current invocation only) | Yes |

</subgraph-checkpointer-scoping-table>

<subgraph-checkpointer-when-to-use>

### When to use each mode

- **`checkpointer=False`** — Subgraph doesn't need interrupts or persistence. Simplest option, no checkpoint overhead.
- **`None` (default / omit `checkpointer`)** — Subgraph needs `interrupt()` but not multi-turn memory. Each invocation starts fresh but can pause/resume.
- **`checkpointer=True`** — Subgraph needs to remember state across invocations (multi-turn conversations).

</subgraph-checkpointer-when-to-use>

<warning-stateful-subgraphs-parallel>

**Warning**: Stateful subgraphs (`checkpointer=True`) do NOT support calling the same subgraph instance multiple times within a single node — the calls write to the same checkpoint namespace and conflict.

</warning-stateful-subgraphs-parallel>

<ex-subgraph-checkpointer-modes>
<python>
Choose the right checkpointer mode for your subgraph.
```python
# No interrupts needed — opt out of checkpointing
subgraph = subgraph_builder.compile(checkpointer=False)

# Need interrupts but not cross-invocation persistence (default)
subgraph = subgraph_builder.compile()

# Need cross-invocation persistence (stateful)
subgraph = subgraph_builder.compile(checkpointer=True)
```
</python>
</ex-subgraph-checkpointer-modes>

<parallel-subgraph-namespacing>

### Parallel subgraph namespacing

When multiple **different** stateful subgraphs run in parallel, wrap each in its own `StateGraph` with a unique node name for stable namespace isolation:

<python>
```python
from langgraph.graph import MessagesState, StateGraph

def create_sub_agent(model, *, name, **kwargs):
    """Wrap an agent with a unique node name for namespace isolation."""
    agent = create_agent(model=model, name=name, **kwargs)
    return (
        StateGraph(MessagesState)
        .add_node(name, agent)  # unique name -> stable namespace
        .add_edge("__start__", name)
        .compile()
    )

fruit_agent = create_sub_agent(
    "gpt-4.1", name="fruit_agent",
    tools=[fruit_info], prompt="...", checkpointer=True,
)
veggie_agent = create_sub_agent(
    "gpt-4.1", name="veggie_agent",
    tools=[veggie_info], prompt="...", checkpointer=True,
)
```
</python>

Note: Subgraphs added as nodes (via `add_node`) already get name-based namespaces automatically and don't need this wrapper.

</parallel-subgraph-namespacing>

---

## Long-Term Memory (Store)

<ex-long-term-memory-store>
<python>
Use a Store for cross-thread memory to share user preferences across conversations.
```python
from langgraph.store.memory import InMemoryStore
from langgraph.runtime import Runtime

store = InMemoryStore()

# Save user preference (available across ALL threads)
store.put(("alice", "preferences"), "language", {"preference": "short responses"})

# Node with store — access via runtime
def respond(state, runtime: Runtime):
    prefs = runtime.store.get((state["user_id"], "preferences"), "language")
    return {"response": f"Using preference: {prefs.value}"}

# Compile with BOTH checkpointer and store
graph = builder.compile(checkpointer=checkpointer, store=store)

# Both threads access same long-term memory
graph.invoke({"user_id": "alice"}, {"configurable": {"thread_id": "thread-1"}})
graph.invoke({"user_id": "alice"}, {"configurable": {"thread_id": "thread-2"}})  # Same preferences!
```
</python>
</ex-long-term-memory-store>

<ex-store-operations>
<python>
Basic store operations: put, get, search, and delete.
```python
from langgraph.store.memory import InMemoryStore

store = InMemoryStore()

store.put(("user-123", "facts"), "location", {"city": "San Francisco"})  # Put
item = store.get(("user-123", "facts"), "location")  # Get
results = store.search(("user-123", "facts"), filter={"city": "San Francisco"})  # Search
store.delete(("user-123", "facts"), "location")  # Delete
```
</python>
</ex-store-operations>

---

## Fixes

<fix-thread-id-required>
<python>
Always provide thread_id in config to enable state persistence.
```python
# WRONG: No thread_id - state NOT persisted!
graph.invoke({"messages": ["Hello"]})
graph.invoke({"messages": ["What did I say?"]})  # Doesn't remember!

# CORRECT: Always provide thread_id
config = {"configurable": {"thread_id": "session-1"}}
graph.invoke({"messages": ["Hello"]}, config)
graph.invoke({"messages": ["What did I say?"]}, config)  # Remembers!
```
</python>
</fix-thread-id-required>

<fix-inmemory-not-for-production>
<python>
Use PostgresSaver instead of InMemorySaver for production persistence.
```python
# WRONG: Data lost on process restart
checkpointer = InMemorySaver()  # In-memory only!

# CORRECT: Use persistent storage for production
from langgraph.checkpoint.postgres import PostgresSaver
with PostgresSaver.from_conn_string("postgresql://...") as checkpointer:
    checkpointer.setup()  # only needed on first use to create tables
    graph = builder.compile(checkpointer=checkpointer)
```
</python>
</fix-inmemory-not-for-production>

<fix-update-state-with-reducers>
<python>
Use Overwrite to replace state values instead of passing through reducers.
```python
from langgraph.types import Overwrite

# State with reducer: items: Annotated[list, operator.add]
# Current state: {"items": ["A", "B"]}

# update_state PASSES THROUGH reducers
graph.update_state(config, {"items": ["C"]})  # Result: ["A", "B", "C"] - Appended!

# To REPLACE instead, use Overwrite
graph.update_state(config, {"items": Overwrite(["C"])})  # Result: ["C"] - Replaced
```
</python>
</fix-update-state-with-reducers>

<fix-store-injection>
<python>
Access store via the Runtime object in graph nodes.
```python
# WRONG: Store not available in node
def my_node(state):
    store.put(...)  # NameError! store not defined

# CORRECT: Access store via runtime
from langgraph.runtime import Runtime

def my_node(state, runtime: Runtime):
    runtime.store.put(...)  # Correct store instance
```
</python>
</fix-store-injection>

<boundaries>
### What You Should NOT Do

- Use `InMemorySaver` in production — data lost on restart; use `PostgresSaver`
- Forget `thread_id` — state won't persist without it
- Expect `update_state` to bypass reducers — it passes through them; use `Overwrite` to replace
- Run the same stateful subgraph (`checkpointer=True`) in parallel within one node — namespace conflict
- Access store directly in a node — use `runtime.store` via the `Runtime` param
- Forget `output_schema` on subgraphs — without it, internal state leaks to the parent
</boundaries>
