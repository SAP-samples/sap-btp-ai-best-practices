---
name: langgraph-persistence
description: "INVOKE THIS SKILL when your LangGraph needs to persist state, remember conversations across turns or restarts, store memory in SAP HANA, serialize concurrent turns across Cloud Foundry instances, bound history with summarization, travel through history, or compose subgraphs. Covers the HANA conversation store, leases and versioned saves, in-process checkpointers, thread_id, time travel, Store, subgraph composition, and subgraph persistence modes. Python-only, SAP Gen AI Hub."
---

<overview>
Two different things are often called "persistence" in LangGraph. Keep them apart:

- **Conversation memory** — the messages of a conversation, kept between user
  turns, process restarts, and Cloud Foundry instances. On SAP BTP this is
  stored in **SAP HANA**.
- **Execution checkpoints** — a LangGraph checkpointer saving graph state at
  every super-step so a paused graph (`interrupt()`, breakpoints, time travel)
  can resume. In this skill set checkpointers are **in-process only**
  (`InMemorySaver`), for tests, local development, and debugging.

**Two memory types:**
- **Short-term**: one conversation's message history (HANA conversation store).
- **Long-term**: facts shared across conversations (a HANA table; `InMemoryStore` only locally).

Co-invoke `langgraph-genai-hub-setup` for LLM initialization details and
`sap-hana-sql` for any new HANA table.
</overview>

<storage-selection>

| Need | Use | Notes |
|------|-----|-------|
| Multi-turn conversation history in a served agent (CF, restarts, several instances) | HANA conversation store from `create-langgraph-react-agent` | Full messages incl. tool calls/results, summarization, lease + version |
| Human approval that spans user turns in production | Pending action in the same HANA row | See `langgraph-human-in-the-loop` |
| Cross-conversation facts or lessons | Your own HANA table (`sap-hana-sql`) | See `langgraph-reflexion` for a lessons table |
| `interrupt()`, breakpoints, time travel inside one process | `InMemorySaver` | Tests, notebooks, local debugging only |

**Not used in this landscape:**

- `SqliteSaver` — a local file: not shared between CF instances and lost when a
  container is replaced.
- `PostgresSaver` — PostgreSQL is not part of the standard SAP BTP AI landscape
  these skills target; HANA is.
- `sap_cloud_sdk.agent_memory.factory.create_checkpointer` — returns
  `InMemorySaver`/`TimedInMemorySaver` (verified in `sap-cloud-sdk` 0.62.0), so it
  is not durable either.
- A HANA LangGraph checkpointer — none exists (`langchain-hana` 1.2.0 ships
  vector store, RDF graph and SPARQL QA only). Do not invent one; use the
  conversation store and the pending-action pattern instead.

</storage-selection>

---

## Checkpointer Setup (In-Process: Tests and Local Development)

Use this for unit tests, notebooks, and local `interrupt()` experiments. State
lives in the Python process only.

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

## Conversation Memory in SAP HANA (Production)

The `create-langgraph-react-agent` template ships `template_agent/memory.py`
(`HanaConversationStore`). Reuse it instead of writing a new store. It:

- creates the table automatically in the runtime user's `CURRENT_SCHEMA`, and
  migrates an older role/content table with `ALTER TABLE ... ADD`;
- stores one row per conversation (`CONTEXT_ID`) with the **full** message
  list serialized by `messages_to_dict` — tool calls and tool results included;
- serializes turns across instances with a row **lease** and fences every save
  with a **version** check (see Concurrency below).

The graph is compiled **without** a checkpointer. History is loaded before the
turn, passed in as input, and the resulting state is saved after the turn.

<ex-hana-conversation-memory>
<python>
Load-run-save one turn with the template's HANA store.
```python
import asyncio
from uuid import uuid4

from langchain_core.messages import HumanMessage

from template_agent.config import MemorySettings
from template_agent.memory import ContextBusyError, HanaConversationStore

store = HanaConversationStore(MemorySettings(enabled=True))
store.ensure()  # create or migrate the table before serving traffic


async def run_turn(graph, context_id: str, user_text: str, lease_seconds: int = 300):
    """Run one turn for one conversation; safe with several app instances."""
    owner = uuid4().hex
    if not await asyncio.to_thread(store.acquire, context_id, owner, lease_seconds):
        raise ContextBusyError(context_id)  # or poll with backoff, as AgentRuntime does
    try:
        stored = await asyncio.to_thread(store.load, context_id)
        final = await graph.ainvoke(
            {"messages": [*stored.messages, HumanMessage(content=user_text)]}
        )
        # Saves only if the version is unchanged and we still hold the lease;
        # releases the lease in the same statement.
        await asyncio.to_thread(
            store.save, context_id, owner, stored.version, final["messages"], None
        )
        return final
    except BaseException:
        await asyncio.to_thread(store.release, context_id, owner)
        raise
```
</python>
</ex-hana-conversation-memory>

`AgentRuntime.ainvoke` in the template already does all of this, plus an
in-process `asyncio.Lock`, lease polling up to `memory.lock_wait_seconds`,
attachment-byte stripping, and a `memory.max_messages` ceiling.

---

## Concurrency Across Instances

Cloud Foundry can run several instances, and clients retry. Two turns on the
same conversation must never run at once, or one overwrites the other.

<concurrency-rules>

- An `asyncio.Lock` (or any in-process lock) only protects one process. Keep it
  to queue same-instance turns cheaply, but never rely on it alone.
- A blind upsert (`MERGE` / `UPSERT`) after the turn is last-writer-wins: a
  turn silently disappears.
- Checkpointers do not help either: two concurrent invocations on one
  `thread_id` both read the latest checkpoint and fork the history.
- Use the HANA **lease + version** pattern:
  1. `acquire`: `UPDATE ... SET LEASE_OWNER=?, LEASE_EXPIRES_AT=ADD_SECONDS(CURRENT_UTCTIMESTAMP, ?) WHERE CONTEXT_ID=? AND (LEASE_OWNER IS NULL OR LEASE_EXPIRES_AT < CURRENT_UTCTIMESTAMP)`, then read `LEASE_OWNER` back to confirm the winner.
  2. Busy: wait with backoff up to a limit, then fail with a retryable error.
  3. `save`: `UPDATE ... SET ..., VERSION = VERSION + 1, LEASE_OWNER = NULL WHERE CONTEXT_ID=? AND VERSION=? AND LEASE_OWNER=?`; zero rows means the turn is stale — raise, do not overwrite.
  4. On any failure: release the lease (`WHERE LEASE_OWNER = ?`).
- `lease_seconds` must exceed the longest turn. If a lease expires mid-turn,
  the version check still prevents the stale save.

</concurrency-rules>

The generic SQL pattern is in `sap-hana-sql`
(`references/hana-ddl-dml-patterns.md`, "Row Leases and Versioned Saves").

---

## Bounding History Size

Full histories grow with every tool result and are replayed on every model call.
Bound them **before each model call** and persist the bounded state:

- Cap individual tool results (the template uses
  `tool_output.max_output_characters`, default 30000).
- Summarize older messages with LangChain's `SummarizationMiddleware` once the
  history passes a token trigger; it keeps the most recent messages verbatim and
  never splits a tool call from its result. With `create_agent` pass it as
  middleware; in a hand-built graph call its `abefore_model` hook from a node,
  as the template's `compact_context` node does.
- Keep a hard message ceiling with
  `trim_messages(..., token_counter=len, strategy="last", start_on="human")` so
  trimming never leaves an orphaned `ToolMessage`.

<ex-summarization-node>
<python>
Summarize long history in a hand-built graph before each model call.
```python
from langchain.agents.middleware import SummarizationMiddleware
from langgraph.graph import MessagesState, StateGraph, START
from langgraph.prebuilt import ToolNode, tools_condition

summarizer = SummarizationMiddleware(
    model=summary_llm,                 # a cheaper Gen AI Hub chat model is fine
    trigger=("tokens", 200_000),
    keep=("messages", 6),
)

async def compact_context(state: MessagesState) -> dict:
    # Returns RemoveMessage(REMOVE_ALL_MESSAGES) + summary + kept messages, or None.
    return await summarizer.abefore_model(state, None) or {}

builder = StateGraph(MessagesState)
builder.add_node("compact_context", compact_context)
builder.add_node("agent", agent_node)
builder.add_edge(START, "compact_context")
builder.add_edge("compact_context", "agent")
builder.add_conditional_edges("agent", tools_condition)
builder.add_node("tools", ToolNode(tools))
builder.add_edge("tools", "compact_context")   # compact again inside long tool loops
```
</python>
</ex-summarization-node>

---

## Thread Management (In-Process Checkpointer)

`thread_id` plays the role `CONTEXT_ID` plays in the HANA store. When an
authenticated caller identity is available, namespace it
(`f"{user_id}:{context_id}"`) so one user cannot read another user's
conversation by guessing its identifier.

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

Time travel needs a checkpointer, so in this landscape it is a local
debugging tool (`InMemorySaver`), not a production feature.

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

`InMemoryStore` is fine for local experiments. In a served agent, keep
cross-conversation facts in a HANA table created and checked at startup
(`sap-hana-sql`), and read/write it from a tool or node.

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
Keep conversation memory in HANA, not in a checkpointer, for served agents.
```python
# WRONG: lost on restart, invisible to other CF instances
graph = builder.compile(checkpointer=InMemorySaver())

# WRONG: a local file is neither shared nor durable on Cloud Foundry
from langgraph.checkpoint.sqlite import SqliteSaver

# CORRECT: compile without a checkpointer; load/save history in HANA per turn
graph = builder.compile()
stored = store.load(context_id)          # HanaConversationStore
final = graph.invoke({"messages": [*stored.messages, HumanMessage(content=text)]})
store.save(context_id, owner, stored.version, final["messages"], None)
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

- Use `InMemorySaver`, `SqliteSaver`, or `sap_cloud_sdk create_checkpointer` for conversation memory in a served agent — use the HANA conversation store
- Recommend `PostgresSaver` or write a custom HANA checkpointer — HANA is the production store; approvals use the pending-action pattern
- Rely on an in-process lock to serialize turns across CF instances — use the HANA lease + version check
- Persist only user/final-assistant text when tools matter — store the full message list and bound it with summarization
- Forget `thread_id` — state won't persist without it
- Expect `update_state` to bypass reducers — it passes through them; use `Overwrite` to replace
- Run the same stateful subgraph (`checkpointer=True`) in parallel within one node — namespace conflict
- Access store directly in a node — use `runtime.store` via the `Runtime` param
- Forget `output_schema` on subgraphs — without it, internal state leaks to the parent
</boundaries>
