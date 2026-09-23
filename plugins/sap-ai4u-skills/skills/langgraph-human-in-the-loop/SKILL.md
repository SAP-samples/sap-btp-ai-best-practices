---
name: langgraph-human-in-the-loop
description: "INVOKE THIS SKILL when implementing human-in-the-loop patterns, pausing for approval, breakpoints, or state editing in LangGraph. Covers interrupt(), Command(resume=...), interrupt_before breakpoints, update_state, approval/validation workflows, and idempotency. Python-only, SAP Gen AI Hub."
---

<overview>
LangGraph's human-in-the-loop patterns let you pause graph execution, surface data to users, and resume with their input:

- **`interrupt(value)`** — pauses execution, surfaces a value to the caller
- **`Command(resume=value)`** — resumes execution, providing the value back to `interrupt()`
- **`interrupt_before`** — compile-time breakpoints that pause before specific nodes
- **`update_state`** — modify graph state between runs
- **Checkpointer** — required to save state while paused
- **Thread ID** — required to identify which paused execution to resume

Co-invoke `langgraph-genai-hub-setup` for LLM initialization details.
</overview>

---

## Requirements

Three things are required for interrupts to work:

1. **Checkpointer** — compile with `checkpointer=InMemorySaver()` (dev) or `PostgresSaver` (prod)
2. **Thread ID** — pass `{"configurable": {"thread_id": "..."}}` to every `invoke`/`stream` call
3. **JSON-serializable payload** — the value passed to `interrupt()` must be JSON-serializable

---

## Basic Interrupt + Resume

`interrupt(value)` pauses the graph. The value surfaces in the result under `__interrupt__`. `Command(resume=value)` resumes — the resume value becomes the return value of `interrupt()`.

**Critical**: when the graph resumes, the node restarts from the **beginning** — all code before `interrupt()` re-runs.

<ex-basic-interrupt-resume>
<python>
Pause execution for human review and resume with Command.
```python
from langgraph.types import interrupt, Command
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict

class State(TypedDict):
    approved: bool

def approval_node(state: State):
    # Pause and ask for approval
    approved = interrupt("Do you approve this action?")
    # When resumed, Command(resume=...) returns that value here
    return {"approved": approved}

checkpointer = InMemorySaver()
graph = (
    StateGraph(State)
    .add_node("approval", approval_node)
    .add_edge(START, "approval")
    .add_edge("approval", END)
    .compile(checkpointer=checkpointer)
)

config = {"configurable": {"thread_id": "thread-1"}}

# Initial run — hits interrupt and pauses
result = graph.invoke({"approved": False}, config)
print(result["__interrupt__"])
# [Interrupt(value='Do you approve this action?')]

# Resume with the human's response
result = graph.invoke(Command(resume=True), config)
print(result["approved"])  # True
```
</python>
</ex-basic-interrupt-resume>

---

## Approval Workflow

A common pattern: interrupt to show a draft, then route based on the human's decision.

<ex-approval-workflow>
<python>
Interrupt for human review, then route to send or end based on the decision.
```python
from langgraph.types import interrupt, Command
from langgraph.graph import StateGraph, START, END
from typing import Literal
from typing_extensions import TypedDict

class EmailAgentState(TypedDict):
    email_content: str
    draft_response: str
    classification: dict

def human_review(state: EmailAgentState) -> Command[Literal["send_reply", "__end__"]]:
    """Pause for human review using interrupt and route based on decision."""
    # interrupt() must come first — any code before it will re-run on resume
    human_decision = interrupt({
        "draft_response": state.get("draft_response", ""),
        "urgency": state.get("classification", {}).get("urgency"),
        "action": "Please review and approve/edit this response"
    })

    if human_decision.get("approved"):
        return Command(
            update={"draft_response": human_decision.get("edited_response", state.get("draft_response", ""))},
            goto="send_reply"
        )
    else:
        return Command(update={}, goto=END)
```
</python>
</ex-approval-workflow>

---

## Validation Loop

Use `interrupt()` in a loop to validate human input and re-prompt if invalid.

<ex-validation-loop>
<python>
Validate human input in a loop, re-prompting until valid.
```python
from langgraph.types import interrupt

def get_age_node(state):
    prompt = "What is your age?"

    while True:
        answer = interrupt(prompt)

        # Validate the input
        if isinstance(answer, int) and answer > 0:
            break
        else:
            prompt = f"'{answer}' is not a valid age. Please enter a positive number."

    return {"age": answer}
```

Each `Command(resume=...)` call provides the next answer. If invalid, the loop re-interrupts with a clearer message.
```python
config = {"configurable": {"thread_id": "form-1"}}
first = graph.invoke({"age": None}, config)
# __interrupt__: "What is your age?"

retry = graph.invoke(Command(resume="thirty"), config)
# __interrupt__: "'thirty' is not a valid age..."

final = graph.invoke(Command(resume=30), config)
print(final["age"])  # 30
```
</python>
</ex-validation-loop>

---

## Breakpoints (interrupt_before)

Breakpoints are compile-time declarations that pause execution before specific nodes. Unlike `interrupt()`, they do not require code changes inside nodes.

<breakpoint-vs-interrupt>

| Feature | `interrupt_before` | `interrupt()` |
|---------|-------------------|---------------|
| Where defined | At `compile()` time | Inside node function |
| When it triggers | Always before the named node | Conditionally, based on code logic |
| Data surfaced | Current graph state | Custom value passed to `interrupt()` |
| Best for | Always-pause gates (e.g., before tool execution) | Dynamic, data-dependent pauses |

</breakpoint-vs-interrupt>

<ex-breakpoint>
<python>
Pause before tool execution and resume after human approval.
```python
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import MessagesState, StateGraph, START
from langgraph.prebuilt import ToolNode, tools_condition
from gen_ai_hub.proxy.langchain.openai import ChatOpenAI

llm = ChatOpenAI(proxy_model_name="gpt-4.1", temperature=0.2)
tools = [calculator, kv_set, kv_get]
llm_with_tools = llm.bind_tools(tools)

def assistant(state: MessagesState):
    system_msg = SystemMessage(content="You are a helpful assistant.")
    return {"messages": [llm_with_tools.invoke([system_msg] + state["messages"])]}

builder = StateGraph(MessagesState)
builder.add_node("assistant", assistant)
builder.add_node("tools", ToolNode(tools))
builder.add_edge(START, "assistant")
builder.add_conditional_edges("assistant", tools_condition)
builder.add_edge("tools", "assistant")

# Compile with breakpoint BEFORE tools node
graph = builder.compile(
    checkpointer=MemorySaver(),
    interrupt_before=["tools"]
)

config = {"configurable": {"thread_id": "approval-1"}}

# Run until breakpoint
for chunk in graph.stream({"messages": [HumanMessage(content="Calculate 15 * 23")]}, config):
    pass

# Inspect what tool is about to execute
state = graph.get_state(config)
print(f"Stopped before: {state.next}")
last_msg = state.values["messages"][-1]
if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
    print(f"Tool: {last_msg.tool_calls[0]['name']}")
    print(f"Args: {last_msg.tool_calls[0]['args']}")

# Resume after human approval
for chunk in graph.stream(None, config):
    pass
```
</python>
</ex-breakpoint>

---

## State Editing

Modify graph state between runs using `update_state`. Useful for correcting agent decisions before resuming.

<ex-state-editing>
<python>
Inspect state, modify it, and resume execution with corrected values.
```python
config = {"configurable": {"thread_id": "edit-1"}}

# Run the graph
result = graph.invoke(initial_state, config)

# Inspect current state
state = graph.get_state(config)
print(f"Current values: {state.values}")
print(f"Next node: {state.next}")

# Modify state before resuming
graph.update_state(config, {"radius": 5.0})  # Correct a value

# Resume with updated state
result = graph.invoke(None, config)
```

Note: `update_state` passes through reducers. Use `Overwrite()` to replace instead of merge — see the persistence skill.
</python>
</ex-state-editing>

---

## Multiple Interrupts

When parallel branches each call `interrupt()`, resume all of them in a single invocation by mapping each interrupt ID to its resume value.

<ex-multiple-interrupts>
<python>
Resume multiple parallel interrupts by mapping interrupt IDs to values.
```python
from typing import Annotated, TypedDict
import operator
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import START, END, StateGraph
from langgraph.types import Command, interrupt

class State(TypedDict):
    vals: Annotated[list[str], operator.add]

def node_a(state):
    answer = interrupt("question_a")
    return {"vals": [f"a:{answer}"]}

def node_b(state):
    answer = interrupt("question_b")
    return {"vals": [f"b:{answer}"]}

graph = (
    StateGraph(State)
    .add_node("a", node_a)
    .add_node("b", node_b)
    .add_edge(START, "a")
    .add_edge(START, "b")
    .add_edge("a", END)
    .add_edge("b", END)
    .compile(checkpointer=InMemorySaver())
)

config = {"configurable": {"thread_id": "1"}}

# Both parallel nodes hit interrupt() and pause
result = graph.invoke({"vals": []}, config)

# Resume all pending interrupts at once using a map of id -> value
resume_map = {
    i.id: f"answer for {i.value}"
    for i in result["__interrupt__"]
}
result = graph.invoke(Command(resume=resume_map), config)
# result["vals"] = ["a:answer for question_a", "b:answer for question_b"]
```
</python>
</ex-multiple-interrupts>

User-fixable errors use `interrupt()` to pause and collect missing data — that's the pattern covered by this skill. For the full 4-tier error handling strategy (RetryPolicy, Command error loops, etc.), see the **fundamentals** skill.

---

## Side Effects Before Interrupt Must Be Idempotent

When the graph resumes, the node restarts from the **beginning** — ALL code before `interrupt()` re-runs. In subgraphs, BOTH the parent node and the subgraph node re-execute.

<idempotency-rules>

**Do:**
- Use **upsert** (not insert) operations before `interrupt()`
- Use **check-before-create** patterns
- Place side effects **after** `interrupt()` when possible
- Separate side effects into their own nodes

**Don't:**
- Create new records before `interrupt()` — duplicates on each resume
- Append to lists before `interrupt()` — duplicate entries on each resume

</idempotency-rules>

<ex-idempotent-patterns>
<python>
Idempotent operations before interrupt vs non-idempotent (wrong).
```python
# GOOD: Upsert is idempotent — safe before interrupt
def node_a(state: State):
    db.upsert_user(user_id=state["user_id"], status="pending_approval")
    approved = interrupt("Approve this change?")
    return {"approved": approved}

# GOOD: Side effect AFTER interrupt — only runs once
def node_a(state: State):
    approved = interrupt("Approve this change?")
    if approved:
        db.create_audit_log(user_id=state["user_id"], action="approved")
    return {"approved": approved}

# BAD: Insert creates duplicates on each resume!
def node_a(state: State):
    audit_id = db.create_audit_log({  # Runs again on resume!
        "user_id": state["user_id"],
        "action": "pending_approval",
    })
    approved = interrupt("Approve this change?")
    return {"approved": approved}
```
</python>
</ex-idempotent-patterns>

<subgraph-interrupt-re-execution>

### Subgraph re-execution on resume

When a subgraph contains an `interrupt()`, resuming re-executes BOTH the parent node (that invoked the subgraph) AND the subgraph node (that called `interrupt()`):

<python>
```python
def node_in_parent_graph(state: State):
    some_code()  # <-- Re-executes on resume
    subgraph_result = subgraph.invoke(some_input)
    # ...

def node_in_subgraph(state: State):
    some_other_code()  # <-- Also re-executes on resume
    result = interrupt("What's your name?")
    # ...
```
</python>
</subgraph-interrupt-re-execution>

---

## Command(resume) Warning

`Command(resume=...)` is the **only** Command pattern intended as input to `invoke()`/`stream()`. Do NOT pass `Command(update=...)` as input — it resumes from the latest checkpoint and the graph appears stuck. See the fundamentals skill for the full antipattern explanation.

---

## Fixes

<fix-checkpointer-required-for-interrupts>
<python>
Checkpointer required for interrupt functionality.
```python
# WRONG
graph = builder.compile()

# CORRECT
graph = builder.compile(checkpointer=InMemorySaver())
```
</python>
</fix-checkpointer-required-for-interrupts>

<fix-resume-with-command>
<python>
Use Command to resume from an interrupt (regular dict restarts graph).
```python
# WRONG
graph.invoke({"resume_data": "approve"}, config)

# CORRECT
graph.invoke(Command(resume="approve"), config)
```
</python>
</fix-resume-with-command>

<boundaries>
### What You Should NOT Do

- Use interrupts without a checkpointer — will fail
- Resume without the same thread_id — creates a new thread instead of resuming
- Pass `Command(update=...)` as invoke input — graph appears stuck (use plain dict)
- Perform non-idempotent side effects before `interrupt()` — creates duplicates on resume
- Assume code before `interrupt()` only runs once — it re-runs every resume
- Use `interrupt_before` without a checkpointer — breakpoints require state persistence
</boundaries>
