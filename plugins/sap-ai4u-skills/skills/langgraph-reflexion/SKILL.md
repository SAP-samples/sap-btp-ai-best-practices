---
name: langgraph-reflexion
description: "INVOKE THIS SKILL when building iterative self-improvement agents with human feedback and persistent lessons in LangGraph. Covers the reflexion loop (act-tools-feedback-reflect-finalize), custom lesson reducers, and SQLite-backed lesson persistence. Python-only, SAP Gen AI Hub."
---

<overview>
Reflexion is an iterative improvement pattern where an agent acts, receives human feedback, reflects to create reusable lessons, and retries guided by those lessons. Lessons persist across runs via a checkpointer, enabling domain-specific competence bootstrapping.

Co-invoke `langgraph-genai-hub-setup` for LLM initialization and `langgraph-persistence` for checkpointer setup.
</overview>

---

## When to Use

<reflexion-design>

| Use Reflexion When | Use Simpler Patterns When |
|-------------------|--------------------------|
| Tasks require iterative quality improvement | One-shot answers suffice |
| Human feedback should guide future agent behavior | No human-in-the-loop needed |
| Domain-specific competence needs bootstrapping | General-purpose tasks |
| Lessons from past runs should persist | Stateless execution is fine |

</reflexion-design>

---

## Reflexion Loop

<reflexion-loop-diagram>

```
START -> act -> (has tool calls?) -> tools -> feedback -> (approved?) -> finalize -> END
                      |                                       |
                      +-> feedback -> reflect -> (retry?) -> act
                                                    |
                                                    +-> finalize -> END
```

Nodes:
1. **act** — LLM with tools, guided by accumulated lessons
2. **tools** — Execute tool calls (ToolNode)
3. **feedback** — Human approves or provides improvement notes
4. **reflect** — Convert feedback into compact, reusable lessons
5. **finalize** — Synthesize final answer from conversation and lessons

</reflexion-loop-diagram>

---

## State Definition

<ex-reflexion-state>
<python>
Reflexion state with message history, lessons (custom reducer), and attempt tracking.
```python
import operator
from typing import Any, Dict, List, TypedDict, Annotated
from langchain_core.messages import BaseMessage


def unique_lessons_reducer(left: List[str] | None, right: List[str] | None) -> List[str]:
    """Merge lessons uniquely, preserving order. Deduplicates by stripped text."""
    left_list = left or []
    right_list = right or []
    seen = set()
    result: List[str] = []
    for lesson in left_list + right_list:
        key = (lesson or "").strip()
        if key and key not in seen:
            seen.add(key)
            result.append(key)
    return result


class ReflexionState(TypedDict, total=False):
    messages: Annotated[List[BaseMessage], operator.add]   # Chat history (accumulates)
    attempt: int                                            # Current iteration
    lessons: Annotated[List[str], unique_lessons_reducer]   # Persisted lessons (deduplicated)
    feedback: str                                           # Human feedback text
    approved: bool                                          # Human approval flag
    history: Annotated[List[Dict[str, Any]], operator.add]  # Audit trail
    final: str                                              # Final synthesized answer
```
</python>
</ex-reflexion-state>

---

## Unique Lessons Reducer

<ex-unique-lessons-reducer>
<python>
Custom reducer that deduplicates lessons while preserving insertion order.
```python
def unique_lessons_reducer(left: List[str] | None, right: List[str] | None) -> List[str]:
    """Merge lessons uniquely, preserving order.

    - Keeps existing lessons from `left` in their original order
    - Appends only new lessons from `right` not already present
    - Compares by stripped text to avoid whitespace duplicates
    """
    left_list = left or []
    right_list = right or []
    seen = set()
    result: List[str] = []
    for lesson in left_list + right_list:
        key = (lesson or "").strip()
        if key and key not in seen:
            seen.add(key)
            result.append(key)
    return result
```
</python>
</ex-unique-lessons-reducer>

---

## Graph Construction

<ex-reflexion-graph>
<python>
Build the reflexion graph with act-tools-feedback-reflect-finalize loop.
```python
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import JsonOutputParser
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from gen_ai_hub.proxy.langchain.openai import ChatOpenAI


def build_reflexion_graph(tools, max_attempts: int = 3, checkpointer=None):
    parser = JsonOutputParser()

    def act(state: ReflexionState) -> dict:
        """LLM acts guided by accumulated lessons."""
        attempt = int(state.get("attempt", 0)) + 1
        lessons = state.get("lessons", []) or []
        lessons_text = "\n".join(f"- {l}" for l in lessons) if lessons else "(none)"

        llm = ChatOpenAI(proxy_model_name="gpt-4.1", temperature=0.2)
        llm_with_tools = llm.bind_tools(tools)

        system = SystemMessage(content=(
            "You are a careful, tool-using assistant.\n"
            f"LESSONS:\n{lessons_text}\n\n"
            "Use lessons to avoid repeating past mistakes."
        ))
        convo = [system] + (state.get("messages", []) or [])
        response = llm_with_tools.invoke(convo)
        return {"messages": [response], "attempt": attempt}

    def feedback(state: ReflexionState) -> dict:
        """Collect human approval and improvement notes."""
        # In production, replace with actual UI interaction
        approved = input("Approve? (y/n): ").strip().lower().startswith("y")
        notes = "" if approved else input("What should improve? ").strip()
        feedback_msg = HumanMessage(content=f"FEEDBACK: approved={approved}; notes={notes}")
        return {
            "approved": approved,
            "feedback": notes,
            "messages": [feedback_msg],
            "history": [{"attempt": state.get("attempt", 1), "approved": approved, "notes": notes}],
        }

    def reflect(state: ReflexionState) -> dict:
        """Convert feedback into compact, reusable lessons."""
        llm = ChatOpenAI(proxy_model_name="gpt-4.1", temperature=0.0)
        system = SystemMessage(content=(
            "Convert feedback into compact lessons (short imperatives).\n"
            "Return JSON: {\"lessons\": string[], \"should_retry\": boolean}"
        ))
        user = HumanMessage(content=f"Feedback: {state.get('feedback', '')}")
        response = llm.invoke([system, user])
        try:
            data = parser.parse(response.content)
            new_lessons = [l for l in data.get("lessons", []) if isinstance(l, str) and l.strip()]
        except Exception:
            new_lessons = ["Verify facts before answering."]
        return {"lessons": new_lessons}

    def finalize(state: ReflexionState) -> dict:
        """Synthesize final answer from conversation and lessons."""
        llm = ChatOpenAI(proxy_model_name="gpt-4.1", temperature=0.2)
        system = SystemMessage(content="Produce the final answer. Be clear and concise.")
        convo = [system] + (state.get("messages", []) or [])
        response = llm.invoke(convo)
        return {"final": response.content.strip(), "messages": [response]}

    # Routing functions
    def route_after_act(state):
        last = (state.get("messages") or [None])[-1]
        return "tools" if last and getattr(last, "tool_calls", None) else "feedback"

    def route_after_feedback(state):
        return "finalize" if state.get("approved") else "reflect"

    def route_after_reflect(state):
        if state.get("approved") or int(state.get("attempt", 0)) >= max_attempts:
            return "finalize"
        return "act"

    # Build graph
    builder = StateGraph(ReflexionState)
    builder.add_node("act", act)
    builder.add_node("tools", ToolNode(tools))
    builder.add_node("feedback", feedback)
    builder.add_node("reflect", reflect)
    builder.add_node("finalize", finalize)

    builder.add_edge(START, "act")
    builder.add_conditional_edges("act", route_after_act, {"tools": "tools", "feedback": "feedback"})
    builder.add_edge("tools", "feedback")
    builder.add_conditional_edges("feedback", route_after_feedback, {"finalize": "finalize", "reflect": "reflect"})
    builder.add_conditional_edges("reflect", route_after_reflect, {"finalize": "finalize", "act": "act"})
    builder.add_edge("finalize", END)

    return builder.compile(checkpointer=checkpointer) if checkpointer else builder.compile()
```
</python>
</ex-reflexion-graph>

---

## Persistent Lessons with SQLite

<ex-reflexion-persistence>
<python>
Use SqliteSaver so lessons persist across process restarts.
```python
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_core.messages import HumanMessage

def run_reflexion(prompt: str, db_path: str = "reflexion.db", thread_id: str = "default"):
    """Run reflexion agent with persistent lessons."""
    with SqliteSaver.from_conn_string(db_path) as checkpointer:
        app = build_reflexion_graph(
            tools=my_tools,
            max_attempts=3,
            checkpointer=checkpointer,
        )
        config = {"configurable": {"thread_id": thread_id}}
        initial = {"messages": [HumanMessage(content=prompt)], "attempt": 0, "lessons": [], "history": []}
        result = app.invoke(initial, config=config)
        return result

# First run: agent learns lessons
result1 = run_reflexion("Analyze this data", thread_id="session-1")

# Second run (same thread): agent starts with previously learned lessons
result2 = run_reflexion("Analyze different data", thread_id="session-1")
```
</python>
</ex-reflexion-persistence>

---

## Fixes

<fix-lesson-format>
<python>
Lessons should be short imperatives, not paragraphs.
```python
# WRONG: Verbose lessons waste context window
lessons = ["When the user asks about data analysis, make sure to verify all facts and provide comprehensive explanations with multiple examples..."]

# CORRECT: Short imperatives that guide behavior
lessons = ["Verify facts before answering.", "Prefer concise structured output.", "Cite sources when uncertain."]
```
</python>
</fix-lesson-format>

<fix-max-attempts>
<python>
Always cap the reflexion loop to prevent infinite iterations.
```python
# WRONG: No cap — can loop forever
def route_after_reflect(state):
    return "act"  # Always retry!

# CORRECT: Cap at max_attempts
def route_after_reflect(state):
    if int(state.get("attempt", 0)) >= max_attempts:
        return "finalize"
    return "act"
```
</python>
</fix-max-attempts>

<fix-reflect-json-parsing>
<python>
Always wrap lesson extraction in try/except with fallback.
```python
# WRONG: Crashes if LLM returns malformed JSON
data = parser.parse(response.content)
new_lessons = data["lessons"]

# CORRECT: Fallback to generic lessons on parse failure
try:
    data = parser.parse(response.content)
    new_lessons = [l for l in data.get("lessons", []) if isinstance(l, str) and l.strip()]
except Exception:
    new_lessons = ["Verify facts before answering.", "Prefer concise structured output."]
```
</python>
</fix-reflect-json-parsing>

<boundaries>
### What You Should NOT Do

- Use reflexion for simple one-shot tasks — it adds unnecessary complexity
- Persist lessons without a thread_id — lessons from different contexts will mix
- Skip the max_attempts guard — the loop can run indefinitely
- Store verbose paragraphs as lessons — they waste context window and degrade quality
- Expect reflexion to work without a checkpointer — lessons need persistence to be useful across runs
</boundaries>
