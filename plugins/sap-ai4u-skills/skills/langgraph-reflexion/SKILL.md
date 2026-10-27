---
name: langgraph-reflexion
description: "INVOKE THIS SKILL when building iterative self-improvement agents with human feedback and persistent lessons in LangGraph. Covers the reflexion loop (act-tools-feedback-reflect-finalize), custom lesson reducers, and SAP HANA-backed lesson persistence shared across restarts and Cloud Foundry instances. Python-only, SAP Gen AI Hub."
---

<overview>
Reflexion is an iterative improvement pattern where an agent acts, receives human feedback, reflects to create reusable lessons, and retries guided by those lessons. Lessons are long-term knowledge: they persist in a SAP HANA table and are loaded into the next run, enabling domain-specific competence bootstrapping.

Co-invoke `langgraph-genai-hub-setup` for LLM initialization, `sap-hana-sql` for the lessons table, and `langgraph-persistence` for conversation memory. The in-graph `feedback` node below uses `input()` for local runs; in a served agent, end the turn and treat the user's next message as feedback (`langgraph-human-in-the-loop`, pending-action pattern).
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

## Persistent Lessons in SAP HANA

Lessons are not graph checkpoints: they are a small, deduplicated list that
every future run should read. Store them in their own HANA table, created
automatically at startup in the runtime user's schema, and pass them into the
graph as initial state. The graph itself is compiled without a checkpointer.

<ex-reflexion-persistence>
<python>
HANA lesson store: idempotent table creation, deduplicated UPSERT, bounded load.
```python
"""Persist reflexion lessons in SAP HANA, shared by all app instances."""

import hashlib
import os
from contextlib import closing

from hdbcli import dbapi
from langchain_core.messages import HumanMessage

LESSONS_TABLE = "AGENT_REFLEXION_LESSONS"


def _connect():
    """Open one hdbcli connection from the standard HANA_* variables.

    hdbcli connections are not context managers; wrap them in closing().
    """
    return dbapi.connect(
        address=os.environ["HANA_ADDRESS"],
        port=int(os.environ["HANA_PORT"]),
        user=os.environ["HANA_USER"],
        password=os.environ["HANA_PASSWORD"],
        encrypt=True,
    )


class HanaLessonStore:
    """Store short lessons per scope (for example a task type) in HANA."""

    def ensure(self) -> None:
        """Create the lessons table in CURRENT_SCHEMA if it does not exist."""
        with closing(_connect()) as conn:
            cur = conn.cursor()
            cur.execute(
                "SELECT COUNT(*) FROM SYS.TABLES "
                "WHERE SCHEMA_NAME = CURRENT_SCHEMA AND TABLE_NAME = ?",
                (LESSONS_TABLE,),
            )
            if not cur.fetchone()[0]:
                cur.execute(f"""
                    CREATE COLUMN TABLE "{LESSONS_TABLE}" (
                      "SCOPE" NVARCHAR(256) NOT NULL,
                      "LESSON_KEY" NVARCHAR(64) NOT NULL,
                      "LESSON" NVARCHAR(1000) NOT NULL,
                      "CREATED_AT" TIMESTAMP DEFAULT CURRENT_UTCTIMESTAMP NOT NULL,
                      PRIMARY KEY ("SCOPE", "LESSON_KEY")
                    )""")
            conn.commit()

    def load(self, scope: str, limit: int = 50) -> list[str]:
        """Return the newest lessons for a scope, oldest first."""
        with closing(_connect()) as conn:
            cur = conn.cursor()
            cur.execute(
                f'SELECT "LESSON" FROM "{LESSONS_TABLE}" WHERE "SCOPE" = ? '
                f'ORDER BY "CREATED_AT" DESC LIMIT {int(limit)}',
                (scope,),
            )
            return [row[0] for row in reversed(cur.fetchall())]

    def add(self, scope: str, lessons: list[str]) -> None:
        """Insert new lessons; a repeated lesson is a no-op (primary key)."""
        with closing(_connect()) as conn:
            cur = conn.cursor()
            for lesson in {l.strip() for l in lessons if l and l.strip()}:
                key = hashlib.sha256(lesson.lower().encode()).hexdigest()
                cur.execute(
                    f'UPSERT "{LESSONS_TABLE}" ("SCOPE", "LESSON_KEY", "LESSON") '
                    "VALUES (?, ?, ?) WITH PRIMARY KEY",
                    (scope, key, lesson[:1000]),
                )
            conn.commit()


lessons = HanaLessonStore()
lessons.ensure()


def run_reflexion(prompt: str, scope: str = "default"):
    """Run reflexion seeded with stored lessons, then persist what it learned."""
    app = build_reflexion_graph(tools=my_tools, max_attempts=3)  # no checkpointer
    known = lessons.load(scope)
    result = app.invoke({
        "messages": [HumanMessage(content=prompt)],
        "attempt": 0,
        "lessons": known,
        "history": [],
    })
    lessons.add(scope, [l for l in result.get("lessons", []) if l not in known])
    return result

# First run learns lessons; a later run (any instance, after restarts) starts with them.
result1 = run_reflexion("Analyze this data", scope="data-analysis")
result2 = run_reflexion("Analyze different data", scope="data-analysis")
```
</python>
</ex-reflexion-persistence>

Choose `scope` deliberately (task type, tenant, or user). A shared scope makes
one user's feedback shape everyone's runs; keep customer or personal data out
of lesson text.

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
- Persist lessons without a deliberate scope — lessons from different tasks or tenants will mix
- Skip the max_attempts guard — the loop can run indefinitely
- Store verbose paragraphs as lessons — they waste context window and degrade quality
- Keep lessons in a checkpointer, `SqliteSaver`, or local files — store them in a HANA table so every instance and restart sees them
- Block a served agent on `input()` for feedback — end the turn and treat the next user message as feedback
</boundaries>
