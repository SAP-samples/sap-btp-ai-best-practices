# Agent memory, concurrency, and approvals (sap-ai4u-skills 0.4.0)

## Purpose

Release 0.4.0 makes SAP HANA the single production store for LangGraph agent
state across the `sap-ai4u-skills` plugin, and upgrades the
`create-langgraph-react-agent` template so that its HANA memory:

- keeps the **full** conversation, including tool calls and tool results,
  instead of only user and final-assistant text;
- stays bounded through tool-output caps, LLM **summarization**, and a hard
  message ceiling;
- is **safe with several Cloud Foundry instances**: two turns on the same
  conversation never run at once, and a stale turn can never overwrite a
  newer one;
- supports **human approval** of guarded tools across user turns, stored as a
  pending action in the same HANA row.

The design borrows the useful parts of the Joule Studio agent template
(full-state memory, summarization with a cheaper model, tool-output caps) while
replacing its storage: that template's `create_checkpointer` returns an
in-memory saver, which is neither durable nor shared between instances.

## Why the previous behaviour was unsafe

The runtime serialized turns with an `asyncio.Lock` per context. That lock lives
in one Python process. With two CF instances, two simultaneous turns on one
`contextId` both loaded the same row, both ran, and the second blind `MERGE`
silently replaced the first turn.

## Inputs and outputs

Configuration (`config/agent.yaml` of a generated agent):

| Key | Default | Meaning |
|-----|---------|---------|
| `memory.enabled` | `false` | Persist conversations in HANA |
| `memory.table_name` | `LANGGRAPH_AGENT_MEMORY` | Table created/migrated in `CURRENT_SCHEMA` |
| `memory.max_messages` | `200` | Hard ceiling; trimming restarts on a user message |
| `memory.lease_seconds` | `300` | Per-conversation lease lifetime; must exceed the longest turn |
| `memory.lock_wait_seconds` | `30` | Wait for a busy conversation before `ContextBusyError` |
| `summarization.enabled` | `true` | Summarize older history |
| `summarization.trigger_tokens` | `200000` | Approximate history size that triggers a summary |
| `summarization.keep_messages` | `6` | Recent messages kept verbatim |
| `summarization.model` | agent model | Optional cheaper deployment for summaries |
| `tool_output.max_output_characters` | `30000` | Cap per tool result |
| `approvals.tools` | `[]` | Tools that need the user's approval first (requires memory) |

Outputs of `AgentRuntime.ainvoke(text, context_id, ...)`:

- `AgentResult.status == "completed"`: normal answer in `output_text`.
- `AgentResult.status == "input_required"`: `output_text` is an approval request
  listing each held tool call and its arguments. The next message on the same
  context approves (`yes`, `ok`, `approve`, `confirm`, `proceed`, `go ahead`) or
  rejects (anything else, passed to the model as feedback).
- `ContextBusyError`: another instance held the conversation longer than
  `lock_wait_seconds`. Over A2A this becomes a failed task with
  "Another request for this conversation is still running. Retry shortly."
- `ConcurrentUpdateError`: the lease expired mid-turn and another turn saved
  first; nothing was overwritten.

## How it works

HANA table (one row per conversation):

```sql
CREATE COLUMN TABLE "LANGGRAPH_AGENT_MEMORY" (
  "CONTEXT_ID" NVARCHAR(256) NOT NULL PRIMARY KEY,
  "MESSAGES" NCLOB NOT NULL,                 -- {"format": 2, "messages": [...]}
  "PENDING_ACTION" NCLOB,                    -- held tool calls or NULL
  "VERSION" BIGINT DEFAULT 0 NOT NULL,
  "LEASE_OWNER" NVARCHAR(64),
  "LEASE_EXPIRES_AT" TIMESTAMP,
  "UPDATED_AT" TIMESTAMP DEFAULT CURRENT_UTCTIMESTAMP NOT NULL
)
```

Older tables (first three columns only) are migrated with `ALTER TABLE ... ADD`
at startup, and their role/content rows remain readable.

Turn lifecycle (`template_agent/runtime.py`):

1. In-process `asyncio.Lock` for the context (cheap queueing on one instance).
2. Acquire the HANA lease: conditional `UPDATE` on a free or expired lease, then
   read `LEASE_OWNER` back; a new context is inserted already leased. Poll with
   backoff while busy.
3. Load messages, pending action, and version.
4. Resolve a pending action, if any, into one `ToolMessage` per held call.
5. Run the graph:
   `START -> load_skill_node -> compact_context -> agent -> tools -> compact_context -> agent -> END`.
   `compact_context` truncates oversized tool results and calls
   `SummarizationMiddleware.abefore_model`. A call to an approval-gated tool
   ends the graph before the tool runs.
6. Save and release in one statement fenced by `VERSION` and `LEASE_OWNER`.
7. On any error or cancellation, release the lease.

Key snippet, the fenced save in `template_agent/memory.py`:

```python
cursor.execute(
    f"""
    UPDATE {self.quoted_table}
    SET "MESSAGES" = ?, "PENDING_ACTION" = ?, "VERSION" = "VERSION" + 1,
        "LEASE_OWNER" = NULL, "LEASE_EXPIRES_AT" = NULL,
        "UPDATED_AT" = CURRENT_UTCTIMESTAMP
    WHERE "CONTEXT_ID" = ? AND "VERSION" = ? AND "LEASE_OWNER" = ?
    """,
    (_encode_messages(messages), pending, context_id, expected_version, owner),
)
if cursor.rowcount != 1:
    raise ConcurrentUpdateError(...)
```

## Skill guidance changes

- `langgraph-persistence`: HANA for conversation memory; `InMemorySaver` only
  in-process; `SqliteSaver`, `PostgresSaver`, and `sap_cloud_sdk`
  `create_checkpointer` listed as not used; new concurrency and history-size
  sections.
- `langgraph-human-in-the-loop`: production approvals as a HANA pending action;
  `interrupt()` scoped to in-process use.
- `langgraph-reflexion`: lessons in a HANA table instead of `SqliteSaver`.
- `sap-hana-sql`: generic "Row Leases and Versioned Saves" pattern.
- `langgraph-fundamentals`, `sap-agent-joule-cf-bootstrap`, `sap-btp-ai`:
  pointers updated.

## Known limits and follow-ups

- The A2A server still uses `InMemoryTaskStore`. Approval requests are therefore
  returned as completed tasks (the pending action is keyed by `contextId`), not
  A2A `input-required` tasks, whose `taskId` would not be found on another
  instance. A shared task store would allow `input-required`.
- Context IDs are not yet namespaced by caller identity. Because tool results are
  now stored, add authentication and `user:context` namespacing before exposing
  the A2A endpoint.
- No lease heartbeat: `memory.lease_seconds` must exceed the longest turn. The
  version check still prevents a stale save if it does not.

## Related files

- `plugins/sap-ai4u-skills/skills/create-langgraph-react-agent/assets/template/template_agent/memory.py`
- `plugins/sap-ai4u-skills/skills/create-langgraph-react-agent/assets/template/template_agent/runtime.py`
- `plugins/sap-ai4u-skills/skills/create-langgraph-react-agent/assets/template/template_agent/config.py`
- `plugins/sap-ai4u-skills/skills/create-langgraph-react-agent/assets/template/template_agent/a2a_server.py`
- `plugins/sap-ai4u-skills/skills/create-langgraph-react-agent/assets/template/docs/mcp-and-hana.md`
- `plugins/sap-ai4u-skills/skills/create-langgraph-react-agent/assets/template/scripts/live_hana_smoke.py`
- `plugins/sap-ai4u-skills/skills/langgraph-persistence/SKILL.md`
- `plugins/sap-ai4u-skills/skills/langgraph-human-in-the-loop/SKILL.md`
- `plugins/sap-ai4u-skills/skills/langgraph-reflexion/SKILL.md`

## How to test

From the template directory, in a Python 3.11/3.12 virtual environment with
`requirements-dev.txt` installed:

```bash
.venv/bin/python -m pytest -q
.venv/bin/ruff check template_agent scripts tests
```

Against a real HANA Cloud instance (credentials in `.env`):

```bash
.venv/bin/python -m scripts.live_hana_smoke
```

The live script uses two store objects as two instances on one temporary row
and checks lease exclusivity, versioned save, stale-save rejection, and lease
expiry, then deletes the row.
