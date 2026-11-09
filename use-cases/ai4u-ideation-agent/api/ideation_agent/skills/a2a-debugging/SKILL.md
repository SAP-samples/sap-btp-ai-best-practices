---
name: a2a-debugging
description: "Use when a user asks to debug tool behavior, inspect past tool calls, or explain why a result may be inconsistent across turns."
---

# A2A Debugging Skill

Use this workflow when debugging request/response behavior.

## Steps

1. Inspect `metadata.toolCalls` and `metadata.executedTools` from A2A task responses when available.
2. If needed, inspect `data/tool_usage_log.jsonl` entries for the relevant context/task IDs.
3. Compare tool inputs against user request wording.
4. If memory behavior is involved, read `/memories/user_profile.md` and relevant files.
5. Explain what happened in plain terms and propose a correction.

## Guardrails

- Do not claim a tool call happened unless it appears in the log.
- If logs are missing, state that clearly.
