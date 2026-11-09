# Ideation Agent

Conversational SAP BTP AI ideation agent exposed over A2A JSON-RPC.

This agent is configured for a focused interview flow:
- starts by asking for `industry` and `challenge`
- asks up to 3 clarification questions total
- uses local skills as context (via `load_skill`)
- returns a final recommendation in a fixed format

## Behavior Contract

The agent behavior is defined in [system_prompt.py](system_prompt.py):

1. Collect initial problem context (`industry` + `challenge`).
2. Ask at most 3 clarification questions.
3. Use local skills to ground capability selection.
4. Return:
   - `Scenario Title`
   - `AI Capabilities` (max 5)
   - `Strategy` (one paragraph)

## Quick Start

1. Create environment and install dependencies:
```bash
cd api/ideation_agent
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Ensure SAP AI Core credentials are available (project `.env` is loaded automatically).

3. Start server from repo root:
```bash
uvicorn ideation_agent.a2a_server:app --reload --port 8000
```

4. Send first user message:
```bash
curl -s http://localhost:8000/a2a \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "id": 1,
    "method": "message/send",
    "params": {
      "message": {
        "role": "user",
        "parts": [{ "kind": "text", "text": "Help me ideate an SAP BTP AI scenario." }],
        "messageId": "msg-1"
      },
      "configuration": { "historyLength": 6 }
    }
  }' | jq
```

## Skills

Skills are discovered dynamically from `skills/*/SKILL.md` at runtime.

Key ideation skills in this agent:
- `ideation-interview`
- `sap-btp-ai-capabilities`
- `access-to-generative-ai-models`
- `document-grounding`
- `vector-rag-embedding`
- `vector-rag-query`
- `knowledge-graph`
- `narrow-ai-anomaly-detection`
- `narrow-ai-classification`
- `narrow-ai-clustering`
- `narrow-ai-linear-regression`
- `narrow-ai-time-series-forecasting`

When full instructions are needed, the agent calls:
- `load_skill("<skill-id>")`

## Tooling Policy

Design intent for this agent is conversational ideation with skill-loading only.

- Expected tool usage: `load_skill`
- Not required for target behavior: calculators, external HTTP tools, MCP tools

If you want hard enforcement (not just prompt guidance), restrict registered tools in:
- [tools/__init__.py](tools/__init__.py)

## Multi-Turn Conversations

Conversation memory is persisted via LangGraph checkpointing (SQLite).

- `contextId` maps to LangGraph `thread_id`
- Reusing the same `contextId` continues the same interview
- Different `contextId` values isolate conversations

## Main Files

- [a2a_server.py](a2a_server.py): A2A JSON-RPC server
- [agent.py](agent.py): LangGraph workflow
- [system_prompt.py](system_prompt.py): ideation behavior and output format contract
- [skill_catalog.py](skill_catalog.py): skill discovery + loading
- [skills/ideation-interview/SKILL.md](skills/ideation-interview/SKILL.md): conversational ideation workflow

## Configuration

Common environment variables:
- `AICORE_PROVIDER` (default: `openai`)
- `AICORE_MODEL` (default: `gpt-4.1`)
- `A2A_CONVERSATIONS_DB` (optional custom SQLite path)
- `MCP_ENABLED` (set `false` to disable MCP tools)
