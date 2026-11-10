# Architecture

## System Overview

```
┌─────────────────────┐         ┌───────────────────────────────────────────┐
│  ui/ (Vite)         │  HTTPS  │  api/ (FastAPI)                           │
│                     │────────▶│                                           │
│  Vanilla JS         │         │  ┌─ app/main.py (CORS, auth, health)     │
│  UI5 Web Components │◀────────│  ├─ app/auth.py (XSUAA + CDC JWT)        │
│  Login (XSUAA/CDC)  │   JSON  │  ├─ app/routers/ideation_agent.py        │
└─────────────────────┘         │  │     ├─ REST API (conversations)       │
                                │  │     ├─ A2A JSON-RPC server             │
                                │  │     └─ Background enrichment polling   │
                                │  │                                        │
                                │  ├─ ideation_agent/ (LangGraph)           │
                                │  │     ├─ agent.py (graph + nodes)        │
                                │  │     ├─ system_prompt.py                │
                                │  │     ├─ provocateur.py                  │
                                │  │     ├─ tools/ (CSE, Perplexity, LI)    │
                                │  │     └─ *_background.py (async tasks)   │
                                │  │                                        │
                                │  └─ app/services/ (HANA, CDC, ObjectStore)│
                                └───────────────────────────────────────────┘
                                         │           │           │
                                         ▼           ▼           ▼
                                   ┌──────────┐ ┌────────┐ ┌──────────┐
                                   │ SAP AI   │ │  HANA  │ │ Object   │
                                   │ Core     │ │ Cloud  │ │ Store    │
                                   │(Gen AI   │ │        │ │ (S3)     │
                                   │  Hub)    │ │        │ │          │
                                   └──────────┘ └────────┘ └──────────┘
```

## LangGraph Agent Flow

The ideation agent is built as a LangGraph `StateGraph`:

```
START
  │
  ▼
┌─────────┐
│  skill  │  ← Scan available skills, start background enrichment
└────┬────┘
     │
     ▼
┌───────────┐     ┌───────────────┐
│ assistant │────▶│ tools (if any)│
│  (LLM)   │◀────│               │
└─────┬─────┘     └───────────────┘
      │
      ▼
   [output]
```

**Nodes:**
- **skill** — Fires once on first turn. Starts CSE, Perplexity, and LinkedIn background tasks (when enabled). Loads skill catalog summaries.
- **assistant** — Main LLM node. Generates clarification questions (turns 1-9) or the final recommendation. Polls and integrates background enrichment results.
- **tools** — Optional tool execution (e.g., MCP-connected tools).

## Conversation Lifecycle

1. **User sends first message** → Creates conversation in HANA, runs `skill` node, then `assistant`
2. **Turns 1-3** → Primary agent interviews solo (Process Architect persona)
3. **After Q3** → Mini-reimagine fires (lightweight direction check)
4. **Turns 4-5** → Provocateur intercepts agent questions (pass/replace/inject)
5. **After Q5** → Full reimagine fires (strategic reframe, visible "thinking card")
6. **Turns 6-9** → Provocateur with reimagine context, converges to final
7. **Final delivery** → Structured JSON recommendation with title, problem, solution, capabilities

## Provocateur + Reimagine Pipeline

```
                     Q1-Q3                    Q4-Q5                   Q6-Q9
              ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
              │  Primary Agent  │     │  Primary Agent  │     │  Primary Agent  │
              │  (solo)         │     │  + Provocateur  │     │  + Provocateur  │
              └────────┬────────┘     │  (intercepts)   │     │  + Reimagine    │
                       │              └────────┬────────┘     └────────┬────────┘
                       │                       │                       │
                After Q3:                 After Q5:               Convergence
              mini_reimagine()         run_reimagine()           to final output
              → "on_track" / "pivot"   → strategic reframe
```

The provocateur operates at three levels:
- **pass** — Agent's question is good enough, let it through
- **replace** — Substitute a better question that challenges assumptions
- **inject** — Add an extra question before the agent's (expands thinking)

## Background Enrichment

Three optional research tasks run in parallel with the interview:

| Task | Trigger | Gate | Data Source |
|------|---------|------|-------------|
| CSE | Turn 1 (company name) | `CSE_URL` + `CSE_KEY` | SAP Company Search Engine |
| Perplexity | Turn 1 (company) + Turn 3 (problem) | `IDEATION_PERPLEXITY_SKILL_ENABLED` | Sonar Pro via AI Core Orchestration |
| LinkedIn | Turn 1 (company name) | `IDEATION_LINKEDIN_SKILL_ENABLED` | LinkedIn public guest APIs |

Results are stored in HANA and injected into the agent's context as they complete.

## A2A Protocol

The agent exposes an [Agent-to-Agent (A2A)](https://google.github.io/A2A/) JSON-RPC interface at `/api/ideation-agent/a2a`:

- **Agent card**: `GET /api/ideation-agent/.well-known/agent-card.json`
- **JSON-RPC methods**: `tasks/send` (send message), `tasks/get` (poll status)

This allows programmatic access from other agents (e.g., Joule Works).

## Authentication Flow

```
┌────────┐         ┌──────┐         ┌───────┐
│  User  │         │  UI  │         │  API  │
└───┬────┘         └──┬───┘         └───┬───┘
    │  Click Login    │                  │
    │────────────────▶│                  │
    │                 │  PKCE redirect   │
    │                 │─────────────────▶│  XSUAA
    │◀────────────────│  auth code       │
    │  Consent        │                  │
    │────────────────▶│                  │
    │                 │  POST /token     │
    │                 │─────────────────▶│
    │                 │◀─────────────────│  access_token
    │                 │  Bearer token    │
    │                 │─────────────────▶│  verify JWT
    │                 │◀─────────────────│  UserIdentity
```

When CDC is enabled, an alternative OTP flow is available for external users (email → verify code → CDC JWT → API validates).

## Data Persistence

| Data | Storage | Table |
|------|---------|-------|
| Conversations | HANA Cloud | `IDEATION_CONVERSATIONS` |
| Messages | HANA Cloud | `IDEATION_MESSAGES` |
| Submissions | HANA Cloud | `IDEATION_SUBMISSIONS` |
| Canvas items | HANA Cloud | `IDEATION_CANVAS` |
| Shared links | HANA Cloud | `IDEATION_SHARES` |
| User profiles | HANA Cloud / CDC | `IDEATION_USERS` (HANA) |
| Analytics events | HANA Cloud | `IDEATION_EVENTS` |
| File uploads | Object Store (S3) | — |
| Background research | HANA Cloud | `IDEATION_*_RESULTS` |

## Observability

LLM usage is logged as structured JSON to stdout:

```json
{
  "type": "btp.llm_usage.v1",
  "model": "gpt-5.4",
  "provider": "sap-ai-core",
  "input_tokens": 1234,
  "output_tokens": 567,
  "latency_ms": 2100,
  "outcome": "ok",
  "step_name": "ideation_agent.assistant",
  "feature": "interview"
}
```

On SAP BTP Cloud Foundry, stdout is automatically collected by Loggregator and available in SAP Cloud Logging service — no additional configuration required.
