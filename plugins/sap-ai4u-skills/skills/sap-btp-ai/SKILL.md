---
name: sap-btp-ai
description: Router and shared conventions for ALL SAP BTP AI work - SAP Gen AI Hub / AI Core LLM calls, LangGraph agents, RAG on HANA vector store, hana-ml (PAL) machine learning, RPT-1 tabular prediction, document grounding, knowledge graphs, and token-usage logging. Invoke FIRST when a task touches SAP AI SDK, gen_ai_hub, hana-ml, or SAP BTP AI services, then follow its pointer to the specialized skill.
---

# SAP BTP AI - Skill Router and Shared Conventions

This skill routes to the right specialized skill and defines conventions shared by all of them. Read the routing table, invoke the matching skill(s), and apply the shared rules below in every implementation.

## Routing Table

| Task involves | Use skill |
| --- | --- |
| Direct LLM calls (OpenAI/Anthropic/Gemini) through Gen AI Hub, structured output, multimodal input, orchestration service | `access-to-generative-ai-models` |
| Any LangGraph code (graphs, state, nodes, edges) | `langgraph-fundamentals` (always) + the ones below |
| LLM initialization for LangGraph on SAP BTP | `langgraph-genai-hub-setup` |
| Tool-calling agents, ReAct, MCP tools in LangGraph | `langgraph-agent-patterns` |
| Checkpointers, thread memory, subgraphs, Store | `langgraph-persistence` |
| Pausing for approval, interrupt/resume, breakpoints | `langgraph-human-in-the-loop` |
| Chunking + embedding documents into HANA vector store | `vector-rag-embedding` |
| Semantic retrieval + grounded answering from HANA vectors | `vector-rag-query` |
| SAP Document Grounding Service (pipelines, repositories, collections) | `document-grounding` |
| RDF triples, SPARQL, knowledge-graph RAG on HANA | `knowledge-graph` |
| Any SQL/DDL/DML against SAP HANA, hdbcli, sqlalchemy-hana | `sap-hana-sql` |
| Classification with hana-ml PAL | `narrow-ai-classification` |
| Regression with hana-ml PAL | `narrow-ai-linear-regression` |
| Clustering / segmentation with hana-ml PAL | `narrow-ai-clustering` |
| Time-series forecasting with hana-ml PAL | `narrow-ai-time-series-forecasting` |
| Anomaly / outlier detection with hana-ml PAL | `narrow-ai-anomaly-detection` |
| RPT-1 tabular prediction (`RPTClient`, `[PREDICT]` rows) | `rpt-1` |
| Logging LLM token usage to SAP Cloud Logging | `token-logger` |
| SAP Document AI extraction | `sap-document-ai-client` |
| Cloud Foundry org/space, app, route, service, or deployment preparation | `cf-cli` |
| BTP account, entitlement, service, access, environment, or Destination operations | `btp-cli` |
| Joule Studio CLI validation, assistant inspection, or deployment preparation | `joule-cli` |
| Connect an existing A2A agent to Joule through CF and a BTP Destination | `sap-agent-joule-cf-bootstrap` + the relevant CLI skills |

Combine skills when the task spans areas. Common combinations:

- LangGraph agent on BTP: `langgraph-fundamentals` + `langgraph-genai-hub-setup` + `langgraph-agent-patterns`.
- RAG app: `vector-rag-embedding` + `vector-rag-query` + `sap-hana-sql` (for any custom tables).
- Production API: add `token-logger` to whichever skill produced the LLM calls.
- Anything writing to HANA: always also apply `sap-hana-sql` (HANA is not PostgreSQL; generic SQL fails after deployment).

## Best-Practices Reference Repo

Reference implementations are maintained in the public
[`SAP-samples/sap-btp-ai-best-practices`](https://github.com/SAP-samples/sap-btp-ai-best-practices/tree/main/best-practices)
repository. If a local clone is available, discover its location from the
current workspace instead of assuming a user-specific absolute path.

When a skill's snippet and a notebook in that repository disagree, prefer the
notebook for API signatures and the skill for policy (safety, idempotency,
logging).

## Shared Environment Setup

All Gen AI Hub / AI Core access uses these variables, loaded with `load_dotenv()` before creating any client:

```bash
AICORE_AUTH_URL=""
AICORE_CLIENT_ID=""
AICORE_CLIENT_SECRET=""
AICORE_BASE_URL=""
AICORE_RESOURCE_GROUP=""
```

HANA access adds:

```bash
HANA_ADDRESS=""      # narrow-ai / knowledge-graph notebooks use lowercase: hana_address
HANA_PORT="443"      # HANA Cloud uses 443 with encryption
HANA_USER=""
HANA_PASSWORD=""
```

Match the casing already used by the project's `.env`; do not introduce a second casing convention into an existing codebase.

Typical installs:

```bash
pip install "sap-ai-sdk-gen[all]" python-dotenv        # Gen AI Hub LLM access
pip install hana-ml                                     # PAL / narrow AI
pip install hdbcli sqlalchemy-hana                      # HANA SQL
pip install langgraph langchain langchain-hana          # LangGraph + HANA vector store
```

## Non-Negotiable Rules

1. **Never bypass the Gen AI Hub proxy.** Import LLM clients from `gen_ai_hub.proxy.*`, never from `openai`, `langchain_openai`, `anthropic`, or `google.genai` directly. Direct imports silently skip SAP AI Core routing and fail on credentials.
2. **Smoke-test before building.** Run one minimal text-only LLM call (or `SELECT * FROM DUMMY` for HANA) and confirm it works before adding tools, chains, RAG, or graphs. Most failures are environment failures; find them in 10 lines, not 300.
3. **Model names are deployment-specific.** Example model names in skills (`gpt-4.1`, `gpt-4o`, `text-embedding-3-small`, `anthropic--claude-4.5-sonnet`) must exist as deployments in the target AI Core resource group. Keep model names in variables and confirm availability with the user or a probe call; do not hard-fail an implementation over a model-name guess.
4. **HANA is its own SQL dialect.** Before writing any DDL/DML, apply `sap-hana-sql`. No `IF NOT EXISTS`, no `ON CONFLICT`, no `SERIAL`.
5. **Tabular app data belongs in HANA**, not local CSV/Excel files, unless the file is a user upload flow or the user says otherwise.
6. **Verify, then deliver.** Each skill ends with expected outputs or a validation checklist - actually run it. Never mark work complete from static reading alone. Never deploy to Cloud Foundry yourself; the user deploys manually.
7. **Document features** in `docs/<feature>.md` and give every function/class a docstring, per user conventions.
