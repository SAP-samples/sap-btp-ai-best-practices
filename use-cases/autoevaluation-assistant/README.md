# Evaluation Assessment Assistant

This repository implements an AI-assisted assessment review assistant. It helps users fill a form/assessment by ingesting per-question evidence documents and returning **AI-suggested answer selections with rationale and evidence-aware reasoning**.

The application has two core capabilities:

- An assessment workflow in UI + API + background worker that reviews user-uploaded documents and proposes verified answers for each question.
- A Joule-ready knowledge agent (A2A) that answers questions about assessment glossary terms, question explanations, and dimensions.

## Current architecture

- Frontend: UI5 Web Components app in `ui/`
- Backend API: FastAPI app in `api/`
- Worker: Async background worker in `api/app/workers/ai_review_worker.py`
- Database: SAP HANA for all runtime tabular and binary persistence (frameworks, questions, jobs, attachments, chunks, retrieval rounds, results)
- AI runtime: SAP AI Core via `sap-ai-sdk-gen`
- Agent runtime: A2A endpoint mounted at `/a2a/` from `api/app/a2a_app.py`

## Main runtime flow

### 1) Framework and knowledge preparation (one-time per tenant)

The app does not use local DB files for runtime state.

- Assessment framework (dimensions, questions, answer items, translations) is loaded into HANA via:
  - `api/scripts/import_assessment_framework.py`
  - source files: `data/sanitized/assessment_framework.xlsx`, `data/sanitized/assessment_question_explanations.csv`, optional `data/sanitized/IT/*`
- Joule knowledge resources (glossary + question/dimension explanations) are loaded into HANA via:
  - `api/scripts/import_joule_knowledge_resources.py`
  - sources are workbooks in `data/`

### 2) User workflow in the UI

1. UI fetches framework:
   - `GET /api/assessment/dimensions`
   - `GET /api/assessment/dimensions/{dimension}/questions`
2. User selects answers and uploads files per question.
3. “Save as draft” creates a review job via `POST /api/ai-review/jobs`.
   - attachments must be named `<question_id>__<filename>` when accepted by backend validation.
   - supported file types include PDF/DOCX/XLSX/XLSM/EML/PNG/JPEG.
4. API stores job/task/attachment metadata in HANA and returns a `job_id`.
5. Frontend polls `GET /api/ai-review/jobs/{job_id}` until tasks finish and renders
   verified answers + level-wise reasoning.

### 3) Worker processing and answer generation

`ai_review_worker.py` leases pending question tasks and writes back final results.

For each question task:

- Save uploaded files locally in temp directories.
- Extract text from PDF/DOCX/XLSX/XLSM/EML and extract raw image bytes for PNG/JPEG.
- Estimate input size and choose review route:
  - **Direct path** when prompt+evidence estimate is within model limits.
  - **Agentic RAG path** for oversized evidence packs.

#### Direct path

The task goes straight into SAP AI Hub structured review (`gpt-5.4` by default via `GenAiReviewClient`) with:
- question + answer items + constraints
- extracted text blocks (PDF/DOCX/XLSX/XLSM/EML)
- image evidence blocks (PNG/JPEG)
- raw low-density PDF when text extraction is sparse

#### Agentic RAG path for large documents

When token estimates exceed direct limit, the worker does retrieval-driven review:

1. Chunk extracted evidence (`EvidenceChunk`) with token-aware chunking.
2. Generate embeddings with `text-embedding-3-large` and persist in HANA.
3. Retrieve top candidates for generated queries.
4. Run a bounded **retrieve -> assess -> refine** loop (max 3 rounds by default):
   - model classifies accepted vs rejected chunks
   - reports evidence gaps and refined queries
5. If sufficient evidence is achieved, the task is reviewed with only accepted chunks
   and persisted as the final result.
6. If not sufficient, worker returns explicit insufficiency state and warnings.

The loop keeps all metadata in HANA so long docs are still auditable and restartable.

### 4) Joule knowledge agent runtime (LangGraph surrogate + A2A)

The Joule path is a separate runtime surface and runs inside the same API app:

1. A Joule function call resolves to a system alias (`ASSESSMENT_KNOWLEDGE_AGENT_API`) and sends the user query to `/a2a/` on the API.
2. FastAPI mounts the A2A server at `/a2a/` in `api/app/a2a_app.py`.
3. A2A request handling invokes `AssessmentKnowledgeAgentExecutor`, which delegates each turn to `JouleKnowledgeGraphAgent.answer`.
4. The agent loads prior turns by context ID, calls a compiled LangGraph graph, appends assistant/user turns to HANA, and returns the final text artifact to A2A.
5. The LangGraph graph is a tool-calling surrogate:
   - It uses a chat model (`JouleKnowledgeRuntimeTranslator` + `ChatOpenAI` via SAP Gen AI Hub).
   - It exposes three tools:
     - question explanation lookup
     - dimension explanation lookup
     - glossary search
   - The tool layer is backed by `JouleKnowledgeService`, which reads question/term/dimension data from HANA and returns only grounded, translated text in the requested language.

The final artifact from the graph is returned to Joule as the function result; no background polling is used for this path.

## API endpoints used by the flow

- `GET /api/assessment/dimensions`
- `GET /api/assessment/dimensions/{dimension}/questions`
- `POST /api/ai-review/jobs`
- `GET /api/ai-review/jobs/{job_id}`
- `DELETE /api/ai-review/jobs?assessment_id=...&dimension=...`

All protected APIs require `X-API-Key`.

## Local development

### Prerequisites

- Node.js and npm
- Python 3.12+
- Cloud Foundry CLI (for deployment)

### Configure environment

From repo root:

```bash
cp api/.env.example api/.env
cp ui/.env.example ui/.env
```

- Set `api/.env` values for HANA + SAP AI Core.
- Keep `VITE_API_KEY` in `ui/.env` equal to `API_KEY` in `api/.env`.

### Run local services

Terminal A – API:

```bash
cd api
python -m venv venv
source venv/bin/activate  # macOS/Linux
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Terminal B – worker:

```bash
cd api
source venv/bin/activate
python -m app.workers.ai_review_worker
```

Terminal C – UI:

```bash
cd ui
npm install
npm run dev
```

Open the UI at `http://127.0.0.1:5173/assessment`.

## Deploy to SAP BTP / Cloud Foundry

`deploy.sh` reads `api/.env` and executes `cf push` with required manifest variables.

### Quick deployment

```bash
# 1) Login to CF
cf login -a https://api.cf.eu10-005.hana.ondemand.com -o <org> -s <space>

# 2) Deploy API + worker + UI
./deploy.sh
```

`deploy.sh` uses `manifest.yaml`, which currently defines three CF apps and
binds the backend API and worker to the `Cloud Logging` service after `cf push`:

- `autoevaluation-assistant-api` (FastAPI)
- `autoevaluation-assistant-worker` (background worker, no-route)
- `autoevaluation-assistant-ui` (Vite preview)

The deployed API and UI routes use `cfapps.eu10-005.hana.ondemand.com`.
LLM token usage telemetry is emitted as compact JSON stdout events by the API
and worker, then collected through the Cloud Logging bindings.

### Manual push (if needed)

```bash
cf push \
  --var api_key="your-shared-api-key" \
  --var hana_address="<hana-host>" \
  --var hana_port="443" \
  --var hana_user="<hana-user>" \
  --var hana_password="<hana-password>" \
  --var aicore_auth_url="<aicore-auth-url>" \
  --var aicore_client_id="<aicore-client-id>" \
  --var aicore_client_secret="<aicore-client-secret>" \
  --var aicore_base_url="<aicore-base-url>" \
  --var aicore_resource_group="<aicore-resource-group>"
```

## Deploy the Joule knowledge agent

Joule uses the A2A stack in:

- `da.sapdas.yaml`
- `joule/a2a/scenarios/*`
- `joule/a2a/function*`
- `joule/a2a/capability_context.yaml`

Recommended flow:

1. Configure a destination in BTP named `ASSESSMENT_KNOWLEDGE_AGENT_API` pointing to the deployed API URL:
   - Set header `X-API-Key` to the same API key used for deployment.
   - Keep destination auth in BTP as `NoAuthentication` (header supplies API key).
2. Deploy the package:

```bash
joule login ...
joule deploy -c -n document_assessment_assistant
joule launch document_assessment_assistant
```

Important:

- The remote function currently targets `/a2a/` and the A2A endpoint is mounted at `/a2a/` on the API app.
- If your destination already ends with `/a2a/`, do not double-append another `/a2a/` in the function path.
- Ensure the API is reachable and returns a valid `AGENT_PUBLIC_URL`/`API_BASE_URL` so the A2A card exposes the correct public endpoint.

## Related docs in this repo

- `docs/assessment-ai-document-review.md`
- `docs/assessment-automatic-evidence-routing-rag.md`
- `docs/assessment-joule-knowledge-agent.md`
- `docs/assessment-ai-review-evidence-reasoning.md`
- `api/app/services/ai_review_graph.py`
- `api/app/services/evidence_routing.py`
- `api/app/services/evidence_indexing.py`
- `api/app/services/evidence_retrieval.py`
- `api/app/services/evidence_rag.py`
- `api/app/services/joule_knowledge_agent.py`
