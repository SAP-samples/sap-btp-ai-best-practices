# GR/IR Procurement Chat

A conversational AI agent for GR/IR (Goods Receipt / Invoice Receipt) reconciliation, built on FastAPI + SAP UI5 Web Components and deployed to SAP BTP Cloud Foundry.

- **Frontend:** [UI5 Web Components](https://sap.github.io/ui5-webcomponents/) (plain JavaScript + Vite)
- **Backend:** [FastAPI](https://fastapi.tiangolo.com/) (Python) + [LangGraph](https://langchain-ai.github.io/langgraph/) ReAct agent
- **LLM:** GPT-4.1 via SAP GenAI Hub proxy
- **Data:** `api/data/Source.csv` — full GR/IR dataset loaded into pandas at startup

## Project Structure

```
AI-in-a-day/
├── api/
│   ├── app/
│   │   ├── main.py                        # FastAPI app factory, CORS, router mount
│   │   ├── security.py                    # API key header dependency
│   │   ├── models/common.py               # HealthResponse, ErrorResponse
│   │   ├── routers/
│   │   │   ├── grir_chat.py               # POST /api/grir-chat/chat + DELETE /session
│   │   │   ├── grir_agent.py              # LangGraph graph singleton + NDJSON streaming
│   │   │   ├── grir_session_store.py      # In-memory session history store
│   │   │   └── grir_system_prompt.md      # Agent system prompt (loaded at startup)
│   │   ├── utils/langgraph/
│   │   │   ├── grir_tools.py              # 9 pandas tools over Source.csv
│   │   │   └── common.py                  # make_llm() SAP GenAI Hub factory
│   │   └── observability/
│   │       └── llm_usage_logging.py       # Structured JSON stdout logging
│   ├── data/
│   │   ├── Source.csv                     # Full GR/IR dataset
│   │   └── Reasoning.csv                  # 7 issue classification examples
│   ├── .env.example
│   └── requirements.txt
├── ui/
│   ├── index.html                         # App shell — mounts #grir-app
│   └── src/
│       ├── grir-chat-entry.js             # Entry point
│       ├── style.css
│       ├── services/api.js                # fetch + streamNDJSON helpers
│       ├── modules/streaming-renderer.js  # renderMarkdown + StreamingRenderer
│       └── pages/grir-chat/
│           ├── grir-chat.html
│           ├── grir-chat.js
│           └── grir-chat.css
├── tests/
│   └── unit/
│       └── test_llm_usage_logging.py
├── manifest.yaml                          # CF push manifest
└── deploy.sh                              # Automated CF deploy with key rotation
```

## Prerequisites

- [Node.js](https://nodejs.org/) (which includes npm)
- [Python](https://www.python.org/) 3.11+
- [Cloud Foundry CLI](https://github.com/cloudfoundry/cli/releases)
- SAP AI Core credentials with GPT-4.1 deployed via SAP GenAI Hub

## Local Development Setup

### 1. Create Your `.env` Files

```bash
cp api/.env.example api/.env
cp ui/.env.example ui/.env
```

### 2. Configure Your Secrets

- **`api/.env`**: Fill in your `AICORE_*` credentials and set a secure `API_KEY`.
- **`ui/.env`**: Set `VITE_API_KEY` to match the `API_KEY` from `api/.env`.

## Running Locally

### Backend (API)

```bash
cd api
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux

pip install -r requirements.txt
uvicorn app.main:app --reload
```

API runs at [http://127.0.0.1:8000](http://127.0.0.1:8000). Swagger UI at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).

### Frontend (UI)

```bash
cd ui
npm install
npm run dev
```

## Deployment to SAP BTP

### Automated Deployment (Recommended)

1. **Login to Cloud Foundry**

   ```bash
   cf login -a <API_ENDPOINT> --sso
   # Example:
   cf login -a https://api.cf.eu10-004.hana.ondemand.com -o btp-ai-sandbox -s Dev
   ```

2. **Update `manifest.yaml`**

   Replace `template-ui5-web-components-fastapi` with your app name throughout `manifest.yaml`.

3. **Run the Deployment Script**

   ```bash
   chmod +x deploy.sh
   ./deploy.sh
   ```

   The script generates a secure API key, then deploys both the UI and API to Cloud Foundry.

### Manual Deployment

<details>
<summary>Click to view manual deployment instructions</summary>

1. Generate a secure API key:
   ```bash
   openssl rand -hex 32
   ```

2. Login to Cloud Foundry (see above).

3. Deploy:
   ```bash
   cf push --var api_key="your-secure-api-key-goes-here"
   ```

</details>
