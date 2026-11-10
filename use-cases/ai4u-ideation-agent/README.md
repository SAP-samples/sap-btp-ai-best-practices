# AI4U Ideation Agent

AI-powered tool that guides users through a structured interview to design SAP BTP AI use cases. Produces machine-actionable JSON briefs.

> **Ready to deploy?** Jump to [Deployment](#deployment) or see the full [Deployment Guide](docs/DEPLOYMENT.md).

## Features

- **Guided Interview** — LangGraph agent asks adaptive clarification questions with suggested answers, then delivers a structured recommendation
- **Process Architect** — Provocateur + Reimagine pipeline challenges assumptions and reframes problems for better solutions
- **Background Research** — Optional CSE company profiling, Perplexity web research, and LinkedIn hiring analysis run in parallel
- **Structured Output** — Final recommendation includes title, problem, solution spec, app type, and AI capabilities (JSON)
- **Conversation Persistence** — Multi-turn sessions stored in SAP HANA Cloud; resume anytime
- **A2A Protocol** — Agent-to-Agent JSON-RPC interface for programmatic access
- **File Upload** — Multimodal analysis of uploaded images, PDFs, and documents
- **Authentication** — XSUAA (primary) + optional SAP CDC OTP (external users via email code)
- **Update Notifications** — Persistent banner when a newer version is available upstream (disable via `UPDATE_CHECK_ENABLED=false`)

## Prerequisites

- SAP BTP account with Cloud Foundry environment
- SAP AI Core instance with Gen AI Hub (LLM access)
- SAP HANA Cloud instance
- Object Store (S3-compatible) service instance
- XSUAA service instance
- Python 3.11+, Node.js 22.12+ within Node 22

## Architecture

```
┌─────────────────┐     ┌─────────────────────────────────────┐
│  ui/ (Vite)     │────▶│  api/ (FastAPI)                     │
│  UI5 Web Comps  │     │  ├─ ideation_agent/ (LangGraph)     │
│  Vanilla JS     │     │  ├─ app/auth.py (XSUAA + CDC)       │
└─────────────────┘     │  ├─ app/services/ (CDC, ObjectStore) │
                        │  └─ app/routers/ideation_agent.py   │
                        └─────────────────────────────────────┘
```

**Deployment:** 2 Cloud Foundry apps on SAP BTP
- `ai4u-ideation-agent-api` — Python backend
- `ai4u-ideation-agent-ui` — Production frontend

## Quick Start

Clone the public repository and enter the agent directory:

```bash
git clone https://github.com/SAP-samples/sap-btp-ai-best-practices.git
cd sap-btp-ai-best-practices/use-cases/ai4u-ideation-agent
```

Run the backend in the first terminal, starting from the agent directory:

```bash
cd api
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # Fill in your credentials
uvicorn app.main:app --reload --port 8000
```

Open a second terminal in the agent directory and run the frontend:

```bash
cd ui
npm install
npm run dev
```

API docs at http://127.0.0.1:8000/docs

## Configuration

See [docs/CONFIGURATION.md](docs/CONFIGURATION.md) for the full reference of all environment variables.

See [`api/.env.example`](api/.env.example) for a ready-to-copy template.

**Required services:**
- XSUAA — authentication (always required)
- SAP AI Core — LLM calls via Gen AI Hub SDK (`sap-ai-sdk-gen[all]` on PyPI)
- HANA Cloud — conversation persistence
- Object Store — file uploads

**Optional features:**
- CDC — external user OTP login (disabled when `CDC_API_KEY` not set)
- CSE — company search enrichment (disabled when `CSE_URL` not set)
- Perplexity — AI research via AI Core Orchestration (disabled when `IDEATION_PERPLEXITY_SKILL_ENABLED` not set)
- LinkedIn — hiring intelligence from public job listings (disabled when `IDEATION_LINKEDIN_SKILL_ENABLED` not set)

**Admin access:** Set `ADMIN_EMAILS` env var (comma-separated list of email addresses).

## Deployment

> **Manual deployment:** Follow the [Manual Setup guide](docs/DEPLOYMENT.md#manual-setup-reference) to configure and deploy the agent in your own SAP BTP environment. The quick-install wizard (`setup.sh`) is **BETA** and is still being finalized.

See [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) for the full guide.

**Prerequisites** (BTP Cockpit → Entitlements & Instances):
- Entitlements: AI Core `extended`, HANA Cloud `hana`, Object Store `s3-standard`, XSUAA `application`, CF Runtime ≥ 3 GB
- AI Core instance with **GPT-5.4** + **Gemini 2.5 Flash** deployed — [tutorial](https://developers.sap.com/tutorials/ai-core-generative-ai)
- HANA Cloud instance running
- [CF CLI](https://docs.cloudfoundry.org/cf-cli/install-go-cli.html) installed

**Quick deployment (BETA):**

Run from `sap-btp-ai-best-practices/use-cases/ai4u-ideation-agent`:

```bash
cf login -a https://api.cf.<region>.hana.ondemand.com --sso
./setup.sh
```

The BETA setup wizard creates services, extracts credentials, generates configs, and deploys.

**Subsequent updates:** `git pull && ./deploy.sh`

## Environment

| File | Purpose |
|------|---------|
| `api/.env.example` | Template with all variables documented |
| `api/.env` | Local development values |
| `api/.env.production` | Production credentials (git-ignored) |
| `manifest.yaml.template` | Template for customer CF routes, services, and non-secret UI env vars |
| `manifest.yaml` | Tracked CF starter configuration for the shared development environment; use the template to configure your own routes and services |
| `xs-security.json.template` | Template for XSUAA configuration |
| `xs-security.json` | Tracked XSUAA starter configuration for the shared development environment; use the template to configure your own routes |

## HANA Cloud

Conversations are persisted in SAP HANA Cloud. Tables auto-create on first request.

Setup: create a dedicated user in HANA Cloud Central SQL Console, then set `HANA_*` vars in `api/.env` for local development or `api/.env.production` for CF deployment.

## Key Files

| Purpose | Path |
|---------|------|
| Production agent (LangGraph) | `api/ideation_agent/` |
| Agent router + A2A server | `api/app/routers/ideation_agent.py` |
| Auth (XSUAA + optional CDC) | `api/app/auth.py` |
| Admin access control | `api/app/admin.py` |
| CDC service (JWT + writes) | `api/app/services/cdc.py` |
| Main app + CORS | `api/app/main.py` |
| LLM usage observability | `api/app/observability/llm_usage_logging.py` |
| Frontend auth | `ui/src/services/auth.js` |
| Login page | `ui/src/pages/login/login.js` |
| Deploy script | `deploy.sh` |
| CF manifest template | `manifest.yaml.template` |
| Environment template | `api/.env.example` |

## License

Apache 2.0 — see [LICENSE](LICENSE).

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup, code style, and submission guidelines.

## Documentation

- [Architecture](docs/ARCHITECTURE.md) — system diagram, LangGraph flow, A2A protocol
- [Deployment](docs/DEPLOYMENT.md) — step-by-step BTP Cloud Foundry deployment
- [Configuration](docs/CONFIGURATION.md) — all environment variables and feature flags
