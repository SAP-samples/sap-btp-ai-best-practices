# Contributing to AI4U Ideation Agent

Thank you for considering contributing to this project! This document covers how to get the development environment running and guidelines for submitting changes.

## Prerequisites

- Python 3.11+
- Node.js 22.12+ within Node 22
- SAP BTP account with:
  - XSUAA service instance
  - SAP AI Core with Gen AI Hub
  - SAP HANA Cloud instance
  - Object Store (S3-compatible) service instance

## Development Setup

Follow the [README Quick Start](README.md#quick-start) to clone the public repository. Start each terminal in `sap-btp-ai-best-practices/use-cases/ai4u-ideation-agent` before running the commands below.

### Backend

```bash
cd api
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Copy and fill in credentials
cp .env.example .env
# Edit .env with your service credentials

# Run the development server
uvicorn app.main:app --reload --port 8000
```

API docs available at http://127.0.0.1:8000/docs

### Frontend

Open a separate terminal in the agent directory:

```bash
cd ui
npm install
npm run dev
```

The frontend uses `http://127.0.0.1:8000` as its default API URL. Set the frontend authentication values described in the [configuration reference](docs/CONFIGURATION.md#application-env-vars) for your XSUAA instance.

## Project Structure

```
api/
├── app/                    # FastAPI application
│   ├── main.py            # App entry, CORS, auth endpoints
│   ├── auth.py            # XSUAA + CDC JWT validation
│   ├── admin.py           # Admin access control
│   ├── routers/           # API route handlers
│   └── services/          # HANA, CDC, Object Store clients
├── ideation_agent/        # LangGraph agent
│   ├── agent.py           # Graph definition + nodes
│   ├── system_prompt.py   # Agent persona + instructions
│   ├── provocateur.py     # Creativity pipeline
│   ├── tools/             # Agent tools (CSE, Perplexity, LinkedIn)
│   └── tests/             # Agent quality tests
└── requirements.txt

ui/
├── src/
│   ├── pages/             # Page components (login, interview, admin)
│   ├── services/          # Auth, CDC, API clients
│   └── components/        # Shared UI components
└── package.json
```

## Code Style

- **Python**: Follow PEP 8. Use type hints. Prefer `async def` for I/O-bound operations.
- **JavaScript**: Vanilla JS with UI5 Web Components. No framework.
- **Comments**: English only. Document "why", not "what".
- **Commits**: Use [conventional commits](https://www.conventionalcommits.org/) (e.g., `feat:`, `fix:`, `docs:`).

## Adding a New Background Skill

The ideation agent supports background research tasks that run in parallel with the interview. To add one:

1. Create a tool module in `api/ideation_agent/tools/your_skill.py`
2. Create a background module in `api/ideation_agent/your_skill_background.py` (follow `perplexity_background.py` pattern)
3. Add a feature flag: `your_skill_enabled()` function gated on an env var
4. Wire it into `agent.py`'s `skill` node
5. Add the status/result polling to the `/enrichment/{context_id}` endpoint
6. Add the env var to `.env.example`

## Running Tests

```bash
cd api
source .venv/bin/activate

# Agent quality tests (requires AI Core credentials)
python -m pytest ideation_agent/tests/ -v

# Specific test suites
python -m pytest ideation_agent/tests/test_quality.py -v
python -m ideation_agent.tests.test_provocateur_4way -v
```

## Deployment

See [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) for step-by-step SAP BTP Cloud Foundry deployment instructions.

## Submitting Changes

1. Fork the repository
2. Create a feature branch from `main`
3. Make your changes with clear commit messages
4. Test locally (backend starts, frontend builds, basic flow works)
5. Submit a pull request with a description of what changed and why

## Backward Compatibility

- Never break existing API contracts
- New features should be gated behind env vars (disabled by default)
- Database schema changes must be additive (new tables/columns, never drop)

## License

By contributing, you agree that your contributions will be licensed under the Apache 2.0 License.
