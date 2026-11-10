# AI4U Ideation Agent

## What This Is

Enterprise AI use-case ideation tool. Users chat with an LLM agent that interviews them to define AI project requirements, then exports structured JSON.

## Architecture at a Glance

- **Backend**: FastAPI (Python 3.11+) at `api/`
- **Frontend**: Vanilla JS + UI5 Web Components + Vite at `ui/`
- **Deployment**: SAP BTP Cloud Foundry. Config in `manifest.yaml`.
- **Auth**: XSUAA (primary) + optional SAP CDC OTP (external users). See `api/app/auth.py`.

## Access Control

- **Admin**: Identified via `ADMIN_EMAILS` env-var allowlist, or CDC `data.role` field when CDC is configured.
- **CSE (Company Search Engine)**: Optional — only active when `CSE_URL` is set. Gated for CDC users (external users don't see it).
  - `/company-suggestions` returns empty for CDC users.
  - `/enrichment/{context_id}` returns `"disabled"` CSE status for CDC users.
- All other features (ideation interview, export, Perplexity research, LinkedIn analysis) are available to all authenticated users.

## Development Rules

1. **Test locally before committing** — iterate until stable.
2. **Backward compatibility** — never break existing APIs or behaviors.
3. **Documentation language** — English only (code, comments, commits, docs).

## Running Locally

```bash
# Backend
cd api && source .venv/bin/activate && uvicorn app.main:app --reload --port 8000

# Frontend
cd ui && npm run dev
```

## Environment Pattern

- `api/.env` — local dev values
- `api/.env.production` — production values (git-ignored)
- `api/.env.example` — template with all available variables
- `manifest.yaml` — Cloud Foundry deployment env vars
- Secrets never committed to git.

## Creativity System (Provocateur + Reimagine)

The ideation agent uses a "Process Architect" approach that adds architectural challenge and problem reframing to the interview. The system behaves like an AI Process Architect in a workshop — it doesn't assume the user is right about their problem framing. This is the default behavior for all users.

### Architecture

| Component | File | When it fires |
|-----------|------|---------------|
| Mini-reimagine | `api/ideation_agent/provocateur.py` → `run_mini_reimagine()` | After Q3 — lightweight direction check ("on_track" or "pivot") |
| Provocateur | `api/ideation_agent/provocateur.py` → `intercept_question()` | Q4+ — pass/replace/inject questions to steer toward better architecture |
| Full reimagine | `api/ideation_agent/provocateur.py` → `run_reimagine()` | After Q5 — deep strategic reframe (2-3 sentence spark) |

### Flow

1. **Q1-Q3**: Primary agent solo (Process Architect persona)
2. **After Q3**: Mini-reimagine fires → direction signal to provocateur
3. **Q4-Q5**: Provocateur intercepts primary agent's questions (pass/replace/inject)
4. **After Q5**: Full reimagine fires → visible "thinking card" to user → context for Q6+
5. **Q6-Q9**: Provocateur active with reimagine context → converge to final
6. **Final delivery**: Reimagine context integrated into core solution (3-case: embrace/acknowledge/reject)

### Eval

Run `python -m ideation_agent.tests.test_provocateur_4way -v` for the 4-way comparison (old baseline / new baseline / reimagine-only / provocateur+reimagine).

## Key Files

| Purpose | File |
|---------|------|
| Backend auth (XSUAA + optional CDC) | `api/app/auth.py` |
| Admin access control | `api/app/admin.py` |
| CDC JWT validation | `api/app/services/cdc.py` |
| HANA user profiles (CDC fallback) | `api/app/services/hana_users.py` |
| Main app + CORS + token proxy | `api/app/main.py` |
| Ideation agent router + A2A | `api/app/routers/ideation_agent.py` |
| Provocateur + Reimagine module | `api/ideation_agent/provocateur.py` |
| Agent system prompt (Process Architect) | `api/ideation_agent/system_prompt.py` |
| Agent state | `api/ideation_agent/state.py` |
| LLM usage observability | `api/app/observability/llm_usage_logging.py` |
| Version check service | `api/app/services/version_check.py` |
| Frontend auth service | `ui/src/services/auth.js` |
| Frontend CDC service | `ui/src/services/cdc.js` |
| Login page | `ui/src/pages/login/login.js` |
| Deployment config | `manifest.yaml` |
| Environment template | `api/.env.example` |

## Feature Flags

Optional features are controlled by environment variables:

| Feature | Gate | Function |
|---------|------|----------|
| CSE (Company Search) | `CSE_URL` + `CSE_KEY` presence | `cse_enabled()` in `tools/company_search.py` |
| CDC Auth | `CDC_API_KEY` presence | `cdc_enabled()` in `app/auth.py` |
| Perplexity Research | `IDEATION_PERPLEXITY_SKILL_ENABLED=true` | `perplexity_enabled()` in `tools/perplexity_research.py` |
| LinkedIn Research | `IDEATION_LINKEDIN_SKILL_ENABLED=true` | `linkedin_enabled()` in `tools/linkedin_research.py` |
| Update Check | `UPDATE_CHECK_ENABLED=true` (default) | `check_for_update()` in `app/services/version_check.py` |

Successful version checks are cached for one hour. Set `UPDATE_CHECK_ENABLED=false` to disable upstream version checks.
