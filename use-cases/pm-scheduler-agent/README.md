# FMI Maintenance Schedule Optimization Agent

> Flask + Angular Fiori · SAP Generative AI Hub (gpt-5.4) · S/4HANA OData data source
> FMI 9-Phase Maintenance Model · Phase 6 – Scheduling

---

## Table of Contents

1. [Business Context](#1-business-context)
2. [Architecture](#2-architecture)
3. [Data Source — S/4HANA OData](#3-data-source--s4hana-odata)
4. [Scheduling Logic](#4-scheduling-logic)
5. [Opportunistic Maintenance](#5-opportunistic-maintenance)
6. [AI Analysis](#6-ai-analysis)
7. [Write-Back to S/4HANA](#7-write-back-to-s4hana)
8. [REST API & Swagger](#8-rest-api--swagger)
9. [Project Structure](#9-project-structure)
10. [Configuration](#10-configuration)
11. [Local Development](#11-local-development)
12. [Deploy to SAP Cloud Foundry](#12-deploy-to-sap-cloud-foundry)

---

## 1. Business Context

Freeport-McMoRan (FMI) uses a **9-Phase Maintenance Model**. This tool supports **Phase 6 — Scheduling**, where the planner turns a pool of "Ready to Schedule" (RTS) maintenance orders into a confirmed weekly plan that fits available workforce capacity.

Key principles:
- Only **P3 (Medium)** and **P4 (Low)** priority orders are considered (P1/P2 are emergencies handled outside this workflow).
- **MN03** (preventive maintenance) orders have a fixed date — the scheduled week must overlap the `BASIC_START_DATE`; the date is never moved.
- **MN01** (corrective / other) orders use a flexible window: any week between `BASIC_START_DATE` and `LATEST_EXECTN_FINISH_DATE` is valid.
- Scheduling is greedy and capacity-constrained per work center.

---

## 2. Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  Angular 19 SPA  (SAP Fiori / @fundamental-ngx / Horizon)   │
│  served as static files from Flask                          │
└──────────────────────────────┬──────────────────────────────┘
                               │ HTTP /api/*
┌──────────────────────────────▼──────────────────────────────┐
│  Flask REST API  (Python 3.11)          Swagger UI: /apidocs │
│  ├── /api/meta            — plants, work centers, equipment  │
│  ├── /api/orders          — raw order-operation detail       │
│  ├── /api/schedule        — generate weekly schedule         │
│  ├── /api/schedule/opportunity — opportunistic orders        │
│  ├── /api/schedule/confirm     — write scheduled dates → S/4 │
│  ├── /api/ai/explain      — LLM schedule explanation         │
│  ├── /api/ai/conflict     — LLM conflict analysis            │
│  ├── /api/ai/query        — free-text LLM query              │
│  ├── /api/schedule/export — CSV download                     │
│  └── /api/schedule/email  — HTML email delivery              │
└───────────────┬───────────────────────────┬─────────────────┘
                │                            │
   ┌────────────▼────────────┐   ┌───────────▼─────────────────┐
   │ SAP Generative AI Hub   │   │ S/4HANA OData v2 services    │
   │ (gpt-5.4 via AI Core)   │   │ (orders + work-center data)  │
   └─────────────────────────┘   └──────────────────────────────┘
```

**Single-app deployment**: Flask serves the pre-built Angular app from `./static/` and falls back to `index.html` for all non-API routes, enabling client-side routing.

**Data is 100% S/4HANA OData** — there is no CSV or direct HANA database access. All order and work-center data is read live from standard OData services (see below).

---

## 3. Data Source — S/4HANA OData

All data is read from standard S/4HANA OData v2 APIs via `backend/s4_client.py` (HTTP basic auth against the S/4 gateway). The former SAP HANA (`hdbcli`) access and the CSV fallback have been **removed** — the app is now OData-only.

| Dataset | OData Service | Entity |
|---------|---------------|--------|
| PM Orders + Operations | `API_MAINTENANCEORDER_SRV` | `MaintenanceOrder` (with `$expand=to_MaintOrderOperation`) |
| Work Center Capacity | `API_WORK_CENTER_SRV` | `WorkCenterCapacity` |

### Field Mapping (OData → internal columns)

`backend/data.py` maps OData fields back to the internal column names the scheduling engine and frontend expect, so no downstream logic changed.

| Internal column | OData field | Notes |
|-----------------|-------------|-------|
| `ORDER_NO` | `MaintenanceOrder` | |
| `OPER_NO` | `MaintenanceOrderOperation` | 4-digit (e.g. `0010`) |
| `OPER_WORK_CENTER` / `WORK_CENTER` | `WorkCenter` | work-center code, used on both orders and capacity for the join |
| `ORDER_TYPE_CODE` | `OrderType` | MN01 / MN03 |
| `PRIORITY` | `MaintPriority` | code → text: `3`→Medium, `4`→Low, … |
| `WO_PHASE` | derived from `SystemStatus` | `REL…`→Released, `CLSD`/`TECO`→Closed |
| `ORDER_SUBPHASE` | `MaintOrdProcessSubPhaseCode` | `E0009`→"Ready to Schedule (Order)" |
| `PLANT_NAME` | `MaintenancePlant` / `WorkCenterPlant` | code → name: `3707`→Miami, `3720`→Sierrita |
| `BASIC_START_DATE` / `BASIC_FINISH_DATE` | `MaintOrdBasicStartDate` / `MaintOrdBasicEndDate` | `/Date(ms)/` → normalised to date |
| `LATEST_EXECTN_FINISH_DATE` | `LatestAcceptableCompletionDate` | |
| `ACTIVITY_WORK_INVOLVE` | `PlannedWork` | |
| `CAPACITY` / `AVAILABLETIME` | `AvailableCapacity` | |
| `MANPOWER` | `NumberOfCapacities` | |

OData v2 dates arrive as `/Date(milliseconds)/` and are parsed and normalised to midnight. Server-driven pagination (`__next`) is followed automatically.

---

## 4. Scheduling Logic

### Candidate Filtering (`backend/scheduling.py`)

An order is a candidate for a given week `[ws, we]` if **all** of:
1. `PRIORITY` ∈ `{Medium, Low}` (P3 / P4)
2. `ORDER_SUBPHASE` contains `"Ready to Schedule (Order)"` (RTS status)
3. Order-type window rule:
   - **MN03**: `ws ≤ BASIC_START_DATE ≤ we` (exact week — date not moved)
   - **MN01**: `BASIC_START_DATE ≤ we` AND `BASIC_FINISH_DATE ≥ ws` (flexible overlap)

### Composite Score (lower = higher priority)

```
score = order_type_rank × 0.4 + criticality_rank × 0.3 + urgency × 0.3
```

| Factor | Values |
|--------|--------|
| `order_type_rank` | MN03 = 1, other = 2 |
| `criticality_rank` | A = 1, B = 2, C = 3 |
| `urgency` | days until `LATEST_EXECTN_FINISH_DATE` (lower = more urgent) |

### Greedy Capacity Fill

For each work center:
1. Candidates are sorted ascending by `score`.
2. Each order is added to the schedule if its `ACTIVITY_WORK_INVOLVE` fits within the remaining capacity for the week.
3. Orders that don't fit are moved to the **Deferred** list.

---

## 5. Opportunistic Maintenance

When equipment is offline (e.g., planned downtime), the planner can pull forward future-week PM orders for that same equipment:

1. Select the offline equipment in the **Equipment Down** section.
2. The API returns all future-week RTS P3/P4 orders for that equipment, grouped by `WEEK_BUCKET` (ISO week string).
3. The planner selects orders to include; they are merged into the current schedule with an `_OPPORTUNISTIC` flag.

This avoids a second equipment shutdown for maintenance that was already planned for a nearby week.

---

## 6. AI Analysis

All AI features use **SAP Generative AI Hub** (`gpt-5.4` via `gen_ai_hub.proxy.langchain`). Three prompts are available:

| Feature | Prompt focus |
|---------|-------------|
| **Explain Schedule** (`/api/ai/explain`) | Why each order was scheduled (or deferred), considering priority, criticality, capacity, and date constraints |
| **Analyze Conflicts** (`/api/ai/conflict`) | Overloaded work centers, orders near latest-finish deadline left deferred |
| **Ask a Question** (`/api/ai/query`) | Free-text query against the current schedule context |

A full scheduling session (explain + conflict + query) consumes roughly **2,300 tokens** total (~$0.015 per session). See `scripts/token_sim.py` for a measurement that runs offline.

---

## 7. Write-Back to S/4HANA

After generating a schedule, the planner can push the confirmed dates back to S/4HANA with the **Confirm & Update S/4** button.

- Endpoint: `POST /api/schedule/confirm`
- For every scheduled operation it pins the operation constraint date `OpEarliestSchedldExecStrtDte` (+ start time) via an OData v2 **MERGE** on `MaintenanceOrderOperation`.
- A CSRF token is fetched once and reused across all writes.
- The response reports per-operation success/failure: `{ total, updated, failed[] }`.

> Scheduled operation dates (`ScheduledBasicStartDate`) are controlled by the S/4 scheduling engine and cannot be set directly via the API. Pinning the **constraint** date is the supported way to force an operation to a specific start. Header basic dates can also be updated via `POST /api/orders/update-basic-start`.

---

## 8. REST API & Swagger

Interactive API docs (Flasgger / OpenAPI) are available at:

```
http://localhost:5001/apidocs/
```

Endpoints are grouped by tag: **Meta**, **Orders**, **Capacity**, **Schedule**, **AI**. Each endpoint documents its request body and responses and can be executed from the UI.

---

## 9. Project Structure

```
fmi/
├── backend/
│   ├── __init__.py
│   ├── app.py          # Flask factory + REST endpoints + Swagger
│   ├── s4_client.py    # S/4HANA OData connection layer (session, paging, CSRF, writes)
│   ├── data.py         # OData → internal-column mapping (cached)
│   ├── scheduling.py   # Pure scheduling engine
│   └── ai.py           # LLM integration (SAP AI Hub)
├── frontend/           # Angular 19 SPA
│   ├── src/
│   │   └── app/
│   │       ├── models/          # TypeScript interfaces
│   │       ├── services/        # HttpClient service
│   │       ├── pages/
│   │       │   └── schedule-page/   # Main page (fd-dynamic-page + tabs)
│   │       └── components/
│   │           ├── schedule-table/  # scheduled operations
│   │           ├── deferred-table/  # deferred operations
│   │           └── data-table/      # raw order-operation detail
│   ├── angular.json
│   └── package.json
├── scripts/
│   └── token_sim.py    # offline LLM token/cost simulation
├── static/             # Angular build output (generated by build.sh)
├── manifest.yml        # CF deployment descriptor (env via ((vars)))
├── Procfile            # gunicorn command
├── .env.example        # template for local .env (no secrets)
├── .cfignore
├── build.sh            # build Angular → cf push
├── dev.sh              # run Flask (:5001) + Angular (:4200) locally
└── requirements.txt
```

---

## 10. Configuration

All credentials are provided via environment variables (locally through a `.env` file, in CF through the manifest `((vars))`). Copy the template and fill in real values:

```bash
cp .env.example .env
# then edit .env
```

| Variable | Purpose |
|----------|---------|
| `AICORE_AUTH_URL`, `AICORE_CLIENT_ID`, `AICORE_CLIENT_SECRET`, `AICORE_BASE_URL`, `AICORE_RESOURCE_GROUP` | SAP Generative AI Hub |
| `GMAIL_USER`, `GMAIL_APP_PASSWORD` | Schedule email delivery |
| `S4_BASE_URL`, `S4_CLIENT`, `S4_USERNAME`, `S4_PASSWORD` | S/4HANA OData connection |
| `S4_VERIFY` | `false` to skip TLS verification (dev), or set `S4_CA_BUNDLE` to a CA path |

> **Never commit `.env`.** It is git-ignored. Only `.env.example` (placeholders) is tracked.

### Test the S/4 connection

```bash
python -m backend.s4_client
```

This runs a standalone connectivity check ($count against the maintenance-order and work-center services) and prints a pass/fail report.

---

## 11. Local Development

The quickest path is the dev script, which sets up the venv, installs deps, and starts both servers:

```bash
./dev.sh
#   Backend :  http://localhost:5001
#   Frontend:  http://localhost:4200
```

### Manual — Backend (Flask)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env   # fill in credentials
flask --app "backend.app:create_app()" run --port 5001
```

### Manual — Frontend (Angular dev server)

```bash
cd frontend
npm install
npm start          # proxies /api to the Flask backend
# open http://localhost:4200
```

---

## 12. Deploy to SAP Cloud Foundry

### Prerequisites

1. `cf` CLI installed and logged in to your BTP space.
2. SAP AI Core service instance bound, or env vars supplied to `manifest.yml`.
3. Angular build output in `./static/` (produced by `build.sh`).

### One-step deploy

```bash
chmod +x build.sh
./build.sh
```

`build.sh` runs `npm run build` (outputs to `./static/`) then `cf push`.

### Environment via a vars file

`manifest.yml` references credentials as `((VAR))` placeholders — supply them with a **git-ignored** vars file:

```bash
cf push --vars-file secrets.vars.yml
```

```yaml
# secrets.vars.yml (NEVER commit)
AICORE_AUTH_URL: https://...
AICORE_CLIENT_ID: ...
AICORE_CLIENT_SECRET: ...
AICORE_BASE_URL: https://...
AICORE_RESOURCE_GROUP: default
GMAIL_USER: ...
GMAIL_APP_PASSWORD: ...
S4_BASE_URL: https://...
S4_CLIENT: "550"
S4_USERNAME: ...
S4_PASSWORD: ...
S4_VERIFY: "false"
```

### Check logs

```bash
cf logs fmi-scheduling --recent
```

The app will be available at the URL shown in `cf apps`.
