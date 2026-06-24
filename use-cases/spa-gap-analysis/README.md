# SPA Gap Analysis Application

## Current POC State (June 2026)

This README is updated for the latest POC cleanup and production smoke baseline. Older MVP sections below are kept only where still useful for setup context.

- Production UI: `https://spa-gap-analysis.cfapps.eu10-005.hana.ondemand.com`
- Production API: `https://spa-gap-analysis-api.cfapps.eu10-005.hana.ondemand.com`
- Latest ETL baseline: exit code `0`, with only the expected capped-potential warning.
- Latest backend test baseline: `108 passed, 5 skipped, 9 warnings`.
- Latest frontend build: passed.
- Production smoke passed for Home, Summary View, Quick Lookup, Agent Chat, and onboarding.

Data is not provided in this repository!

---

## 🎯 Application Features

### Quick Lookup (Tab 1)
- Fast SPA gap analysis for a single customer
- Find similar customers based on SOff, PLType, COGS, RFM
- Identify missing SPAs with confidence scores
- Material coverage analysis

### Agent Chat (Tab 2)
- Conversational AI interface
- Natural language queries
- Quick action buttons
- Conversation memory

---

## 📁 Project Structure

```
spa-gap-analysis/
├── api/                      # FastAPI backend
│   ├── app/
│   │   ├── data/            # Data files
│   │   │   ├── raw/         # Excel files (place here)
│   │   │   └── processed/   # Parquet files (generated)
│   │   ├── etl/             # ETL pipeline
│   │   ├── services/        # Business logic (7 modules)
│   │   ├── routers/         # API endpoints
│   │   ├── models/          # Pydantic models
│   │   └── main.py
│   ├── requirements.txt
│   └── .env.example
├── ui/                       # UI5 Web Components frontend
│   ├── src/
│   │   ├── pages/
│   │   │   ├── spa-quick-lookup/    # Tab 1
│   │   │   └── spa-agent-chat/      # Tab 2
│   │   ├── services/
│   │   └── main.js
│   ├── package.json
│   └── .env.example
├── claude/analysis/          # Documentation
│   ├── Phase_1_Complete.md  # ETL Report
│   ├── Phase_2_Complete.md  # Services Report
│   └── Phase_3_Complete.md  # API Report
└── manifest.yaml            # Cloud Foundry deployment
```

---


### Prerequisites

- [Python 3.8+](https://www.python.org/) (Anaconda recommended)
- [Node.js](https://nodejs.org/) (v16+)
- Excel data files (provided separately)




### 3️⃣ Configure Environment

**Backend (`api/.env`):**
```bash
cp api/.env.example api/.env
```

Edit `api/.env`:
```env
# API Key for authentication
API_KEY=your-secret-key-here

# Optional: AI Core credentials (for future LangGraph integration)
AICORE_AUTH_URL=your-auth-url
AICORE_CLIENT_ID=your-client-id
AICORE_CLIENT_SECRET=your-secret
AICORE_BASE_URL=your-base-url
AICORE_RESOURCE_GROUP=default
```

**Frontend (`ui/.env`):**
```bash
cp ui/.env.example ui/.env
```

Edit `ui/.env`:
```env
# Must match API_KEY from api/.env
VITE_API_KEY=your-secret-key-here

# API URL (use default for local dev)
VITE_API_URL=http://127.0.0.1:8000
```

### 4️⃣ Run Backend API

**Terminal 1:**
```bash
cd api

# Install dependencies (first time only)
pip install -r requirements.txt

# Run FastAPI server
python -m app.main
# or
uvicorn app.main:app --reload
```

**Server starts on:** `http://127.0.0.1:8000`

**Swagger UI:** `http://127.0.0.1:8000/docs`

**Test health endpoint:**
```bash
curl http://127.0.0.1:8000/api/spa/health
```

### 5️⃣ Run Frontend UI

**Terminal 2:**
```bash
cd ui

# Install dependencies (first time only)
npm install

# Run Vite dev server
npm run dev
```

**UI starts on:** `http://localhost:5173`

---

## 🧪 Testing

Current automated baseline:

```powershell
cd api
$env:PYTHONPATH='app'
.\.venv\Scripts\python.exe -m pytest tests -q -p no:tmpdir
```

Expected result: `108 passed, 5 skipped, 9 warnings`.

Frontend:

```bash
cd ui
npm run build
```

Expected result: Vite build passed.

## LLM Usage Logging

The backend emits centralized LLM usage events as structured JSON to stdout for SAP Cloud Logging. Events use `event_type="llm_usage"` and schema `btp.llm_usage.v1`, so they are compatible with the `logs-cfsyslog-*` data view and the LLM Usage Dashboard.

### Event Coverage

- Agent Chat model calls (`/api/spa/agent-chat`)
- Onboarding Sonar/Sonar-Pro calls
- Customer strategic insight calls
- Grounding completion utility calls
- Test event endpoint: `GET /api/test/log-llm-usage`

### User Counting

The logger hashes the first available stable user identity from:

1. `x-client-user-id`, `x-user-id`, `x-forwarded-user`, `x-authenticated-user`
2. Already-authenticated Bearer JWT claims: `user_name`, `email`, `user_uuid`, `sub`
3. `None` if no user identity is available

Set `LOG_USER_HASH_SALT` in Cloud Foundry to make user hashes stable and non-reversible across restarts. Raw user ids, emails, JWTs, API keys, cookies, prompts, and completions are never logged.

### Validate In Cloud Foundry

```bash
cf set-env spa-gap-analysis-api LOG_USER_HASH_SALT "<stable-secret-salt>"
cf restart spa-gap-analysis-api
curl -H "X-API-Key: <api-key>" -H "x-client-user-id: smoke-user" \
  https://spa-gap-analysis-api.cfapps.eu10-005.hana.ondemand.com/api/test/log-llm-usage
cf logs spa-gap-analysis-api --recent | grep llm_usage
```

In SAP Cloud Logging / OpenSearch Dashboards:

- Data view: `logs-cfsyslog-*`
- Search: `llm_usage`
- Confirm fields such as `event_type`, `schema_version`, `app_name`, `route`, `provider`, `model`, `input_tokens`, `output_tokens`, `total_tokens`, `outcome`, `latency_ms`, `user_hash`, and `actor_type`.
- Count unique users with Unique Count / Cardinality on `user_hash`.

If SSO/JWT identity is present, unique users are based on hashed SSO/JWT identity. If SSO is not present, unique users are based on `x-client-user-id` or equivalent client headers and represent unique clients/browsers, not guaranteed individual people.

### Test API Endpoints

**1. Health Check:**
```bash
curl http://127.0.0.1:8000/api/health
```

**2. Quick Lookup (requires API key):**
```bash
curl -X POST http://127.0.0.1:8000/api/spa/quick-lookup \
  -H "X-API-Key: your-secret-key-here" \
  -H "Content-Type: application/json" \
  -d '{
    "customer_id": "CUSTOMER_ID",
    "top_n_similar": 10
  }'
```

**3. Agent Chat:**
```bash
curl -X POST http://127.0.0.1:8000/api/spa/agent-chat \
  -H "X-API-Key: your-secret-key-here" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Analyze customer CUSTOMER_ID"
  }'
```

### Test Business Logic (Python)

```bash
cd api/app/services
python test_services.py
```

**Expected:** 5/6 tests pass

### Test UI

1. Open `http://localhost:5173`
2. Navigate to "SPA Quick Lookup"
3. Enter a valid customer ID from your local source data
4. Click "Analyze Customer"
5. View results (profile, missing SPAs, similar customers)

---

## 📊 API Endpoints

### SPA Analysis

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/spa/quick-lookup` | Fast SPA gap analysis |
| POST | `/api/spa/agent-chat` | Conversational AI |
| GET | `/api/health` | Health check |
| GET | `/api/spa/customer/{id}` | Get customer profile |
| DELETE | `/api/spa/agent-chat/{conv_id}` | Clear conversation |

**Authentication:** All endpoints require `X-API-Key` header

**Full API docs:** `http://127.0.0.1:8000/docs`

---

---

## 🏗️ Architecture

### Data Flow

```
Excel Files (25 MB)
    ↓
ETL Pipeline (Phase 1)
    ↓
Parquet Files (3.8 MB)
    ↓
Data Loader (LRU Cache)
    ↓
Business Logic Services (Phase 2)
    ├── Similarity Engine
    ├── Gap Detector
    ├── Material Matcher
    ├── RFM Analyzer
    └── Confidence Scorer
    ↓
API Endpoints (Phase 3)
    ├── Quick Lookup API
    └── Agent Chat API
    ↓
UI5 Pages (Phase 4)
    ├── Quick Lookup (Tab 1)
    └── Agent Chat (Tab 2)
```

### Key Algorithms

**Similarity Engine:**
1. Mandatory: SOff (Sales Office) exact match
2. Mandatory: PLType (Price List Type) exact match
3. Rank by: COGS similarity (log scale, 0-100 pts)
4. Boost by: RFM segment (Champions +30, Need Attention +5)
5. Boost by: Same Price Group (+10 pts)
6. **Total score:** 0-140 points

**Confidence Scoring:**
- SPA Type: Blanket (30) vs Targeted (20)
- Similar Count: % of similar customers with SPA (0-25)
- Material Coverage: % COGS covered (0-25)
- RFM Quality: Avg segment weight (0-20)
- **Total:** 0-100 points (High: 80+, Medium: 60-79, Low: <60)

**RFM Segmentation:**
- **Champions:** 1,403 (22.7%)
- **Loyal:** 1,015 (16.4%)
- **Promising:** 697 (11.3%)
- **At Risk:** 405 (6.5%)
- **Need Attention:** 2,671 (43.1%)

---

## 📖 Documentation

### Phase Reports

- **Phase 1:** [ETL Pipeline Report](claude/analysis/Phase_1_Complete.md)
- **Phase 2:** [Business Logic Report](claude/analysis/Phase_2_Complete.md)
- **Phase 3:** [API Endpoints Report](claude/analysis/Phase_3_Complete.md)

### Analysis Documents

- [Business Requirements (ENG)](claude/analysis/business_requirements_eng.md)
- [UX Design Specification](claude/analysis/UX_Design_Specification_eng.md)
- [Data Mapping Issues](claude/analysis/Data_Mapping_Issues.md)
- [Dataset Update Analysis](claude/analysis/Dataset_Update_Analysis_3.13.md)

---

## 🐛 Troubleshooting


### API Issues

**Problem:** "API key not configured"
- **Solution:** Set `API_KEY` in `api/.env`

**Problem:** "Module not found: app.services"
- **Solution:** Run from `api/` directory: `python -m app.main`

**Problem:** "Customer not found"
- **Solution:** Check customer_id exists in customer_master.parquet

### UI Issues

**Problem:** "CORS error"
- **Solution:** Ensure backend is running on port 8000

**Problem:** "401 Unauthorized"
- **Solution:** Check `VITE_API_KEY` matches backend `API_KEY`

**Problem:** "No results displayed"
- **Solution:** Check browser console (F12) for errors

---

## 🚀 Deployment to Cloud Foundry

### Automated Deployment

1. **Login to CF:**
```bash
cf login -a https://url --sso
```

2. **Run deployment script:**
```bash
chmod +x deploy.sh
./deploy.sh
```

### Manual Deployment

1. **Generate API key:**
```bash
openssl rand -hex 32
```

2. **Deploy:**
```bash
cf push --var api_key="your-generated-key"
```


---

## 📊 Performance Metrics

- **ETL Processing:** ~2 minutes (25 MB → 3.8 MB)
- **API Response Time:** <200ms	
  - Data Loading: <1ms (cached)
  - Similarity Search: ~50ms
  - Gap Detection: ~100ms
- **Data Compression:** 84% (25 MB → 3.8 MB)
- **RFM Calculation:** 6,191 customers in <1s

---

## 🔐 Security

- **API Key Authentication:** Required for all /spa/* endpoints
- **CORS:** Configured for localhost (dev) + production origin
- **Environment Variables:** Secrets stored in .env (not committed)
- **Data Privacy:** All processing local, no external API calls (except AI Core optional)

---

---

## 👥 Support

**Contact:** AI COE Team

---

**Status:** 🟢 Production Ready (Phases 1-3 complete, Phase 4 in progress)

**Last Updated:** June 19, 2026
