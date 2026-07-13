# Section 232 Agent

Trade compliance application for SAP BTP that helps teams determine the metal composition of industrial products, classify them under the U.S. Harmonized Tariff Schedule (HTS), and assess their eligibility under Section 232 steel and aluminum tariffs.

The system combines PDF-grounded document analysis, GCC Tracker material data, and official HTS catalog data into an orchestrated classification pipeline. Users manage a product worklist backed by GCC Tracker data, assign supporting product PDFs (technical drawings, spec sheets, certificates), and run single or batch classifications that produce metal composition breakdowns, HTS code recommendations, and Section 232 assessments.

## What the Application Does

### Products Worklist

The main screen lists read-only products sourced from the GCC Tracker workbook. Users can filter by priority, business segment, product code, document status, and classification status. Selected GCC items can be classified individually or in batch.

### Product Detail and Classification

Each product has a detail view showing its master data, uploaded PDF documents, and the latest classification result. From this screen users can:

- Upload and assign product PDFs (technical drawings, material certificates).
- Run text-only or document-assisted GCC item-id classification to produce metal composition, HTS code, and Section 232 results.
- Review the full reasoning behind each classification step.

Classification supports two document modes:

- `text_only` uses GCC item data and prepared gram columns without requiring PDFs.
- `with_documents` requires uploaded PDF assignments and uses those PDFs as additional HTS and evidence context.
- Product-code multipart prediction and user-created product APIs have been removed from the active runtime.

### Settings and Corpus Management

Administrators use the settings page to:

- Upload Section 232 source PDFs (Federal Register notices, BIS guidance) that feed the Section 232 eligibility reasoning.
- Review, publish, and maintain Section 232 rulesets derived from uploaded source PDFs.
- Upload HTS catalog CSV files from USITC to refresh the official tariff tree used during classification.
- Reset saved classification snapshots when needed.

## Classification Pipeline

When a product is classified, the backend runs a LangGraph workflow through four sequential phases.

### Phase 1 -- Signal Gathering

The active document signal step is:

| Agent | What it does |
| --- | --- |
| **Diagram Analysis** | Converts assigned PDFs to images, runs a full-page vision pass, and can trigger one targeted zoom-refinement round when tiny text blocks are blocking matched identifiers, explicit weight evidence, material standards, or alloy grades. The workflow extracts metal cues, gram-level composition, preliminary HTS hints, and whether the metal-share reading is exact or estimated, then derives the final composition directly from this PDF-backed output. |

### Phase 2 -- HTS Fact Profile

The system synthesizes a structured fact sheet from the GCC item context, PDF-backed composition result, and diagram material cues. This includes an article summary, function summary, material profile, and a set of HTS heading hypotheses that guide the next phase.

### Phase 3 -- Legal Evidence

One HTS retrieval path is used:

| Agent | What it does |
| --- | --- |
| **HANA Tree Search** | Routes through the official HTS catalog stored in HANA using a three-stage LLM classifier: chapter selection, heading selection, and family (8-10 digit code) selection. |

### Phase 4 -- Trade Decision

The final step merges HTS candidates from the HANA tree search and diagram hints. Candidates are ranked by source reliability and confidence, then an LLM selects the best HTS code with full reasoning. The system also assesses Section 232 eligibility against the published Section 232 ruleset, then applies a deterministic last-layer exemption: if the combined steel, aluminum, and copper weight is below 15% of total item weight, the item is marked not subject. Exact PDF proof clears the item confidently, while estimate-based low-metal cases are marked not subject with an explicit human-review recommendation.

The complete response includes the metal composition breakdown, the ranked HTS classification with citations, the Section 232 assessment, all intermediate agent outputs, and a timing breakdown showing phase durations and critical path analysis.

## Data Architecture

The application persists six datasets in SAP HANA Cloud. Three are reference datasets loaded from source files; three are operational datasets written during normal application use.

### Reference Datasets

| HANA Table | Content | Source |
| --- | --- | --- |
| `METAL_COMPOSITION_SERVING` | Denormalized GCC workbook data used for legacy GCC item lookup, GCC-only disambiguation, and source context. The runtime consumes business fields directly and tolerates legacy `prepared__*` columns for backward compatibility. | `data/GCC Tracker.xlsb`, sheet `Material Master` |
| `HTS_2026_CATALOG` | Structured HTS hierarchy: codes, chapters, headings, families, descriptions, duty rates, and search text. Used by the HANA tree search agent during classification. | `data/hts_chapters/chapter*.csv` |
| `HTS_2026_CODE_MAP` | Legacy-to-current HTS code mappings. Used to validate and normalize historical codes during tree search. | `data/hts_chapters/hts_code_map.csv` |

Reference datasets are refreshed via dedicated scripts (`api/scripts/refresh_metal_composition_hana.py`, `api/scripts/refresh_hts_hana_catalog.py`) or through the settings page for HTS catalog sources.

### Operational Datasets

| HANA Table | Content | Written When |
| --- | --- | --- |
| `METAL_COMPOSITION_SECTION232_SOURCES` plus ruleset tables | Uploaded Section 232 PDF corpus, extracted text, draft review rows, published rulesets, and runtime rules. | When administrators upload source PDFs, review draft candidates, and publish a ruleset. |
| `METAL_COMPOSITION_UI_DOCUMENT_ASSIGNMENTS` | Saved per-item PDF assignments with revision history. Append-only. | When users save document assignments or upload PDFs for a product. |
| `METAL_COMPOSITION_UI_CLASSIFICATION_HISTORY` | Full classification payloads, agent outputs, timing, status, and timestamps. Append-only. | After each single-item or batch classification completes. |

After API startup, most request paths use the in-memory serving store loaded from HANA rather than live per-request HANA queries. The Section 232 source corpus and classification history are queried on demand.

## Current implementation state (`api/` and `ui/`)

`api/` runtime:

- `app/main.py`: FastAPI entrypoint, CORS setup, API key auth wiring, startup/shutdown behavior, and background classification worker configuration.
- `app/routers/metal_composition.py`: Main application API under `/api/metal-composition` (items, document upload/assignment, predictions/classification, job status, reports, Section 232 workflow, settings, and HTS catalog source management).
- `app/routers/metal_composition_admin.py`: Admin setup endpoint under `/api/metal-composition/admin` used to refresh GCC Tracker data in HANA (`/gcc-tracker/refresh-hana`).
- `app/services/metal_composition/`: Service layer, orchestration, workflow nodes, HANA integration, item/job management, settings, reporting, and persistence helpers.
- `app/models/`: Pydantic request/response models for all major endpoints.
- `app/utils/`: Shared utility modules (HANA helpers, shared LangGraph utilities).
- `scripts/`: Operational refresh tooling for HANA tables and CAD scan helpers.
- `tests/`: API and workflow unit tests covering model validation, service behavior, and router paths.

`ui/` runtime:

- `src/main.js`: UI bootstrap and page router initialization.
- `src/services/api.js`: API client wrapper with `X-API-Key` handling, JSON/form request support, and classification job polling.
- `src/modules/`: Router and navigation modules.
- `src/pages/`: Active pages:
  - `products/` (worklist + filtering + bulk actions)
  - `product-detail/` (item details, document assignment, per-item results)
  - `settings/` (GCC refresh, HTS catalog upload/reset, source corpus upload workflow)
  - `section-232-review/` (Section 232 draft and published ruleset review)
- `src/routes.js`: Route map used by the SPA.
- `vite.config.js`: Preview host allowance (`VITE_APP_HOST`).

## Architecture

```
ui/          Vite + UI5 Web Components frontend (products, product detail, settings)
api/         FastAPI backend
  app/
    routers/           HTTP endpoints
    services/
      metal_composition/
        service.py     Business logic (item management, classification dispatch)
        workflow/       LangGraph pipeline (orchestrator + step modules)
        config.py      Frozen-dataclass settings from env vars
        serving_store.py   GCC data loading and in-memory store
        hts_catalog.py     HTS catalog loading and resolver
        section_232_sources.py   Section 232 corpus management
        ui_state.py    HANA-backed UI state (assignments, jobs, ownership, history)
        documents.py   Document discovery and upload management
    utils/             HANA connection, security, shared helpers
  scripts/             Data refresh scripts for HANA tables
  tests/               pytest test suite
data/        Source files (GCC workbook, HTS chapter CSVs)
docs/        Documentation
```

## Local Setup

Create local environment files before starting either service:

```bash
cp api/.env.example api/.env
cp ui/.env.example ui/.env
```

Update the secrets after copying:

- `api/.env`: SAP AI Core / Gen AI Hub credentials, HANA credentials where needed, and `API_KEY`.
- `ui/.env`: `VITE_API_BASE_URL`, `VITE_APP_HOST`, and `VITE_API_KEY` matching the backend `API_KEY`.

Optional diagram zoom refinement settings in `api/.env`:

- `METAL_COMPOSITION_DIAGRAM_ZOOM_ENABLED=true` enables the second-pass zoom workflow.
- `METAL_COMPOSITION_DIAGRAM_ZOOM_MAX_REQUESTS=6` caps executed zoom regions per analysis.
- `METAL_COMPOSITION_DIAGRAM_ZOOM_RENDER_DPI=600` controls PDF crop rerender resolution.
- `METAL_COMPOSITION_DIAGRAM_ZOOM_PADDING_RATIO=0.03` expands each requested crop before rendering.
- `METAL_COMPOSITION_DIAGRAM_ZOOM_IMAGE_MAX_DIMENSION=4096` bounds processed zoom image dimensions.
- `METAL_COMPOSITION_DIAGRAM_ZOOM_IMAGE_MAX_BYTES=2097152` bounds processed zoom image payload size.

Zoom traces are stored only in the classification payload under `agent_outputs.diagram.zoom_diagnostics` and the diagram timing details. External API contracts stay unchanged.

## Running Locally

Backend (`api/`):

```bash
cd api
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
export PORT=8000
uvicorn app.main:app --reload --host 0.0.0.0 --port "${PORT}"
```

Frontend:

```bash
cd ui
npm install
npm run dev -- --host 0.0.0.0 --port 5173
```

Useful local endpoints:

- API docs: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- UI dev server: [http://127.0.0.1:5173](http://127.0.0.1:5173)

## How to use the application

1. Start the API and keep API_KEY aligned between `api/.env` and `ui/.env`.
2. Open the UI at `http://127.0.0.1:5173`.
3. On **Settings**, upload/refresh:
   - GCC Tracker source via HANA refresh.
   - HTS CSV source files.
   - Section 232 source PDFs.
4. Open **Products**, filter and select items to classify.
5. For document-assisted classification, upload PDF(s) on the item detail screen before running classification.
6. Run:
   - single-item prediction/classification (`Predict`/`Classify`) or
   - batch prediction/classification from the products list.
7. Use the job status/polling path to monitor completion, then open item detail to review:
   - metal composition result,
   - HTS recommendation and evidence,
   - Section 232 outcome.
8. Optional: export PDF report and check review workflows for draft/published Section 232 candidates in the settings/review section.

Minimal API examples:

```bash
export API_BASE="http://127.0.0.1:8000/api/metal-composition"
export API_KEY="test-api-key"

curl -s "$API_BASE/items?limit=20" \
  -H "X-API-Key: $API_KEY"

curl -s -X POST "$API_BASE/items/predict" \
  -H "X-API-Key: $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"item_id":"gcc:1001","document_mode":"text_only"}'

curl -s "$API_BASE/classification-jobs/<job_id>" \
  -H "X-API-Key: $API_KEY"
```

## Verification

Backend test suite:

```bash
cd api
pytest -q
```

Frontend production build:

```bash
cd ui
npm run build
```

## Deployment

`manifest.yaml` contains Cloud Foundry app names and routes for the current product shape:

- `section-232-agent-api`
- `section-232-agent-ui`

Before deployment, update those names and routes if your target space requires different hostnames.

Automated deployment:

```bash
chmod +x deploy.sh
./deploy.sh
```

Manual deployment:

```bash
cf push --var api_key="your-secure-api-key"
```
