# Commodity Code Pipeline

The Commodity Code Pipeline extracts invoice data from PDFs, matches line items to HANA-hosted reference codes, verifies suggestions with an LLM, and exposes the results through a Streamlit UI and SAP Joule.

## Architecture

The solution has three user-facing components:

1. **API** (`commodity-code-pipeline-api`): FastAPI service that accepts asynchronous extraction jobs, runs document extraction and code matching, and stores job state, uploaded PDFs, result metadata, and generated Excel workbooks in SAP HANA.
2. **UI** (`commodity-code-pipeline-ui`): Streamlit application that uploads one or more PDFs, polls the API, previews completed results, and downloads an Excel workbook.
3. **Joule assistant** (`commodity_code_assistant`): Direct SAPDAS assistant that accepts one PDF, submits it to the same API, checks job status on demand, and displays structured results in pages of up to 30 line items.

```text
Streamlit UI ─┐
              ├─> FastAPI ─> HANA job storage ─> PDF extraction ─> HANA reference data
SAP Joule ────┘                                                      │
                                                                      └─> embedding match ─> LLM verification
```

The Joule module contains dialog orchestration only. Extraction, matching, persistence, and LLM processing remain in `api/`.

## Repository Layout

```text
.
├── api/                         # FastAPI service and extraction pipeline
│   ├── app/                     # API routes, models, and services
│   ├── doc_extraction/          # PDF extraction, embeddings, and LLM processing
│   ├── scripts/                 # Reference-data and sample-PDF utilities
│   └── tests/                   # Backend tests
├── ui/                          # Streamlit application
│   ├── src/                     # UI pages and API client
│   └── tests/                   # UI tests
├── joule_agent/                 # SAPDAS assistant and Commodity Code capability
│   ├── assistant.da.sapdas.yaml # Assistant deployment descriptor
│   └── commodity_code_capability/
├── generated_reference_data/    # Synthetic reference-data artifacts
├── docs/                        # Architecture and feature documentation
├── manifest.yaml                # Cloud Foundry API/UI manifest
└── deploy.sh                    # API/UI deployment helper
```

## Processing Flows

### Streamlit

1. The user uploads one or more PDFs.
2. The UI submits a multipart request to `POST /api/extraction/run`.
3. The API stores the job and PDFs in HANA and returns `202 Accepted` with a `job_id`.
4. The UI polls `GET /api/extraction/jobs/{job_id}` until the job is `SUCCEEDED` or `FAILED`.
5. A successful run shows a preview and enables the Excel download endpoint.

### Joule

1. Joule accepts exactly one PDF of at most 10 MB.
2. The capability sends the resolved PDF bytes to `POST /api/extraction/jobs`.
3. The returned `job_id` is stored as `last_job_id` for the active conversation.
4. The user checks the job explicitly; Joule does not poll automatically.
5. After the job succeeds, Joule retrieves `GET /api/extraction/jobs/{job_id}/result?page={page}` and displays the structured line items.

Recognized job states are `QUEUED`, `RUNNING`, `SUCCEEDED`, and `FAILED`.

## Prerequisites

### API and UI

- Cloud Foundry CLI, authenticated against the target org and space
- Permission to push applications and configure environment variables
- SAP HANA credentials with permission to read the reference tables and create or use the extraction job tables
- SAP Gen AI Hub credentials or service binding available to the API runtime
- Python 3 for local validation

### Joule

- A deployed and reachable API application
- Joule Studio CLI 1.5.21 authenticated against the target tenant
- Permission to deploy and launch digital assistants
- Permission to create or update an SAP BTP destination

## Runtime Configuration

### API environment variables

| Variable | Required | Purpose |
| --- | --- | --- |
| `API_KEY` | Production | Protects all extraction endpoints through `X-API-Key`. |
| `APP_ENV` | Production | Set to `production`; the API fails closed when `API_KEY` is missing. |
| `ALLOWED_ORIGIN` | Production UI | Allowed Streamlit origin for CORS. |
| `hana_address`, `hana_port`, `hana_user`, `hana_password` | Yes | HANA connection used by reference data and extraction jobs. |
| `hana_encrypt` | No | Enables encrypted HANA connections; defaults to `true`. |
| `hana_ssl_validate_certificate` | No | Enables HANA certificate validation; defaults to `false`. |
| `HANA_SCHEMA` | No | Overrides the current HANA schema. |
| `HANA_REFERENCE_DATA_VERSION` | No | Requires all three reference tables to contain the expected shared `DATA_VERSION`. |
| `HANA_COMMODITY_CATALOG_TABLE` | No | Overrides `REFERENCE_COMMODITY_CATALOG`. |
| `HANA_UNSPSC_MAPPING_TABLE` | No | Overrides `REFERENCE_UNSPSC_MAPPING`. |
| `HANA_SUPPLIER_GROUPS_TABLE` | No | Overrides `REFERENCE_SUPPLIER_GROUPS`. |
| `HANA_EXTRACTION_JOBS_TABLE` | No | Overrides `EXTRACTION_JOBS`. |
| `HANA_EXTRACTION_JOB_FILES_TABLE` | No | Overrides `EXTRACTION_JOB_FILES`. |
| `EXTRACTION_JOB_WORKERS` | No | Concurrent in-process workers; defaults to `1`. |
| `EXTRACTION_MAX_QUEUED_JOBS` | No | Maximum queued plus running jobs; defaults to `20`. |
| `LLM_MODEL`, `LLM_MODEL_NAME` | No | Extraction and verification model overrides; defaults are `gpt-4.1`. |
| `EMBEDDING_MODEL` | No | Embedding model override; defaults to `text-embedding-3-large`. |

The API creates and validates the HANA extraction job tables on first use. The three reference tables must already be populated and must share one `DATA_VERSION`. `api/scripts/generate_and_load_reference_data.py --help` documents the available generation and HANA-load options when authorized source datasets are available.

### UI environment variables

| Variable | Required | Purpose |
| --- | --- | --- |
| `API_BASE_URL` | Yes | Base URL of the API application. |
| `API_KEY` | Production | Must match the API application's key. |

## Deploy the API and UI

Review `manifest.yaml`, configure the HANA and SAP Gen AI Hub runtime in the target landscape, and deploy manually from the repository root.

For API/UI-only deployment, the helper generates a new API key and passes it to Cloud Foundry:

```bash
chmod +x deploy.sh
./deploy.sh
```

For a deployment that will also be used by Joule, retain the generated key because the same value must be added to the BTP destination:

```bash
API_KEY="$(openssl rand -hex 32)"
cf push --var api_key="$API_KEY"
```

Configure any HANA variables that are not supplied through the target landscape's service or secret management, then restage the API. For example:

```bash
cf set-env commodity-code-pipeline-api hana_address "<host>"
cf set-env commodity-code-pipeline-api hana_port "<port>"
cf set-env commodity-code-pipeline-api hana_user "<user>"
cf set-env commodity-code-pipeline-api hana_password "<password>"
cf set-env commodity-code-pipeline-api HANA_SCHEMA "<schema>"
cf restage commodity-code-pipeline-api
```

Do not commit credentials or API keys.

### Verify the Cloud Foundry applications

```bash
curl https://commodity-code-pipeline-api.cfapps.eu10-004.hana.ondemand.com/api/health
cf app commodity-code-pipeline-api
cf app commodity-code-pipeline-ui
cf logs commodity-code-pipeline-api --recent
```

The Streamlit UI is available at:

```text
https://commodity-code-pipeline.cfapps.eu10-004.hana.ondemand.com
```

## Deploy the Joule Assistant

### 1. Create the BTP destination

Create or update the destination used by `joule_agent/commodity_code_capability/capability.sapdas.yaml`:

```text
Name: CommodityCodePipelineAPI
Type: HTTP
URL: https://commodity-code-pipeline-api.cfapps.eu10-004.hana.ondemand.com
Proxy Type: Internet
Authentication: NoAuthentication
URL.headers.X-API-Key: <same API_KEY configured on the API application>
```

Keep the API key in the destination; do not add it to the SAPDAS YAML files.

### 2. Validate the SAPDAS files

Run these commands from the repository root:

```bash
joule lint joule_agent/assistant.da.sapdas.yaml
joule compile joule_agent/commodity_code_capability /tmp/commodity_code_joule_compile
```

### 3. Deploy and launch

Deployment must be run manually by a user authenticated to the target Joule tenant:

```bash
joule deploy --compile \
  joule_agent/assistant.da.sapdas.yaml \
  --name commodity_code_assistant

joule launch commodity_code_assistant
```

## Use the Joule Assistant

Keep the upload, status checks, and result navigation in the same Joule conversation because `last_job_id` is conversation-scoped.

1. Ask Joule to upload and classify a PDF, then attach one PDF of at most 10 MB.
2. Select **Check status**. If the job is still `QUEUED` or `RUNNING`, select **Check again** later.
3. When the job is complete, select **Show page 1**.
4. Use **Next page** and **Previous page**, or ask `Show page 2 of the latest commodity code results`.
5. Uploading another PDF in the same conversation replaces `last_job_id` with the new job.

Each Joule line item contains:

- description
- net amount
- quantity
- unit price
- AI-suggested commodity code
- AI confidence score
- AI reasoning

For the complete destination, API, response, and troubleshooting contract, see [docs/joule-agent-integration.md](docs/joule-agent-integration.md).

## API Endpoints

All extraction endpoints require `X-API-Key` when `API_KEY` is configured.

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/api/health` | API health probe. |
| `GET` | `/api/extraction/defaults` | UI extraction defaults. |
| `POST` | `/api/extraction/run` | Submit one or more multipart PDF uploads from the UI. |
| `POST` | `/api/extraction/jobs` | Submit one raw PDF body from Joule. |
| `GET` | `/api/extraction/jobs/{job_id}` | Read status and progress. |
| `GET` | `/api/extraction/jobs/{job_id}/result?page=1` | Read one structured Joule result page. |
| `GET` | `/api/extraction/jobs/{job_id}/download` | Download the completed Excel workbook. |

## Local Development

Create an `api/.env` with the required HANA and SAP Gen AI Hub configuration, then install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r api/requirements.txt -r ui/requirements.txt
```

Start the API:

```bash
cd api
python -m app.main
```

In another terminal, start the UI:

```bash
cd ui
API_BASE_URL=http://127.0.0.1:8000 python streamlit_app.py
```

When a local `API_KEY` is configured, pass the same value to the UI process.

## Tests

```bash
cd api
python3 -m unittest discover -s tests -p 'test_*.py'

cd ../ui
python3 -m unittest discover -s tests -p 'test_*.py'
```

Validate Joule separately with the lint and compile commands in the deployment section.

## Troubleshooting

- **`503 API authentication is not configured`**: set `API_KEY` when `APP_ENV=production`, then restage the API.
- **HANA storage or reference-data errors**: verify the lowercase HANA connection variables, schema permissions, reference-table names, and shared `DATA_VERSION`.
- **Joule receives `401`**: verify `URL.headers.X-API-Key` in `CommodityCodePipelineAPI` matches the API application key.
- **Joule cannot prepare a status or result path**: redeploy the current SAPDAS files and begin a new conversation so an older `last_job_id` value is not reused.
- **Results are unavailable**: check the job first; structured result pages are returned only after `SUCCEEDED`.
- **CORS errors**: verify `ALLOWED_ORIGIN` matches the deployed Streamlit route.
- **Capacity errors**: check Cloud Foundry quota and adjust `memory` or `disk_quota` in `manifest.yaml` when needed.
