# IBP Optimizer Log Extractor

Automated extraction of Supply Planning Logs from SAP IBP Application Jobs.
One click in a browser replaces 5–8 minutes of manual navigation through the SAP Fiori Launchpad.

---

## What it does

- **Navigates SAP IBP automatically** — finds the latest finished Application Job, opens the log, downloads the Supply Planning Logs attachment
- **Parses and stores the data** — extracts two CSV files from the download, inserts all rows into SAP HANA Cloud with metadata (timestamp, job name, checksum)
- **Serves a Fiori UI** — displays downloaded files with inline CSV preview, real-time extraction progress, and HANA management
- **Works in two modes** — manual trigger from a local machine, or fully automatic via a Kyma scheduled job

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  User's machine  (Mac or Windows)                                           │
│                                                                             │
│  IBP-Agent.command  /  IBP-Agent.bat                                        │
│    └─ local_agent.py  (Flask :5001)                                         │
│         └─ extract.py  (Playwright + Chrome — visible browser)              │
│              └─ navigates IBP → downloads Supply Planning Logs              │
│                   └─ POST /api/upload ──────────────────────────────────►   │
└──────────────────────────────────────────────────┬──────────────────────────┘
                                                    │
                             OR (scheduled/automated)│
                                                    │
┌──────────────────────────────────────────────────┤
│  SAP BTP Kyma (Kubernetes)                       │
│                                                  │
│  CronJob "ibp-extractor"  (Mon–Fri 6 AM UTC)     │
│    └─ kyma_wrapper.py  (xvfb-run + Chrome)       │
│         └─ extract.py                            │
│              └─ POST /api/upload ────────────────┘
│              └─ POST /api/run/status (streams log to CF backend)
└─────────────────────────────────────────────────────────────────────────────┘
                                     │ HTTPS
┌───────────────────────────────────▼─────────────────────────────────────────┐
│  SAP BTP — Cloud Foundry                                                    │
│                                                                             │
│  ibp-extractor-backend  (Python / python_buildpack)                         │
│    Flask REST API — upload, files, status, HANA, Kyma trigger, sessions    │
│                                                                             │
│  ibp-extractor-ui  (Angular 18 / staticfile buildpack / Nginx)              │
│    SAP Fiori Horizon UI — Fundamental NGX                                   │
│                                                                             │
│  SAP HANA Cloud                                                             │
│    IBP_OPTEXPLLOG1   optimizer explanation log — customer demand            │
│    IBP_OPTEXPLLOG2   optimizer explanation log — inventory/supply           │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Why the extraction runs locally (or via Kyma)

SAP IBP detects automated browsers via `navigator.webdriver` and blocks JavaScript execution in headless Chromium — the Fiori UI never renders. The extraction must run with a **real visible Chrome browser**.

- **Local mode:** Chrome opens on the user's machine — SSO is automatic via the persistent profile.
- **Kyma mode:** Chrome runs inside the container using `xvfb-run` (virtual display) — same rendering as a real screen, no webdriver detection.

---


## Extraction flow (step by step)

1. User clicks **Extract Now** in the UI (or Kyma CronJob fires)
2. Chrome opens, navigates to the configured IBP tenant
3. Filters Application Jobs by name, selects the most recent **Finished** row
4. Waits for the busy indicator to clear, clicks the Log icon
5. Finds the **Supply Planning Logs** attachment on the "Optimizer explanation log created" row
6. Downloads the zip file containing two CSVs (`optexpllog1.csv`, `optexpllog2.csv`)
7. POSTs the zip to the CF backend `/api/upload`
8. CF backend extracts, parses (semicolon-delimited), and inserts into HANA
9. UI refreshes — files appear in the table with inline preview and HANA row counts

---

## Repository layout

| Area | Key files |
|---|---|
| Local agent | `IBP-Agent.command`, `Windows_new/IBP-Agent.bat`, `local_agent.py`, `extract.py` |
| CF backend | `backend/app.py`, `backend/requirements.txt`, `manifest.yml` |
| CF frontend | `frontend/src/app/app.component.ts`, `frontend/dist/ibp-ui/browser/` |
| Kyma | `Dockerfile.kyma`, `requirements-kyma.txt`, `kyma_wrapper.py`, `kyma/cronjob.yaml`, `kyma/job-trigger.yaml`, `kyma/secret-template.yaml` |

---

## Deployment from scratch (after `git clone`)

The repository is **fully re-deployable**, but four pieces are intentionally kept **outside** the repo for security. You must provide them yourself before deploying.

### What is NOT included (and why)

| Missing item | Why | How to provide it |
|---|---|---|
| **Docker image** | Only the `Dockerfile.kyma` recipe is versioned; the built image is not | Build it and push to your own container registry (step 3) |
| **IBP session cookies** (`session_cookies.json`) | Sensitive — grants access to the IBP tenant | Export from Chrome with the *Cookie-Editor* extension after logging in (step 5) |
| **Kyma kubeconfig** (`kyma-kubeconfig.yaml`) | Contains a service-account token | Download from BTP Cockpit → your Kyma environment → *Kubeconfig* |
| **HANA credentials** | Secrets | Inject via `cf set-env` or a user-provided service instance (step 4) |

### Placeholders in the repo

The following files contain `<your-...>` placeholders you must replace before deploying:

| File | Placeholders |
|---|---|
| `manifest.yml` | `<your-ibp-tenant>`, `<your-ibp-job-name>`, `<your-hana-host>`, `<your-hana-user>`, `<your-hana-password>`, `<your-hana-schema>` |
| `kyma/cronjob.yaml` | `<your-registry>`, `<your-ibp-job-name>`, `<your-cf-backend-host>` |
| `kyma/job-trigger.yaml` | `<your-registry>`, `<your-cf-backend-host>` |

> Recommendation: never commit real values. Keep `manifest.yml` anonymized and inject secrets via `cf set-env` or a user-provided service.

---


### Step 1 — Clone

```bash
git clone https://github.tools.sap/sap-btp-ai-services-coe/ibp-agent.git
cd ibp-agent
```

### Step 2 — Prerequisites

- **BTP subaccount** with:
  - Cloud Foundry runtime entitlement
  - Kyma runtime enabled
  - SAP HANA Cloud instance
- **CLIs:** `cf`, `kubectl`, `docker` (with `buildx`), `node` 18+, `python` 3.11+
- **Container registry** you can push to (Docker Hub, GitHub Container Registry, or the Kyma-integrated registry)

### Step 3 — Build and push the Docker image

The image bundles Chromium, Xvfb and Playwright — required because IBP blocks headless browsers.

```bash
docker buildx build --platform linux/amd64 \
  -t <your-registry>/ibp-extractor-agent:latest \
  -f Dockerfile.kyma --push .
```

Then replace `<your-registry>` in `kyma/cronjob.yaml` and `kyma/job-trigger.yaml`.

### Step 4 — Deploy CF backend and frontend

```bash
# Log into CF
cf login -a https://api.cf.<your-region>.hana.ondemand.com

# Edit manifest.yml — replace all <your-...> placeholders.
# For HANA_PASSWORD, prefer cf set-env over committing it:
cf push ibp-extractor-backend
cf set-env ibp-extractor-backend HANA_PASSWORD "<real-password>"
cf restage ibp-extractor-backend

# Build and push the Angular frontend
cd frontend
npm install
npx ng build --configuration production
echo "pushstate: enabled" > dist/ibp-ui/browser/Staticfile
cd ..
cf push ibp-extractor-ui
```

Note the backend URL returned by `cf push` — you will need it in step 6.

### Step 5 — Prepare IBP session cookies

The agent authenticates against IBP by reusing browser cookies (IBP uses SSO and blocks automated login):

1. Open Chrome, log into IBP manually with the service user
2. Install the **Cookie-Editor** extension (Chrome Web Store, free)
3. On the IBP tab, click Cookie-Editor → *Export* → *Export as JSON*
4. Save the file as `session_cookies.json` locally (it is gitignored)

### Step 6 — Deploy to Kyma

```bash
# 1. Download the kubeconfig from BTP Cockpit and set it
export KUBECONFIG=./kyma-kubeconfig.yaml

# 2. Create the namespace
kubectl create namespace ibp-agent

# 3. Create the cookies Secret from your exported JSON
kubectl create secret generic ibp-cookies \
  --from-file=session_cookies.json=./session_cookies.json \
  -n ibp-agent

# 4. Edit kyma/cronjob.yaml and kyma/job-trigger.yaml:
#    - <your-registry>/ibp-extractor-agent:latest    -> your image
#    - <your-cf-backend-host>                        -> CF backend URL from step 4
#    - <your-ibp-job-name>                           -> IBP Application Job name

# 5. Apply the CronJob (scheduled extraction, Mon-Fri 6 AM UTC)
kubectl apply -f kyma/cronjob.yaml -n ibp-agent
```

### Step 7 — Enable the "Run Now" button in the CF UI

To let the CF backend trigger on-demand Kyma Jobs, add these env vars and redeploy:

```bash
cf set-env ibp-extractor-backend KYMA_NAMESPACE      "ibp-agent"
cf set-env ibp-extractor-backend KYMA_JOB_IMAGE      "<your-registry>/ibp-extractor-agent:latest"
cf set-env ibp-extractor-backend KYMA_COOKIES_SECRET "ibp-cookies"
cf restage ibp-extractor-backend
```

### Rotating cookies when they expire

Re-export from Cookie-Editor, then:

```bash
kubectl create secret generic ibp-cookies \
  --from-file=session_cookies.json=./new_cookies.json \
  -n ibp-agent \
  --dry-run=client -o yaml | kubectl apply -f -
```

Or, for local/CF mode, use the **Upload IBP Cookies** button in the UI.

---


## Running locally (development / manual mode)

### macOS — first time
```bash
./setup.sh    # creates .venv, installs deps, installs Playwright Chrome
```

### macOS — daily use
Double-click `IBP-Agent.command` in Finder, or:
```bash
bash IBP-Agent.command
```

### Windows — first time
1. Install Python 3.11+ from https://www.python.org — check **Add Python to PATH**
2. Double-click `Windows_new/setup.bat`

### Windows — daily use
Double-click `Windows_new/IBP-Agent.bat`

> Extract the ZIP before running — do not double-click `.bat` files from inside a ZIP.

---

## IBP session management

SAP IBP uses SSO. Chrome authenticates automatically via the persistent profile on first run.

If the session expires, the UI shows **"Upload IBP Cookies"**:
1. Open Chrome, log into IBP with the service user
2. Install **Cookie-Editor** extension (Chrome Web Store)
3. Export as JSON
4. Upload using the **Upload IBP Cookies** button in the UI

For Kyma: update the `ibp-cookies` Kubernetes Secret with the new JSON (see "Rotating cookies" above).

---

## HANA schema

| Table | Source CSV | Description |
|---|---|---|
| `<SCHEMA>.IBP_OPTEXPLLOG1` | `optexpllog1.csv` | Customer demand explanation log |
| `<SCHEMA>.IBP_OPTEXPLLOG2` | `optexpllog2.csv` | Inventory / supply explanation log |

Where `<SCHEMA>` is the value of `HANA_SCHEMA` in `manifest.yml`.

Metadata columns on every row: `EXTRACTED_AT`, `SOURCE_FILE`, `CSV_FILE`, `JOB_NAME`.
Schema auto-evolves — new CSV columns are added via `ALTER TABLE` on each run.

---

## SAP BTP services used

| Service | Usage |
|---|---|
| **Cloud Foundry Runtime** | Hosts Flask backend (python_buildpack) and Angular UI (staticfile_buildpack) |
| **SAP HANA Cloud** | Stores parsed CSV data from every extraction run |
| **Kyma (Kubernetes)** | Runs scheduled/on-demand extraction in a container with virtual display |

---

## Is this an Agent?

Yes — technically a **Web Automation Agent** / **Task Agent**. It has a goal, makes decisions to achieve it (which row to select, how to handle busy indicators, which attachment to download), and acts autonomously. It is **not RPA**.

| RPA | This Agent |
|---|---|
| Records and replays fixed coordinates | Uses semantic DOM selectors + conditional logic |
| Breaks when layout changes | Has fallbacks for multiple navigation paths |
| No reasoning | Verifies Status = Finished, waits for busy indicators, handles two attachment rows |
| Screen bot | DOM-based automation via Playwright |

The local agent exposes a REST interface — any AI orchestrator (LangGraph, SAP AI Core, n8n) can invoke it as a **tool**:

```http
POST http://localhost:5001/api/run
{ "jobName": "<your-ibp-job-name>" }
```

---

## Tech stack

| Layer | Technology |
|---|---|
| Browser automation | Python · Playwright · Chrome |
| Local agent | Python · Flask |
| Kyma agent | Python · Playwright · xvfb · Docker |
| CF backend | Python · Flask · hdbcli · python_buildpack |
| CF frontend | Angular 18 · @fundamental-ngx/core · SAP Fiori Horizon |
| Database | SAP HANA Cloud |
| Cloud platform | SAP BTP Cloud Foundry + Kyma (Kubernetes) |

---

## Security notes

- **Never commit** `session_cookies.json`, `kyma-kubeconfig.yaml`, `.env` files, or HANA credentials. They are gitignored — keep it that way.
- **HANA password:** always inject via `cf set-env` or a user-provided service instance, never in `manifest.yml`.
- **Container registry:** use a corporate/private registry for production deployments. The `<your-registry>` placeholder is intentionally generic.
- **Cookies rotation:** IBP sessions expire regularly. Plan for periodic cookie refresh (manual re-export via Cookie-Editor).

