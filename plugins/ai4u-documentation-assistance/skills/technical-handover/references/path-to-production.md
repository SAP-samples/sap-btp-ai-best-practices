# Path to production on SAP Business AI Platform

Use this checklist to build the gap register in Part VI. The register is deliberately narrow: it lists only what certainly has to change before real users and real data touch the solution. That is **security** and **integration**. With a BPD, it also lists the customer's own requirements the PoC does not meet (see [bpd-merge.md](bpd-merge.md), section 6).

Why so narrow: we build PoCs outside the customer's landscape, on exports (usually Excel workbooks). We can't know how the customer runs production: their CI/CD, landscapes, monitoring, availability targets or UI standards. Gaps in those areas are guesses, and they bury the few rows that matter. The implementation team decides them with the customer.

For each check, look for the evidence in the repo, record the **current state** honestly and recommend the **target** on SAP Business AI Platform. An area with no finding is fine; don't invent rows to fill it. A typical register has 5–10 rows. Prioritise with Must / Should / Could. Never add time or effort estimates.

## 1. Security

Raise a row only for a concrete weakness found in the code or config.

| Check | Evidence to look for | Typical PoC state | Production target |
|---|---|---|---|
| User authentication | `xs-security.json`, approuter (`xs-app.json`), XSUAA/IAS libs, JWT validation code, `X-API-Key` headers | Shared static API key, or no authentication; no users | Application Router + SAP Authorization and Trust Management (XSUAA) or SAP Cloud Identity Services (IAS) with the corporate IdP; JWT validation in every backend app |
| API key in the browser | `VITE_*`, `REACT_APP_*`, `NEXT_PUBLIC_*` vars holding keys; `X-API-Key` set in frontend code | Key compiled into the SPA bundle, so anyone with the URL can call the API | Remove it; the browser uses the approuter session and the backend receives the user JWT |
| Authorization | Scopes/role templates, role checks in handlers, unprotected DELETE/reset/settings routes | No roles: every caller can delete data or change settings; decisions not attributable to a user | Role templates + role collections; scope checks on destructive and administrative routes; the signed-in user recorded on decisions |
| Secrets | `manifest.yml` env blocks, `.env`, `deploy.sh`, `cf push --var`, `vars.yml` | Credentials in env vars or deploy scripts, sometimes committed | Service bindings (HANA, AI Core) and SAP Credential Store; nothing secret in manifests, scripts or the repo |
| Dangerous exposed surface | Experimental routers, generated-code execution, debug or admin endpoints reachable with the shared key | Reachable in the deployed app | Removed from production, or restricted to an admin scope |
| Identity towards the back end | Calls to ECC / S/4 (usually absent in a PoC) | Not implemented | Destination with principal propagation (on-premise) or OAuth2SAMLBearer (cloud); a technical user only where justified and documented |

## 2. Integration

The PoC works on exports; production reads from and writes back to the customer's systems. The target systems come from the interview (e.g. "data comes from and goes back to ECC"). Don't assume S/4HANA when the customer runs ECC. One row per direction and system.

| Check | Evidence | Typical PoC state | Production target |
|---|---|---|---|
| Inbound data | File upload endpoints, Excel/CSV parsers | Manual workbook/CSV export and upload | Extraction from the source system named in the interview: **ECC** (RFC/BAPI, IDoc, or via SAP Integration Suite / SLT / Datasphere), **S/4HANA** (OData, CDS views), **APO/IBP** (IBP OData / APO RFCs). Keep the upload as a fallback |
| Write-back | Export endpoints, "simulated" updates | Excel/CSV export re-keyed by users, or no output into the system | Write-back to the target system (ECC via BAPI/IDoc, S/4 via OData) behind a human approval step, with read-back reconciliation |
| Connectivity | None | Not present | SAP Connectivity Service + Cloud Connector for on-premise systems; one SAP BTP Destination Service entry per system and client |
| Non-SAP sources | Uploaded files from PLM, SharePoint, etc. | Manual upload | The named system's API through a destination; only if the interview or the BPD names it |

## Do not raise

These never become gap rows, open questions or roadmap items, even when the code shows the PoC state plainly. Describe the current state in Part V where the blueprint asks for it, and stop there.

- High availability, instance counts, worker resilience, job durability, statelessness, sizing, load tests, performance and run times
- CI/CD, deploy scripts, MTA, landscapes and transport
- Logging, monitoring, alerting, dashboards, health checks, runbooks
- UI hosting, theme, Fiori look, localisation, UX defects
- Tests, eval suites, dependency pinning, licence scans, dead code, template leftovers
- Model choice, orchestration, prompt management, usage quotas and telemetry
- Data retention and purge, audit logging, master-data governance, multi-user data separation
- Algorithm or optimizer backlog, solver choice

Exception: a BPD requirement the PoC fails (e.g. a mandatory audit record) is a row, because the customer asked for it (one row per requirement, see [bpd-merge.md](bpd-merge.md), section 6).

## Writing the recommendation
Good row: *"G-01 · Identity · The shared API key is compiled into the UI bundle (`ui/src/services/api.js` reads `VITE_API_KEY`), so anyone who can load the UI can call every endpoint, including DELETE. → Introduce the approuter + XSUAA, forward the user JWT, validate it in a FastAPI dependency, and remove `VITE_API_KEY`. · Must"*

Good row: *"G-04 · Integration (ECC outbound) · The approved plan is only exported as XLSX and re-keyed by planners. → Write the approved plan to ECC through a BAPI behind a planner approval step, via Cloud Connector and a destination, with read-back reconciliation. · Must"*

Bad row: *"Improve security."*

Bad row: *"G-12 · Resilience · One API instance. → Run two instances. · Should"* (an operations decision for the implementation team, not a gap)
