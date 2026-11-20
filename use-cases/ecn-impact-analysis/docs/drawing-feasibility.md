# PS4 drawing-feasibility assessment

## Purpose and scope

`POST /api/drawing-feasibility/assessments` provides a **transient PS4 technical-feasibility assessment** for one pipe drawing. Its report answers whether the submitted configuration is feasible against the assessed PS4 requirements. It is not a general manufacturing, quality, tooling, tolerance, cost, capacity, or release approval.

For the current PoC, a new committed ECN also schedules the same assessment automatically when one of its source-backed technical materials has an exact match in the local fixture catalog. This changes only the drawing source and orchestration: it reuses the same `DrawingFeasibilityService`, PS4 policy, `DrawingFeasibilityReport`, and Material Dialog report renderer as the manual route. The stored automatic result contains the report JSON and safe drawing metadata, never the PDF bytes.

## Data boundary

- The endpoint accepts exactly one PDF, PNG, or JPEG as `drawing_file` in multipart form data.
- It validates filename, declared media type, magic bytes, and the configured streamed byte limit. Uploaded bytes exist only in request memory, are not persisted, logged, echoed, or returned to the browser.
- The material-popup report renders only controlled statuses, criteria, and calculator traces. It intentionally omits drawing-source-reference indicators and post-analysis limitation/disclaimer paragraphs; drawing bytes, input data URLs, raw evidence text, prompt text, and policy documents are never rendered.
- All `/api` business routes, including this one, require `X-API-Key`. This is existing application protection, not fine-grained drawing authorization.

## Approved policy context

The provider loads a fixed allowlist only, in deterministic order, from `api/app/drawing_feasibility/policy/`:

1. `pipe-drawing-to-ps4-specification.md`
2. `pipes-guideline-ps4.md`
3. `ps4-pipe-evaluation-rubric.md`
4. `tools-readme.md`

The loader does not enumerate the `drawings/` directory. It records hashes for these four Markdown files and instructs the model to ignore instructions in the submitted drawing and never open, retrieve, quote, or infer content from PDF, ZIP, archive, or other out-of-request files. No reference PDF or ZIP is packaged as model policy context.

The original skills specify XML exchange documents. This application preserves their evidence and PS4 decision semantics in strict Pydantic models and projects the final report as validated JSON for the UI; it does not expose free-form XML or model prose to the browser.

## Workflow

The purpose-built finite LangGraph is:

```text
extract_pipe_specification → evaluate_ps4_requirements → render_report
```

1. **Extraction** — one SAP Generative AI Hub Responses call converts the approved drawing into a strict `PipeSpecification` with direct evidence. It does not decide feasibility.
2. **Evaluation** — packaged `ps4_measurement_calculator.py` performs all numeric normalization, table lookup, arithmetic, comparisons, bender-schedule mapping, automatic-crimp, and robot-pickup checks. For the PS4 raw-material list, the calculator derives its lookup key from source-backed diameter and wall thickness; a drawing's EN material-grade text or note is retained as evidence but is not used as that key. The automatic-crimp objective is derived from the calculator-backed schedule geometry: **Pass** when all required L1/L2/L3/L4 checks pass, **Fail** when any fails, and **Warning** when the geometry cannot be fully assessed; LEFT/RIGHT callout omission does not downgrade passing geometry. Tube length uses the source-backed diameter, wall thickness, and total length: Ø8 and Ø10 with 1 mm wall thickness are limited to 200–1500 mm; other listed diameter/thickness combinations are limited to 200–3000 mm. In the calculation trace, diameter and wall thickness are neutral **Input** values; only total tube length receives the calculator's Pass/Fail result. The LLM never performs these operations or selects substitute schedule rows.
3. **Report** — deterministic code turns controlled calculator outcomes and policy rules into the JSON contract shown by the material popup. A single source-stated end-option set retains every option: if it includes a free/B4 option alongside EO-2 or metal-seal alternatives, the end check is **Warning**, explicitly names the compatible and non-compatible source options, and the candidate remains feasible. A set with no compatible end option fails. As an extraction safeguard, the evaluator may recover this result only from complete source-backed `or` callouts already retained as evidence for end-related prohibited features; it never invents an unrecorded B4/free option.

The graph has no tool-selecting/ReAct loop. A configured concurrency semaphore defaults to one, graph recursion is capped at eight, deterministic calculator subprocesses have a 15-second timeout, and the full request has a bounded timeout.

## Automatic ECN fixture assessment (PoC)

After a newly committed ECN, the upload workflow schedules the existing Change Summary job and then resolves source-backed technical material numbers against this exact local allowlist. The PDF fixtures are packaged at `api/app/ecn/fixtures/drawings/`, so they are included in the API application artifact for the controlled Cloud Foundry PoC deployment:

```text
QKX5632660 → QKX5632660_2_AFS.pdf
QKX5525860 → QKX5525860_2_AFS.pdf
QKX3896510 → QKX3896510_2_IW_old.pdf
QKX0971490 → QKX0971490_2_IW_old.pdf
```

The fixture resolver uses no filename similarity or inferred revision matching. A matching PDF is sent to SAP Generative AI Hub only while the existing assessment executes. The automatic runtime persists one report JSON record per ECN/material/drawing SHA-256 and displays that report using the existing Material Dialog presentation. It never persists the PDF bytes. An unmatched material does not receive an invented report. The manual endpoint remains available as the transient fallback path.

Matching material details also expose `engineering_drawing` with the packaged
PDF's filename and media type. **View engineering drawing** retrieves the exact
PDF through `GET /api/ecns/{ecn_row_id}/drawing?number=...`, using the existing
API authentication and ECN membership check. The viewer opens these sample
bytes without generating a report or changing the assessment. The file
chooser is absent when a drawing is available. Materials without an available
PDF keep one shared file chooser under **Engineering drawing**. The selected
file is used for **View engineering drawing** and, when no saved report exists,
**Analyze Drawing**. Selection and preview do not run analysis. Choosing a new
file clears any transient report so its result stays associated with the
selected drawing. This flow omits the former attachment and local-preview
disclaimers. PDF.js renders pages in the viewer with page and zoom controls;
the original PDF remains downloadable. A stored assessment report alone does
not supply a drawing.

For historic ECNs, run the controlled backfill only after explicit approval because `--execute` sends matching local PDFs to SAP Generative AI Hub and writes reports:

```bash
# Read-only: list exact ECN/material fixture matches.
PYTHONPATH=api python scripts/backfill_fixture_manufacturing_feasibility.py

# External transmission + report persistence for the listed matches.
PYTHONPATH=api python scripts/backfill_fixture_manufacturing_feasibility.py --execute
```

The script does not infer mappings, does not retain raw PDF bytes, and is idempotent for an already stored ECN/material/drawing SHA-256 report.

## Configuration

The drawing workflow is active whenever the API is running. It intentionally reuses the existing ECN LLM runtime configuration, so it uses the same `LLM_MODEL`, `LLM_REQUEST_TIMEOUT_SECONDS`, and `LLM_MAX_CONCURRENCY` values. `LLM_MODEL` must therefore be an approved multimodal deployment in the target SAP AI Core resource group.

Only the drawing-specific streamed upload bound is configured separately:

```dotenv
DRAWING_MAX_UPLOAD_BYTES="20971520"
```

The standard AI Core credentials (`AICORE_AUTH_URL`, `AICORE_CLIENT_ID`, `AICORE_CLIENT_SECRET`, `AICORE_BASE_URL`, and `AICORE_RESOURCE_GROUP`) are loaded by the application before the SAP proxy client is created. A real multimodal smoke test still requires an approved, sanitized test drawing and explicit confirmation before transmission. No such test is included in this repository validation.

## API contract

### Request

```http
POST /api/drawing-feasibility/assessments
X-API-Key: <configured-key>
Content-Type: multipart/form-data

 drawing_file=<one PDF, PNG, or JPEG>
```

The response is synchronous and is not a durable job. Safe errors use the application envelope:

```json
{"code":"DRAWING_FILE_TYPE_UNSUPPORTED","message":"Only PDF, PNG, and JPEG drawings are supported","details":null}
```

Expected codes include `DRAWING_FIELDS_INVALID`, `DRAWING_FILE_TYPE_UNSUPPORTED`, `DRAWING_CONTENT_INVALID`, `DRAWING_FILE_TOO_LARGE`, `DRAWING_MODEL_UNAVAILABLE`, `DRAWING_ANALYSIS_TIMEOUT`, and `DRAWING_ANALYSIS_FAILED`.

### Response

The response is a strict `DrawingFeasibilityReport` with `overall_result`, `candidate_reports`, controlled criteria statuses, safe source metadata, scope notice, and limitations. Each criterion retains up to 60 deduplicated source references so a large schedule cannot invalidate report construction; the original drawing bytes remain transient and are never returned. The material popup deliberately presents only the decision summary and calculator trace: it does not display source-reference disclosures or post-analysis limitation/disclaimer paragraphs. For calculator-backed criteria it includes deterministic calculator provenance. The material popup projects that provenance into an open **Geometry and PS4 calculation trace** with source/mapped values, calculator-derived values, minimum/maximum requirements, formulas/lookups, and semantic result indicators. It never recomputes a geometry result in the browser. Multiple explicitly associated drawing candidates result in `Configuration-dependent`; alternatives are never merged into an invented configuration. An end callout's own option set is retained within its source-associated candidate, so compatible free/B4 and incompatible EO-2/metal alternatives produce a warning rather than an error.

## Observability and operations

Each actual model extraction call emits one `btp.llm_usage.v1` success or error event via `app.observability`. It includes token/cache counters when available, latency, model, endpoint, stage, and correlation ID. It does not log the drawing, prompt, model raw output, evidence raw text, or a raw user identity.

The measured per-drawing token and CU baseline, cache caveats, planning allowance, and reproduction inputs are documented in [`drawing-feasibility-token-cost-estimate.md`](drawing-feasibility-token-cost-estimate.md).

Before production use, add a retention/authorization decision, malware scanning, page/image limits, approved representative evaluation drawings, provider capability validation, durable queueing/retry, and explicit reproducibility/version policy. The automatic PoC persists only ECN-linked report JSON plus safe fixture metadata; it has no raw-drawing retention, general drawing history, or download capability.

## Validation

Run focused tests from the repository root:

```bash
PYTHONPATH=api python -m pytest -q tests/unit/test_drawing_feasibility.py tests/unit/test_app_lifespan.py
(cd ui && npm test -- --run tests/material-dialog.test.js)
```

These tests use synthetic byte signatures and fake services only; they do not call SAP AI Core or send a drawing externally.
