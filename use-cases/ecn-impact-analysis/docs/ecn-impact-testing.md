# ECN impact testing

## Purpose

The acceptance suite proves the ECN workflow at three increasingly external
boundaries. The default suite is deterministic and free of HANA and model
credentials. Development-fixture checks read the supplied source files without
copying them into application logic. Live checks are opt-in because they use
customer-connected infrastructure and may make a billable model request.

## Default non-live workflow

`tests/integration/test_ecn_workflow.py` sends multipart requests through the
real FastAPI routes and uses the production JSON/workbook parsers, evidence
derivation, ingestion/history services, generation runtime, structured-output
validator, retry behavior, and response mapping. Only persistence and the
external provider are substituted:

- `MemoryWorkflowStore` and its locked adapters emulate the HANA repository
  contracts. They do not parse sources or derive domain facts.
- `ControlledImpactProvider` returns evidence-scoped strict JSON and can hold or
  fail selected calls deterministically.
- No SQLite database, local CSV/Excel runtime store, external service, or second
  event loop is used.

Run it from the repository root:

```bash
PYTHONPATH=api .venv/bin/python -m pytest -q tests/integration/test_ecn_workflow.py
```

The scenarios cover first publication, immediate evidence visibility, bounded
parallel generation, immutable stored-result reuse, BOM reuse/replacement,
duplicate and missing business identifiers, completed-with-gaps output,
sibling failure isolation, failed-only retry, active reset protection, rule
preservation, restart recovery, and same-ECN evidence authorization.

## Browser workflow

The Playwright acceptance suite uses a stateful same-origin API double whose
list, detail, progress, retry, upload, and reset routes share one row-state
collection. This proves the full user journey and HTTP contract without HANA
or provider calls:

```bash
npm --prefix ui run test:e2e -- ecn-workflow.spec.js
npm --prefix ui run test:e2e -- shell.spec.js
npm --prefix ui run build
```

The workflow covers the multipart ECN/BOM upload, immediate deterministic
evidence while generation is running, stored six-section output, controlled
evidence inspection, duplicate/missing business IDs, failed-only retry,
query-state restoration, and protected idle-only reset. Completed detail reads
are also asserted to issue no generation mutation.

The Cloud Foundry manifest pins the API to one instance and one Uvicorn process
because scheduling is process-local in this MVP. It allocates 1 GiB memory,
sets the shared model concurrency to four, and records the accepted upload,
workbook, prompt, timeout, and `gpt-5.4` model defaults. These settings are
configuration only; test execution never deploys the application.

## Supplied development files

`tests/integration/test_development_fixtures.py` is marked
`development_fixture`. It reads source files only from
`DEVELOPMENT_FIXTURE_DIR`, defaulting to the ignored repository `data/`
directory. A missing file means the tests skip; it does not mean the source
contract passed.

```bash
DEVELOPMENT_FIXTURE_DIR='/absolute/path/to/data' \
  PYTHONPATH=api .venv/bin/python -m pytest -q -m development_fixture
```

These checks lock the observed source facts: 35 corrected ECNs, the retained
JSON syntax defect at line 318816/column 1, the ECN345274 workbook-only
crosswalk, the ECN126772 raw/enriched hierarchy and component-specific MBOM
gap, preservation of 1900 migration dates, 4,575 parsed conditional rules and
6,081 assignments, and the three-role workbook parser staying below a peak
850 MiB RSS in an isolated child process.

Application code never opens these fixture paths. Production input arrives by
browser upload and is persisted in HANA.

## Live provider and HANA checks

The provider-only smoke uses the configured model (default `gpt-5.4`),
validates all six sections, checks usage availability, and passes returned usage
fields to the structured logger:

```bash
RUN_LLM_INTEGRATION_TESTS=1 \
  PYTHONPATH=api .venv/bin/python -m pytest -q -m llm_live \
  tests/integration/test_provider_live.py
```

This is a provider/schema/logger plumbing check. Phase 03 unit tests remain the
authoritative proof that every production generation outcome emits exactly one
payload-free telemetry event with response-endpoint, correlation, cache, input,
output, and provider-total fields.

The full live acceptance creates UUID-suffixed HANA tables, loads the supplied
rules, uploads the real JSON and BOM through FastAPI, performs one real model
analysis, verifies HANA/API readback, and drops only its isolated tables:

```bash
RUN_FULL_LIVE_TESTS=1 \
  PYTHONPATH=api .venv/bin/python -m pytest -q -s \
  tests/integration/test_generation_api_live.py
```

The API loads `api/.env` with `python-dotenv`; do not source that file in a
shell. A skipped live test means its explicit opt-in flag or credentials/files
were unavailable, not that connectivity was verified.

The RDF-only HANA acceptance does not make a model call. It creates only a
UUID-suffixed relational namespace and graph, then exercises the application's
projection service, fixed SPARQL reads, verification failure handling, retry,
and cleanup:

```bash
RUN_HANA_INTEGRATION_TESTS=1 PYTHONPATH=api \
  .venv/bin/python -m pytest -q \
  tests/integration/test_hana_rdf_projection.py -m hana_live
```

This command passed on 2026-09-18 in 69.27 seconds against the prepared
RDF-capable HANA instance. It also caught and led to fixes for HANA's required
SPARQL autocommit mode, HANA's `typed-literal` response form, and exact
field-level JSON Pointer evidence. The test removes only its UUID-scoped graph
and tables. A separate normal application-lifespan check returned `/api/health`
with HTTP 200 after loading `api/.env`, creating or validating maintained
unsuffixed tables, and completing the RDF probe; it did not upload data or call
a model.

## Default regression command

The normal regression suite excludes all credentialed and customer-file gates:

```bash
PYTHONPATH=api .venv/bin/python -m pytest -q \
  -m "not hana_live and not llm_live and not development_fixture"
```

## Approved acceptance scenarios

| # | Approved scenario | Required verification |
|---:|---|---|
| 1 | Corrected JSON plus workbook publishes history/jobs | `test_first_upload_exposes_evidence_then_stores_and_reuses_one_analysis`; opt-in fixture parser |
| 2 | Malformed copy rejected atomically with location | `test_corrected_and_original_malformed_json`; ingestion atomicity unit test |
| 3 | First workbook required; later snapshot reused | `test_reuse_then_replace_bom_keeps_old_evidence_hashes`; ingestion service unit test |
| 4 | Replacement snapshot does not mutate old evidence/results | `test_reuse_then_replace_bom_keeps_old_evidence_hashes` |
| 5 | D-document rows inspectable but non-seeding | Development-fixture ECN345274 assertions; evidence-matching unit test |
| 6 | Dotted hierarchy reconstructed before filtering | BOM hierarchy unit test; development-fixture raw path assertion |
| 7 | Year-1900 value preserved and qualified | Effectivity unit test; development-fixture `02.01.1900` assertion |
| 8 | Exact trimmed joins only; no fuzzy/zero rewrite | Evidence-matching unit test; development-fixture ECN/component rows |
| 9 | Duplicate business ECN IDs create separate rows/routes | Integration duplicate test; Playwright duplicate-route test |
| 10 | Missing business ID remains navigable | Integration missing-ID test; Vitest/Playwright missing-ID assertions |
| 11 | Process-wide provider concurrency and sibling isolation | Configurable semaphore barrier tests; the application currently clamps active provider calls to two |
| 12 | Deterministic evidence visible before AI completion | Analysis-section/detail Vitest; Playwright pending assertion |
| 13 | Mixed statuses save/render as completed with gaps | Analysis contract unit test; integration mixed-state test; Playwright workflow |
| 14 | Unknown/cross-ECN evidence rejected | Impact-validation unit test; integration cross-ECN test |
| 15 | Summary/warnings defaults and N/A behavior | Analysis-section Vitest; detail Playwright assertions |
| 16 | Restarted unfinished jobs fail and can retry | Generation restart unit test; integration restart test |
| 17 | Retry only failed; completed returns 409 | Analysis API unit test; integration and detail 409 tests |
| 18 | Markdown tables contained; malicious content removed | Markdown Vitest; narrow-width shell/detail Playwright |
| 19 | Active reset rejected; idle reset preserves tables/rules | History unit/integration tests; Settings Vitest/Playwright |
| 20 | HANA create/validate/incompatible/privilege behavior | Schema unit tests; uniquely suffixed full live test — `LIVE_PASS_2026-09-15` |
| 21 | Every provider attempt has correlated token telemetry, no rates | Generation-service event-capture/token tests; full live one-call provider/logger path — `LIVE_PASS_2026-09-15` |
| 22 | RDF projection stays rebuildable, scoped, and source-grounded | `test_hana_rdf_projection_rebuilds_without_cross_ecn_leakage` — `LIVE_PASS_2026-09-18` |

Current non-live evidence includes 877 backend tests, 47 UI unit tests, eight
Playwright tests, Python compilation, and the Vite production build. A
production-dependency npm audit reports zero known vulnerabilities after
pinning the retained router's transitive
`path-to-regexp` dependency to the patched 1.9 line. The supplied
development-fixture suite also passed all six checks.

The explicit full-live command passed on 2026-09-15 in 122.04 seconds. It used
UUID-suffixed HANA tables, uploaded 35 real ECNs, made exactly one native
Responses call to `gpt-5.4`, verified persisted/API readback, emitted one
`btp.llm_usage.v1` event, and removed its isolated tables. The measured call
reported 16,945 input tokens, 1,062 output tokens, and 18,007 total tokens with
14,079 ms provider latency. The provider-only smoke was intentionally not run
as a second call because the fuller HANA-plus-provider acceptance already
proved the same live boundary. Mocks and in-memory adapters never count as live
HANA or provider evidence.
