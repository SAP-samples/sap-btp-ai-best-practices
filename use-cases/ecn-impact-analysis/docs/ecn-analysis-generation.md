> **Current workflow:** See [Staged ECN impact analysis](ecn-staged-impact-analysis.md).
> The one-call final synthesis described below applies to jobs pinned to the
> previous prompt version. New explicit generation uses three parallel impact
> calls plus Data Readiness when issues exist, followed by one summary call.

# ECN analysis generation

## Purpose

The generation pipeline turns one immutable deterministic `EvidenceBundle` into one validated six-section ECN impact explanation. The LLM is a constrained renderer of admitted evidence: deterministic parsing, BOM matching, conditional-rule evaluation, and evidence identity remain outside the model.

The implementation first lets a source-only RDF discovery agent use its fixed tools until it requests no more tools or its bounded request window expires, then makes one grounded final-analysis call. There is no discovery iteration-count limit; every individual tool remains scoped and paginated. New jobs default to `gpt-5.4`. When `LLM_MODEL` selects another deployment, a provider timeout triggers one retry of the same model request using `gpt-5.4`; it does not replay tools or a completed outer agent step. Rate limits and provider 5xx responses retain bounded retries with exponential backoff and jitter. Authentication, authorization, missing-model, and schema failures are terminal.

## Inputs and outputs

Input is the evidence bundle persisted with one ECN occurrence plus the generation configuration pinned when its job was created:

- prompt and output-schema versions;
- model name and request timeout;
- maximum projected evidence items and Unicode characters;
- strict structured-output requirement.

New jobs default to `gpt-5.4` through `LLM_MODEL`. The selected primary model is pinned on publication, so changing the environment affects only newly created jobs; manual job retries keep the original job-local primary model. The timeout fallback is fixed at `gpt-5.4`.

`build_synthesis_projection()` keeps complete aggregates, fairly samples evidence categories, adds a bounded snapshot context, and caps the final call at 100 grouped records and 120,000 characters. Discovery can invoke only the fixed, server-scoped RDF tools; it cannot submit SQL/SPARQL or choose another ECN. Omitted groups and deterministic missing evidence force the affected section to return `INSUFFICIENT_EVIDENCE`.

The provider output must contain exactly these sections:

1. `executive_summary`
2. `change_intent`
3. `affected_materials`
4. `ebom_mbom_impact`
5. `configuration_scope`
6. `uncertainties_and_missing_evidence`

Each section contains a status, grounded Markdown assessment, same-section evidence IDs, and explicit missing evidence when incomplete. `NOT_APPLICABLE` remains visible and requires supporting evidence. A valid output becomes either `COMPLETED` or `COMPLETED_WITH_GAPS`.

## Evidence and inference boundaries

The model receives the canonical prompt projection only. It never receives source bytes, complete JSON objects, raw workbook rows, full conditional-rule artifacts, meeting notes, or other ECNs.

The system instruction prohibits deriving redesign from revision, replacement from co-occurrence, add/remove semantics from one-sided lists, material impact from document nesting, no impact from a missing BOM match, applicability from unknown rule facts, or Make/Buy from descriptions. `RULE_PLANT_FOOTPRINT` is only a rule-declared brand-scope hint; it is not active applicability, confirmed production plant identity, or a global-series conclusion.

## Provider and telemetry contract

`NativeImpactProvider` uses `gen_ai_hub.proxy.native.openai.AsyncOpenAI` with SDK retries disabled, `store=False`, the pinned per-request timeout, and a strict final JSON schema. A timeout on the primary model immediately makes one same-payload request to `gpt-5.4`; a timeout on that fallback is terminal. Rate limits and provider 5xx responses use bounded primary-model retries. RDF discovery and user chat apply the same timeout-only fallback to the individual model invocation. Their outer deadlines allow the fallback request to run. Returned incomplete responses, refusals, and missing output remain received provider calls so their usage can be persisted before application validation rejects them. Provider exception and refusal text never enters persistence, telemetry, or public errors.

The `btp.llm_usage.v1` stdout event records nullable provider usage and an explicit `usage_available` flag. It includes upload, job, attempt, correlation, provider, model, deployment, endpoint, outcome, latency, uncached input, cache-read input, cache-write input, inclusive input, output, and total counts. When fallback succeeds, usage is attributed to `gpt-5.4`, the model that returned the response, while the job retains its pinned primary model. It contains no prompt, evidence, source payload, pricing, currency, or rate data.

## Attempt lifecycle and persistence order

Every job is claimed together with a HANA `STARTED` attempt before this service runs. Processing is isolated per occurrence:

1. Build the bounded deterministic synthesis projection. A pre-call projection failure stores aggregate `NOT_CALLED`, emits no usage, and fails safely.
2. Run timeout-bounded RDF discovery and persist each completed model stage as `DISCOVERY_1`, `DISCOVERY_2`, and so on. Only source facts returned by tools survive; model prose is discarded.
3. Recheck RDF currentness, merge the complete discovery result into the final citation allowlist, and discard partial/stale discovery facts safely.
4. Call the final generator, allowing its provider adapter one `gpt-5.4` retry on timeout, and persist its `FINAL` child row using the next call number.
5. Aggregate per-stage token counts and latency into the existing attempt row. A received refusal, incomplete response, or malformed payload is still a provider `SUCCESS` followed by application validation `INVALID`.
6. Validate section and severity-factor citations, replace model-authored data readiness with deterministic readiness, then atomically store the immutable analysis and mark the job complete.

Provider errors, timeout, cancellation, invalid output, and database failures use fixed safe codes. Exception text and source content are never persisted or emitted. If provider-outcome persistence fails, validation and analysis insertion are skipped so an analysis can never refer to an attempt whose billable outcome is absent.

All synchronous HANA calls run through `run_blocking()`. Cancellation waits for the worker thread and its transaction to finish before propagating; this prevents lifespan shutdown from disposing the SQLAlchemy engine while a transaction is still active.

## Concurrency, scheduling, and restart behavior

One `GenerationRuntime` instance owns the API process's task registry and semaphore.
Direct summary batches and manual retries share the process-wide provider semaphore,
capped at two calls. A worker obtains a semaphore slot before claiming its job, so
excess work remains visibly `QUEUED` rather than appearing `RUNNING` while waiting
for provider capacity.

Import code publishes deterministic `READY` jobs and never schedules AI work.
`POST /api/summary-batches` accepts exactly one selection mode: explicit IDs, or
the active list predicates plus excluded IDs. Filter-wide membership is resolved
server-side with the list repository's predicates at submission time; pagination
and sorting are ignored, and the resolved IDs are persisted in the batch. Completed
jobs are queued again and their previous analysis is removed atomically;
already-running jobs are reused. Retry code schedules
the explicit job ID returned by the committed `FAILED` to `QUEUED` transition.
Duplicate scheduling is harmless because the HANA claim is an atomic
`WHERE STATE = 'QUEUED'` transition with a single winner.

Shutdown first prevents new scheduling, then cancels and gathers every tracked task. Cancellation-safe repository bridging waits for any active synchronous transaction, after which the shared provider is closed; the engine is disposed later by the application lifespan. At startup, the HANA repository changes leftover `QUEUED` and `RUNNING` jobs to `FAILED` with `PROCESS_RESTARTED`. Already completed analyses and terminal attempt usage are preserved. This accepted MVP recovery makes interrupted work visible and manually retryable without introducing a second queue database.

Manual retry is deliberately narrow: only an existing `FAILED` job without a stored analysis can move back to `QUEUED`. It preserves the job's pinned prompt, schema, model, and model configuration, then competes for the same process-wide provider capacity as newly uploaded work. Reads never claim, retry, or regenerate jobs.

## Related code

- `api/app/ecn/llm/synthesis.py`: bounded deterministic and discovered-evidence projection.
- `api/app/ecn/assistant/discovery.py`: timeout-bounded fixed-tool source discovery.
- `api/app/ecn/llm/two_pass.py`: shared projection records and hard final bounds.
- `api/app/ecn/llm/prompts.py`: legacy one-pass projection retained for historical adapter tests.
- `api/app/ecn/llm/provider.py`: native asynchronous stage adapter with bounded transient retries.
- `api/app/ecn/llm/validation.py`: response-status, schema, gap, and citation validation.
- `api/app/ecn/services/blocking_io.py`: cancellation-safe synchronous HANA bridge.
- `api/app/ecn/services/generation.py`: discovery, final, per-stage audit, aggregation, and completion sequence.
- `api/app/ecn/services/runtime.py`: shared concurrency cap and tracked background tasks.
- `api/app/ecn/services/retry.py`: guarded failed-only requeue transition.
- `api/app/main.py`: schema validation, restart recovery, container construction, and ordered shutdown.
- `api/app/routers/analysis_progress.py`: progress and explicit manual retry HTTP boundaries.
- `api/app/routers/workspace_actions.py`: explicit/filter-wide submission and durable batch APIs.
- `api/app/observability/llm_usage_logging.py`: provider-neutral usage normalization and stdout event.
- `api/app/ecn/analysis_models.py`: strict six-section schema.
- `api/app/persistence/analysis_repository.py`: immutable job, attempt, usage, and analysis persistence.

## Testing

Run the deterministic provider boundary tests without live credentials:

```bash
PYTHONPATH=api python -m pytest -q \
  tests/unit/test_prompt_projection.py \
  tests/unit/test_impact_provider.py \
  tests/unit/test_impact_validation.py
```

Live SAP Gen AI Hub tests are opt-in and marked `llm_live`; HANA integration tests are marked `hana_live`. Neither runs as part of the ordinary unit suite.

Attempt ordering and cancellation safety can be checked directly:

```bash
PYTHONPATH=api python -m pytest -q \
  tests/unit/test_blocking_io.py \
  tests/unit/test_analysis_attempts.py
```

Concurrency and shutdown races are barrier-driven rather than timing-only:

```bash
PYTHONPATH=api python -m pytest -q \
  tests/unit/test_generation_service.py \
  tests/unit/test_generation_restart.py
```
