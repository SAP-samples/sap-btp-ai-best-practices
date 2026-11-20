> **Current workflow:** See [ECN snapshot review redesign](snapshot-review-redesign.md).
> Registration/import now make no model calls; explicit generation uses one synthesis
> stage. The older flow below remains historical implementation context.

# Bounded two-pass ECN generation

## Purpose

This historical two-pass flow used `gpt-5.4` as an evidence-grounded renderer without sending the complete stored evidence bundle in one prompt. Current analyses default to `gpt-5.4`; the one-retry timeout fallback applies only when another primary model is configured.

## Inputs and outputs

The input is the immutable `EvidenceBundle` pinned to one ECN occurrence. Pass 1 receives only the ECN header, business-section counts, material keys, and compact grouped summaries. It returns a strict `EvidencePlan` containing one or more allowlisted slices and optional exact material identifiers.

The deterministic resolver translates that plan into pass-2 records. The output is a strict `AnalysisResponse` containing six narrative sections, manufacturing/procurement/configuration severity factors, separate data readiness, and deterministic overall impact.

Overall impact excludes data readiness:

- `HIGH` if any business factor is high.
- `UNKNOWN` if no factor is high and at least one is unknown.
- `MEDIUM` if none is high or unknown and at least one is medium.
- `LOW` only when all applicable business factors are low.

## Safety and bounds

- Planner input: maximum 40,000 characters.
- Final evidence: maximum 100 grouped records and 120,000 characters.
- Provider calls: maximum two per attempt (`PLANNER`, then `FINAL`).
- Resolver slices: material changes, BOM paths, plant rules, procurement context, and grouped data-quality evidence.
- No arbitrary SQL, unrestricted tools, source BLOBs, raw ECN payload, or full workbook rows are accepted from the planner.
- Oversized slices retain counts and representative grouped records; omitted groups become explicit evidence gaps.
- Every narrative section and business factor may synthesize across the grouped records included in pass 2. Citation validation accepts only internal IDs from that exact bounded projection, so fabricated, omitted, or foreign-ECN references are rejected without imposing the obsolete one-source-section-per-fact restriction.

Each initiated call is stored in `ECN_IMPACT_PROVIDER_CALL` with stage, safe HTTP/provider diagnostics, latency, deployment, retry hint, and nullable token usage. The existing `ECN_IMPACT_LLM_ATTEMPT` row stores the aggregate outcome and summed usage. Provider messages, headers, credentials, prompts, and response bodies are never persisted in diagnostic columns.

Retries are performed only for 429, timeout, and 5xx conditions, using bounded exponential backoff and jitter. Authentication, authorization, missing-model, and invalid-response failures are terminal.

## Data readiness

The final model may discuss data quality, but it does not control the stored readiness severity or overall impact. Before strict response validation, application logic replaces data readiness from missing-evidence groups, data-quality groups, and bounded omissions, validates the three model-authored business factors, and derives overall impact from the fixed rule. Data readiness is displayed separately and cannot reduce business impact severity.

The provider schema is normalized for strict OpenAI-compatible structured output: every object property is required, objects are closed with `additionalProperties=false`, and application compatibility defaults are removed. This provider-only normalization does not change the persisted Pydantic contract used to read historical analyses.

## Related files

- `api/app/ecn/llm/two_pass.py`: compact planner projection and allowlisted resolver.
- `api/app/ecn/llm/provider.py`: strict stage-specific Responses API calls and retries.
- `api/app/ecn/llm/validation.py`: response, citation, factor, and gap validation.
- `api/app/ecn/services/generation.py`: two-stage orchestration and usage aggregation.
- `api/app/ecn/analysis_models.py`: severity factors and fixed overall-impact logic.
- `api/app/persistence/analysis_repository.py`: parent-attempt and provider-call persistence.
- `api/app/persistence/table_contracts.py`: `ECN_IMPACT_PROVIDER_CALL` schema.

## Testing

Run the focused deterministic tests from the repository root:

```bash
PYTHONPATH=api .venv/bin/python -m pytest -q \
  tests/unit/test_two_pass_generation.py \
  tests/unit/test_analysis_attempts.py \
  tests/unit/test_analysis_repository.py \
  tests/unit/test_impact_provider.py \
  tests/unit/test_impact_validation.py \
  tests/unit/test_analysis_contract.py
```

Run the in-memory end-to-end workflow without credentials:

```bash
PYTHONPATH=api .venv/bin/python -m pytest -q \
  tests/integration/test_ecn_workflow.py
```

Live model acceptance remains opt-in and requires a successful provider preflight for the exact configured primary deployment.
