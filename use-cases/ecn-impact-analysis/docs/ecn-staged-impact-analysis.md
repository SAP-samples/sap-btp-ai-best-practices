# Staged ECN impact analysis and What needs attention

## Purpose

An ECN assessment now separates Manufacturing, Procurement, Configuration, and Data Readiness model work. The first four calls run concurrently against the same pinned evidence when data issues exist; the Data Readiness call is skipped when there are no issues to describe. A sequential Summary call writes the executive summary and five supporting narratives using their validated outputs. This keeps each perspective independently retryable and makes partial progress visible without weakening source grounding.

The page presents two related sets of follow-ups:

1. **Readiness:** numbered, deterministic issue titles, affected materials, source rows, and prescribed next steps. The Data Readiness call writes only one clear description per issue code. It cannot add, delete, reorder, or rename issues. If its output is absent or invalid, the deterministic description remains visible.
2. **Impact next steps:** complementary actions appear in expandable sections on their Manufacturing, Procurement, and Configuration impact cards. These calls return actions in `additional_next_steps`, separately from factor severity and `missing_evidence`. The Summary call does not add actions. What needs attention shows the shared Readiness list, including grouped source warnings.

The model must not infer missing units, validity dates, successor authorization, BOM placement, or active production applicability. Factor citations may refer to deterministic evidence or exact source-only RDF facts supplied in the stage prompt; both the stage and final validators enforce that same allowlist. Human Engineering review remains the place to record confirmed corrections and disposition decisions.

## Inputs and outputs

Input is the selected ECN's HANA-pinned evidence bundle, bounded snapshot projection, deterministic `snapshot.questions`, optional source-only RDF discovery facts, and job-pinned model configuration. Data Readiness receives each issue's code, title, deterministic explanation, affected object numbers, representative issue details, all distinct missing-field labels, and an omitted-object count when the source group is large. Full source rows remain in the snapshot and object APIs.

| Stage | Structured output | Validation |
| --- | --- | --- |
| Manufacturing / Procurement / Configuration | One cited severity factor and `additional_next_steps` for that perspective | Strict schema, projected citation IDs, one or two Markdown bullets |
| Data Readiness | One `{code, description}` for every deterministic issue | Strict schema and exact, unique issue-code coverage |
| Summary | Executive summary and five supporting narratives | Strict six-section schema, section-specific citations, required gap status, three labeled executive bullets plus an optional open-points bullet |

The service derives overall severity from the three business factors and derives the Data Readiness factor deterministically. It validates the combined response before writing the completed analysis. The public detail response carries the AI descriptions and perspective-specific actions. `GET /api/ecns/{ecn_row_id}/analysis-stages` returns only validated stage outputs compatible with the job's evidence hash, prompt version, and output schema, so the page can show partial progress.

If the latest job fails after saving compatible stages, the detail and inbox both label it **Partial analysis**. The inbox still exposes the underlying `FAILED` job state for filtering and counts; its `has_partial_analysis` field is computed from the same evidence, version, and accepted-attempt boundary as the stage endpoint. A previous accepted assessment may remain visible, clearly labeled as previous, until Summary and final validation complete.

## Lifecycle and recovery

New imports pin `ecn-snapshot-v4` and `ecn-impact-sections-v3`. Explicit generation upgrades an existing selected job to these versions, even when its last completed analysis used the previous prompt. A manual retry keeps the failed job's pinned versions. Historical v2 jobs therefore remain retryable through their original one-call provider contract.

Each successful stage result and provider-call audit row is stored in HANA against its job and attempt. If one parallel call fails, successful siblings remain available. A retry reuses only stage results with the same evidence hash and versions, calls the missing stages, then calls Summary. Provider calls share a process-wide concurrency limit of four; the job admission limit remains two. Import and read-only page requests make no model calls.

During explicit regeneration, the last accepted analysis remains visible and is labeled as previous while the new job runs or fails. Successful completion replaces it in one HANA transaction with the validated new analysis. A failed regeneration can be retried without removing the last accepted result. Clearing history deletes stage rows before their attempt parents.

## Related code

- `api/app/ecn/llm/staged.py`: stage schemas and evidence-bounded prompts.
- `api/app/ecn/llm/validation.py`: final factor citation validation using the same supplied evidence boundary.
- `api/app/ecn/services/staged_generation.py`: concurrent stage execution, audit, retry reuse, and sequential synthesis.
- `api/app/ecn/services/generation.py`: attempt lifecycle, aggregate telemetry, and final validation.
- `api/app/ecn/analysis_models.py`: final response, Data Readiness descriptions, and attributed next steps.
- `api/app/persistence/analysis_repository.py`, `table_contracts.py`: HANA stage rows and atomic completion.
- `api/app/persistence/ecn_repository.py`: list-level partial-stage availability for consistent status labels.
- `api/app/persistence/workspace_repository.py`: explicit regeneration with previous-result retention.
- `api/app/routers/snapshots.py`: validated partial-result read endpoint.
- `ui/src/components/assessment-context.js`: numbered deterministic issues with AI description fallback.
- `ui/src/pages/ecn-detail/ecn-detail.js`: partial results, previous assessment labeling, and expandable next steps on each impact card.
- `ui/src/pages/ecns/ecns.js`: list-level partial status while retaining failed-job filtering.

## Verification

Run `PYTHONPATH=api .venv/bin/python -m pytest -q tests/unit` and `npm --prefix ui test`. The staged attempt test verifies four concurrent calls, independent stage persistence, and a retry that calls only the failed perspective before Summary. Repository tests verify that regeneration retains an accepted analysis until the replacement transaction and that history reset deletes stage rows safely. Live HANA/provider acceptance and Cloud Foundry deployment are separate manual operations.

## Plain language (prompt v4)

Every stage and the chat assistant use the shared `STYLE` instructions from `llm/staged.py`. Use short engineering sentences, part numbers, dates, plants, and phrases such as new part, replaced by, removed, and quantity changed. Avoid internal vocabulary such as resulting object, occurrence, supplied scope and evidence in generated prose. State an unknown once.

The executive summary uses exactly **What changes**, **What is impacted**, and **Where**, with optional **Still open**. Provider JSON schema patterns enforce that shape and at most two factor bullets. Sentence simplicity is also checked during live acceptance; prompt instructions alone are not a factual guarantee.

The model receives `bom_changes` and `plant_impact`, including both engineering and manufacturing dates. Earlier analyses stay readable. Use Regenerate summary to create v4 output. Part/document lists are excluded from the model projection because their facts are already represented in the bounded source context.

A live v4 regeneration exposed a Manufacturing response with an invented citation ID. Stage prompts now carry explicit citation allowlists into the provider JSON schema: factor IDs are limited to the supplied records/header/RDF facts, and each narrative has its own list. Empty lists require an empty citation array. Server-side citation checks remain in place. The provider contract test verifies these constraints; a new regeneration exercises them against the configured model.

Live acceptance also exposed a summary that marked a required source gap as
sufficient. Required-gap sections now have a provider-schema status enum fixed
to `INSUFFICIENT_EVIDENCE` and at least one missing-fact item. Runtime validation
remains enabled. Factor cards additionally have a 550-character limit, using
counts and representative parts; prompts distinguish engineering release from
manufacturing cut-over and name new VC groups as new assemblies.

Explicit replacement clauses are normalized against directed `REPLACED` BOM
pairs before stage results are saved. This repairs the observed model inversion
“new is replaced by old” without changing severity or citations. The original
provider result is unchanged. Only known unidirectional pairs qualify; unrelated,
unpaired, bidirectional, hypothetical and question clauses are left alone.
Implementation: `api/app/ecn/llm/grounded_language.py`; regressions:
`tests/unit/test_grounded_language.py`. This guard is narrowly scoped to explicit
replacement direction and does not claim to validate all natural-language facts.
