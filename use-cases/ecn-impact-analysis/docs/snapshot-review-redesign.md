# ECN snapshot review workspace

This implementation follows the approved snapshot-review plan, amended after
profiling all 557 ECNs: start with the objects explicitly referenced by an ECN.
Offer their observed BOM occurrences and navigable ancestors; do not start with
a presumed KMAT change. Source-declared type, observed depth, and inferred
ancestor context are separate facts. Affected/resulting does not prove add/remove.

## Inputs and outputs

Inputs are user-uploaded ECN JSON exports and versioned BOM, conditional-rule,
material-master, purchasing and stock workbooks. Runtime tabular data lives in
HANA. Importing must make zero LLM calls. Users open a fresh dialog, parse the
complete export in a short-lived browser worker, and upload only selected ECNs
in batches of at most 25. Summaries can be generated for explicit IDs or every
ECN matching the active filters; provider concurrency remains two. GET requests
never generate summaries.

Outputs are deterministic snapshot assessments, paginated referenced objects and
BOM locations, complete aggregate coverage, grouped questions, saved AI summaries,
and append-only human reviews. Summaries retain executive, manufacturing,
procurement, configuration and data-readiness explanations. Supported severity
is separate from incomplete coverage. The read-only sidebar shares assessment
facts and keeps conversation only for the browser session, separated by ECN.

Material-group descriptions and supplier names are primary display fields.
Commodity is optional. Stock quantities have source plant/storage coordinates;
missing rows are unavailable, not zero, and no obsolescence or common units are
inferred. Make/Buy and automatic deployment are excluded.

## Delivery checklist

- [x] New reference layouts and stock, including compatible HANA initialization.
- [x] Preview in a worker and import selected-only JSON batches without automatic AI.
- [x] Shared object-first assessment and hierarchy/reference navigation.
- [x] Explicit bounded AI batches, one synthesis call, version reuse and cancellation.
- [x] Session-only read-only LangGraph sidebar and snapshot-bound human reviews.
- [x] UI integration, regression checks, source-data validation and focused commits.

## Validation

Use `PYTHONPATH=api .venv/bin/python -m pytest tests/unit` and `npm test --prefix ui`
plus `npm run build --prefix ui`. New tests cover explicit selection boundaries,
zero automatic model calls, exact source coordinates, object/occurrence counts,
multiple depths, ancestor membership, reference-layout changes, saved review
provenance, chat isolation and partial-evidence summaries. Dataset profiling must
be read-only; live acceptance uses isolated HANA tables and bounded model calls.
Record completed checks and any remaining limitations here as work progresses.

## Implemented flow and boundaries

1. **Settings:** upload BOM/rules and relevant master, purchasing and stock snapshots.
   The new material-group description and supplier-name columns are nullable HANA
   additions; old layouts remain readable. Source statuses keep their original scope.
2. **Select and import:** the full JSON array stays inside a short-lived Web Worker.
   The dialog provides search, pagination, row selection, and whole-file
   Select/Deselect all. Only selected records cross the API boundary through
   `POST /api/uploads`, in original-order batches of at most 25. Every batch keeps
   the original filename and a stable idempotency UUID. Completed batches remain
   committed after a later failure; retry continues the failed and remaining
   batches with the same UUIDs. Jobs are stored as `READY` and are not scheduled.
   Closing or completing the dialog terminates the worker and forgets unselected ECNs.
4. **Inspect:** the primary table groups ECN objects by number and technical/document
   identity. Unnumbered objects receive a stable inspection key. Source object types,
   Affected/Resulting roles and actual observed depths remain separate. A second
   category exposes material-like raw BOM rows carrying the ECN number. These rows
   are not silently relabeled as direct JSON references. Exact ancestors link to
   the ECN's referenced objects under that ancestor, without selecting unrelated
   descendants. Material, supplier and stock records retain file/sheet/row provenance.
5. **Generate summaries:** submit directly from the inbox or ECN detail page with no
   confirmation screen. Selection is either explicit occurrence IDs retained across
   pages, or all IDs matching the active list predicates minus explicit exclusions.
   The server resolves filtered membership at submission time and persists the
   resolved IDs; pagination and sorting do not affect membership. Each ECN uses one
   structured synthesis call, complete aggregates and bounded representative evidence.
   A completed summary remains visible until a selected regeneration succeeds;
   in-flight work is reused.
   Queued cancellation is
   owned by its batch, so an old cancel cannot stop a later resubmission. Running
   calls finish normally. Provider concurrency is capped at four across summaries
   and assistant turns. Transient summary retries can increase actual usage.
6. **Ask about this ECN:** the read-only LangGraph sidebar shares the snapshot service.
   Tools inspect overview, objects, exact hierarchy, configuration, material master,
   purchasing and stock. They accept no ECN switch, SQL, file path or write operation.
   A turn has no model/tool iteration-count limit, but retains a 180-second
   request timeout, 3,000 output tokens per call and bounded tool/history input.
   Chat is browser-memory only, separated by assessment, and clears on reload.
   A collapsed sanitized trace and backend logs expose tool names, arguments,
   timing, outcome and result counts without conversation or source payloads.
7. **Review:** append an Accept ECN or Reject decision (see
   [engineering-review.md](engineering-review.md)), comments and self-reported reviewer name to the exact assessment ID, evidence
   hash and snapshot version. API identity is stored separately. Review statements
   never rewrite source facts or AI results. Repeated submission IDs are idempotent;
   conflicting payloads are rejected. A new assessment explicitly reuses the exact
   original ECN position with the current active datasets; earlier versions remain
   linked and their reviews/summaries remain unchanged. No AI is triggered by this action.

An observed High/Medium/Low impact can coexist with open evidence questions. An
unestablished dimension does not erase another supported dimension's severity.
The UI presents summary availability separately from data coverage and places
needed evidence next to each impact explanation. Historical result readers remain
compatible with previous severity/completeness semantics.

## Related implementation files

| Area | Files |
|---|---|
| Selected-only import | `ui/src/components/export-import.js`, `ui/src/components/ecn-import-worker.js`, `ui/src/modules/ecn-import-data.js`, `api/app/routers/uploads.py` |
| Shared snapshot | `api/app/ecn/evidence/snapshot.py`, `api/app/ecn/services/snapshots.py`, `api/app/routers/snapshots.py` |
| Synthesis | `api/app/ecn/llm/synthesis.py`, `api/app/ecn/services/generation.py`, `api/app/ecn/llm/validation.py` |
| Batch and review storage | `api/app/persistence/workspace_contracts.py`, `workspace_repository.py`, `api/app/routers/workspace_actions.py` |
| Assistant | `api/app/ecn/assistant/runtime.py`, `tools.py`, `skills.py`, `agent_skills/inspect-ecn/SKILL.md`, `api/app/routers/assistant.py` |
| UI | `ui/src/components/export-import.js`, `snapshot-workspace.js`, `ecn-chat.js`, `ui/src/services/workspace-api.js` |
| Reference layouts | `api/app/ecn/reference_data/parsers.py`, `models.py`, `api/app/persistence/reference_dataset_repository.py`, `schema.py` |
| Bounded database writes | `api/app/persistence/bulk.py`, `reference_dataset_repository.py`, `upload_repository.py` |

The assistant adapts the minimal `sap-ai4u-skills:create-langgraph-react-agent`
0.2.1 graph and full-text skill loader. It preserves `START → load_skill_node →
agent ⇄ tools → END`, with unused A2A/MCP/attachments omitted. The former planner
helpers remain only for legacy compatibility; production summary execution invokes
`build_synthesis_projection` and a single final stage.

Example HTTP actions (all require the configured API key):

```http
POST /api/uploads
Content-Type: multipart/form-data
request_id=<uuid>&ecn_file=<selected-only-json-batch>

POST /api/summary-batches
{"request_id":"<uuid>","selection":{"mode":"explicit","ecn_ids":["<assessment-uuid>"]}}

POST /api/summary-batches
{"request_id":"<uuid>","selection":{"mode":"filtered","query":"","ecn_number":"","analysis_state":null,"has_warnings":null,"excluded_ecn_ids":[]}}
```

## Acceptance evidence — 17 September 2026

- Backend offline suite: **801 passed, 6 opt-in live tests skipped**.
- UI unit/component suite: **43 passed**. Browser acceptance: **7 passed**, including
  the real 557-record ECN JSON, selected-only multipart serialization, direct
  generation, lazy evidence, retry/reset, duplicate identifiers, and reload safety.
- Production UI build passed; existing UI5 bundle-size advisory remains.
- Development-fixture suite: **7 passed** against the supplied source datasets.
- Read-only full-data parsing: **557 ECNs and 70,798 BOM rows**. Revised reference
  exports parsed successfully. ECN346863 has **9 directly referenced technical
  objects**, all located, plus **7 objects in material-like BOM rows carrying that
  ECN**. QKX7702260 is in the second group and exposes its supplied stock rows.
  ECN345274 has **6 directly referenced technical objects** and **2 BOM change
  objects**, preserving the difference between references and observed change rows.
- Database publication now passes lazy batches of at most 500 reference/BOM rows
  to the driver and one wide ECN NCLOB row at a time. These writes share the original
  transaction, preserving atomic publication while avoiding large parameter arrays.
  The selected ECN source NCLOB is explicitly hydrated inside its connection;
  the metadata-only detail reader cannot silently produce an empty object view.
- The API manifest now allocates **4 GiB**, as agreed with the user. The live test
  checks a configurable `SNAPSHOT_MEMORY_BUDGET_MIB` (default 4096). This is a measured
  single-process workflow check, not a concurrent-user capacity guarantee.
- `tests/integration/test_snapshot_workspace_live.py` creates only UUID-suffixed
  test tables, checks additive schema initialization, selects positions 26–50 plus
  the two representative ECNs in memory, imports selected-only 25-record batches,
  checks review replay
  and reassessment, and removes its test tables. It requires explicit live opt-in.
  `RUN_SNAPSHOT_LLM=1` additionally enables one summary and one bounded assistant turn.
- Isolated HANA selected-only acceptance: **1 passed in 279.30 seconds**. It parsed
  all **557** source records, imported **27** selected records in at-most-25-record
  batches, verified stock on the separately identified BOM-change object QKX7702260,
  saved/replayed an attributed review, and created a separate reassessment version.
  Peak RSS was **2,824.8 MiB (2.76 GiB)** under the configured 4 GiB budget. All
  suffixed test tables were removed; no AI summaries were generated and existing
  application tables were unchanged.
- Separate real-provider smoke test: **1 passed**. Synthetic evidence exercised one
  structured synthesis and a two-call LangGraph tool exchange: **3 model calls,
  4,611 total tokens**. This validates provider integration without sending customer
  source files. Customer acceptance of the generated conclusions remains a review task.

No deployment is performed. No bulk generation occurs as part of
import. Stock lacks supplied units/extraction date; snapshot membership does not
prove active production applicability, add/remove direction or obsolete stock.
