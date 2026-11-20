# Rebuildable ECN RDF Projection

## Purpose

The application keeps the imported ECN JSON in relational HANA as the system
of record. The RDF layer is a derived, disposable read model of the complete
JSON for one ECN occurrence. It enables bounded discovery of fields that were
not anticipated by fixed evidence projections, without copying relational BOM,
material master, stock, purchasing, or conditional-rule data into a graph.

Fact and field searches cover one ECN or several ECNs in one query (see
[Global ECN chat](ecn-global-chat.md)); node expansion and replacement inputs
stay per ECN. Authoritative replacement relationships are not included.

## Ownership and persisted data

| Layer | Owner | Contents | Recovery |
| --- | --- | --- | --- |
| Relational core | HANA tables | Source artifact, `ECN_IMPACT_ECN.SOURCE_JSON`, operational records, and `ECN_IMPACT_RDF_PROJECTION` status | Authoritative; imported normally |
| RDF read model | `urn:ecn-impact:graph:ecns` | Deterministic JSON-to-RDF triples for one ECN membership subset | Delete and rebuild from relational source |

`ECN_IMPACT_RDF_PROJECTION` stores the source hash, mapper version, five
structural baseline counts, lifecycle state, verified graph counts, bounded
safe failure fields, and attempt timestamps. It is not a duplicate source of
truth. It makes incomplete or failed graph work durable and recoverable.

## Import and lifecycle

The input is one validated `UploadPublication`, whose ECN occurrences each
already have canonical `source_json`. The output is one relational `PENDING`
status per newly committed ECN and a trusted ECN ID list for background work.

```text
relational artifact + ECN rows + PENDING status rows commit atomically
    -> FastAPI background task rebuilds trusted ECN IDs through one graph writer
    -> deterministic JSON-to-RDF mapping
    -> delete only that ECN membership, insert ordered N-Triples
    -> graph read-back verifies provenance, counts, array indexes, and isolation
    -> CURRENT or safe FAILED status
```

States are `PENDING`, `CURRENT`, `FAILED`, and `STALE`. A graph-dependent
operation is allowed only from `CURRENT`; graph tools return a safe HTTP 409
code such as `RDF_PROJECTION_PENDING`, `RDF_PROJECTION_FAILED`, or
`RDF_PROJECTION_STALE` otherwise. Detail, deterministic evidence, and the
baseline impact synthesis remain readable in every state.

At startup the application creates/validates the relational schema, then
probes fixed graph create/write/read/delete capability before becoming ready.
It starts one reconciliation task that selects only missing, non-current,
hash-drifted, or version-drifted records. Import, retry, and startup batches
share one process-local graph-mutation lock, matching the single configured API
instance while retaining per-ECN failure isolation. No separate worker queue is
used; use a database-owned lock before intentionally scaling the API beyond one
instance.

New ECN occurrences append to the shared graph. Rebuild deletes and replaces
only the selected occurrence's membership subset, never the whole graph. A
replayed request ID remains a no-op; separate source imports remain immutable
occurrences rather than being collapsed by mutable business metadata.

## Diagnostics and rebuild API

The detail page fetches status independently, so the existing ECN detail
payload and relational join remain stable.

- `GET /api/ecns/{ecn_row_id}/rdf-projection` returns one bounded status view.
- `POST /api/ecns/{ecn_row_id}/rdf-projection/retry` accepts no body and
  returns `202` after marking the selected record `PENDING`; it schedules a
  rebuild from stored `ECN.SOURCE_JSON`.

The page displays lifecycle state, mapper version, shortened source hash,
verified node/triple counts, a bounded safe error, and a retry action only for
`FAILED` or `STALE`. Relational detail, baseline synthesis, and non-graph chat
questions remain available while graph work is pending or failed.

## Security boundary

`RdfRepository` is the only HANA RDF boundary. It uses the fixed named graph
and server-built templates executed through `SYS.SPARQL_EXECUTE`. There is no
generic SPARQL endpoint, graph ORM, graph query DSL, client-selected graph,
client-selected HANA object name, or client-supplied source path. RDF literals
and descriptions are evidence, never instructions.

Retry routes receive only a route ECN UUID. Import and reassessment schedule
only ECN IDs returned by their committed relational transaction. HANA failures
are reduced to bounded safe application errors; raw source JSON and generated
SPARQL are not exposed in API errors or projection lifecycle logs.

## Single-ECN assistant discovery

Chat receives three additional read-only tools only after the selected
projection is `CURRENT`. They are closed over the route-selected ECN; the
model cannot choose an ECN, graph, SQL/SPARQL statement, database object, or
file path.

| Tool | Input | Output | Hard boundary |
| --- | --- | --- | --- |
| `list_ecn_fields` | Optional field-name text and page | Exact projected JSON field names, occurrence counts, subject types, and example RFC 6901 paths | Field catalogue only; no source values, at most 50 entries per page |
| `find_ecn_facts` | Predicate/value text, optional known node type or affected/resulting role, page | Ten field facts ordered by JSON pointer, with source artifact, source hash, mapper version, and continuation metadata | Case-insensitive lexical search with exact identifier matching; query text is trimmed and capped at 200 characters |
| `expand_ecn_nodes` | Root or graph nodes already issued during this chat turn, direction, up to ten field/structural relation filters, depth 1–2, page | Up to 100 adjacent source/structural facts, with the same JSON-path provenance and continuation metadata | At most five starting nodes; every node must belong exactly once to the selected ECN before traversal |

The repository has only fixed `SYS.SPARQL_EXECUTE` templates. It decodes
reversible field IRIs back to exact JSON keys, converts projected literals to
native JSON scalar values, uses deterministic ordering, and counts distinct
triples before pagination. Search and expansion return graph facts to a
per-turn in-memory collector for the later controlled impact-discovery stage;
ordinary chat does not persist them.

For a field fact, its returned JSON pointer is the exact field coordinate, not
only the containing object's pointer. For example, `REQUESTER_NAME` on the
root object returns `/REQUESTER_NAME`. RFC 6901 escaping is applied while
reconstructing that coordinate from the reversible field predicate.

The `inspect-ecn` skill directs broad questions to the relational overview,
unanticipated fields (such as `REQUESTER_NAME`) to graph search, nearby JSON
context to expansion, and exact identifier enrichment back to the relational
BOM/material tools. BOM observations remain Q900V-only and every fact must cite
its returned RFC 6901 JSON pointer.

Interactive chat preloads `inspect-ecn`; loading it is not a model-selected tool
step. A direct source-field question about a person, date, identifier, status,
title, owner, requester, or similar JSON property first discovers the exact key
with `list_ecn_fields`, then retrieves its value with `find_ecn_facts`. An
unestablished answer is allowed only after field discovery and exact fact
retrieval both return no match. This prevents spelling or naming variants from
making the fixed relational overview look like the complete source.

Chat has no model-call or tool-call count budget. A 300-second request timeout
allows a timed-out primary model call and one fallback call before termination;
every graph tool remains individually scoped,
paginated, and size-bounded. Each response includes a collapsed, sanitized tool
trace; backend logs carry the same call name, arguments, duration, outcome, and
count/pagination metadata without duplicating source values.

## On-demand replacement candidates

The single-ECN chat also provides `find_replacement_candidates`. Its inputs are
one fixed, selected-ECN RDF read of affected/resulting source objects (number,
name, description, object type, role, source path, enclosing link path, and a
non-null `OemReplacesAffected` scalar when present) plus the existing
relational snapshot's Q900V BOM locations. It never copies BOM rows into RDF.

The scorer considers a pair only if their exact part numbers share an observed
immediate Q900V parent. It then applies deterministic ordering: matching local
item/position when available, compatible source object type, replacement words,
description-token overlap, and stable RDF references. It exposes those signals,
conflicts such as incompatible type, multiple local alternatives, and divergent
extra parent context, exact JSON paths, and an integer rank. It does not expose
a probability or confidence score.

The graph repository returns source inputs in pages of 100. The tool refuses a
candidate result above 2,000 source objects instead of silently deriving a
partial ECN result. It returns at most ten candidates per requested page and
always labels the scope `Q900V only`.

Derived `replacementCandidate` results are computed on demand only: they are
not written to HANA tables or RDF triples and are never promoted to a
replacement mapping. A future non-null source `OemReplacesAffected` value is
returned separately as `authoritative_source_relation` with its enclosing link
RFC 6901 path; it is excluded from candidate ranking.

## Timeout-bounded impact discovery

Before the final impact-synthesis request, an explicitly generated ECN runs a
small tool-directed discovery step. The input is the fixed deterministic
synthesis projection: ECN header, aggregate snapshot counts, and the number of
already admitted evidence groups. It does not receive raw source JSON, other
ECNs, SQL/SPARQL, a graph identifier, file paths, or model-authored discovery
prose.

```text
CURRENT projection check
    -> DISCOVERY_1..N model calls and fixed graph-tool reads
    -> retry an individual model timeout once using GPT-5.4
    -> stop when the model requests no tools or twice the pinned request timeout expires
    -> source-only facts receive deterministic E- identifiers
    -> FINAL structured impact synthesis call
```

The assistant-local `assess-impact` skill restricts discovery to source facts.
It prohibits Make/Buy inference from descriptions, replacement inference from
affected/resulting co-occurrence, and scope claims beyond the Q900V BOM
population. The second discovery response's prose is ignored; only facts
returned by the fixed graph tools can reach the final model.

Each `GraphEvidenceFact` has a deterministic `E-` identifier derived from the
selected ECN, subject, predicate, literal/object, JSON pointer, source hash,
and mapper version. The final strict citation validator receives those exact
identifiers in addition to deterministic evidence IDs. It still rejects
fabricated, omitted, foreign-ECN, or non-returned citations.

Discovery has two terminal states:

- `COMPLETE` stores canonical source facts, their SHA-256 discovery hash, and
  every sequential `DISCOVERY_N` provider-call audit row.
- `FAILED` discards all partial graph facts and stores only a bounded error
  code and the hash of the empty evidence array. Baseline synthesis still runs
  from deterministic relational evidence.

The aggregate attempt's success follows the final synthesis response. A failed
discovery call remains visible in `ECN_IMPACT_PROVIDER_CALL` and contributes
to total token/latency usage, but it cannot invalidate a successful final
analysis.

`ECN_IMPACT_ANALYSIS` persists `DISCOVERY_STATUS`,
`DISCOVERY_ERROR_CODE`, `DISCOVERY_HASH`, and
`DISCOVERED_EVIDENCE_JSON`. This is an auditable cached projection, not a
source of truth: it can be recreated from relational ECN JSON plus a new
bounded discovery attempt.

## History reset

An idle history reset deletes `ECN_IMPACT_RDF_PROJECTION` rows before their
`ECN_IMPACT_ECN` parents in the same relational transaction. After that
authoritative commit, it best-effort drops only the application-owned named
graph. If graph cleanup fails, the completed relational reset remains valid;
missing status rows still prevent graph-dependent reads.

## Key implementation files

- `api/app/ecn/rdf/projection.py` — pure deterministic JSON-to-RDF mapper.
- `api/app/persistence/rdf_repository.py` — fixed HANA graph/status operations.
- `api/app/ecn/rdf/service.py` — project, verify, retry, gate, reconcile, clear.
- `api/app/ecn/assistant/tools.py` — bounded single-ECN graph search and expansion.
- `api/app/ecn/evidence/replacement_candidates.py` — deterministic on-demand candidate scorer.
- `api/app/ecn/assistant/discovery.py` — timeout-bounded discovery and source-fact identity.
- `api/app/ecn/assistant/agent_skills/assess-impact/SKILL.md` — discovery guardrails.
- `api/app/ecn/assistant/runtime.py` — per-turn graph-node/fact scope for chat.
- `api/app/ecn/services/generation.py` — discovery audit, final-call ordering, and completion persistence.
- `api/app/persistence/upload_repository.py` — atomic `PENDING` status insert.
- `api/app/routers/rdf_projections.py` — closed diagnostics/retry HTTP surface.
- `api/app/main.py` — capability probe and one startup reconciliation task.
- `ui/src/pages/ecn-detail/` — independently fetched status and retry display.

## Verification

Run the focused non-live checks from the repository root:

```bash
PYTHONPATH=api .venv/bin/python -m pytest -q \
  tests/unit/test_rdf_projection.py \
  tests/unit/test_rdf_repository.py \
  tests/unit/test_rdf_tools.py \
  tests/unit/test_replacement_candidates.py \
  tests/unit/test_rdf_service.py \
  tests/unit/test_impact_discovery.py \
  tests/unit/test_analysis_attempts.py \
  tests/unit/test_upload_repository.py \
  tests/unit/test_ingestion_service.py \
  tests/unit/test_upload_api.py \
  tests/unit/test_app_lifespan.py \
  tests/unit/test_history_repository.py \
  tests/unit/test_history_service.py \
  tests/unit/test_ecn_api_models.py \
  tests/unit/test_ecn_api.py
npm --prefix ui test -- tests/api.test.js tests/ecn-detail-page.test.js
```

The isolated live HANA check is opt-in and creates only UUID-suffixed tables
and a UUID-suffixed graph, which it removes in `finally`:

```bash
RUN_HANA_INTEGRATION_TESTS=1 PYTHONPATH=api \
  .venv/bin/python -m pytest -q \
  tests/integration/test_hana_rdf_projection.py -m hana_live
```

It passed against the prepared RDF-capable HANA instance on 2026-09-18 in
69.27 seconds. It verified repeatable schema bootstrap, capability probing,
two scoped projections, requester retrieval at `/REQUESTER_NAME`, rejected
cross-ECN nodes, failure-to-retry recovery, source preservation, and scoped
rebuild equality. HANA requires `SYS.SPARQL_EXECUTE` to run with DB-API
autocommit enabled; `RdfRepository` enables it only for a short-lived graph
connection and resets it before the connection returns to SQLAlchemy's pool.

The normal application lifespan also returned `/api/health` with HTTP 200
against that instance on 2026-09-18. It loaded `api/.env` through the
application bootstrap, created or validated the maintained unsuffixed tables,
and completed the graph capability probe without importing source data or
making a model call.

Do not deploy this application automatically; Cloud Foundry deployment is
manual and owned by the user.
