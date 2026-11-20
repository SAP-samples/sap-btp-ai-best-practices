# ECN Agent Discovery and Detail Loading

## Purpose

This feature lets the single-ECN chat and impact workflow investigate any
field present in the selected ECN's rebuildable RDF projection, shows which
tools the chat actually used, and removes the full-BOM read from cold snapshot
loading. It does not add generic SPARQL or a second source of truth. Cross-ECN
search was added later; see [Global ECN chat](ecn-global-chat.md).

## Inputs and outputs

Inputs are a route-selected ECN occurrence, its `CURRENT` RDF projection, the
pinned BOM artifact, and the existing deterministic evidence. The model never
supplies an ECN ID, graph, SQL/SPARQL, HANA object name, or file path.

Outputs are:

- exact RDF field names and example RFC 6901 paths from `list_ecn_fields`;
- exact source values from `find_ecn_facts`, with graph-node expansion only
  when context is needed;
- a sanitized per-chat `tool_trace` plus structured backend diagnostics;
- sequential `DISCOVERY_N` audit rows followed by one strict `FINAL` call;
- the existing snapshot shape, built from matching BOM rows and ancestors;
- independently rendered snapshot, object, review, and version panels.

## Agent workflow and boundaries

`inspect-ecn/SKILL.md` and `assess-impact/SKILL.md` are loaded into their
respective system instructions. For an unknown source property the required
workflow is:

```text
list_ecn_fields(query="request")
  -> find_ecn_facts(query="REQUESTER_NAME")
  -> expand_ecn_nodes(...) only if surrounding structure is needed
```

The tool docstrings describe inputs, outputs, pagination, examples, and absence
rules because LangChain exposes those descriptions to the model. Both chat and
impact discovery have no iteration-count budget. Chat stops at 300 seconds;
impact discovery stops at twice the job's pinned request timeout to allow one
fallback request after a timeout. Each tool response remains individually
bounded and server-scoped.

The chat trace contains only sanitized arguments, duration, success/error,
counts, pagination, truncation, and safe error codes. It excludes source values,
raw tool pages, prompts, credentials, SQL/SPARQL, file paths, and internal ECN
identifiers. Discovery emits the same diagnostic shape to backend logs and
persists every provider call with its sequence.

## Targeted BOM loading

`ArtifactRepository.snapshot_bom_rows()` uses fixed parameterized HANA SQL:

```python
rows = artifacts.snapshot_bom_rows(
    bom_artifact_id,
    referenced_component_numbers,
    ecn_number,
)
```

It selects enriched eBOM/MBOM rows for referenced component numbers and raw
eBOM material-like rows carrying the ECN change number. It then follows only
their trusted `PARENT_BOM_ROW_ID` chain in batches of at most 500. The existing
explicit projection converts NCLOB values inline and the existing codec decodes
every row. No `SELECT *`, workbook BLOB, complete BOM transfer, or persistent
process cache is involved.

No secondary HANA index was added speculatively. Add one only after a live
query plan on larger customer BOMs shows these predicates are the remaining
bottleneck.

## Parallel browser loading

`snapshotWorkspace()` starts the overview, first object page, version history,
and review history requests in the same synchronous turn. Each panel keeps its
own existing error handling and abort signal. Review saving still awaits the
overview promise so it can bind the exact evidence hash and snapshot version.

## Related files

- `api/app/ecn/assistant/tools.py` — fixed tool schemas and model-facing descriptions.
- `api/app/ecn/assistant/runtime.py` — timeout-bounded chat and sanitized trace.
- `api/app/ecn/assistant/discovery.py` — timeout-bounded impact discovery.
- `api/app/ecn/assistant/agent_skills/` — source-inspection workflows and examples.
- `api/app/persistence/rdf_repository.py` — fixed field-catalogue and graph reads.
- `api/app/persistence/artifact_repository.py` — targeted BOM and ancestor query.
- `api/app/ecn/services/snapshots.py` — shared targeted snapshot construction.
- `api/app/ecn/services/generation.py` — sequential discovery/final persistence.
- `ui/src/components/ecn-chat.js` — collapsed diagnostic trace.
- `ui/src/components/snapshot-workspace.js` — concurrent panel requests.

## Testing

Run the focused checks from the repository root:

```bash
PYTHONPATH=api .venv/bin/python -m pytest -q \
  tests/unit/test_rdf_repository.py \
  tests/unit/test_rdf_tools.py \
  tests/unit/test_impact_discovery.py \
  tests/unit/test_analysis_attempts.py \
  tests/unit/test_artifact_repository.py \
  tests/unit/test_snapshot_workspace.py
npm --prefix ui test -- tests/snapshot-workspace.test.js
```

The concurrency test holds the overview promise unresolved and asserts that
the object, version, and review requests have already started. Repository tests
assert server-side filters, ancestor retrieval, bound parameters, explicit
NCLOB projection, and the absence of a full-BOM fallback. No deployment is
part of this feature.
