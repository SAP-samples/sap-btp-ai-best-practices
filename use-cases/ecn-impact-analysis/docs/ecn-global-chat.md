# Global ECN chat

## Purpose

The ECN chat works on every page, not only on an ECN's detail page.

- **Global mode** (Incoming ECNs list and any page without an open ECN): the
  chat reads every stored ECN. It answers questions such as "which ECNs have
  high impact?", "which ECNs touch QKX7702260?", "which ECNs were requested by
  Ada?" or "compare the BOM changes of ECN306072 and ECN304940". It is
  read-only: to accept an ECN or send it back, the user opens that ECN.
- **ECN mode** (detail page): the chat answers about the open ECN, as before,
  and can still save an engineering review the user confirmed. It reaches other
  ECNs only when the user explicitly asks ("what ECN has the same materials as
  this ECN?").

Both modes use one tool set. The tools keep their names, so the assistant
skills (`inspect-ecn`, `assess-impact`, `review-decision`) need no change.

## Inputs and outputs

| Endpoint | Body | Mode |
|---|---|---|
| `POST /api/chat` | `{message, history}` | Global, read-only |
| `POST /api/ecns/{id}/chat` | `{assessment_id, message, history}` | Open ECN (unchanged) |

Both return `answer`, `consulted_objects` (objects of the open ECN, empty in
global mode), `consulted_ecns` (`[{ecn_number, assessment_id}]` whose data was
read; the UI renders them as links to their ECN pages), `tool_trace`,
`model_calls`, `correlation_id`, `review_saved`, and `assessment_id` (`null` in
global mode). Message and history limits are the same for both endpoints
(4000-character message, 12 turns, 24000 characters of history).

## How the chat reaches several ECNs

Every read tool has an `ecn` argument holding ECN numbers. Empty means the open
ECN. When no ECN is open, empty means "every ECN" for search tools and is an
`ECN_REQUIRED` error for detail tools. The server resolves numbers
(case-insensitive, newest upload when a number was imported more than once);
the model never supplies an internal ID, SQL, SPARQL or a file path.

Every tool result is fed back into the model and capped at 24,000 characters.
Search tools return one compact row per match, paged, so they need no ECN cap.
Detail tools return evidence pages per ECN, so they read at most five ECNs and
split the cap evenly.

| Tool | `ecn` | Behaviour across ECNs |
|---|---|---|
| `list_ecns` (new) | n/a | Stored ECNs with impact, plants, review decision, counts; filters: text, plant, impact, review |
| `find_ecns_with_part` (new) | list, empty = all | Every ECN touching one part number, with its role (named in the ECN, outgoing or incoming BOM side), changes, assemblies, plants, dates |
| `find_related_ecns` (new) | one, empty = open | Other ECNs sharing part numbers, ranked by shared count |
| `find_ecn_facts`, `list_ecn_fields` | list, empty = open or all | One SPARQL query over all selected ECN graphs; facts name their ECN, fields count their ECNs; ECNs without a current graph are listed in `not_current_ecns` |
| `get_impact_analysis`, `get_review_status` | list, no cap | Several ECNs return compact rows (severities / decision) |
| `get_ecn_overview`, `list_bom_changes`, `list_material_changes`, `get_data_quality_issues`, `inspect_object`, `list_referenced_objects` | up to 5 | `{"results": [...]}`, one bounded result per ECN |
| `inspect_hierarchy`, `inspect_configuration`, `expand_ecn_nodes`, `find_replacement_candidates` | one | Per-ECN data |
| `draft_review_decision`, `submit_review_decision` | none | Open ECN only; not offered in global mode |

Error codes the model can receive: `ECN_REQUIRED`, `ECN_NOT_FOUND`,
`TOO_MANY_ECNS`, `ECN_SCOPE_FIXED` (impact-analysis discovery stays on its one
ECN), `RDF_PROJECTION_NOT_READY`.

### Cross-ECN graph search

All ECN RDF projections share one HANA named graph; each ECN is a membership
root. One ECN keeps the original fixed root; several ECNs bind `?root` with a
`VALUES` list built only from canonical UUIDs:

```sparql
VALUES ?root { <urn:ecn-impact:ecn:...1> <urn:ecn-impact:ecn:...2> }
?subject <urn:ecn-impact:vocab:belongsToEcn> ?root .
```

### Part matching

`SnapshotService.part_index(ecn_row_id)` is a small cached summary of one ECN
(`api/app/ecn/evidence/part_index.py`): per part number its roles, change types,
assemblies, plant codes and dates. Only four full snapshots are cached, so the
matching tools read this summary instead.

## Related files

- `api/app/ecn/assistant/catalogue.py`: `EcnCatalogue`, ECN number resolution and target selection.
- `api/app/ecn/assistant/global_tools.py`: `list_ecns`, `find_ecns_with_part`, `find_related_ecns`.
- `api/app/ecn/assistant/tool_scope.py`: output budget, `_for_ecns` (multi-ECN shape and budget split).
- `api/app/ecn/assistant/tools.py`, `graph_tools.py`: the existing tools with the `ecn` argument.
- `api/app/ecn/assistant/runtime.py`: scoped and global prompt intros; `ainvoke(..., ecn_row_id=None)`.
- `api/app/routers/assistant.py`: `POST /api/chat`.
- `api/app/persistence/rdf_repository.py`, `api/app/ecn/rdf/service.py`: multi-ECN fact and field reads, `current_ids`.
- `api/app/ecn/evidence/part_index.py`, `api/app/ecn/services/snapshots.py`: part index.
- `ui/src/components/ecn-chat.js`, `ui/src/modules/shell-chat.js`, `ui/src/services/workspace-api.js`: global mode in the shell chat.

Key snippet, one tool body run for the ECNs a call names:

```python
async def run(row_id: str) -> dict:
    """Page one ECN's material changes."""
    result = await run_blocking(snapshots.changes, row_id, page=page, page_size=5)
    ...
return await _for_ecns(catalogue, ecn, run)  # empty ecn = open ECN
```

## Limits and when to change them

- The ECN catalogue (up to 1000 ECNs, 100 per SQL page) is read once per chat
  turn. Add server-side filtering if stored ECNs grow far beyond that.
- The first part search builds each ECN's snapshot once (the same cost as the
  list page's first plant fill). If that is slow, store the part summary on
  snapshot build, like `AFFECTED_PLANTS_JSON`, and query it with SQL.
- The graph `VALUES` list has one row per ECN; chunk it into several queries
  if stored ECNs reach many hundreds.
- Duplicate imports of one ECN number: the chat uses the newest upload.

## Testing

```bash
PYTHONPATH=api .venv/bin/python -m pytest -q tests/unit/test_global_chat_tools.py tests/unit/test_rdf_repository.py tests/unit/test_rdf_service.py tests/unit/test_rdf_tools.py tests/unit/test_ecn_api.py
npm --prefix ui test -- --run tests/snapshot-workspace.test.js
```

Manual check with the API and UI running: on Incoming ECNs open the chat and
ask "Which ECNs have high impact?", "Which ECNs include QKX7702260?" and
"Compare the BOM changes of ECN306072 and ECN304940"; the tool trace shows
`list_ecns`, `find_ecns_with_part` and one `list_bom_changes` call with both
numbers. Open ECN306072 and ask "What is the impact of this ECN?" (no cross-ECN
tool) and "What ECN has the same materials as this ECN?" (`find_related_ecns`).
