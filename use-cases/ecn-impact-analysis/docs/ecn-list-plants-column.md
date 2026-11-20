# Plants column in the ECN inbox

## What this is

The "Incoming ECNs" table on the ECN inbox has a **Plants** column right after
**Description**. It lists the plants affected by each ECN, one per line, using
the same labels as the **Plants affected** table on the ECN overview page
(`PL10 Plant A`, or the plant code alone when the name is unknown). Plants
are derived from the conditional rules and the Plant Code Mapping frozen at
import (see [Rule-based plant assignment](rule-based-plant-assignment.md)); the
"Plant not determined" row of the overview is not a plant and is not listed.

The inbox can also be **filtered** by plant (a "Plant" text box that matches
any part of a plant code or name, case-insensitively) and **sorted** by plant
("Plants" in the Sort list).

A row shows `—` when:

- its plants have not been assessed yet (tooltip: "Not assessed yet"), or
- the assessment linked no plant (tooltip: "No plant linked").

## Why it is built this way

The overview's plant table is not stored anywhere: `SnapshotService.get`
rebuilds it on demand from the ECN's BOM changes and pinned material master,
purchase info and stock rows (`build_plant_impact`). That takes several HANA
reads per ECN, which is too slow to repeat for every row of every list page.

The snapshot of one ECN occurrence (`ECN_ROW_ID`) never changes: its dataset
pins are fixed, and "Reassess with current data" creates a new occurrence.
So the plant labels are computed once and cached on the ECN row.

## Storage

| Table | Column | Type | Meaning |
|---|---|---|---|
| `ECN_IMPACT_ECN` | `AFFECTED_PLANTS_JSON` | `NVARCHAR(2000)`, nullable | JSON array of labels, e.g. `["PL10 Plant A"]`. `NULL` = not assessed yet; `[]` = no plant linked. |

The column is created automatically at API startup. New tables get it from the
contract in `table_contracts.py`; existing tables get it through the approved
additive `ALTER TABLE ... ADD` in `ensure_schema` (`schema.py`, `additive`
dict). It is the last column so both paths give the same column order.

Only labels are stored (no JSON keys), so a text filter cannot match field
names. Label lists longer than the column (about 90 plants) keep the leading
labels only; the overview page still shows every plant.

## When the labels are written

1. **Whenever a snapshot is built** (opening the ECN overview, summary
   generation, reviews): `SnapshotService.get` calls `_store_plant_labels`,
   an `UPDATE ... WHERE "AFFECTED_PLANTS_JSON" IS NULL`, so the value is
   written once and never overwritten.
2. **When the list shows a row that has no labels yet**: `GET /api/ecns`
   calls `_fill_affected_plants` (`routers/ecns.py`), which builds the snapshot
   of each such row on the returned page (at most `page_size` rows). A failing
   snapshot is logged and the row stays `—`; the page still loads.

Known limit: the first load of a page full of never-opened ECNs pays one
snapshot build per row. Later loads are a plain column read. If that first
load becomes too slow, fill the column in a background job after import.

## Filter and sort semantics

- `plant` filter: `UPPER("AFFECTED_PLANTS_JSON") LIKE '%<VALUE>%'` with a bound
  parameter. Rows not assessed yet (`NULL`) never match. This is intended: an
  ECN that does affect PL10 but was never assessed is excluded from a `PL10`
  search until its plants are known.
- `affected_plants` sort: orders by the stored JSON text, i.e. by the first
  plant label; `NULLS LAST` keeps unassessed rows at the end in both orders.
  Rows filled lazily during a page load were unassessed when the query ran,
  so they stay at the end of that page until the next load.
- "Select all matching ECNs" sends the plant filter along
  (`FilteredSummarySelection.plant`), so summary generation and deletion act
  on exactly the plant-filtered set.

## Inputs and outputs

- Input: `GET /api/ecns?plant=PL10&sort=affected_plants&order=asc`
  (`plant`: 1-80 characters).
- Output: each `EcnListItem` has `affected_plants: string[] | null`.
- Input: `POST /api/summary-batches` / `POST /api/ecns/delete` with
  `selection.mode = "filtered"` accept `selection.plant`.

## Files

| File | Role |
|---|---|
| `api/app/persistence/table_contracts.py` | `AFFECTED_PLANTS_JSON` column contract |
| `api/app/persistence/schema.py` | Additive column for existing tables |
| `api/app/ecn/services/snapshots.py` | `plant_labels`, `plant_labels_json`, `_store_plant_labels`, `affected_plants` |
| `api/app/routers/ecns.py` | `plant` parameter, `affected_plants` sort, `_fill_affected_plants` |
| `api/app/persistence/ecn_repository.py` | Filter, sort expression, projected column |
| `api/app/persistence/codec.py` | JSON to `affected_plants` in `ecn_list_item_from_row` |
| `api/app/ecn/api_models.py` | `EcnListQuery.plant`, sort literal, `EcnListItem.affected_plants` |
| `api/app/routers/workspace_actions.py` | `plant` in filter-wide selections |
| `ui/src/pages/ecns/ecns.{html,js,css}` | Column, filter box, sort option, `createPlantList` |
| `ui/src/modules/list-state.js` | `plant` URL state, `affected_plants` sort |

Key snippet (`snapshots.py`):

```python
def plant_labels(snapshot):
    plants = [p for p in (snapshot.get("plant_impact") or {}).get("plants") or [] if p.get("plant_code")]
    return [f"{p['plant_code']} {p['plant_name']}" if p.get("plant_name") else p["plant_code"] for p in plants]
```

## Deployment note

The startup schema check rejects extra columns. Once this build has added
`AFFECTED_PLANTS_JSON` to a HANA schema, an older build would fail to start
against that schema. Deploy this build to every app that shares the schema.

## How to test

```bash
PYTHONPATH=api .venv/bin/python -m pytest -q tests/unit/test_affected_plants.py tests/unit/test_ecn_repository.py tests/unit/test_hana_table_contracts.py
```

```bash
cd ui && npx vitest run tests/ecns-page.test.js tests/list-state.test.js
```

Live HANA (isolated, suffixed tables; simulates an old table, then checks the
upgrade, filter and sort):

```bash
RUN_HANA_INTEGRATION_TESTS=1 HANA_ADDRESS=<host> HANA_PORT=<port> HANA_USER=<user> HANA_PASSWORD=<pass> PYTHONPATH=api .venv/bin/python -m pytest -q tests/integration/test_hana_persistence.py -k plant_column
```

Manual: open the inbox; rows show plants (never-opened ECNs are filled on first
load). Type `PL10` in Plant, apply, and only ECNs with an PL10 plant remain.
