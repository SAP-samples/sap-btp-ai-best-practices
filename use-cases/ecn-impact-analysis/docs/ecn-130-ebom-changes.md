# 130% EBOM rows in BOM changes

## Purpose

The BOM workbook carries the engineering BOM twice: the 150% sheet (`EBOM Q900V`)
and the raw 130% sheet (`EBOM 130% KM_Q900VPXA`). BOM changes used to read only the
150% sheet and the MBOM, so an ECN whose parts exist only in the 130% sheet showed
"no BOM change" even though its change number is in the workbook. The change builder
now also uses 130% material rows, but only where the 150% sheet has no row for the
same position.

## Inputs and outputs

- Input: parsed BOM rows already stored in HANA (no new table, no schema change).
- A 130% row is "raw-only" when it is a material-like part row (not a document, not a
  level-1 VC group header) with a parent, a component and an item, and no 150% row of
  the same upload has the same VC group, parent component, component and position
  (zero padding ignored: `0140` equals `140`). The VC group of a 130% row is the root
  of its hierarchy path; the `Main` column is not used.
- Output: raw-only rows behave like 150% rows in `build_bom_changes` (pairing, MBOM
  join, dates, part flags). Their positions are padded to four digits like the 150%
  sheet (`10` becomes `0010`). Their lines carry the qualifier `EBOM_130_SOURCE`, shown
  in the Notes column as "From the 130% EBOM only", and `sources[].kind` stays
  `RAW_EBOM_130`. Plants and `vc_groups` come from the conditional rules, as for any
  other row; `assembly` is the direct parent of the part. Object locations (flag
  `raw_only: true`) and program matches list them too.
- Not handled: document rows (an ECN found only on drawings still has no BOM change),
  quantity differences between the sheets, alternative items, the `Main` column.

## Why the 130% sheet is safe to use

On the real workbook, the 11,734 positions present in both sheets agree on change
number and start date in every case. Quantities differ on 897 of them (418 blank in
the 130% sheet), so the 150% row always wins and 130% quantities are flagged, not
trusted silently. 489 raw-only material rows exist; 60 carry a JSON ECN across 17
ECNs. Four ECNs exist only in the 130% sheet: ECN193101, ECN301489, ECN304579,
ECN307099. They now yield 6, 1, 2 and 2 added lines.

## Files

- `api/app/ecn/evidence/raw_ebom.py`: `raw_only_row_ids`, `promote_raw_only`.
- `api/app/ecn/evidence/bom_changes.py`: promotes raw-only locations, adds `EBOM_130_SOURCE`.
- `api/app/ecn/evidence/scoped_usage.py`: stamps `raw_only` on every location and keeps
  raw-only rows in object locations.
- `api/app/ecn/evidence/snapshot.py`: `program_matches` includes raw-only rows.
- `ui/src/components/bom-changes-table.js`: note text and empty-state wording.
- Tests: `tests/unit/test_raw_ebom.py`, `tests/unit/test_raw_ebom_snapshot.py`,
  `tests/unit/test_bom_changes.py`, `tests/integration/test_real_bom_changes_fixture.py`,
  `ui/tests/ecn-views.test.js`.

## Key code

```python
covered = {_twin_key(row) for row in rows if row.bom_kind is BomKind.ENRICHED_EBOM}
raw_only = frozenset(row.bom_row_id for row in rows
                     if row.bom_kind is BomKind.RAW_EBOM_130
                     and row.row_classification is BomRowClassification.MATERIAL_LIKE_OBSERVED
                     and row.parent_material_number and row.component_number and row.item_value
                     and _twin_key(row) not in covered)
```

```python
def build_bom_changes(ecn_number, locations, ...):
    # 130%-only material rows count as 150% rows from here on.
    locations = promote_raw_only(locations)
```

## How to test

```bash
PYTHONPATH=api .venv/bin/python -m pytest -q tests/unit/test_raw_ebom.py tests/unit/test_raw_ebom_snapshot.py tests/unit/test_bom_changes.py
PYTHONPATH=api DEVELOPMENT_FIXTURE_DIR=data .venv/bin/python -m pytest -q -m development_fixture tests/integration/test_real_bom_changes_fixture.py
cd ui && npx vitest run tests/ecn-views.test.js
```
