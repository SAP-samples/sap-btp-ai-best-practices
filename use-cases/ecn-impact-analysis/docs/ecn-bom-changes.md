# ECN BOM changes and plant impact

The overview answers what changes in each assembly, when the change starts, and which plants need a check. The facts come from the assessment's pinned HANA datasets. AI explains these facts; it does not decide replacement pairs.

## Inputs and outputs

Inputs are the ECN JSON, dated engineering and manufacturing BOM rows, declared model-range scope, the conditional rules plus the Plant Code Mapping frozen at import, and material master, purchasing and stock records. Files enter through Settings or the manifest publisher; application requests read the normalized rows from HANA.

The snapshot and `/api/ecns/{id}/bom-changes` expose change lines and material flags. Each line contains the outgoing/incoming part, quantities, dates, engineering position, assembly (and its description when observed in the BOM hierarchy), rule-derived plants and VC groups, model range, alternative, BOM levels and exact source coordinates. One line may belong to several plants. The snapshot also exposes `plant_impact`, `programs`, `parts`, `reference_documents`, and `counts.attention_items`.

## Pairing and dates

Exact ECN codes work for every BOM kind, including synthetic IDs. Only MBOM rows can use an embedded ECN number. `MBOM_CHANGE_NUMBER_PATTERN` defaults to `^[A-Z]{4}(?P<digits>\d{6})[0-9A-Z]{2}$`; its named `digits` group is required. For example, `MPXA3452740A` maps to `ECN345274`.

Rows are paired within the same source scope, BOM kind, full parent, alternative and sequence. Engineering rows also retain their position. Rules are applied in order:

1. The same part ends and starts: show quantity or validity change.
2. The ECN provides a unique replacement pair.
3. The part numbers provide a unique revision pair.
4. Exactly one unpaired outgoing and incoming row remain in that group.

Every match must be unique in both directions. Repeated rows are retained. Ambiguous many-to-many changes remain separate additions/removals with an explanation.

Manufacturing changes are joined with engineering release rows for their positions and dates. The EBOM release and MBOM start remain separate dates. EBOM-only changes are supported and flag the missing manufacturing update. A released assembly can be established in MBOM through dated rows starting under that assembly.

Engineering rows come from the 150% EBOM. A 130% EBOM material row is used as an engineering row only when the 150% EBOM has no row at the same VC group, parent, component and position; such lines carry the qualifier `EBOM_130_SOURCE`. See [130% EBOM rows in BOM changes](ecn-130-ebom-changes.md).

A part is fully phased out only when it has a known finite end date, no later use remains in the pinned current BOMs, and their declared coverage is complete. This is a statement about the assessed BOM coverage, not every possible plant or program. Unknown timing stays unknown. Earlier current/comparison use prevents a `NEW_PART` flag. No historical use found is limited to available source history.

## Plant table and Readiness

Plants and their names come from the conditional rules and the Plant Code Mapping (see [Rule-based plant assignment](rule-based-plant-assignment.md)). A line counts for every plant of its VC group. The Plants & BOM tab gives one compact per-plant total before its single aligned BOM evidence table. `EBOM and MBOM aligned` requires both levels on a line; `Manufacturing BOM update pending` requires the explicit `MBOM_NOT_UPDATED` qualifier; all other combinations remain `BOM alignment needs review`. Material master, purchasing and stock rows count for a plant only when the rules place that material there; other SAP records are audit evidence. Lines or materials without a plant form a last "Plant not determined" row with the reason.

Green means dated BOM changes were found for that plant. Amber means a BOM update/date needs checking, the plant only has ECN objects without BOM changes, or no plant could be determined. Old-part stock is reported without inventing a unit. Missing records do not mean zero stock.

Readiness combines deterministic questions and grouped source warnings from the pinned evidence bundle. `counts.attention_items` is exactly the number of collapsible findings. The overview does not add the historical upload warning count. Part lists and BOM matches use complete snapshot membership, including BOM-only removals absent from the ECN JSON.

## Files and examples

- `api/app/ecn/evidence/bom_changes.py`: dates, engineering/manufacturing joins and flags.
- `api/app/ecn/evidence/bom_pairing.py`: unique row pairing without losing duplicates.
- `api/app/ecn/evidence/raw_ebom.py`: which 130% EBOM rows have no 150% twin, and their presentation as engineering rows.
- `api/app/ecn/evidence/plant_impact.py`: plant status, stock and purchasing.
- `api/app/ecn/evidence/snapshot_views.py`: complete parts, released versions and attention counts.
- `api/app/ecn/services/snapshots.py` and `api/app/persistence/artifact_repository.py`: pinned HANA retrieval, including all uses of changed parts.
- `ui/src/components/`: change lists, plant/BOM tables, Readiness and reference lines.

```python
changes = build_bom_changes(ecn_number, locations, lifecycle_changes,
                            coverage_complete=True)
plants = build_plant_impact(changes, reference_rows, source_json)
```

## Verification

```bash
PYTHONPATH=api .venv/bin/python -m pytest -q tests/unit/test_bom_changes.py tests/unit/test_plant_impact.py tests/unit/test_snapshot_views.py
DEVELOPMENT_FIXTURE_DIR=data PYTHONPATH=api .venv/bin/python -m pytest -q tests/integration/test_real_bom_changes_fixture.py
```

The optional real-file test also parses the real rules workbook. It covers ECN345274 replacements (positions 0150/0160, 2026-06-29 end, 2026-06-30 manufacturing start) in Plant A only, ECN346863 removals and quantity changes in Plant A plus the Plant B change in `Q900VV0080301`, inspectable BOM-only parts, PK01 purchasing for `QKX0187890` only, and five QKX2833570 units at PL10 (source unit unspecified).

## Date and size safeguards

An engineering/manufacturing replacement consumes both matching engineering
sides at the same position, preserving all source rows and the engineering end
date. It therefore produces one change instead of a duplicate engineering removal.
A part-wide new/fully-phased-out flag requires a date on every participating
start/end row. Mixed dated and undated rows stay `TIMING_UNKNOWN`.

Plant confirmation requires the endpoint for a standalone addition/removal and
both endpoints for a replacement, quantity change or validity change. Readiness
uses manufacturing usage on the assessment date, including assemblies observed
as MBOM parents: future introductions remain scheduled follow-ups.

`api/app/ecn/llm/bom_context.py` caps illustrative change, material and plant lists
at 25 each and samples across categories. The complete change totals and plant
status totals remain available, with explicit total/omitted counts. Extra
examples can be removed to meet the shared synthesis character budget; UI and
API records remain complete. `tests/unit/test_synthesis_bom_context.py` checks
150 additions across 150 plants and verifies that the input snapshot is unchanged.

If the ECN declares `PHASE_OUT` but its matched BOM rows only start the material,
the engine adds `DECLARED_PHASE_OUT`. The part is outgoing with status
`PHASE_OUT_NOT_CONFIRMED`, never a new part, and the plant stays amber. This is the
original synthetic feedback case: a BOM change code does not override the ECN's
conflicting phase-out intent. The dated BOM row remains visible for inspection;
no missing expiration date is invented.
