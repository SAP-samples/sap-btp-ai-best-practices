# Rule-based plant assignment

## What this is

Each ECN material is assigned to the plants that build it. Plants are no longer
typed when a workbook is uploaded. The application derives them from the
Windchill **conditional rules** and a Settings table called the **Plant Code
Mapping**:

```text
ECN material -> level-3 VC group (EBOM) -> rules that select the VC group
             -> brands in those rules (E_BRAND_JB) -> Plant Code Mapping (JB -> PL10 Plant A)
```

## Why

At the 28-09 customer touchpoint OEM confirmed three things:

- SAP extracts are not filtered per plant. For example, the material master
  holds every plant a material is set up in: 89 of 172 materials exist in both
  PL10 and PK01.
- The plant a VC group belongs to is decided by the conditional rules. Those
  rules live in Windchill and are not fully held in SAP.
- Plant selection happens at level 3 of the 150% EBOM (the VC group). Levels 1
  and 2 are common and can belong to several plants. Every component below a
  VC group belongs to that VC group's plants.

The first mapping supplied by OEM:

| Brand suffix | Plant |
|---|---|
| `JB` (Brand A) | PL10 Plant A |
| `IS` | PL10 Plant A |
| `JD` | PK01 Plant B (Plant B) |

When a rule names several brands, its VC group is relevant for every mapped plant.

## How the rules are read

The rules workbook is the `CONDITIONAL_RULES` dataset (sheet `Default`, 4,575
rules in the September extraction). It holds two families of rules.

**VC-group selection** (2,891 rules). Each rule includes exactly one VC group:

```text
(GEN_E_BRAND = E_BRAND_IS / GEN_E_BRAND = E_BRAND_JB) + GEN_E_FRONT_LINKAGE = E_F_LINKAGE_W
    --> Inclure (Q900VS1030 = Q900VV1030207)
```

- `Q900VS1030` is the level-2 slot.
- `Q900VV1030207` is the level-3 VC group.
- The VC groups belong to several model ranges: Q900V, Q938V, Q957V, Q932V,
  Q923V, Q947V and Q910V.

**Option derivation** (1,684 rules). These map sales options to engineering
options, for example `... --> Inclure (GEN_E_DOC_PACK = E_DOC_PACK_005)`. They
are indexed but never looked up as VC groups. This is the "full sales-option
logic", which is deliberately not modelled yet.

Every VC-selection rule names a brand. The brand is always a positive AND term:

- it is OR-ed only with other brands;
- it is never negated;
- it is never OR-ed with a non-brand term.

A rule's brands are therefore a necessary condition, and **a VC group's plants
are the union, over its admitted rules, of the plants mapped to their brands.**
Other terms such as gearbox, tyres or country narrow the configuration, not the
plant. Negated brands are ignored defensively.

**Lifecycle states.** The `Etat` column holds each rule's lifecycle state (App
For Series, In Work, Valid For Collaboration, Under Review, To be Obsolete).
Settings chooses which states are used. By default every state except "To be
Obsolete" is used; excluding it changes the plant set of 118 VC groups.

## Which VC group a BOM row belongs to

| BOM row | VC group(s) |
|---|---|
| 150% EBOM, level 3 or deeper | the `VC` column |
| 150% EBOM, level 2 (slot) | every VC group the rules can select for the slot |
| 150% EBOM, level 1 | none: common to all plants (`LEVEL_1_COMMON`) |
| 130% EBOM | root (`.1`) of the row's hierarchy path |
| MBOM | `Item` without its model suffix (`Q900VV0120800_Q900` -> `Q900VV0120800`) |

Rules decide the plants of **every** row, including the Plant A 130% EBOM and
MBOM sheets. A VC group valid for both plants therefore shows Plant A MBOM
cut-over dates under Plant B too.

## Inputs and outputs

### Inputs

- The pinned `CONDITIONAL_RULES` dataset (Settings upload).
- The Plant Code Mapping and the excluded rule states. They are read from HANA
  when an ECN is imported and **frozen** into its assessment
  (`EvidenceContext.plant_assignment`). Later Settings edits apply to new imports
  or to "Reassess with current data".
- The pinned BOM, material master, purchase info record and stock datasets. None
  of these declares a plant any more.

### Outputs

- Every snapshot BOM location has `plant_scope = {vc_groups, plants, reason, unmapped_brands}`.
- Every BOM change line has `plants`, `vc_groups`, `plant_reason` and `unmapped_brands`.
- `plant_impact.plants[]` has one row per rule plant, plus a last row
  `plant_code: null` / "Plant not determined" when something could not be
  placed. Each row carries `materials: [{number, description, role, vc_groups,
  reason, reason_label, unmapped_brands}]`.
- `plant_impact.records_outside_rule_plants` lists SAP records found at plants
  that the rules do not assign to that material.

### How master data is matched

SAP rows keep their own plant code. A material master, purchasing or stock row
counts under plant P only when the rules place that material in P, matched by
plant code. This is why a mapping row needs the SAP plant code.

A material's plants are those of the BOM change lines it takes part in, which
are the assemblies the ECN changes. When a material has no lines, its plants
fall back to its current BOM occurrences.

### Reasons when no plant is found

| Reason | Meaning |
|---|---|
| `NO_RULES_PINNED` | No rules workbook was active when the ECN was imported |
| `MAPPING_EMPTY` | The Plant Code Mapping was empty at import |
| `NOT_IN_BOM` | The material is not in the supplied BOMs |
| `LEVEL_1_COMMON` | Level-1 item, common to all plants |
| `NO_RULE_FOR_VC_GROUP` | No admitted rule selects the VC group |
| `NO_BRAND_IN_RULES` | The VC group's rules name no brand |
| `BRAND_NOT_MAPPED` | The brand has no mapping row (listed in `unmapped_brands`) |

## Settings: Plant Code Mapping

The Settings page shows **Plant Code Mapping** right below the procurement
status-code legend. It contains:

- an editable table: Brand suffix | Plant code | Plant name | remove;
- an **Add row** button;
- **Rule states used for plant assignment** checkboxes (checked means used);
- a **Save mapping** button.

Validation rules (the server returns a 400 with the message):

- A suffix contains only letters and digits (`JB`, not `E_BRAND_JB`).
- A suffix appears at most once.
- Plant code and plant name are required.
- One plant code has one name. Several suffixes may share a plant.

API (API-key protected):

```http
GET /api/plant-settings
PUT /api/plant-settings
{"mappings": [{"suffix": "JB", "plant_code": "PL10", "plant_name": "Plant A"}],
 "excluded_rule_states": ["To be Obsolete"]}
```

The response also returns `available_rule_states`, the states seen in uploaded
rule workbooks.

## ECN overview

- **Plant rollup**: one compact sentence names each affected plant, its
  assembly and component-change totals, and any BOM-alignment follow-up.
- **BOM change evidence**: one aligned table has an **Assembly** and an
  **Affected plants** column, because one line can belong to several plants.

## HANA tables

Both tables are created automatically by `ensure_schema` at API startup.

| Table | Columns |
|---|---|
| `ECN_IMPACT_PLANT_MAPPING` | `BRAND_SUFFIX` (PK), `PLANT_CODE`, `PLANT_NAME`, `UPDATED_AT` |
| `ECN_IMPACT_APP_SETTING` | `SETTING_KEY` (PK), `VALUE_JSON`, `UPDATED_AT`; key `excluded_rule_states` |

`ECN_IMPACT_REFERENCE_DATASET.SCOPE_CODE` / `SCOPE_NAME` stay in the schema but
are no longer filled. "Reset analysis history" keeps the mapping.

## Key code

```python
# api/app/ecn/rules/plant_scope.py
index = build_vc_index(rules, excluded_states=("To be Obsolete",))
resolution = resolve_row(bom_row, index, mappings)   # {"vc_groups", "plants", "reason", "unmapped_brands"}

# api/app/ecn/services/snapshots.py (per assessment)
plant_context = {"index": self._vc_index(rule_artifact_ids, excluded), "mappings": frozen_mappings,
                 "rules_pinned": bool(rule_artifact_ids)}
snapshot = build_snapshot(..., plant_context=plant_context)
```

## Related files

- `api/app/ecn/plant_settings.py`: mapping model, validation, default exclusion.
- `api/app/persistence/plant_settings_repository.py`: HANA read and atomic replace.
- `api/app/routers/plant_settings.py`: `GET` / `PUT /api/plant-settings`.
- `api/app/ecn/rules/plant_scope.py`: brands, VC-group index, per-row VC groups, reasons.
- `api/app/ecn/services/ingestion.py`: freezes the settings into each assessment.
- `api/app/ecn/evidence/scoped_usage.py`, `bom_changes.py`, `plant_impact.py`: plants per location, line and plant row.
- `api/app/ecn/services/snapshots.py`: rule index cache and plant context.
- `ui/src/pages/settings/plant-mapping.js`: Settings editor.
- `ui/src/components/plant-impact-table.js`, `bom-changes-table.js`: overview drill-down and Plants column.

## How to test

```bash
PYTHONPATH=api .venv/bin/python -m pytest -q tests/unit/test_plant_settings.py tests/unit/test_plant_scope.py \
  tests/unit/test_plant_impact.py tests/unit/test_bom_changes.py
cd ui && npm test -- --run tests/settings-page.test.js tests/ecn-views.test.js
```

Real-data acceptance uses the ignored files under `data/`, including the rules
workbook:

```bash
DEVELOPMENT_FIXTURE_DIR=$PWD/data PYTHONPATH=api .venv/bin/python -m pytest -q -m development_fixture \
  tests/integration/test_real_bom_changes_fixture.py
```

Expected results:

- **ECN345274** is Plant A only. `Q923VV2526401` and `Q923VV2526404` are
  selected only by JB rules.
- **ECN346863** is Plant A plus Plant B. Only `Q900VV0080301` has a JD rule, so
  its new part `QKX0187890` is the one Plant B change.
- `QKX7702260` has PK01 records, but its VC group `Q900VV0080402` is
  Plant A-only. Those records are listed as outside the rule plants.

## Known limits

- Only the simple brand-based model is implemented. The full sales-option ->
  VC-group logic is future work.
- 30 VC groups in the Plant A 130% EBOM are selected only by JD rules (29
  outright, one because its JB/IS rules are "To be Obsolete"). Rules win, so
  they are shown as Plant B. See
  [Plant A 130% EBOM: JD-only VC groups](plant_a-130-jd-only-vc-groups.md).
- The brand-to-plant key is only partially known (JB/IS/JD). Unmapped suffixes
  are reported, never guessed.
