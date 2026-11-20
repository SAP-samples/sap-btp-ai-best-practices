# SAP BOM hierarchy behavior and UI implications

**Question.** Can a material hierarchy page legitimately show multiple EBOM and/or MBOM occurrences, and can their hierarchies differ?

**Short answer.** **Yes.** A production UI must treat a row as a *component occurrence in a specific BOM header and effectivity context*, not as “the hierarchy of a material”. Multiple alternatives, usages, plants, versions, effectivity intervals, parent assemblies, and planning-scope alternatives can produce distinct, valid hierarchy paths for the same material.

## Decision for this PoC

Do not normalize the currently displayed paths into a single hierarchy in the browser. The backend should eventually return an explicit, normalized occurrence DTO and row identity. The UI should render the returned hierarchy uniformly; it must not infer EBOM/MBOM semantics or configuration stages from the path shape.

## SAP findings

### 1. Multiple BOM alternatives are standard

SAP’s BOM model supports multiple alternatives for a material/plant/usage context. Per the SAP sources returned by Perplexity, alternatives may be selected by **lot size**, **explosion date**, or **production version**. Therefore, more than one BOM can be valid for a material context and the alternatives can have different component structures.

- SAP Help: [Selecting a Bill of Material Alternative](https://help.sap.com/docs/SAP_S4HANA_ON-PREMISE/34de0103497c4b80a7c7fbf6952ff971/c701b753128eb44ce10000000a174cb4.html)
- SAP Help: [Settings for Bill of Material Selection](https://help.sap.com/docs/SAP_S4HANA_ON-PREMISE/34de0103497c4b80a7c7fbf6952ff971/c101b753128eb44ce10000000a174cb4.html)
- SAP Learning: [Creating Multiple BOMs](https://learning.sap.com/courses/exploring-basic-data-for-manufacturing-and-product-management-in-sap-s-4hana/creating-multiple-boms)
- SAP S/4HANA Cloud Help: [Purpose](https://help.sap.com/docs/SAP_S4HANA_CLOUD/d5699934e7004d048c4801b552f3b013/8b76540d4fae4d119b9d74fe1c44a696.html)

### 2. Multiple MBOMs can be associated with one EBOM

SAP documentation cited by Perplexity describes planning-scope alternatives that can create **multiple MBOMs associated with a single EBOM**. It also allows planning-scope alternatives for subassemblies. Those alternatives can therefore produce different manufacturing paths for the same engineering component.

- SAP Help: [Prerequisites](https://help.sap.com/docs/SAP_S4HANA_ON-PREMISE/36802406aebb4b96b1598246e1d316ee/27700454543d1a5de10000000a44538d.html)
- SAP Learning: [Identifying BOM Types](https://learning.sap.com/courses/s-4hana-peo-foundational-setup-and-basic-processes/identifying-bom-types)
- SAP Help: [EBOM and MBOM Scenarios](https://ekg.cloud.sap/SAP/UA/HELP/DT/EBOMAndMBOMScenariosBb65200cca3b43448f30ed062e7ca800_5f9310ed8ebd4b88a5255a8fe0073b39_PLM_system_integration_for_SAP_S_4HANA_30_enUS_11)

### 3. Effectivity can differentiate otherwise similar paths

SAP’s parameter-effectivity guidance describes transferring effectivity from EBOM to MBOM and allowing MBOM component effectivity to be modified. A component can be allocated across multiple MBOM assemblies subject to quantity/effectivity rules. Therefore, dates alone may not be the full business identity for a BOM occurrence in a PEO/variant scenario.

- SAP Learning: [Understanding the Concept of Parameter Effectivity](https://learning.sap.com/courses/s-4hana-peo-advanced-production-engineering-processes/understanding-the-concept-of-parameter-effectivity)
- SAP Help: [How to Display BOM Maturity Levels](https://help.sap.com/docs/SAP_S4HANA_ON-PREMISE/9c4986bda35f4840ae438960ffbef64d/229162e753ff43f1b86c14fa1fca8a2c.html)

### 4. What this means for EBOMs

Whether the customer models more than one “EBOM” depends on its S/4HANA/PLM/PEO configuration and terminology. The standard evidence supports multiple BOM alternatives/usages and version/effectivity contexts; it does **not** justify assuming that every customer calls each engineering alternative a separate EBOM. Validate the target source system’s exact BOM header keys and usage codes before making UI labels or uniqueness claims.

## EKX MCP evidence

The SAP EKX MCP was reachable and its index returned SAP concepts and SAP Help/Enterprise Architecture Reference Library artifacts for:

- `multiple BOM`, `alternative BOM`, `BOM usage`, and `alternative BOM item`;
- `EBOM-MBOM Linkages`, `Create MBOM from EBOM`, `EBOM and MBOM Scenarios`, and `EBOM to MBOM transfer`;
- `BOM Selection Using Production Version`.

One SPARQL lookup directly verified the SAP Enterprise Architecture Reference Library solution activity **“Copy Engineering BoM to Manufacturing BoM”**:

- [EKX solution-activity record](https://ekg.cloud.sap/SAP/EARL/SAD/SA/CopyEngineeringBoMToManufacturingBoMERLF7938c68e3ef4afab8968153335e1a6a10)

The EKX index is useful corroboration and discovery metadata. The SAP Help and SAP Learning links above are the stronger sources for semantic/product claims.

## Current application assessment

### Current data path

1. `ui/src/services/workspace-api.js:12-14` calls `GET /api/ecns/{id}/object`.
2. `api/app/routers/snapshots.py:64-87` forwards the query to `SnapshotService.object_details`.
3. `api/app/ecn/services/snapshots.py:59-180` maps source `locations` into the UI-oriented `material_brief.linked_use` rows.
4. `api/app/ecn/evidence/snapshot.py:71-128` constructs each source occurrence with more information than the UI currently retains: occurrence ID, parent/path, item position, alternative, artifact, validity, and source coordinates.
5. `ui/src/components/material-dialog.js:24-38` then applies a browser-side visual rule: enriched EBOM paths suppress `Main`, `Sub`, and `Product context` nodes and derive `150%`/`130%`/`100%` labels from source levels; non-enriched paths are rendered whole and untagged.

### Current risks

The current `linked_use` projection and deduplication omit key BOM identity fields such as source occurrence ID, item position, BOM/header ID, alternative, scope/artifact, change-number range, and most raw effectivity metadata. Distinct valid SAP occurrences can therefore collapse if their current display fields are equal.

There is also a presentation mismatch: `RAW_EBOM_130` is displayed as `EBOM` in `material_brief`, but the UI applies the enriched EBOM tag/filter behavior only to `ENRICHED_EBOM`. Two rows labelled EBOM can therefore look different.

### Current linkage availability

The repository contains **no verified EBOM-to-MBOM relationship key today**. `BomRow` preserves a stable `bom_row_id`/`occurrence_id`, BOM kind, path, parent, item position, alternative, plant, dates, change references, and source provenance (`api/app/ecn/models.py:248-342`; `api/app/ecn/evidence/snapshot.py:82-126`). It does not carry an EBOM occurrence foreign key, an MBOM occurrence foreign key, a handover/planning-scope key, or a component-mapping status.

`build_snapshot()` groups enriched EBOM and MBOM records under the same component number and attaches every matching occurrence to that object (`api/app/ecn/evidence/snapshot.py:148-284`). That is an aggregate view of observed occurrences—not evidence of a paired relationship. The current `/object` endpoint is number-based, so it cannot request a selected occurrence independently (`api/app/routers/snapshots.py:45-68`).

**PoC-safe interim UI:** title the view **“Observed BOM occurrences”**, display separate EBOM and MBOM counts, and group/sort occurrences by kind, path/parent, item position, alternative, effectivity, and `occurrence_id`. Keep raw sheet/row provenance in audit details rather than the end-user table. Do not draw connectors, use an `EBOM → MBOM` label, or imply a match until the backend receives a source-provided linkage.

## Recommended production DTO

Return one row per **component occurrence in a BOM header/effectivity context**. Keep a stable machine key separate from display labels.

```json
{
  "occurrence_key": "stable-source-occurrence-key",
  "component": { "material_id": "…", "description": "…" },
  "bom_header": {
    "bom_id": "…",
    "header_material_id": "…",
    "plant_id": "…",
    "usage": "…",
    "alternative": "…",
    "version": "…",
    "category": "EBOM|MBOM|other"
  },
  "item": {
    "item_key": "…",
    "position": "…",
    "parent_material_id": "…",
    "quantity": "…",
    "unit": "…"
  },
  "effectivity": {
    "valid_from": "YYYY-MM-DD|null",
    "valid_to": "YYYY-MM-DD|null",
    "change_number": "…|null",
    "type": "date|parameter|serial|configuration|other|null",
    "value": "…|null"
  },
  "hierarchy": [
    { "node_key": "…", "material_id": "…", "item_position": "…", "configuration_stage": "…|null" }
  ]
}
```

### Minimum identity and display rules

- Use the server’s `occurrence_key` for rendering and deduplication; do not use a material number, description, or rendered path as identity.
- Preserve BOM header **usage**, **alternative**, **plant**, and **version** where the source system uses them.
- Preserve the parent/item position and full ordered path because the same component can appear in multiple places.
- Preserve the applicable effectivity dimensions. Date, change number, and parameter/configuration effectivity should not be silently merged.
- Render the path consistently as static evidence. Show a configuration-stage label only when the API explicitly supplies one.
- Make BOM identity discoverable in the UI: for example `MBOM · usage V · alternative 02 · plant PL10`, with optional values omitted only when the source does not provide them.

## Recommendation for the next UI decision

Keep the table at one row per dynamic occurrence/context. Add a compact **BOM identity** field or expandable row detail for usage, alternative, version, and effective interval. Do not attempt to force multiple EBOM/MBOM rows into one canonical hierarchy.

## Research method and limitations

- **SAP EKX MCP:** health check passed; indexed concept and SAP artifact searches were run for BOM hierarchy, EBOM/MBOM, alternatives, planning scope, parameter effectivity, and BOM selection.
- **Perplexity MCP (sonar-pro):** queried official `help.sap.com` and `learning.sap.com` material for BOM alternatives, EBOM/MBOM, planning scope, effectivity, plant, usage, and version.
- Direct readable fetches of three SAP Help pages were blocked because those pages are JavaScript-rendered. Claims above are therefore tied to the official sources surfaced by Perplexity and should be revalidated against the customer’s exact SAP release/customizing before production.
- No customer system, database, or BTP resource was accessed or changed.

## EBOM-to-MBOM linkage: how a UI should show it

### Research conclusion

In SAP Product Handover / PEO scenarios, the relationship is **not a visual similarity between two paths**. A handover explicitly identifies a selected source EBOM and a planned target MBOM. SAP Help describes planning-scope alternatives that can result in multiple MBOMs associated with one EBOM, including alternatives for subassemblies. A one-to-one UI assumption is therefore incorrect.

At component level, the relation may also be non-one-to-one: a source EBOM component can be allocated, split across, or consumed by multiple MBOM assemblies during manufacturing planning. Reconciliation checks the aggregated target quantity and parameter effectivity against the source item. The UI must therefore use the source-system handover/component mapping when it is available; it must never infer a connection simply because product IDs, descriptions, or hierarchy text happen to match.

### Required relationship data

For Product Handover / PEO data, add an explicit relationship object in addition to occurrence rows:

```json
{
  "handover_key": "stable-source-handover-or-planning-scope-key",
  "planning_scope_alternative": "…|null",
  "source_ebom": {
    "product": "…",
    "plant": "…|null",
    "usage": "…",
    "bom_number": "…",
    "alternative": "…",
    "version": "…|null"
  },
  "target_mbom": {
    "product": "…",
    "plant": "…|null",
    "usage": "…",
    "bom_number": "…",
    "alternative": "…",
    "version": "…|null"
  },
  "component_mappings": [
    {
      "source_occurrence_key": "…",
      "target_occurrence_key": "…",
      "mapping_status": "mapped|split|unmapped|mismatch"
    }
  ]
}
```

The exact field names vary with the customer’s product and integration APIs. The requirements are the durable handover/planning-scope relation and source/target component occurrence keys, not any assumed public SAP API name.

### Recommended single-table presentation

Use one table with **one row per verified EBOM-to-MBOM relationship**, not one row per isolated material occurrence:

| Source EBOM | Target MBOM | Planning context | Component mapping | Effective interval | Status |
|---|---|---|---|---|---|
| `Q900V` · Engineering · Alt 01 | `Q900V-PL10` · Production · Alt 01 | PL10 · Scope A | 24 mapped; 1 split | 19 May 2026– | Mapped |
| `Q900V` · Engineering · Alt 01 | `Q900V-PK01` · Production · Alt 02 | PK01 · Scope B | 23 mapped; 2 unmapped | 19 May 2026– | Needs review |

- Place a human-readable product/description first and compact technical identity second: `Usage`, `Alternative`, optional `BOM number`, and `Version`.
- Render an arrow (`EBOM → MBOM`) only for a verified relationship.
- Use an explicit `No verified handover link` status for EBOM-only or MBOM-only records; do not fabricate a relationship.
- Put source/target component paths in an expandable mapping detail or child table, rather than duplicating complete hierarchy paths in every relationship row.

### Default ordering and quick scanning

1. Group by **handover/planning-scope key**; this keeps one source EBOM and all of its target MBOM alternatives together.
2. Within a group, sort by **source EBOM product, usage, and alternative**.
3. Then sort targets by **plant, target alternative, production version, and effective-from date**.
4. Surface `Needs review`, `Unmapped`, and `Split` before fully mapped rows when the page is used for ECN impact triage.
5. Offer filters for `Mapped`, `Unmapped`, `Split`, `EBOM only`, `MBOM only`, plant, and effective date.

For a classic BOM scenario without Product Handover/PEO mapping data, retain the source and target BOM identities but present them as separate occurrences. Do not claim that an EBOM is connected to an MBOM until the source system provides a verified link.

### Sources for the linkage conclusion

- SAP Help: [Prerequisites](https://help.sap.com/docs/SAP_S4HANA_ON-PREMISE/36802406aebb4b96b1598246e1d316ee/27700454543d1a5de10000000a44538d.html) — planning scope and its alternatives; Perplexity’s SAP-only retrieval reports that several planning-scope alternatives can result in multiple MBOMs for one EBOM.
- SAP Help: [Creating a Handover](https://help.sap.com/docs/PLM_EPD/b183c303b8fc4254a08e565cdee4e163/4f2423a99f0c473687c5bb097588b7e5.html) and [Hand Over Engineering Bills of Material (EBOM)](https://help.sap.com/docs/PLM_EPD/b183c303b8fc4254a08e565cdee4e163/ac4414b0d8ab460cb0bd75d17c6861c5.html) — source EBOM and target MBOM handover context.
- SAP Learning: [Understanding the Concept of Parameter Effectivity](https://learning.sap.com/courses/s-4hana-peo-advanced-production-engineering-processes/understanding-the-concept-of-parameter-effectivity) — component allocation/restructuring and reconciliation of quantity/effectivity.
- SAP Help: [Data Used to Identify a BOM](https://help.sap.com/docs/SAP_S4HANA_ON-PREMISE/18ee18df146f46e9a7738186eebceaa7/4e05c453f57eb44ce10000000a174cb4.html) — BOM identity dimensions.
- SAP Help: [Production Version Selection](https://help.sap.com/docs/SAP_S4HANA_ON-PREMISE/21aead0c98bd4755abdacd91c99e3393/938bbf53f106b44ce10000000a174cb4.html) — production-version-based selection context.

The SAP Help pages are JavaScript-rendered and could not be directly extracted in this session. The links are first-party sources surfaced by a Perplexity query restricted to `help.sap.com`, `learning.sap.com`, and `api.sap.com`; validate exact availability and field names against the customer’s SAP release and customizing before production implementation.
