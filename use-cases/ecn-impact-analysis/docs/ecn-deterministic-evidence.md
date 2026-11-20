# Deterministic ECN evidence

This document defines the deterministic evidence prepared for one normalized ECN occurrence. The evidence layer preserves what the uploaded sources say, makes exact cross-source joins reproducible, and records unavailable conclusions as gaps. It does not decide engineering intent or invent links that are absent from the source data.

## Evidence tiers

| Tier | Meaning | What it does not prove |
|---|---|---|
| A | exact business ECN to raw EBOM `Change No.` row | material classification or production activity |
| B | exact technical WTPart where-used, including raw-component plus same-ECN enriched confirmation | replacement or plant validity |
| C | exact component occurrence with parent/path/effectivity context | direct ECN-to-MBOM change relation |
| D | a documentation row nested below its nearest material-like ancestor | that the owner material changed |

Tier labels express evidence strength and provenance, not business conclusions. In particular, an MBOM match is component presence only, and a document-owner bridge is structural context only.

## Exact join sequence

For one normalized ECN occurrence, matching proceeds in this order:

1. Trim external whitespace from the business ECN number and join it exactly and case-sensitively to raw EBOM `Change No.`.
2. From those raw rows, admit a component seed only when the row is classified `MATERIAL_LIKE_OBSERVED` and has a component number. Independently admit normalized JSON source objects only when they are validated technical WTParts with a number.
3. Join each admitted seed exactly and case-sensitively to enriched-EBOM component occurrences. A raw-seed occurrence is confirmed only when the enriched row also has the same ECN number; all other exact component hits remain context.
4. Join each admitted seed exactly to MBOM component occurrences. These results are presence evidence and never a direct ECN-to-MBOM relation.
5. For raw rows classified as confirmed documentation, follow normalized parent row IDs to the nearest `MATERIAL_LIKE_OBSERVED` ancestor. This records nesting without asserting that the ancestor changed.
6. Use an enriched occurrence's exact adjacent `Sub`/`VC` pair to select rules whose `Include` action contains that assignment. Traverse rule dependencies only forward, from a producer action to later conditions containing the same exact assignment. Selected rules are evaluated independently; one rule's output never mutates another rule's facts.

Identifiers are not case-folded, fuzzy-matched, zero-normalized, or otherwise rewritten. Document rows and unresolved raw rows never become structural seeds.

## Occurrence and citation identity

A BOM occurrence is identified by the complete tuple:

```text
(bom_kind, artifact_id, sheet_name, source_row_number, parent_path,
 item_or_position, material_number, parent_material_number, component_number,
 plant, alternative, valid_from_raw, valid_to_raw)
```

This prevents separate parent paths, positions, plants, alternatives, or validity intervals from collapsing into one result.

An evidence identity contains the section, kind, tier, sorted complete source coordinates, sorted related IDs, the occurrence tuple when applicable, and a short deterministic discriminator. Its prose summary is deliberately excluded, so copy edits do not invalidate citations. The public citation is `E-` followed by the first 16 lowercase hexadecimal characters of the canonical SHA-256. The builder retains the full identity for every generated citation and fails with `EVIDENCE_ID_COLLISION` if two different identities truncate to the same public ID.

A source coordinate contains `artifact_id`, `artifact_kind`, and the applicable JSON path or workbook sheet, source row, and normalized BOM row ID. Every source-backed item retains all contributing coordinates; only a synthetic missing-input item may have no source.

The bundle hash is the full canonical SHA-256 of the normalized ECN row ID, immutable `EvidenceContext`, and all six ordered evidence sections. Dates and enums use their stable string forms, decimals remain exact strings, mapping keys are sorted, and sequences retain their deterministic order.

## Six evidence sections

1. `EXECUTIVE_SUMMARY` contains the uploaded ECN occurrence header and complete normalized `description_text`. The latter preserves labeled `DescriptionSummary`, `Description`, and `SOLUTION_DESCRIPTION` prose so later evidence-only generation can cite the detailed intent; it is not replaced by the bounded display summary.
2. `CHANGE_INTENT` contains conservative affected/resulting candidate facts.
3. `AFFECTED_MATERIALS` contains each normalized source object and its identity quality.
4. `EBOM_MBOM_IMPACT` contains Tier A-D raw, enriched, MBOM-presence, and document-owner facts.
5. `CONFIGURATION_SCOPE` contains independently evaluated or explicitly unevaluated relevant rules, and available BOM scope fields.
6. `UNCERTAINTIES_AND_MISSING_EVIDENCE` contains matching diagnostics, contradictions, warnings, and ordinary missing inputs.

Partial results remain usable when another section has a gap.

## Change candidates

`AFFECTED_ONLY`, `RESULTING_ONLY`, `CO_OCCURRENCE`, and `REVISION_ONLY` are candidate labels. They do not mean remove, add, replace, or physical change. Co-occurrence is set-level evidence and does not create a Cartesian pairing between endpoints. Revision continuity is not proof of a physical redesign.

Version 1 emits no `EXPLICIT_SOURCE_RELATION`: the current source has no verified typed relationship containing two technical endpoint IDs. Free text, booleans, and provider exception text do not meet that contract.

## Counts

`AFFECTED_ITEM_COUNT` counts unique affected objects with a validated technical WTPart identity. `RAW_ECN_MATCH_COUNT` counts unique Tier A raw occurrences.

`EBOM_MATCH_COUNT` counts unique enriched occurrences reached through an eligible raw ECN component and corroborated by the same ECN number. `EBOM_CONTEXT_COUNT` counts other exact enriched occurrences. A JSON-only same-number occurrence may be Tier B evidence, but it is not a confirmed raw-to-enriched crosswalk count.

`MBOM_MATCH_COUNT` counts unique exact component presences regardless of how many raw or JSON seeds found them; it is not an ECN-to-MBOM link. `WARNING_COUNT` includes only warnings and contradictions, not normal missing inputs or not-applicable conclusions.

## Missing evidence and quality signals

The deterministic preparation can emit these ordinary gap codes without failing the whole ECN:

- ECN and source identity: `BUSINESS_ECN_MISSING`, `ECN_SOURCE_OBJECTS_MISSING`, `TECHNICAL_WTPART_IDENTITY_MISSING`, `WTPART_NUMBER_MISSING`.
- Exact matching: `RAW_ECN_NO_MATCH`, `RAW_COMPONENT_NUMBER_MISSING`, `EBOM_CONFIRMED_NO_MATCH`, `MBOM_COMPONENT_NO_MATCH`.
- Document hierarchy: `DOCUMENT_OWNER_NOT_FOUND`, `DOCUMENT_OWNER_IDENTIFIER_MISSING`.
- Rules: `RULE_ARTIFACT_MISSING`, `RULE_PATH_NOT_ESTABLISHED`; `NO_RELEVANT_RULE_MATCH` is a deterministic result rather than an error.
- Scope: `BOM_CONFIGURATION_SCOPE_MISSING`, `PLANT_SCOPE_UNAVAILABLE`, `ALTERNATIVE_SCOPE_UNAVAILABLE`, `EFFECTIVITY_SCOPE_UNAVAILABLE`, `ANALYSIS_AS_OF_DATE_MISSING`.

Source normalization and matching may additionally surface warnings or contradictions such as `TECHNICAL_IDENTITY_CONTRADICTION`, `ENRICHED_PATH_CONTRADICTION`, and `SUSPICIOUS_OR_OF_NEGATIONS`. Missing inputs are not counted as warnings.

Raw year-1900 validity values are retained verbatim and, when parsed, retain their ISO date alongside `MIGRATION_VALID_FROM_PLACEHOLDER`. They are migration placeholders, not true engineering-event timestamps and not warnings. When `analysis_as_of_date` is absent, BOM intervals remain visible but `is_active` stays unknown (`null`). Blank cells likewise remain unknown rather than false.

## Plant scope

Evidence bundles do not assign plants. `PLANT_SCOPE_UNAVAILABLE` is raised only
when the assessment pinned no rules workbook or the Plant Code Mapping frozen in
`EvidenceContext.plant_assignment` is empty. Plants themselves are derived in the
snapshot from the pinned rules and that frozen mapping; see
[Rule-based plant assignment](rule-based-plant-assignment.md).

## Explicitly disabled hypotheses

Version 1 does not normalize an enriched material suffix such as `_Q900` to another material number. It also does not map ECN-style change numbers to MPXA-style numbers. Those relationships are unsupported hypotheses until an authoritative crosswalk is supplied, so they cannot participate in a join, evidence tier, or count.

## Source-observed acceptance facts

`ECN345274` is a workbook-only validation case and is absent from the current corrected JSON. Its 34 raw rows split into 30 D-prefixed documentation rows and four material-like occurrences at positions 150/160; the positions are not type codes. Components `QKX463881A` and `QKX891416A` each repeat under parents `Q923VV2526401` and `Q923VV2526404`, leading to same-ECN enriched occurrences on rows 9311, 9312, 9484, and 9485.

`ECN126772` at current JSON ordinal 3 is the cross-source golden case. Resulting WTPart `495452W1` retains exact source version `6.3 (Design)`, joins exact raw row 2605 through parent-chain rows 2517, 2518, and 2604, and reaches same-ECN enriched row 1392. That target component has no exact MBOM occurrence. The complete ECN nevertheless has 213 unique MBOM component occurrences through its other validated WTPart objects, so its ECN-level evidence does not emit `MBOM_COMPONENT_NO_MATCH`. This distinction prevents a target-specific absence from being misreported as absence for the whole ECN.

Year-1900 valid-from values are retained legacy-migration placeholders. They are neither date errors nor true engineering-event timestamps, and no creation or expiration date replaces them.

## Related implementation and tests

- `api/app/ecn/evidence/identifiers.py` defines canonical candidate, evidence, and bundle hashes.
- `api/app/ecn/evidence/matching.py` defines exact indexes, occurrence identity, seed admission, and BOM matching.
- `api/app/ecn/evidence/change_candidates.py` defines conservative candidate grouping.
- `api/app/ecn/evidence/builder.py` selects relevant rules, projects all six sections, and computes counts.
- `tests/unit/test_evidence_matching.py`, `tests/unit/test_change_candidates.py`, and `tests/unit/test_evidence_bundle.py` exercise the contract.
