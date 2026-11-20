# Plants & BOM Story Assessment

## Scope

This report assesses every tab-specific section visible while **Plants & BOM** is selected. It applies question 2 of the [ECN Demonstration Story](ecn-demonstration-story.md): **Where and to whom does it matter?** It also retains BOM transition facts only where they substantiate that answer.

It does not reassess the shared ECN header, summary-generation control, tab navigation, or decision footer. Those are page-level controls assessed in the [Overview Story Assessment](ecn-overview-story-assessment.md).

**Implementation status:** The Plants & BOM target below is implemented. The detailed tables record the prior content decisions and remain the acceptance criteria for future changes.

**Decision labels:**

| Decision | Meaning |
| --- | --- |
| **Keep** | Directly answers the affected-plants/BOM question or provides a bounded drill-down for it. |
| **Revise** | Has the right evidence but leads with technical detail, counts, or identifiers rather than a clear answer. |
| **Defer** | Valid traceability or operational evidence, but not the Plants & BOM narrative. |
| **Remove** | Does not support the selected ECN story on this tab. |

## Evidence Boundary

Plant names are derived from the imported conditional rules and frozen plant mapping. A green plant state means dated BOM changes were found; it does **not** by itself prove that the plant manufactures every listed part or that EBOM and MBOM are aligned. An amber state means the BOM, date, or plant assignment needs checking.

The tab must therefore distinguish three things in plain language:

1. The named plants and assemblies in the ECN's assessed scope.
2. Whether each affected assembly has an EBOM/MBOM alignment result or a specific gap.
3. The detailed BOM rows that substantiate those conclusions.

It must not turn rule mechanics, source-record exceptions, purchasing completeness, or stock metadata into the primary plant-impact story.

## Main-Section Verdict

The implemented tab starts with one compact plant rollup, followed by a single aligned BOM evidence table. The rollup answers the plant-scope question without repeating every assembly; the table carries the assembly, material, date, and status detail. The prior content decisions below remain the pruning record: rule mechanics, price completeness, material inventories, and record exceptions do not appear in the default narrative.

| Current main section | Decision | Comment |
| --- | --- | --- |
| **Plant rollup** (`#plant-impact`) | Keep, compact | One sentence names each affected plant, its assembly and component-change totals, and any alignment follow-up. It does not repeat individual assemblies. |
| **BOM changes** (`#bom-changes`) | Keep | One aligned table contains only rows carrying this ECN. It uses descriptions for assemblies and makes the EBOM/MBOM outcome explicit for every row. |
| **Browse all parts and documents** (`#object-browser`) | Defer | A pageable inventory of all ECN objects, including documents, is an audit/search capability. It does not answer where the BOM change applies and should live in History & Audit or open only from a contextual inspection action. |

## Plants Affected

| Current content | Decision | Story support | Comment |
| --- | --- | --- | --- |
| **Plants affected** heading | Revise | Question 2 | Rename to **Affected plants and assemblies** so the user expects the required plant-to-assembly answer, not a generic plant list. |
| **“Plants from the conditional rules. Green … Amber …”** | Defer | Provenance only | This is a technical explanation and color legend. Keep the rule basis in History & Audit or a brief “How this scope was determined” disclosure. Do not make it the first explanation of the ECN's impact. |
| **Plant name: `PK01 Plant B`, `PL10 Plant A`** | Keep | Question 2 | These satisfy the named-plant requirement and must remain visible without expansion. Use the full plant name before the code where space permits. |
| **Green “BOM change confirmed” pill** | Revise | Question 2 evidence | The underlying fact is useful, but “confirmed” is too broad. State the actual result, for example “Dated BOM changes found,” and separately show whether EBOM and MBOM are aligned. |
| **Amber check state** | Revise | Question 2 | A warning must name the assembly, the precise missing alignment/date fact, and the owner if it prevents the decision. “Check needed” is not actionable by itself. |
| **Cutover date** | Keep, when dated | Questions 2 and 3 evidence | A BOM effective date helps relate scope to the transition. Label it as the relevant BOM effective date, not the universal ECN cutover, when the dates differ. Do not show “Cutover not dated” as primary narrative. |
| **“New parts: 1 priced / 1 missing price”** | Defer | Question 4 evidence | Price completeness is a procurement fact. It neither names the affected assembly nor explains plant scope. Keep the detailed fact on Procurement and promote it to the Overview only when it changes the decision. |
| **“3 materials” / “17 materials” expansion** | Revise | Question 2 evidence | Counts hide the relationship the story requires. Replace the primary control with a named assembly/product summary, such as “M1V decal assemblies: 3 component changes.” Detailed component rows may remain behind an evidence disclosure. |
| **Expanded material list, role, and VC groups** | Defer to contextual evidence | Traceability | The part description and old/new role can substantiate a plant claim. VC groups, rule reason, and raw scope mechanics are too detailed for the default tab view. Show them only after the user asks to inspect the affected component. |
| **SAP records outside the rule plants** | Defer | Scope-audit evidence | This sentence lists opaque material numbers and explains a rule exception, but does not change the displayed ECN answer. If a record creates a real scope conflict, surface one named plant, component, consequence, and owner as the direct action; otherwise retain it in audit evidence. |
| **ECN inventory disposition and unit warning** | Defer | Question 3 evidence | “Use Existing” without a named outgoing part, plant, quantity unit, or decision consequence does not establish a use-up risk. Show it only when it affects the phase-out decision; otherwise keep the source limitation in audit evidence. |

## BOM Changes

| Current content | Decision | Story support | Comment |
| --- | --- | --- | --- |
| **BOM changes** heading | Revise | Question 2 | Rename to **BOM change evidence** or **Affected assemblies and BOM alignment**. The heading should make clear that the table substantiates the plant summary above. |
| **Helper: one change per assembly, BOM dates, open part for source rows** | Revise | Drill-down orientation | Keep only the concise statement that rows are ECN-specific BOM changes. “Open a part to check its source rows” is traceability instruction and belongs in the inspection interaction, not the section introduction. |
| **Model-range group heading** | Revise | Question 2 | A model range helps establish product context, but an opaque code alone is not enough. Display its product/program description where available, with the code as secondary detail. |
| **Assembly column** | Revise | Question 2 | This is the critical missing story fact. The current value is a raw assembly identifier. Show an assembly description first, retain the identifier secondarily, and make the same assembly visible in the corresponding plant summary. |
| **Plants column** | Keep | Question 2 | Named plants directly substantiate scope. Preserve the full names and do not replace them with a count. |
| **Plain-language change label** | Keep | Questions 2 and 3 evidence | “Added,” “Replaced,” “Removed,” and quantity changes make the structural effect understandable. |
| **Old/new part columns with description, dates, and lifecycle status** | Keep, but limit default detail | Questions 2 and 3 evidence | Descriptions before identifiers, outgoing/incoming dates, and translated lifecycle labels are appropriate evidence. Detailed source rows remain on inspection. |
| **Notes for quantity differences, pairing uncertainty, MBOM gaps, and 130% source** | Revise | Question 2 evidence | Keep only facts that qualify the displayed assembly relationship. Replace the generic **Notes** column with **BOM status and caveats**. “MBOM update pending” is a valid specific gap; the remaining qualifiers should use short plain-language labels. |
| **No BOM changes empty state** | Defer for the curated demonstration | Operational feedback | The wording is factual and useful for non-demo ECNs. The demonstration must instead select an ECN with usable BOM evidence, per the story requirement. |

## Minimum Plants & BOM Target

The tab should answer question 2 in two layers, not require the user to construct the answer from cards, counts, and a separate raw table.

### 1. Plant Rollup

Display one compact sentence covering every named plant. It must state affected-assembly and component-change totals, plus any alignment follow-up. Do not repeat individual assemblies, dates, or material names here.

For the ECN shown in the screenshot, the shape should be similar to:

> Plant B (PK01): 1 affected assembly, 1 component change; BOM evidence aligned. Plant A (PL10): 4 affected assemblies, 9 component changes; 1 BOM row needs alignment review.

The bracketed result must be computed from the BOM rows. It must say either that the relevant BOMs are aligned or identify the concrete gap, such as **Engineering change released; manufacturing BOM update pending**. A dated change found is not sufficient evidence to claim alignment.

Do not show price counts, raw-material counts, rule-color explanations, unqualified stock disposition, or SAP-record exceptions in this layer.

### 2. BOM Change Evidence

Below the rollup, retain one aligned table. The default table should answer, for every ECN-touched BOM row:

| Required field | Content |
| --- | --- |
| **Assembly** | Description first; identifier as secondary detail. |
| **Affected plants** | Full plant names, not a count. |
| **Change** | Added, replaced, removed, or quantity changed. |
| **Transition** | Described outgoing and incoming component, with the relevant effective date. |
| **BOM status** | EBOM/MBOM aligned, manufacturing-BOM update pending, or a concise source caveat. |

The table is valid because it is bounded to BOM rows carrying the selected ECN, rather than an unbounded BOM tree. Keep one table so that assembly, plant, change, transition, and BOM status columns remain aligned across the full evidence set.

## Content Placement

| Content | Location after pruning | Comment |
| --- | --- | --- |
| Named-plant scope, totals, and alignment follow-up | Plants & BOM, visible immediately | This is the tab's direct answer to question 2 without duplicating the evidence table. |
| ECN-specific BOM transition rows | Plants & BOM, beneath the scope summary | Bounded evidence for the named claim. |
| Price completeness, purchasing records, sourcing risk, and cost | Procurement | These answer make-or-buy and sourcing questions, not plant/BOM scope. |
| Rule mappings, VC groups, unassigned brands, records outside rule plants, and source coordinates | History & Audit or contextual part inspection | Preserve traceability without making data-processing mechanics the story. |
| Stock quantities, use-existing disposition, and unit limitations | Procurement or History & Audit; Overview only when decision-relevant | They matter only when they create a named phase-out/use-up consequence. |
| All referenced parts and documents browser | History & Audit or on-demand inspection | It is a searchable record inventory, not default plant/BOM evidence. |

## Acceptance Criteria

The Plants & BOM tab meets the demonstration story when all of the following hold:

1. A non-specialist can see every affected plant, its assembly and component-change totals, and any alignment follow-up without reading raw identifiers.
2. The table states an EBOM/MBOM alignment result or specific gap for every ECN-specific BOM row.
3. The displayed BOM status does not infer plant manufacture or BOM alignment solely from conditional-rule membership or a dated row.
4. BOM rows use part and assembly descriptions before identifiers and explain lifecycle/source qualifiers in common language.
5. Procurement counts, inventory caveats, rule mechanics, and record inventories are absent from the default tab narrative.
6. The detailed table remains bounded to ECN-specific BOM rows, with audit/source inspection available on demand.

## Relevant Implementation Files

| File | Current responsibility |
| --- | --- |
| `ui/src/components/plant-impact-table.js` | Compact per-plant rollup derived from deterministic assembly and alignment evidence. |
| `ui/src/components/bom-changes-table.js` | One ECN-specific BOM table with description-first assembly, transition, and status columns. |
| `ui/src/components/snapshot-workspace.js` | Searchable all-parts/documents inventory currently mounted in this tab. |
| `ui/src/pages/ecn-detail/ecn-detail.html` | Plants & BOM tab structure and object-browser placement. |
| `api/app/ecn/evidence/bom_changes.py` | Deterministic BOM pairing, dates, EBOM/MBOM joins, and qualifiers. |
| `api/app/ecn/evidence/plant_impact.py` | Rule-derived plant scope, plant status, purchasing, and stock evidence. |
| `docs/ecn-bom-changes.md` | Evidence semantics and the limits on plant confirmation/alignment claims. |
