# Material Dialog Story Assessment

## Scope

This report assesses the modal opened by a material link in the ECN workspace.
It applies the five questions in [ECN Demonstration Story](ecn-demonstration-story.md), with emphasis on the material-level questions: what the material is, what ECN transition and BOM relationships it has, and whether it should be made or bought.

The prior dialog was a traceability viewer, not a decision brief. It exposed all available BOM occurrences, source coordinates, and source references before clearly stating the material's ECN role or make-or-buy story. The screenshot demonstrates that former failure mode: four large BOM rows and two empty evidence sections left the reader to infer the material's relationship and see no procurement answer.

**Implementation status:** The compact Material and ECN Change, Linked Use, and Make or Buy brief is implemented. Raw occurrence, source-record, and ECN-reference tables now remain under one optional Evidence disclosure. The detailed decisions below remain the pruning record and data-contract target for a governed proposal and make-cost comparison.

**Decision labels:**

| Decision | Meaning |
| --- | --- |
| Keep | Directly answers the material's identity, relationship, lifecycle, make/buy, cost, or direct next action. |
| Revise | The source fact is useful, but its current presentation is too technical, repetitive, or incomplete. |
| Defer | Valid traceability or source evidence that belongs behind an explicit evidence disclosure. |
| Remove | Does not change the material-level ECN decision and should not consume dialog space. |

## Material Story

The dialog must let a non-specialist answer the following without reading a raw BOM row or opening a disclosure:

1. **What is this material?** Show its description first and identifier secondarily.
2. **What does this ECN do to it?** State phase-in, phase-out, replacement, addition, or removal, and name the counterpart material when one is known.
3. **What is it linked with?** Name the affected assembly or product and plant, with the relevant effective or future state. A BOM occurrence is structural evidence, not proof of configured production use.
4. **Can the plant make, buy, or both?** State the recorded plant-specific procurement permission in plain language. Do not turn an `E`, `F`, or `X` master-data value into a recommendation.
5. **If both are allowed, which option is proposed and what does each cost?** Show the proposal, concise rationale, comparable make and buy unit costs, and the delta only when the values share a unit, quantity basis, currency, plant, and effective period. Label a curated demonstration value at the claim it supports.
6. **What must happen next?** Show one named action only when unavailable sourcing, cost, BOM alignment, or lifecycle evidence blocks the ECN decision.

The dialog is a focused drill-down. It must not repeat the complete ECN Overview or become a general source-record browser.

## Main-Section Verdict

| Prior dialog content | Decision | Comment |
| --- | --- | --- |
| **Identifier heading** (`obj.number`) | Revise | The visible heading is an opaque material number. Lead with the material description, followed by `Material: <number>` as secondary detail. |
| **Description paragraph** | Keep, but merge | This is the correct identity fact. Merge it into the header rather than giving the identifier visual priority. |
| **Change action, usage label, counterpart, revision, pairing label, and qualifiers** | Revise and merge | These fragments describe one transition but are rendered as several disconnected paragraphs. Replace them with one plain-language transition sentence that names the counterpart and gives the relevant lifecycle status. Keep a pairing qualifier only when it limits that statement. |
| **Assessment date and production-use caveat** | Defer by default | The caveat is correct, but it appears before the material story in every dialog. Attach it to an uncertain usage claim or place it in the evidence disclosure. Do not state it when no production-use assertion is made. |
| **Usage and validity** | Revise | Relationship evidence is essential, but the five-column table starts with workbook names, row numbers, hierarchy percentages, and repeated validity details. Replace the primary table with distinct material-to-assembly-and-plant relationships. |
| **Material and procurement** | Replace | This table mixes material master, purchasing, and stock source records. It does not state the permitted make/buy mode, proposed outcome, or comparable cost. The empty state in the screenshot consumes substantial space without answering a question. |
| **Source evidence** | Defer | Source roles, revisions, and raw ECN paths support auditability, not the material decision. Keep them in a collapsed Evidence disclosure. |
| **Supporting objects** | Defer | Related documents or non-part objects may be useful inspection links, but are not part of the default material narrative. |
| **Technical source details** | Defer | Raw source validity values are needed only to resolve a date-parsing or traceability question. |
| **Pagination and Close** | Keep as controls | They support safe navigation and bounded loading but are not story content. |

## Minimum Material Brief

The default dialog must have only three visual areas, in this order. It should be dense enough that the complete answer commonly fits above the fold. Do not render an empty area.

### 1. Material And ECN Change

Answers: **What is this material, and what does the ECN do to it?**

Display the description as the heading, its identifier as compact secondary text, and one transition sentence. For example:

> **3-point control warning decal**  
> Material: `Q900VV0080402`  
> **Phase in** to replace **previous 3-point control warning decal** on this ECN. Effective 19 May 2026.

If the relationship is not explicit, say `Replacement relationship needs review`; do not imply that adjacent ECN objects are a confirmed replacement. If the material has no ECN change semantic, retain only its description and identifier. Do not create an empty change area.

### 2. Linked Use

Answers: **What product or assembly is this material linked to, at which plant, and what is its current or future state?**

Display one compact line for each distinct assembly-and-plant relationship, not every source-row occurrence. Use descriptions before identifiers. For example:

| Linked assembly or product | Plant | BOM state |
| --- | --- | --- |
| Q900V control-panel assembly (`Q900VV0080402_Q900`) | Plant B (PK01) | MBOM effective from 19 May 2026; quantity 2 PC |
| Q900V control-panel assembly (`Q900VV0080402_Q900`) | Plant B (PK01) | EBOM effective from 13 Mar 2026; quantity 2 PC |

The table must deduplicate identical relationships and retain separate EBOM and MBOM facts when they change the answer. It must use the path-node descriptions and `plant_scope` already held on a BOM location; the current renderer discards both in favor of identifiers, workbook rows, and a generic `Plant not supplied` label.

Do not say the plant manufactures the material merely because a BOM row is present. State `BOM evidence effective`, `future`, `expired`, or `needs review` as appropriate. Show the production-use limitation only for a conclusion that would otherwise overclaim.

### 3. Make Or Buy Decision

Answers: **What may the plant do, what is proposed, what does each option cost, and is an action required?**

Display one compact, plant-specific decision line or a two-row comparison when both options are permitted:

| Plant | Permitted | Proposed outcome and basis | Make cost | Buy cost | Decision consequence |
| --- | --- | --- | --- | --- | --- |
| Plant A (PL10) | Make or buy | **Buy**: compatible supplier quote is available; curated demo basis | EUR 1.10 / PC | EUR 0.85 / PC | Buy saves EUR 0.25 / PC |

Rules for this area:

- `In-house production` means make is permitted; `External procurement` means buy is permitted; `In-house and external procurement allowed` means both are permitted. Those facts are not a recommendation.
- A proposal requires its own source or a clearly labeled demonstration assumption. Do not derive `Buy` merely because a purchase record or supplier exists.
- Show `Make cost`, `Buy cost`, and the delta only after normalizing each value to the same plant, currency, unit of measure, price quantity, and effective period. State the shared basis, for example `EUR / PC, valid through 31 Dec 2026`.
- If only a purchase price exists, show `Buy cost: EUR 0.85 / PC; make cost unavailable` rather than a comparison or a saving claim.
- If costs are missing but the omission does not block the ECN decision, omit this area. If it does block the decision, show one specific action, for example: `Hold: obtain make and buy unit costs for 3-point control warning decal at Plant A before the 19 May cutover. Owner: Procurement.`
- Do not show supplier names, inventory, storage location, master-data status, information-record numbers, or pricing history unless one changes the proposed option or the action. They belong in evidence.

## Prior Usage Table

| Current field or behavior | Decision | Story support | Required change |
| --- | --- | --- | --- |
| **Source**: model code, workbook, row number, `Plant not supplied` | Revise | Linked use evidence | Replace the workbook coordinate with a described assembly/product and resolved named plants. Put workbook and row coordinates in Evidence. |
| **Hierarchy**: multi-level identifiers and 150%/130% tags | Revise | Linked use evidence | Reduce each occurrence to its relevant assembly/product relationship. Retain configuration percentage only when it distinguishes applicability and label it in plain language. |
| **Quantity** | Keep, conditionally | Linked use evidence | Show it next to the linked assembly only when it distinguishes the BOM relationship or cost basis. |
| **Validity**: start/end dates per source row | Keep, conditionally | Lifecycle evidence | Show the effective or future date that changes the transition. Remove repeated dates and omit an open-ended end date. |
| **Usage**: `effective` / `future` | Revise | Lifecycle evidence | Keep the fact but use `BOM evidence effective from ...` or `scheduled from ...`; it is not configured-production proof. |
| Repeated occurrences of the same parent | Remove from default | None after deduplication | Keep the complete occurrence list only in Evidence. |
| **Technical source details** disclosure | Defer | Traceability | Retain raw dates and parsing detail behind Evidence. |

## Prior Procurement And Evidence Tables

| Current field or behavior | Decision | Story support | Required change |
| --- | --- | --- | --- |
| Dataset type, filename, worksheet, and row | Defer | Traceability | Place in collapsed Evidence. |
| Plant code and material-master status | Revise | Question 4 evidence | Resolve the code to a named plant. Keep a status only when it prevents the chosen make/buy outcome. |
| Supplier or vendor | Defer by default | Buy-option evidence | Show only when the supplier availability is the rationale, risk, or direct action. |
| Stock, storage location, snapshot date | Defer | Question 3 evidence | Show only when a named use-up or retirement decision depends on stock. |
| Procurement type | Revise | Question 4 | Translate its existing canonical label to `Make allowed`, `Buy allowed`, or `Make or buy allowed`; distinguish permission from proposal. |
| Empty `Material and procurement` section | Remove | None | Do not render an empty table or generic missing-record message. Surface a missing fact only as a direct decision blocker. |
| ECN source role, revision, and state | Defer | Traceability | Preserve in Evidence with readable source labels. |
| Empty `Source evidence` section | Remove | None | Do not render it; an absent audit record is not a material decision. |

## Data And Implementation Gaps

The desired make-or-buy comparison requires data not supplied to this dialog today. The UI must not fabricate the missing answer.

| Needed fact | Current state | Required contract |
| --- | --- | --- |
| Material description | Available on the object | Use as the dialog heading. |
| ECN action, counterpart, revision, and qualifiers | Available as `obj.change` | Synthesize one plain-language transition without overstating an uncertain pairing. |
| Assembly/product descriptions and parent relationship | Available on each location path | Return and render the relevant described parent/assembly instead of only `node.number`. |
| Plant scope | Available on each enhanced BOM location as `plant_scope` | Render named plant mappings beside the linked assembly. Do not render `Plant not supplied` when the deterministic scope is available. |
| Make/buy permission | Available on material-master records as canonical `procurement_type_label` | Render the label, not the raw `procurement_type`; keep it plant-specific. |
| Buy price | Available only through the curated `snapshot.purchase_prices` overlay | Include the material's price records in object details or provide a bounded material-decision endpoint. The current object-details response returns only the object, paginated locations, reference records, and source references. |
| Make cost | Not supplied by the current reference datasets or mock overlay | Add an approved internal-cost source or curated demo field with plant, currency, unit, price quantity, validity, basis, and assumption label. Without it, show no make-versus-buy comparison. |
| Proposed make/buy outcome and rationale | Not an inferred source fact | Add an authorized decision/recommendation contract or a clearly labeled curated demonstration claim. A material-master permission is insufficient. |
| Comparable cost delta | Not derivable today | Calculate only from normalized make and buy records with the same plant, currency, unit, price quantity, and effective period. |

The current purchase-price overlay is explicitly illustrative demonstration data. Any material-dialog price or recommendation derived from it must carry that label next to the value, not in a generic dialog disclaimer.

## Content Placement

| Content | Location after pruning | Comment |
| --- | --- | --- |
| Material identity, ECN transition, linked assembly/plant, lifecycle state | Material dialog, visible immediately | This is the direct material story. |
| Plant-specific make/buy permission, proposed outcome, comparable costs, and direct blocker | Material dialog, visible immediately when available | This answers question 4 at the point the user asks about the material. |
| Complete BOM occurrence tree, configuration-level hierarchy, source row coordinates, raw validity values | Collapsed Material dialog Evidence or History & Audit | Preserves traceability without making the user reconstruct the story. |
| Supplier record, information-record number, price history, master-data statuses, stock and storage | Procurement or collapsed Material dialog Evidence | Promote only the one fact that changes the proposal, lifecycle risk, or action. |
| Source roles, revisions, states, supporting documents, and dataset provenance | History & Audit or collapsed Material dialog Evidence | Audit content, not material narrative. |

## Acceptance Criteria

The material dialog meets the demonstration story when all of the following hold:

1. The material description is the primary heading; the identifier is secondary.
2. The default view states the ECN role and names a counterpart only when the relation is explicit or appropriately qualified.
3. The default view names each distinct affected assembly/product and plant relationship, using descriptions before identifiers.
4. It distinguishes BOM evidence from configured production use and does not repeat raw BOM rows or generic caveats.
5. It states plant-specific make/buy permission without calling it a recommendation.
6. When both modes are permitted, it shows a proposal and make/buy costs only when a governed or labeled-demo source supplies comparable values; otherwise it states the precise unavailable evidence only if that blocks the decision.
7. It shows no empty sections, generic missing-data inventories, source coordinates, raw status codes, or duplicated evidence in the default view.
8. Any default action names the material, plant, decision consequence, and accountable owner.
9. Source records and full BOM occurrence evidence remain available on demand.

## Relevant Implementation Files

| File | Current responsibility |
| --- | --- |
| `ui/src/components/material-dialog.js` | Renders the material title, change fragments, paginated usage, procurement/source-record tables, and supporting objects. |
| `api/app/ecn/services/snapshots.py` | Supplies the paginated object-details payload used by the dialog. |
| `api/app/ecn/evidence/snapshot.py` | Builds object descriptions, location paths with descriptions, material-master references, and canonical procurement-type labels. |
| `api/app/ecn/evidence/scoped_usage.py` | Enriches BOM locations with plant scope and lifecycle/usage status. |
| `api/app/ecn/mock_overlay.py` | Supplies curated demonstration purchase-price data; it does not supply make cost. |
| `ui/src/components/procurement-risk.js` | Current Procurement-tab consumer of purchase-price and make/buy evidence. |
| `docs/reference-datasets.md` | Defines canonical `E`/`F`/`X` procurement-type meaning. |
