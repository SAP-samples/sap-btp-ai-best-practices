# Overview Story Assessment

## Scope

This report assesses every rendered section visible while the **Overview** tab is selected. It applies the five questions in [ECN Demonstration Story](ecn-demonstration-story.md): change, affected plants/BOMs, lifecycle, make-or-buy impact, and next decision.

**Implementation status:** The streamlined Overview now implements the **Minimum Overview** target. The detailed tables below remain the pruning record for the sections that were removed or moved to drill-down evidence.

**Decision labels:**

| Decision | Meaning |
| --- | --- |
| Keep | Directly answers a story question in the Overview. |
| Revise | Has the right purpose but does not currently answer the question clearly enough. |
| Defer | Valid operational control or drill-down evidence, but not Overview narrative content. |
| Remove | Does not support a story question and should not appear in the demonstration flow. |

## Main-Section Verdict

The current Overview is not streamlined because it tells the same ECN story four times in four different forms: the page header, the blocker dashboard, the supporting-interpretation accordion, and the component/BOM accordion. The user must reconcile those forms before understanding the decision.

| Current main section | Decision | Comment |
| --- | --- | --- |
| **Header summary and context** | Keep, but merge | Comment: This is the only existing section that begins with the ECN itself. Merge the title, one-line change, reason, owner, and cutover into one compact **Change** area. Remove status pills and raw source text from this area. |
| **Decision blockers and next actions** | Remove and replace | Comment: It starts with data readiness, counts, and generic missing fields before explaining the change or impact. Replace the entire dashboard with one concise **Decision** area at the end of the page. |
| **Impact analysis details** | Remove from Overview | Comment: Three generic AI factor cards plus five narrative accordions repeat and fragment the impact story. Extract only the final, source-backed facts needed for the Impact area; move the rest to the relevant drill-down tab. |
| **Affected components and BOM context** | Remove from Overview | Comment: This is a record inventory, not a decision summary. Keep detailed components and BOM rows in Plants & BOM. |
| **Full AI analysis** | Remove from Overview | Comment: A long generated narrative is explicitly excluded by the story standard and competes with the factual change story. Preserve it only as optional supporting evidence outside the Overview. |
| **Floating decision footer** | Keep as control, not content | Comment: Accept and send-back actions are needed after the decision is understood. The footer must not add a second status/count narrative. |

## Minimum Overview

The five story questions do **not** require five cards or five accordions. The Overview must have only three visual areas, in this order. Every fact in an area must answer its stated question; if it does not, it moves to a drill-down tab or audit.

### 1. Change

Answers: **What changed and why? Who owns it, and when does it take effect?**

Display one plain-language sentence and one compact fact line:

> Replace the metal shim ring with the flexible shim ring to correct the released design data.
>
> Owner: Dana Whitfield | Cutover: 19 May 2026

Do not display analysis status, severity, raw description, record completeness, duplicate pills, or a separate AI label here.

### 2. Impact At A Glance

Answers: **Where does it matter? What is phased in/out? What is the make-or-buy and cost/risk implication?**

Display exactly three named rows, not cards, counts, charts, or accordions:

| Topic | Required content |
| --- | --- |
| **Affected** | Named plants and affected assembly or product. Include the EBOM/MBOM outcome only when it changes the decision. |
| **Transition** | Named incoming part, named outgoing part, phase-in/out status, and relevant date. |
| **Decision factor** | Proposed make/buy outcome, one-sentence reason, and one high-level cost, lead-time, sourcing, or manufacturing consequence. |

Example:

> **Affected**: Plant A, Plant C, and [plant] use the [assembly]. MBOM alignment is pending at Plant A.
>
> **Transition**: Phase in flexible shim ring and phase out metal shim ring on 19 May 2026.
>
> **Decision factor**: Buy the flexible shim ring because Plant A lacks compatible tooling; the quoted unit cost is [value].

If a row has no curated fact for the selected ECN, do not show an empty row or a missing-data warning. Choose an ECN with the required evidence for the demonstration.

### 3. Decision

Answers: **What should happen next?**

Show one outcome first, followed by no more than the direct actions that prevent it:

> **Hold**: Confirm the purchase setup for flexible shim ring at Plant A before the 19 May cutover. Owner: Procurement.

or:

> **Approve**: The new flexible shim ring is sourced, BOMs are aligned, and the phase-out has no remaining use-up risk.

Do not show generic confirmation tasks, readiness labels, severity chips, notes, coverage statements, affected counts, or a list of all data gaps. A valid action always names the object, place, consequence, and owner.

## Content Placement

| Content | Location after pruning | Comment |
| --- | --- | --- |
| Detailed plant, assembly, and EBOM/MBOM rows | Plants & BOM | Comment: These substantiate the named **Affected** row; they are not Overview content. |
| Sourcing records, price history, and detailed manufacturing evidence | Procurement | Comment: These substantiate the **Decision factor** row; show only the conclusion on Overview. |
| Original description, AI narrative, datasets, source rows, and provenance | History & Audit | Comment: These provide traceability and explain how a conclusion was derived. |
| Generation state, source preparation, retry, and errors | Contextual operational feedback | Comment: Show only while the user invokes or is blocked by that operation. |

## Header And Page Controls

| Current section | Decision | Story support | Comment |
| --- | --- | --- | --- |
| **Back to ECNs** (`#back-to-list`) | Keep | Navigation only | Comment: This is required page navigation, not story content. Keep it visually secondary and do not count it as an Overview story section. |
| **ECN number and title** (`#detail-title`) | Revise | Question 1 | Comment: It identifies the ECN, but the title must use a plain-language part or assembly description rather than an opaque record title. |
| **One-line change story** (`#decision-story`) | Keep | Question 1 | Comment: It is the correct primary answer to what changed. It must remain factual, use descriptions before identifiers, and state only the change. |
| **Original ECN description** (`.source-description`) | Defer | Traceability for question 1 | Comment: Raw source prose is not the decision narrative. Keep it as a source drill-down in History & Audit or on demand after the concise change statement. |
| **Summary-available status** (`#detail-status`, snapped duplicate) | Defer | None | Comment: Analysis-generation state is an operational detail, not the change, impact, or decision. Show it only near generation controls or in audit history. |
| **Observed-impact pill** (`#detail-impact`, snapped duplicate) | Remove | None as currently expressed | Comment: A generic severity such as “Medium” does not explain the affected plant, cost, lifecycle, or decision consequence. Replace it only if it becomes a specific story signal. |
| **Generate/regenerate summary** (`#generate-summary`) | Defer | Operational control | Comment: Summary generation supports the workflow but does not tell the ECN story. Keep the control outside the narrative hierarchy. |
| **Context band: owner, cutover, why, cost** (`#ecn-context-band`) | Revise | Questions 1 and 4 | Comment: Owner and cutover directly complete the change story. Reason and cost are valid only when they are short, decision-relevant, and supported by the selected snapshot. |
| **Source preparation notice** (`#source-preparation`) | Defer | Operational control | Comment: RDF preparation is technical state. It must not compete with the decision story. |
| **Generation status and error** (`#detail-generation-status`, `#detail-error`) | Defer | Operational control | Comment: Show these only while the user has initiated generation or when an error prevents the requested action. |
| **Overview / Plants & BOM / Procurement / History & Audit tabs** (`#ecn-tabs`) | Keep | Drill-down navigation | Comment: The tabs support the same story when Plants & BOM provides question 2 evidence, Procurement provides question 4 evidence, and History & Audit provides traceability. They must not introduce separate narratives. |

## Decision And Action Section

| Current section | Decision | Story support | Comment |
| --- | --- | --- | --- |
| **Decision blockers and next actions** (`#attention`) | Remove and replace | Question 5 | Comment: The section starts from data readiness rather than the ECN story. Replace the complete dashboard with the single Decision area described in Minimum Overview. |
| **Readiness heading and “9 of 9 parts” coverage sentence** | Remove | None | Comment: Coverage is a data-quality metric, not an ECN decision. It neither names the plants/assemblies affected nor changes the required action. |
| **Blocking / advisory / note chips** | Remove | None | Comment: Severity counts duplicate the cards and force the analyst to interpret a metric instead of seeing the decision consequence. |
| **New parts have no work instruction** | Defer; reuse only as a direct action | Questions 4 and 5 evidence | Comment: Do not retain this as a card. It may become the single Decision action only when it names each affected part and plant, explains the consequence, and assigns an owner. |
| **New parts have no purchase info record** | Defer; reuse only as a direct action | Questions 4 and 5 evidence | Comment: Do not retain this as a card. It may become the single Decision action only when it names the part and plant, states the sourcing/cost consequence, and assigns procurement ownership. |
| **Confirm the scheduled manufacturing rollout** | Defer; reuse only as a direct action | Questions 3 and 5 evidence | Comment: Do not retain this as a card. It may become the single Decision action only when it names the plant, incoming/outgoing part, and cutover that need confirmation. |
| **Confirm missing stock fields** | Remove | None by default | Comment: Missing-field inventory is not a decision. Keep it only when it becomes a specific phase-out or use-up risk for a named part and plant. |
| **Confirm the model and plant coverage** | Remove | None as written | Comment: Replace this generic validation task with the actual named plants and affected assemblies in the Plants & BOM story section. |
| **Affected-count badges** | Remove | None | Comment: Counts hide the part and plant needed to understand the consequence. Use names in the action itself; detailed counts belong in drill-down evidence. |
| **Expanded follow-up explanation, source rows, datasets, and “View data used”** | Defer | Evidence for questions 2 through 5 | Comment: These are useful to substantiate a valid blocker, but must remain behind the specific action or in History & Audit. |

## Supporting Interpretation

| Current section | Decision | Story support | Comment |
| --- | --- | --- | --- |
| **Impact analysis details** (`#impact-analysis-details`) | Remove from Overview | Questions 2 through 4 | Comment: Do not replace this accordion with more Overview sections. Extract only the three final facts for Impact At A Glance and move the underlying interpretation to drill-down evidence. |
| **Manufacturing impact factor** | Defer | Question 4 evidence | Comment: Extract only a high-level schedule, capacity, tooling, or lead-time implication into the one Decision factor row; do not show this factor card. |
| **Procurement impact factor** | Defer | Question 4 evidence | Comment: Extract only the proposed make/buy outcome, reason, and cost or sourcing implication into the one Decision factor row; do not show this factor card. |
| **Configuration impact factor** | Defer | Question 2 evidence | Comment: Extract only a named product, assembly, or BOM alignment result into the Affected row; technical configuration detail belongs in Plants & BOM. |
| **Factor “Next steps” lists** | Defer | Question 5 evidence | Comment: Promote only actions that block the decision and meet the named part/plant/owner rule; defer all other suggestions. |
| **Change intent AI narrative** | Defer | Question 1 evidence | Comment: Extract its verified one-line fact into Change. Do not retain a duplicate AI narrative in Overview. |
| **Affected materials AI narrative** | Defer | Question 2 evidence | Comment: Extract named incoming/outgoing parts and affected assemblies into Impact at a glance. The complete material narrative is drill-down evidence. |
| **EBOM and MBOM impact AI narrative** | Defer | Question 2 evidence | Comment: Extract only a specific alignment result or named gap into Affected. The full generated interpretation must not be a separate Overview section. |
| **Configuration scope AI narrative** | Defer | Question 2 evidence | Comment: It is evidence only when it explains a named product or assembly impact; otherwise it is technical detail. |
| **Uncertainties and missing evidence AI narrative** | Defer; reuse only as a direct action | Question 5 evidence | Comment: Do not show the narrative. A decision-blocking uncertainty may become the single named Decision action; defer all other data gaps to audit evidence. |

## Change Scope And Full Narrative

| Current section | Decision | Story support | Comment |
| --- | --- | --- | --- |
| **Affected components and BOM context** (`#parts-overview`) | Remove from Overview | Questions 2 and 3 evidence | Comment: The Overview needs a concise named-plant and incoming/outgoing-part summary, not the component inventory. Keep the full scope under Plants & BOM. |
| **Parts in this ECN list** | Defer | Questions 1 through 3 evidence | Comment: It supports inspection of a selected part but is too granular for the Overview narrative. |
| **BOM matches list** | Defer | Question 2 evidence | Comment: EBOM/MBOM rows substantiate the story but should live in Plants & BOM after the overview states the alignment result or named gap. |
| **Items needing attention count** | Remove | None | Comment: This duplicates the decision section and is a count without a decision consequence. |
| **Full AI analysis** (`.executive-details`) | Remove | None in the Overview | Comment: A long generated narrative is explicitly excluded by the story standard. Retain source-backed detail only in a drill-down or audit context, not the Overview. |

## Decision Footer

| Current section | Decision | Story support | Comment |
| --- | --- | --- | --- |
| **Open-action count** (`#decision-bar-status`) | Remove | None | Comment: Do not add a second footer status. The current decision state and its one reason belong in the Decision area, such as “Hold: sourcing approval required for [part] at [plant].” |
| **Reviewer-name field** (`#reviewer-name-footer`) | Defer | Governance only | Comment: This is required to record a decision but does not help the analyst understand the ECN. Reveal it when an action is selected. |
| **Accept ECN** (`#accept-ecn-footer`) | Keep | Question 5 | Comment: This is the final decision action. It must be enabled only when the story's direct blockers are resolved and the recommendation is clear. |
| **Send back to engineering** (`#send-back-footer`, dialog) | Keep | Question 5 | Comment: This is a valid alternative decision. The dialog must preface free text with the named decision-blocking reason so the response remains tied to the story. |

## Enforcement Rule

The Overview may contain only the three areas in **Minimum Overview**. It is complete when it answers all five story questions, not when it exposes all available analysis. Anything else is navigation, an operational control, or drill-down evidence and must not create another Overview section.
