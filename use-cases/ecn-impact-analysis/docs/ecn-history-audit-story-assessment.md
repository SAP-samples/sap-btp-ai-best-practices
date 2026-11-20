# History & Audit Story Assessment

## Scope

This report assesses every section rendered in the **History & Audit** tab against the [ECN Demonstration Story](ecn-demonstration-story.md). This tab is the traceability drill-down for the five story questions; it is not a sixth decision narrative, a duplicate Procurement tab, or a complete source-data browser.

The shared ECN title, change statement, and footer decision controls are assessed in the [Overview Story Assessment](ecn-overview-story-assessment.md). Plants & BOM and Procurement remain the primary evidence views for their respective story questions. This report concerns what must remain available when a reviewer challenges a conclusion, needs to see a recorded decision, or needs to create a new immutable assessment.

**Implementation status:** The compact audit view is implemented. It renders a saved-decision record first, keeps all source and AI evidence beneath a closed disclosure, keeps reassessment in a closed administrative disclosure, and lazily loads the all-record browser only on request. The detailed tables below remain the pruning record and audit-data target.

**Decision labels:**

| Decision | Meaning |
| --- | --- |
| Keep | Required to record a human decision or trace a specific claim back to its assessment evidence. |
| Revise | Useful audit fact whose presentation is too broad, repetitive, or disconnected from the claim it supports. |
| Defer | Valid evidence or administrative control that must be available only after an explicit audit request. |
| Remove | Repeats the decision story or has no traceability purpose in the demonstration flow. |

## Audit Role

The first three tabs answer the decision. History & Audit must answer only these follow-up questions:

1. Has a human recorded an approve, hold, or send-back decision, and why?
2. Which immutable assessment, source records, and generated assessment support a challenged statement?
3. Can an authorized user inspect an earlier assessment or create a new assessment from newer data?

It must not ask a non-specialist to rediscover what changed, rebuild BOM scope, interpret price completeness, or choose make versus buy. Those answers are already concise in Overview, Plants & BOM, and Procurement. A review record may link to the decision evidence, but must not repeat it.

The screenshot's `Can't accept: 7 new parts have no price` banner and its large part table violate this boundary. A count does not name the part, plant, consequence, or accountable owner. The price check also duplicates Procurement's recommendation and the named action already required on Overview. It is acceptance-plan implementation evidence, not the audit story.

## Main-Section Verdict

| Current main section | Decision | Comment |
| --- | --- | --- |
| **Review history** (`#engineering-review`) | Replace | It combines the current acceptance check, all changed parts, data-maintenance instructions, reviewer form, and prior decisions. Replace it with a compact **Decision record**. |
| **Reference documents** (`#reference-documents`) | Keep, collapsed | Released ECN objects and the EBOM/MBOM rows are valid traceability evidence. They must not be a default record inventory. |
| **Original ECN description** (`.executive-details`) | Keep, collapsed | The raw source prose can substantiate the one-line Change statement, but must never compete with it. |
| **Full AI analysis** (`#executive-summary`, `#analysis-root`) | Defer, collapsed | Preserve the generated result for review, but do not display its executive summary and five repeated narratives by default. |
| **Data used** (`.provenance-card`) | Keep, collapsed | Source files, extraction dates, dataset scope, and limitations are necessary to establish assessment provenance when requested. |
| **Assessment history and reassess** (`#assessment-context`) | Defer | Version navigation and reassessment are audit-workflow controls, not evidence for the displayed ECN decision. Keep them available in a subordinate disclosure. |
| **Browse all parts and documents** (`#object-browser`) | Defer | This is an investigation tool. It must be opened by an explicit record search or a material drill-down, never rendered as a default tab narrative. |

## Minimum History & Audit

The tab must have two visual areas, in this order. Both must be compact enough that the default content usually fits above the fold. The tab introduces no new finding, blocker, recommendation, or metric.

### 1. Decision Record

Answers: **Has a decision been recorded, by whom, when, and what was the recorded reason?**

Show one compact record, not the acceptance plan that preceded it:

> **No decision recorded.** Use the decision controls after reviewing the Overview, Plants & BOM, and Procurement evidence.

or, when saved:

> **Approved** by Jordan Lee on 19 May 2026, 14:22.  
> Recorded against assessment dated 13 May 2026.

or, when sent back:

> **Sent back to engineering** by Jordan Lee on 19 May 2026, 14:22.  
> Reason: Obtain a supplier quote for JB greasing sticker at Plant A before cutover.

Rules:

- Show the latest saved outcome first. Use the recorded comment only when it exists; do not render a dash-filled Comments field.
- Put older recorded outcomes in a closed **Previous decisions** disclosure. A timestamped outcome, reviewer, and comment are sufficient there.
- Treat the reviewer name, saved time, assessment version, and evidence hash as audit attributes. The version may appear in the compact line; the hash remains available only in the detailed audit metadata.
- The floating **Accept ECN** and **Send back to engineering** controls remain the single way to record a decision. Do not duplicate reviewer, accept, reject, comment, or confirmation controls in the tab.
- Before a decision is saved, do not show a readiness banner, a purchase-price count, or a part plan. The current proposed decision and direct action are already on Overview.

### 2. Evidence For This Assessment

Answers: **What source or assessment record supports a challenged claim?**

Show one compact, closed **Evidence for this assessment** disclosure containing these claim-specific disclosures. Do not fetch or render their record lists until the reviewer opens them.

| Disclosure | Retained content | Rules |
| --- | --- | --- |
| **Original ECN source** | Raw ECN description and the source artifact/date | This substantiates the concise Change statement. Preserve the source prose exactly; do not create a second summary. |
| **Released ECN records** | Resulting part, drawing, CAD, and other released-object records | Use description first and identifier second. Keep revision, state, and source row only as record detail. |
| **BOM evidence** | Separate EBOM and MBOM records used to substantiate the affected assembly, plant, and alignment outcome | Retain raw rows, quantities, effectivity, source file, sheet, and row only here. This is not configured-production proof. |
| **Generated assessment** | The immutable AI result, section status, missing-evidence text, and model/run metadata when available | This is review evidence, not a second executive summary. Preserve the result that was reviewed; do not replace it with a newly generated narrative. |
| **Assessment inputs and limits** | Assessment date, ECN/BOM/rule artifact names, dataset scope/extraction dates, and applicable limitations | Expose no hashes by default. A limitation is useful only to qualify a claim being challenged. |

The all-object browser may appear as a control inside this disclosure, for example **Search assessment records**. It is for a known identifier or an investigation started from the Material dialog. It must remain closed and unpopulated until deliberately used.

### Administrative Tools

Place a closed **Assessment versions and reassessment** disclosure after the two content areas. It may contain the earlier immutable assessment links, the optional assessment date, and **Create new assessment**. It must not present source-refresh instructions, generation states, or a new-assessment form as part of the audit story.

## Current Review History And Acceptance Plan

| Current content or behavior | Decision | Story support | Required change |
| --- | --- | --- | --- |
| **Blocked/ready/accepted/rejected banner** (`.review-banner`) | Revise | Recorded decision only | After a saved review, render the recorded outcome, reviewer, time, and comment in Decision Record. Before a review, show only `No decision recorded`; do not calculate or narrate the acceptance plan here. |
| **`Can't accept: 7 new parts have no price`** | Remove | None as presented | It is an aggregate data-completeness statement, not a named ECN decision. The exact material-and-plant sourcing action belongs in Overview and Procurement. |
| **Part / Material master / Purchase price table** (`.review-plan`) | Defer | Acceptance implementation evidence | Do not show it in the default audit tab. If retained for a governed acceptance workflow, place it behind a closed acceptance-evidence disclosure and translate any promoted blocker into the named Overview action. |
| **`Create and release in PL10` instructions** | Defer | Operational execution evidence | Preserve only where an authorized material-master workflow needs it. It is not a review narrative and must not precede the decision record. |
| **Has price / Missing pills** | Defer | Procurement evidence | A price existence flag is not a make-or-buy conclusion and lacks price, unit, plant context, and owner. Procurement owns the bounded price and recommendation evidence. |
| **`N BOM item changes are handled outside the material master`** | Remove | None | It is an implementation disclaimer with no decision or traceability value. BOM evidence already has a dedicated audit disclosure. |
| **Review form, reviewer field, Accept ECN, Reject, and reject comment form** | Remove from tab | Footer control duplicates | The global footer is the only decision control. Keep the saved reviewer/comment as the audit record, not a second form. |
| **Current review-history table** | Revise and collapse | Decision-record traceability | Show the latest record inline. Retain prior outcomes in a closed disclosure with outcome, reviewer, timestamp, and non-empty comment. |

## Current Evidence Sections

| Current content or behavior | Decision | Story support | Required change |
| --- | --- | --- | --- |
| **Reference documents heading and explanatory sentence** | Revise | Traceability | Rename the parent to **Evidence for this assessment**. Remove the generic introductory prose; the nested disclosure labels state the evidence purpose. |
| **ECN documents (released versions)** (`added_materials`) | Keep, collapsed | Questions 1 and 3 evidence | Retain lazy pagination. Make descriptions primary and expose type, revision, state, and source coordinate only in the opened records. |
| **EBOM matches** (`ebom_matches`) | Keep, collapsed | Question 2 evidence | Retain as raw support for the named assembly and alignment outcome, not a second BOM narrative. |
| **MBOM matches** (`mbom_matches`) | Keep, collapsed | Question 2 evidence | Retain separately because an EBOM/MBOM difference can change the decision. Do not display an empty or unopened record count as a story signal. |
| **`Not loaded` and record-count labels** | Remove from default | None | They are loading and inventory status, not evidence. Show progress or an error only after the reviewer opens a disclosure. |
| **Original ECN description** | Keep, collapsed | Question 1 traceability | Keep the raw source beneath **Original ECN source**. The source description must not appear beside the decision headline or be restated as an audit summary. |
| **Full AI analysis executive summary** | Defer | Review traceability | Retain under **Generated assessment** only. It repeats the concise Change, Impact, and Decision story and must never be initially expanded. |
| **Five AI narrative accordions** (`change_intent`, `affected_materials`, `ebom_mbom_impact`, `configuration_scope`, `uncertainties_and_missing_evidence`) | Defer | Review traceability | Group beneath the same closed Generated assessment disclosure. Promote only a direct, named blocker to Overview; do not surface generic missing-evidence lists in the tab. |
| **AI placeholders, retry states, and generation diagnostics** | Defer to contextual feedback | Operational state | Show a concise error only when a user initiated generation or the missing result prevents the requested inspection. Do not reserve tab space for it. |
| **Data used: resolution/upload times and artifact names** | Keep, collapsed | Assessment provenance | Combine as immutable metadata in Assessment inputs and limits. Avoid repeating an artifact name under separate headings. |
| **Dataset table and Scope and limitations** | Keep, collapsed | Assessment provenance | Retain source, scope, extraction date, use, and applicable limitation for audit inspection. Do not lead with generic scope warnings. |
| **Assessment date** | Keep, merge | Assessment provenance | Put it in the Decision Record's version reference and the Assessment inputs metadata; do not make it a separate visual area. |
| **Assessment version history** | Defer | Historical traceability | Keep it under Administrative Tools. Mark the current version clearly and preserve earlier immutable links. |
| **Reassess with current data** | Defer | Workflow control | Keep the action behind Administrative Tools. Explain that it creates a new assessment rather than changing the evidence reviewed in this decision. |
| **Browse all parts and documents** | Defer | Investigation tool | Keep searchable, paginated records only after explicit request. It must not automatically load or visually compete with the evidence disclosures. |

## Content Placement

| Content | Location after pruning | Comment |
| --- | --- | --- |
| One-line change, named impact, make-or-buy conclusion, and direct action | Overview | These answer the ECN story before the reviewer enters History & Audit. |
| Plant, assembly, EBOM/MBOM alignment, and BOM transition details | Plants & BOM | The audit tab retains only the raw records supporting a challenged Plants & BOM conclusion. |
| Recommendation, comparable cost, purchase-history price, and decision basis | Procurement | The audit tab may retain a reviewed evidence snapshot, but does not reinterpret price completeness or make/buy permissions. |
| Latest recorded decision, reviewer, saved time, recorded reason, prior outcomes | History & Audit | This is the primary visible content of the audit tab. |
| Source ECN prose, released-object records, raw BOM rows, source coordinates, AI output, dataset provenance, and limitations | Closed History & Audit evidence disclosures | These must be available without becoming the default page. |
| Assessment versions, reassessment, source preparation, generation retry, and object search | Closed administrative or contextual controls | These enable audit work but are not evidence or decision narrative. |

## Audit Data Contract And Gaps

The retained evidence must be immutable for the assessment a reviewer evaluated. The current review save already records `evidence_hash` and `snapshot_version`; the streamlined UI must preserve those attributes even when it does not show the hash by default.

| Required audit fact | Current state | Required behavior |
| --- | --- | --- |
| Recorded outcome, reviewer, timestamp, and comment | Available through review history | Show the latest compact record and retain prior records on demand. Do not substitute a currently recalculated readiness state for a saved decision. |
| Reviewed assessment version and evidence hash | Submitted when saving a review | Retain and make inspectable as detailed audit metadata. The saved decision must identify the immutable evidence set it reviewed. |
| Source ECN, BOM, rule artifact, dataset scope, and extraction date | Available in provenance and snapshot datasets | Keep inside Assessment inputs and limits. Distinguish assessment data from historical-comparison data. |
| Released ECN records and EBOM/MBOM source rows | Available through the lazy evidence endpoint or snapshot | Keep grouped by their claim, paginated, and loaded only on opening. |
| AI result, model/run version, prompt, evidence references, and reviewer override reason | Narrative result is available; complete model/run audit metadata is not currently rendered | Preserve the exact generated result reviewed. A governed production workflow must retain and expose these details on demand, including an override reason where a human departs from the recommendation. |
| Reassessment lineage | Earlier assessment links are available | Preserve the predecessor/successor relationship and make clear that reassessment creates a new immutable result. |

## Acceptance Criteria

History & Audit meets the demonstration story when all of the following hold:

1. The default tab contains only Decision Record and a closed Evidence for this assessment area; Administrative Tools are visually subordinate and closed.
2. The latest saved decision states outcome, reviewer, time, and a recorded reason when one exists. Earlier decisions remain available on demand.
3. Before a decision is saved, the tab does not show price counts, material-master instructions, readiness cards, part inventories, or duplicate accept/reject forms.
4. The tab introduces no new blocker, recommendation, make/buy conclusion, plant/BOM narrative, or generic missing-data warning.
5. Original ECN prose, released records, EBOM/MBOM rows, generated analysis, data provenance, and limitations are preserved as lazily loaded or closed evidence, not removed.
6. Raw BOM evidence is not presented as proof of configured production use.
7. Every audit record remains attributable to the immutable assessment/version reviewed; hashes, source coordinates, model/run detail, and data lineage are available only when the reviewer requests detail.
8. Reassessment and all-object search remain available for authorized audit work without becoming primary page content.
9. The shared footer remains the sole visible accept/send-back control, and the Overview remains the sole default statement of what must happen next.

## Relevant Implementation Files

| File | Current responsibility |
| --- | --- |
| `ui/src/pages/ecn-detail/ecn-detail.html` | Defines the History & Audit surfaces, provenance panel, assessment-history panel, object browser, and shared decision footer. |
| `ui/src/components/engineering-review.js` | Renders the acceptance-plan table, review form, banner, and saved review history; must be reduced to the compact decision record while retaining the save API. |
| `ui/src/components/reference-documents.js` | Lazily renders released-object, EBOM, and MBOM record evidence. |
| `ui/src/components/analysis-sections.js` | Renders the five generated narrative sections that must move below one closed generated-assessment disclosure. |
| `ui/src/components/assessment-context.js` | Renders assessment date, immutable versions, reassessment, dataset provenance, and limitations. |
| `ui/src/components/snapshot-workspace.js` | Mounts the all-object browser, assessment context, review state, and provenance data. |
| `ui/src/pages/ecn-detail/ecn-detail.js` | Wires shared footer decisions, source provenance, generated analysis, snapshot evidence, and audit navigation. |
| `docs/ecn-demonstration-story.md` | Defines the five-question decision story and exclusions for raw audit, diagnostics, and source-ingestion detail. |
