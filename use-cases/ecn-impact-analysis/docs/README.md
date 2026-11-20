# ECN Impact Analysis Documentation

## Purpose

This documentation defines the first implementation epics for the ECN Impact Analysis proof of concept. It converts the customer-supplied process description into explicit contracts between ingestion, applicability discovery, structural reconciliation, and industrial impact assessment.

The documents are design specifications, not evidence that the described application behavior has already been implemented. Phase 1 is review-first, operates at Autonomy Level 0, and does not write changes directly to source SAP or PLM systems.

## Architecture at a glance

The epics form a dependency chain. Each business epic consumes a versioned output from the previous epic rather than independently reinterpreting the same source data.

```text
Released ECN and source extracts
            |
            v
Epic 1: NormalizedChangeSet
            |
            v
Epic 2: CandidateScope
            |
            v
Epic 3: ReconciledStructureDelta + DataReadinessReport
            |
            v
Epic 4: IndustrialImpactPackage
            |
            v
Phase 1 selected outcome: MakeBuyRecommendation + MaterialMasterDraft

Epic 0: Foundation and Governance supports every stage above
```

The selected Phase 1 outcome was clarified in the meeting consolidation received on 12 September 2026. The existing detailed Epic 0–4 documents stop at the governed impact package; the Make-or-Buy recommendation and material-master draft still require an explicit contract and implementation. A live recommendation also needs an approved workflow point before release or an explicit review/reopen point, because the current Plant C process records the decision on the ECN before release.

This ordering preserves two essential distinctions:

- `potentially affected` is a discovery result; `confirmed affected` requires sufficient evidence and any human confirmation explicitly required by policy or business criticality.
- a structural difference says what changed in a product/plant structure; it does not by itself prove an industrial consequence, required action, severity, or owner.

## Current sample-data evidence

- `data/ECN.json` contains 35 Windchill change notices with affected and resulting object relations, but has one extra standalone comma and several exception-valued fields. It is an intake sample, not yet a valid production contract or proven release-time snapshot.
- The file contains 37 distinct ECN-like strings because `ECN120450` and `ECN131984` appear only as narrative references inside other notices. They are not additional top-level notice objects.
- `data/EBOM MBOM_Q900V_04.09.2026.xlsx` provides the raw/enriched engineering hierarchy and one manufacturing view. All 35 JSON notice numbers match raw EBOM `Change No.`, but most direct rows are documents; material context must be recovered through the preserved hierarchy or exact class-controlled part where-used joins.
- In the raw 130% sheet, meeting-confirmed column-D `Item` values beginning with `D` are documentation. For `ECN345274`, the delivered workbook contains four retained material occurrences—two unique components under two VC parents—and all four match enriched EBOM rows carrying the same ECN in column J. The spoken three-row count remains unresolved.
- Year-1900 BOM validity dates are confirmed migration placeholders. Retain the records, raw dates and revision chain; add a qualifier and do not substitute another date or use 1900 as the true engineering-event date.
- For Plant C, Make-or-Buy is recorded on the ECN before release. The Power App is downstream implementation coordination for purchased-part availability, PPAP, stock use-up and assembly-line readiness.

## Current implementation readiness

The repository now implements the ECN upload, relational HANA persistence,
Q900V-bounded evidence/snapshot flow, review UI, and controlled analysis
workflow. The [staged impact analysis](ecn-staged-impact-analysis.md) separates
business perspectives, Data Readiness descriptions, and final synthesis. The
authoritative ECN JSON remains relational; a versioned,
rebuildable RDF projection makes the complete selected ECN discoverable through
bounded chat and impact-discovery tools. See [Rebuildable ECN RDF
Projection](ecn-rdf-projection.md) for the current runtime contract, recovery
model, and verified HANA acceptance command.

The older epic documents below remain design specifications where they describe
future customer integrations or business-decision workflows. They must not be
read as evidence that those future capabilities are already implemented.

The following do not block writing code and deterministic unit tests, but they gate customer-valid outputs:

- corrected point-in-time ECN export and approved `Released`/`Resolved` eligibility;
- authoritative affected/resulting pairing and operation semantics;
- plant/KMAT/footprint and configuration dictionaries;
- approved `_Q900` and ECN-to-MPXA mapping rules;
- labeled Make/Buy data with predecessor/successor relations and edge cases;
- separate Make and Buy material-master templates and validation rules;
- HANA credentials and initialization privileges for a live integration run.

## Epic contracts

| Epic | Business question | Contract output | Boundary that must remain explicit |
| --- | --- | --- | --- |
| [Epic 0 - Foundation and Governance](epic-00-foundation-and-governance.md) | What controls, evidence, states, and persistence must every epic use? | Shared HANA-backed data, audit, rule, access, review, and workflow foundation | Provides reusable controls; makes no ECN business decision |
| [Epic 1 - ECN Intake and Normalization](epic-01-ecn-intake-and-normalization.md) | What released change must be analyzed? | `NormalizedChangeSet` or an insufficient-information block | AI may interpret text, but cannot confirm release or object existence |
| [Epic 2 - Applicability Scope Determination](epic-02-applicability-scope-determination.md) | Where might the change apply? | `CandidateScope` with included and excluded candidates and path evidence | Discovers potential scope; does not claim industrial consequences |
| [Epic 3 - Structure Reconciliation and Data Readiness](epic-03-structure-reconciliation-and-data-readiness.md) | What technically changes in each product, configuration, and plant structure? | `ReconciledStructureDelta` plus `DataReadinessReport` | Describes structural deltas; does not prescribe industrial actions |
| [Epic 4 - Industrial Impact Derivation](epic-04-industrial-impact-derivation.md) | What consequences and required actions follow from the reconciled delta? | Governed `ImpactDomainProfile` plus `IndustrialImpactPackage` | Only selected domains are evaluated; consequences require rules, evidence, and human confirmation where judgment is needed |

Epic 0 is foundational and cross-cutting. It is implemented before or alongside the first business epic, then reused by every later epic. It is intentionally not deferred until the end because provenance, status semantics, authorization, and review behavior cannot be retrofitted safely after business decisions have already been produced.

## Shared delivery principles

All five epics follow these principles:

- Store application tabular data, rule versions, workflow state, evidence references, and audit records in SAP HANA. Do not make local CSV or Excel files part of runtime decision logic. User-uploaded files may be accepted as intake artifacts, but their parsed and validated contents are persisted before use.
- On startup or through an idempotent initialization step, check whether required HANA schemas and tables exist and create the missing structures automatically. Initialization must be safe to repeat and must not delete customer data.
- Record evidence at the field, relation, rule, and decision level. A result without traceable source observations and deterministic relations is not promoted to `confirmed`.
- Use deterministic rules for source status, referential integrity, graph traversal, effectivity, structural comparison, and agreed business policies.
- Use AI only for bounded assistance such as interpreting unstructured documents, retrieving comparable cases, and explaining evidence. AI output is a labeled interpretation or assistance record, not evidence. The underlying source excerpt or object and deterministic relations remain the evidence.
- Require authorized human review when policy requires it, including recommendations, exceptions, ambiguous or conflicting source resolution, business-critical confirmed conclusions, overrides, and downstream action decisions. Reviewers can confirm, reject, or override with a recorded rationale.
- Keep Phase 1 at Autonomy Level 0. The application prepares analysis and recommendations; authorized people make decisions. Direct write-back to SAP, Windchill, or other operational systems is outside the initial scope.
- Treat SAP capabilities, transactions, and interfaces mentioned in the sources as reference points or integration candidates until connectivity, permissions, data contracts, and scope are approved.

Technical/data validation readiness is separate from business-decision approval. A deterministically validated intermediate contract may be published to the next epic when its contract is satisfied, while retaining its audit history and any explicit review requirement. Autonomy Level 0 still requires authorized human decisions for business-critical conclusions and actions; it does not impose a blanket approval gate on every normalization, traversal, or structural-comparison step.

## UI and backend relationship

UI navigation is a user workflow, not the backend architecture. A tab may combine read models from several epics, and one epic may feed several tabs. For example, an ECN detail page may show the normalized change, candidate scope, data-readiness warnings, and downstream impact summary without collapsing their separate contracts.

The source-inspired views map approximately as follows:

| UI area | Primary epic | Supporting epics |
| --- | --- | --- |
| ECN Inbox and ECN detail | Epic 1 | Epic 0 |
| Product, configuration, and plant scope | Epic 2 | Epics 0 and 1 |
| Structure comparison and data readiness | Epic 3 | Epics 0, 1, and 2 |
| Impact, actions, evidence, and review | Epic 4 | All earlier epics |

These mappings guide presentation only. They must not create shortcuts around backend validation or provenance.

## Result-state vocabulary

Every externally visible result uses one of the shared result states defined in Epic 0:

- `confirmed`: evidence satisfies the contract and any explicitly required review is complete.
- `potential`: evidence supports investigation, but one or more confirmation conditions remain open.
- `no impact`: deterministic evaluation and sufficient source coverage support exclusion.
- `insufficient information`: required data or evidence is missing, stale, contradictory, or inaccessible.

Workflow state and result state are separate. A record can be awaiting review while its analytical result is `potential`, for example.

## Remaining Phase 1 and future designs

The following areas are acknowledged but are not yet designed in depth here:

- Impact review and CIB coordination, including local and global review views.
- Phase 1 Make-or-Buy recommendation using predecessor/successor, material description, available ECN text and governed historical labels, with calibrated confidence and expert-review abstention.
- Phase 1 selection and population of the approved Make or Buy material-master template, followed by engineer review/edit and a human SAP save.
- Implementation planning, execution tracking, and closure.

Those remaining designs consume the governed outputs defined here. They must not bypass the `NormalizedChangeSet`, `CandidateScope`, `ReconciledStructureDelta`, or `IndustrialImpactPackage` contracts.

Detailed pricing, routing, capacity and internal pricing are deferred for the agreed first Make-or-Buy model. Downstream SIB/CIB execution, production logistics and Power-App coordination remain outside the confirmed Phase 1 endpoint.

## Primary source set

The primary customer source set for these documents is:

1. `SAP_OEM_AI_PoC_Process_Draft_V1.docx` - target scope, operating model, governance, AI opportunities, data categories, user stories, and acceptance criteria.
2. `TDM - AI WS Presentation 1.pdf` - plant process variants and the draft global change-management process, including explicit process inputs and outputs.
3. `TDM - AI WS Presentation.pdf` - byte-identical to the preceding PDF at the time of review and therefore not treated as an independent source.
4. `OEM-SAPAI4U Daily Touchpoint - 9-09.md` plus the user-provided follow-up meeting consolidation received on 12 September 2026 - current data-delivery status, confirmed raw-EBOM document filter, Make-or-Buy inputs and Phase 1 endpoint.

No precedence between the customer Word and PDF sources has been agreed. When they conflict or use incompatible process definitions, the conflict remains explicit and requires OEM resolution.

Vault Markdown notes and supplied screenshots are interpretation aids only. Screenshots inform navigation and information hierarchy, not behavior, system authority, field names, policies, or scope. When a design statement goes beyond an explicit source statement, the epic labels it as an architectural proposal or an open question.

- [ECN snapshot review redesign](snapshot-review-redesign.md): current staged import, object hierarchy, summaries, assistant and review workflow.
- [UI/UX remediation delivery plan](ui-ux-remediation-plan.md): proposed, agent-ready implementation sequence for responsive Inbox triage, decision clarity, assistant accessibility, Settings information architecture, component consistency, and UX validation. Read this before starting a UX remediation work package.
- [ECN demonstration story](ecn-demonstration-story.md): the single curated-ECN story, required decision content, and the pruning gate for UI, data, and generated text.
- [Overview story assessment](ecn-overview-story-assessment.md): the implemented three-area Overview and the pruning record for deferred evidence.
- [Plants & BOM story assessment](ecn-plants-bom-story-assessment.md): the required named plant-to-assembly scope, BOM-alignment evidence, and deferred technical detail for the Plants & BOM tab.
- [Procurement story assessment](ecn-procurement-story-assessment.md): the make-or-buy decision, bounded sourcing evidence, curated-data boundary, and deferred lifecycle/BOM detail for the Procurement tab.
- [Material dialog story assessment](ecn-material-dialog-story-assessment.md): the compact material identity, relationship, and make-or-buy story, plus the evidence and data needed for a defensible cost comparison.
- [Global ECN chat](ecn-global-chat.md): the chat on every page, cross-ECN tools (`ecn` argument, part matching, multi-ECN graph search) and their limits.
- [Engineering review](engineering-review.md): Accept ECN / Reject, the per-part acceptance plan with its missing-price block, and the list Review column.
- [Writing an accepted ECN into SAP ECC](ecc-material-master-posting.md): the BAPI per line, status columns, info records and the Cloud Connector path to on-premise ECC (simulated today).
