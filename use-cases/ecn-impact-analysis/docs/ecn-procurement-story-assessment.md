# Procurement Story Assessment

## Scope

This report assesses the **Procurement** tab against question 4 of the [ECN Demonstration Story](ecn-demonstration-story.md): **Should the new or changed part be made or bought?** The tab is a decision brief, not a purchase-record browser or a manufacturing-readiness dashboard.

It does not reassess the shared ECN header, the Overview conclusion, Plants & BOM scope/alignment evidence, History & Audit traceability, or the decision footer. Those areas are covered by the [Overview Story Assessment](ecn-overview-story-assessment.md) and [Plants & BOM Story Assessment](ecn-plants-bom-story-assessment.md).

**Implementation status:** The Procurement tab now presents an AI-assisted make-or-buy recommendation for each selected material-and-plant pair, with a compact unit-cost comparison and an optional decision-basis disclosure.

## Decision Story

The tab must let a non-specialist answer the following without reading source rows, machine inventories, or a full purchase history:

1. Which material at which plant is being decided?
2. Does the AI recommend **make**, **buy**, or **buy pending quote**?
3. What is the available buy unit price and estimated make unit cost, or which cost is unavailable?
4. What is the one-sentence reason for the recommendation and what must happen next?
5. Which factors did the recommendation consider, if the reviewer chooses to inspect them?

A material-master `ProcType` is a permission, not the recommendation. `E`, `F`, and `X` must never become Make, Buy, or Make-or-buy recommendations without the dedicated decision input described below.

## Main-Section Verdict

The Procurement tab has one visible decision area. It does not repeat MBOM gaps, all phase schedules, supplier records, stock, or machine inventory. The recommendation considers those relevant facts, but the default view communicates only their decision consequence.

| Content | Decision | Comment |
| --- | --- | --- |
| **AI-assisted make-or-buy recommendation** | Keep | States that the recommendation considers manufacturing history, tooling, capacity, raw materials, and unit cost. It also makes clear that human review is required. |
| **Material-and-plant recommendation table** | Keep | Bounded to recommendation records, not all procurement records. It identifies the material and plant, outcome, unit-cost comparison, and concise decision/next-action statement. |
| **Decision basis disclosure** | Keep, collapsed | Shows the comparable-history, tooling, capacity, raw-material, and cost inputs for one material and plant only when the reviewer asks for them. |
| **Purchase setup, supplier, price-validity, information-record, stock, full tooling, work-instruction, and phase-schedule tables** | Defer | These remain traceability or operational evidence. Promote only the one fact that changes the recommendation or direct action. |
| **MBOM gaps** | Remove from Procurement | BOM alignment answers where the change applies and remains in Plants & BOM. |

## Minimum Procurement View

### 1. Recommendation Context

Display one concise statement before the table:

> Recommendations weigh comparable manufacturing history, tooling, factory time, raw-material availability, and unit cost. Human review is required before an operational decision.

This explains the scope of the AI assessment without making the analyst reconcile factor cards, readiness counts, or raw evidence.

### 2. Recommendation Table

Show only one row per material and plant with a curated recommendation:

| Required field | Content |
| --- | --- |
| **Material** | Description first; material number secondary. |
| **Plant** | Full plant name first, code secondary. |
| **AI recommendation** | Make, Buy, or Buy pending quote. This is a proposed outcome, not a material-master permission. |
| **Unit cost** | `Buy: EUR 0.85 / PC` or `Buy: unavailable`, plus the equivalent make estimate. Show a saving only when currency, unit, and price quantity match. |
| **Decision** | One factual rationale, direct next action, and accountable owner. |

Do not show every criterion in this table. Do not show a supplier name, price-validity period, information-record number, raw capacity percentage, tooling identifier, stock quantity, or schedule note unless it changes the concise decision.

### 3. Decision Basis

Place one collapsed disclosure below the table for each recommendation:

| Required field | Content |
| --- | --- |
| **Comparable history** | Whether this plant or a comparable process has made the part or equivalent before. |
| **Tooling and process** | Whether the required manufacturing process is qualified and available. |
| **Factory time** | The capacity conclusion, not a raw utilization dashboard. |
| **Raw materials** | Whether the material needed to make the part is available. |
| **Buy and make unit cost** | The normalized unit values or the exact unavailable value. |
| **Next action** | The direct action and accountable owner that prevents the recommendation from progressing. |

This is decision evidence, not proof of configured production use, approved supplier availability, or a final automatic operational change.

## Curated Demonstration Contract

`prototype/mock_data/mock_make_buy_recommendations.csv` is the curated demonstration contract for this capability. It is keyed by ECN number, material number, and plant code, and supplies the reviewed recommendation, concise rationale, owner, make cost, and five decision inputs. `mock_purchase_prices.csv` supplies the buy side independently, as one normalized order-history unit price or no price.

The current overlay is a demonstration stand-in for a governed AI output. A production implementation must send these bounded, attributable inputs to a dedicated recommendation workflow; preserve the prompt/result, evidence references, model version, reviewer, final human decision, and override reason. It must not reuse the ECN impact-summary model to infer a sourcing decision from descriptions or procurement permissions.

## Demonstration Outcomes

| ECN / material / plant | Recommendation | Story conveyed |
| --- | --- | --- |
| ECN346863 / diesel fuel sticker / Plant B (PK01) | **Make** | Comparable decal history, qualified equipment, available time, and material stock support EUR 0.42 / PC make cost versus EUR 0.55 / PC buy price. |
| ECN346863 / diesel fuel sticker / Plant A (PL10) | **Buy** | The recorded EUR 0.85 / PC buy price is below the EUR 1.10 / PC local make estimate; the plant can apply but does not print the decal. |
| ECN346863 / JB greasing sticker / Plant A (PL10) | **Buy pending quote** | The plant has application capacity but no in-house print process or printed stock; a supplier quote and purchase setup are required. |
| ECN345274 / injection plastic cover / Plant A (PL10) | **Buy pending quote** | The compatible injection press is unavailable before phase-in and the preceding 3D-print process costs EUR 38 / PC; obtain an external production quote. |

## Acceptance Criteria

The Procurement tab meets the demonstration story when all of the following hold:

1. The default view is about a make-or-buy recommendation, not purchasing-record completeness.
2. Every recommendation names the material and plant and states Make, Buy, or Buy pending quote.
3. The table shows one buy unit price or `unavailable`, one make unit cost or `unavailable`, and a savings claim only after a shared basis is confirmed.
4. The table gives a concise rationale, direct next action, and accountable owner without exposing every decision factor.
5. Comparable history, tooling, factory time, raw-material availability, and cost inputs are available in a collapsed disclosure for the exact material-and-plant decision.
6. Material-master procurement permissions are not displayed or interpreted as recommendations.
7. MBOM alignment, full schedules, supplier records, stock, tooling inventory, and work instructions do not form separate Procurement narratives.
8. The AI recommendation remains subject to human review and does not trigger an operational purchase or manufacturing change.

## Relevant Implementation Files

| File | Current responsibility |
| --- | --- |
| `ui/src/components/procurement-risk.js` | Renders the recommendation table, normalized unit-cost comparison, and collapsed decision-basis disclosures. |
| `ui/src/pages/ecn-detail/ecn-detail.js` | Promotes the leading Procurement recommendation into the Overview decision factor and directs pending-quote actions into the Overview decision. |
| `api/app/ecn/mock_overlay.py` | Adds the curated recommendation input, buy prices, schedules, and other demonstration values to selected snapshots. |
| `prototype/mock_data/mock_make_buy_recommendations.csv` | Defines the material-and-plant recommendation, make-cost, rationale, owner, next action, and decision inputs. |
| `prototype/mock_data/mock_purchase_prices.csv` | Defines the one available normalized buy unit price or missing purchase-history state. |
| `docs/ecn-demonstration-story.md` | Defines the required decision story and the human-review boundary. |
