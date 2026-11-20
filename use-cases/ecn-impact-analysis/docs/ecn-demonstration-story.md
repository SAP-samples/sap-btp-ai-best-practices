# ECN Demonstration Story

## Purpose

Demonstrate one curated ECN from change through the decision a Change Integration Board needs to make. The application must let a non-specialist understand the story at a glance; it is not a complete ECN-data browser.

The chosen ECN must have usable part descriptions, named plant impact, BOM relationships, lifecycle status, and make-or-buy/cost evidence. Demonstration assumptions are permitted where source data is absent, but they must be labeled at the claim they support.

## The Story

1. **What changed and why?**
   The ECN changes a plainly named part or assembly from its current state to its proposed state. Show a short change description, business reason, owner, and effective/cutover date.

2. **Where and to whom does it matter?**
   Name the affected plants and the products or assemblies affected at each. State whether the engineering and manufacturing BOMs are aligned, or name the gap that requires attention.

3. **What is the lifecycle impact?**
   Show what is being phased in and what is being phased out, with the relevant status or date. Call out a retirement/use-up risk only when it changes the decision.

4. **Should the new or changed part be made or bought?**
   State the proposed make-or-buy outcome, its plain-language rationale, and the relevant cost, lead-time, sourcing, or manufacturing implication. A recommendation may rely on curated demonstration data and comparable history, but must identify that basis.

5. **What must happen next?**
   End with the decision or the smallest set of actions preventing it. Every action must name the affected plant, part, or assembly and its owner. The result is an approve, hold, or investigate decision, not a generic readiness score.

## Required Page Sections

Each section exists only to answer one story question.

| Section | Supports | Must show | Must not show |
| --- | --- | --- | --- |
| **Change at a glance** | 1 | Plain-language change, reason, owner, cutover date | Raw ECN prose, unexplained identifiers, duplicate AI summaries |
| **Affected plants and BOMs** | 2 | Named plants, affected assemblies, EBOM/MBOM alignment or a specific gap | A plant count without plant names, an unbounded BOM tree, generic coverage metrics |
| **Phase-in / phase-out** | 3 | Incoming and outgoing parts, status/date, decision-relevant risk | Lifecycle codes without their meaning, unrelated history |
| **Make or buy and impact** | 4 | Proposed outcome, why, cost/lead-time implication, sourcing or manufacturing risk | Unsupported precision, exhaustive procurement records, calculations unrelated to the outcome |
| **Decision and actions** | 5 | Recommendation, direct blockers, named next action and owner | Generic notes, generic "confirm" tasks, readiness percentages, duplicate warnings |

The overview must show the plain-language change and named affected plants without requiring a click. Plant, BOM, and procurement pages may provide drill-down evidence for the same story; they must not introduce a separate story.

## Content Rules

- Prefer part and assembly descriptions. Show an identifier only as secondary detail or when no description exists.
- Use one short, factual sentence for the change. Do not present the raw description as the primary narrative.
- Show a fact, a risk, or an action only if it changes the answer to one of the five story questions.
- A blocker is valid only when resolving it is necessary for the decision. It must say what is blocked, where, and who acts.
- Keep manufacturing impact high-level unless the demonstration specifically relies on a time, capacity, or tooling claim.
- Do not claim plant tooling, work instructions, procurement availability, or costing as source facts when they are demonstration assumptions.
- Preserve traceability and detailed evidence behind a drill-down; do not put it in the decision narrative by default.

## Exclude or Defer

Remove from the demonstration flow unless it directly supports one of the five questions:

- Aggregate completeness, coverage, readiness, or note counts.
- Empty-state warnings and missing-field inventories that do not block the selected decision.
- Raw source descriptions, long generated summaries, opaque part numbers, and unexplained status codes.
- Full audit trails, generation diagnostics, source-ingestion detail, and technical model state.
- Exhaustive BOM, procurement, plant, tooling, stock, scheduling, or cost records.
- Charts or KPIs that do not change the decision or lead directly to its evidence.

## Validation Gate

Before retaining a UI element, data field, API response, generated statement, or test fixture, answer all of the following:

1. Which one of the five story questions does this support?
2. Does it make that answer clearer or provide the required drill-down evidence?
3. Is it expressed in plain language with a named part, plant, or action?
4. If the data is assumed, is the assumption attached to the claim?

If any answer is no, remove the item from the demonstration flow or defer it to a supporting drill-down.

## Decision-Blocker Check

The current **Decision blockers and next actions** section is only partly aligned.

- **New parts have no work instruction** and **New parts have no purchase info record** can support questions 4 and 5 only when they name the new part, affected plant, decision consequence, and accountable owner.
- **Confirm the scheduled manufacturing rollout**, **Confirm missing stock fields**, and **Confirm the model and plant coverage** are not decision-ready story elements as written. They are generic data-validation work, not a clear ECN impact or action.
- The `9 of 9` coverage statement, `3 notes`, and affected counts are supporting metrics rather than the story. Replace them with the named plants, affected assembly/part, and the single decision consequence, or defer them to drill-down evidence.

## Source Basis

This scope follows the call with Dana Whitfield:

- Keep the original ECN description only to the degree needed to explain the change, owner, and high-level overview (`00:01:06`-`00:01:51`).
- Surface named affected plants without making the user click to discover them (`00:03:17`-`00:04:14`).
- Answer the analyst's who, what, where, when, how, why, and cost questions immediately (`00:04:46`-`00:05:29`).
- Select a few ECNs, especially one complete enough to tell the full story; do not "boil the ocean" (`00:08:41`-`00:10:26`).
- Explain the make-or-buy rationale and resulting cost/risk at a high level, rather than modeling every operational variable (`00:33:11`-`00:41:25`).
- Make the experience simple enough that a non-specialist immediately knows what changed and what must happen next (`00:42:25`-`00:43:50`).
