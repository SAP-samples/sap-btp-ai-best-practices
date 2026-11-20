# ECN detail review clarity

The ECN detail page guides an engineer from impact to specific evidence and a
next action. Data readiness is derived from the assessed source data, available
without an AI summary, and never lowers the business-impact severity.

## Inputs and outputs

Inputs are the existing HANA-pinned assessment, its technical objects, current
and historical BOM occurrences, stock rows, source publication metadata, saved
AI response and assessment/review history. No persistence schema changes or
new runtime file-based datasets are introduced.

The snapshot's existing `questions` collection retains its question, explanation,
resolution, code and count. Each group now includes the affected `subjects`
(material number, inspection identity, readable source facts), `scopes` checked
and related impact `factors`. For example:

```python
{
    "code": "BOM_LOCATION_UNAVAILABLE",
    "question": "Confirm intended BOM placement",
    "subjects": [{"number": "SIM9800", "object_key": "…", "details": ["…"]}],
    "scopes": ["bom.xlsx · SIM-RANGE-B · P101 · 2026-09-22"],
    "factors": ["manufacturing", "configuration"],
    # Existing affected_count, why and resolution fields remain present.
}
```

Missing occurrences, future or expired validity, unresolved validity, continued
phase-out use, missing manufacturing introduction, lifecycle ambiguity and
incomplete declared coverage have different next actions. An absent occurrence
does not establish an expected parent, plant or production configuration.
Historical occurrences never prove current use. A confirmed coverage-bounded
phase-out is not turned into a missing-placement question.

Stock follow-ups identify material, plant, storage location, file, sheet and row.
They request only missing unit/date fields, retain known extraction dates and
reported zero quantities, and exclude comparison records. The current stock
import shape does not carry a unit column: confirming a unit in Engineering
review is a reviewer statement, not a mutation of the source or an automatic
clearance of the evidence gap.

## Page behavior

- Source Record contains Change summary followed by Assessment: the BOM
  assessment date, History and Reassess with current data. History labels use
  creation timestamps, clearly separate from the BOM assessment date.
- Reassessment creates a new snapshot using current published sources. Prior
  summaries and reviews remain available. AI summary generation is a separate
  action on the new assessment. Retrying the same date uses the same request ID.
- Three impact cards link to a full-width What needs attention section. Named
  deterministic findings include numbered titles, material inspection buttons,
  checked scope, source facts and explicit next steps. The AI writes one
  description per deterministic issue code, with the source explanation as
  fallback. Complementary actions from the three impact analyses appear under
  Additional Next Steps, grouped across factors when their text matches.
  Review next steps scrolls and moves keyboard focus to that section without
  navigating through the router, preserving unsaved reviews and chat state.
  Missing-material findings explicitly say no matching component was found in
  the supplied EBOMs or MBOMs, so those files cannot establish its parent or BOM
  hierarchy. Manufacturing-introduction findings distinguish present EBOM
  evidence from the absence of an MBOM row valid on the assessment date.
- Supporting analysis opens the saved detailed narrative in a native dialog
  beside the impact heading, with Escape and focus return.
- Explore evidence groups the technical-object inventory, additional evidence
  and Data used. Source filenames, scopes and extraction dates remain visible
  on request. Hashes, raw JSON paths and duplicate enum fields are absent from
  the user-facing displays; their underlying immutable records remain intact.
- Source preparation is visible only when pending, failed, stale or unavailable.
  Retry remains available for failed/stale preparation; reload refreshes chat
  availability after recovery. Healthy diagnostics are hidden.
- The Review status / Decision readiness card remains removed. The current footer clarification is documented below; it does not restore this card or change acceptance/review rules.
- Source breadcrumbs use one-based positions with or without an export-array
  prefix: `Implementation plan 1 › Resulting link 2 › Object 1` and
  `Affected link 1 › Object 2`.
- The demo-only write-back suffix is removed from the displayed description and
  future generated demo inputs. Imported original source evidence is unchanged.

Stored AI summaries keep their original text. The new synthesis instructions
request concrete named next actions for future generations; reading an existing
assessment never silently regenerates or rewrites its summary. Shared readiness
logic and UI changes apply on read. Synthesis bounds subject examples while
retaining full affected counts and complete material evidence in the workspace.

## Related code

- `api/app/ecn/evidence/review_questions.py`: builds follow-ups from final snapshot facts.
- `api/app/ecn/evidence/snapshot.py`, `scoped_usage.py`: shared pinned evidence and usage.
- `api/app/ecn/llm/synthesis.py`, `validation.py`, `two_pass.py`: bounded context,
  deterministic readiness and actionable model instructions.
- `ui/src/components/assessment-context.js`: assessment controls, dataset display and follow-ups.
- `ui/src/components/snapshot-workspace.js`: material inspection, evidence grouping and review.
- `ui/src/components/material-dialog.js`: shared source breadcrumbs and material evidence.
- `ui/src/pages/ecn-detail/ecn-detail.{html,js,css}`: page hierarchy, impact and narrative dialog.

## Current acceptance footer (feedback 08)

The verified current paths are `ui/src/pages/ecn-detail/ecn-detail.{html,js,css}`,
`ui/src/components/audit-log.js`, `api/app/ecn/evidence/acceptance_plan.py` and
`api/app/ecn/services/reviews.py`. This clarification supersedes any older
Engineering review card/ready-state description; no such card is restored.

- `#accept-ecn-footer` still uses `blocked || decided`; `#send-back-footer`
  still uses `decided`. No source-follow-up or geometry blocker is introduced.
- `#accept-ecn-reason` retains the server's specific `block_reason` literally.
  The existing fallback is unchanged. The nearby link scrolls to
  `#bom-changes` for blocked acceptance or `#audit-log-card` for a recorded
  decision. Existing blocked/decided priority is unchanged.
- `#acceptance-rules-open` is a compact native UI5 button in the Bar. It opens
  `#acceptance-rules`, a native UI5 Dialog outside DynamicPage, with Close,
  initial focus, Escape and focus return. Content is a named, focusable scroll
  region bounded to 60vh. Public Button `accessibilityAttributes` describes
  controls/expanded/dialog semantics; `accessibleDescription` describes the
  actual acceptance button. Host-only ARIA is not relied on.
- The literal blocker/evidence link and prompted reviewer input are outside the
  Bar, within its `footerArea` wrapper, so multiline content does not leak from
  the Bar's fixed height. Long reasons scroll, and page content reserves footer
  space. This supersedes the earlier fixed help overlay, which was obscured by
  sticky section navigation. **Send back** retains the accessible name **Send
  back to engineering**. No Review status / Decision readiness card is restored.
- Incoming/new material requires membership in `plant_impact.priced_new_parts`,
  derived from purchase-info-record data in any plant. This is the current
  server surrogate for price evidence, not a numeric/current-price guarantee.
  Missing incoming entries block acceptance; outgoing/old material requires
  no price evidence. The footer does not recompute the plan.
- Accept still prompts for reviewer name, requires the existing browser
  confirmation, and saves the current snapshot hash/version. The server checks
  snapshot identity, reviewer name and acceptance preconditions again. The
  SAP update is simulated: the action records an engineering review only.
  Send-back still requires comments and reviewer name; save/API shapes are
  unchanged.
- A pending, failed or `available: false` plan is not labeled ready. Existing
  missing-plan enablement is deliberately preserved; explanatory help remains
  available. This clarification does not repair the existing save-error UI.

Focused synthetic regression: `ui/tests/oem-feedback-08-acceptance.test.js`.
Native UI5 browser regression: `ui/tests/e2e/oem-feedback-interactions.spec.js`.
Parent checks passed; final results are in [the delivery ledger](ai/oem-feedback-delivery.md).
Re-run:

```sh
cd ui && npm test -- tests/oem-feedback-08-acceptance.test.js tests/ecn-detail-page.test.js
cd ui && npm test
cd ui && npm run build
```

No typecheck script exists in `ui/package.json` (plain ESM). Parent synthetic
browser checks verify real UI5 forwarding, keyboard dialog behavior, long-reason
footer containment, open-help hit-testing/scrolling, BOM navigation and reviewer
prompting at 1280×720 and 390×844. Mocked unit tests preserve confirmation,
server-bound save, Audit destination and cleanup contracts. Screen-reader,
high-contrast and native PDF-plugin checks remain manual. No customer-backed
browser validation was performed.

## Material dialog: usage and validity table

The "Usage and validity" table in `ui/src/components/material-dialog.js` presents each BOM
occurrence as follows:

- **Hierarchy, EBOM rows** (`kind` `ENRICHED_EBOM`): starts at the VC group. Main, Sub and
  product-context nodes are dropped and the summary line is removed. The remaining nodes are
  stacked one per line, still clickable, and tagged by each node's own EBOM `Level`, not by
  its position in the chain: level 3 (VC group) is `150%`, level 4 is `130%`, level 5 and
  deeper are `100%`. A part directly under its VC group therefore reads `130%`. Nodes
  without a level (parent context) are untagged. See `hierarchyNodes`.
- **Hierarchy, MBOM rows**: unchanged, the full chain with its collapsible summary and no tags.
- **Validity**: `Start date: 25 Feb 2022`. `End date: 20 Apr 2022` appears only when a real end
  exists; an empty end or `9999-12-31` renders nothing. Legacy placeholders and uninterpretable
  end values stay visible. See `endDateLine`. The raw "Technical source details" block is unchanged.

Covered by `ui/tests/material-dialog.test.js`.

## Verification

```sh
PYTHONPATH=api .venv/bin/python -m pytest -q
npm --prefix ui test
E2E_PORT=4187 npm --prefix ui run test:e2e
```

Focused regressions cover missing material identity, future-use distinction,
source coordinates and dates for stock, readable paths in both dialogs,
history placement, exact-snapshot reviews, independent reads, safe text rendering,
dialog behavior and desktop/mobile containment. The browser suite builds the
production UI before running. Live HANA/provider suites require their existing
explicit opt-ins; deployment is manual.
