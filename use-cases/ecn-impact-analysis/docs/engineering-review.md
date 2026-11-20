# Engineering review

The Engineering review card on the ECN overview is where an engineer closes the
loop on one exact assessment. The card shows:

- one banner line that says where the ECN stands;
- one row per part: what changes in the material master, and whether the part
  has a purchase price;
- at most two actions: **Accept ECN** and **Reject**.

| State | When | Banner | Actions |
|---|---|---|---|
| Blocked | A new part has no price | "Can't accept: 3 new parts have no price." (red) | Reject |
| Ready | Every new part has a price | "Ready to accept: 5 parts change in the material master." | Accept ECN, Reject |
| Accepted | Newest review is `ACCEPTED` | "Accepted by <name> on <date>. SAP updated (simulated)." (green); every row shows "Done" | None |
| Rejected | Newest review is `CHANGES_REQUESTED` | "Rejected by <name> on <date>." (red) plus the comment | None |

- **Accept ECN** asks for a browser confirmation, then saves an `ACCEPTED`
  review. The SAP write is simulated; see
  [ecc-material-master-posting.md](ecc-material-master-posting.md) for how the
  real write would work.
- **Reject** opens a "What must change?" box. The text is
  required, and is saved as a `CHANGES_REQUESTED` review.
- After either decision the actions disappear. A reassessment creates a new
  assessment, which starts undecided.
- Older reviews with a retired outcome (`REVIEWED`, ...) count as undecided.

The newest decision is shown in three places: the card itself, a pill in the
ECN detail header, and the **Review** column of the ECN list. While the ECN is
undecided and blocked, the header pill says "Blocked". "Rejected" uses the red
`error` tone everywhere (`CHANGES_REQUESTED` in `ui/src/modules/review-outcomes.js`),
"Accepted" the green `success` tone.

## "Your engineering review is needed" callout

While an ECN has no saved review at all, an amber callout sits directly under
the ECN detail header (title and status pills):

- **What:** "Your engineering review is needed" with the hint "Review the
  impact analysis, then accept or reject the ECN." and a **Review and decide**
  button.
- **When shown:** the review history for the displayed assessment is empty.
  Any saved review hides it, including a retired outcome such as
  `CORRECTION_RECORDED`, and including a review saved by the ECN assistant
  (the card re-renders on `ecn-review-saved`). Blocked ECNs still show it,
  because Reject stays available. It stays hidden while the card loads and if
  the acceptance plan fails to load.
- **Button:** opens the collapsed **Engineering review** `<details>` section
  and smooth-scrolls it into view.

It is built in `reviewNeededCallout` and toggled in the card's `render()`, so
it always agrees with the header pill:

```js
// ui/src/components/engineering-review.js (mountReview)
const callout = badgeHost ? reviewNeededCallout(host) : null;
badgeHost?.closest(".detail-header")?.after(callout);
// ...inside render()
if (callout) callout.hidden = reviews.length > 0;
```

The callout is only created when the card is given the header badges
(`badgeHost`), that is on the ECN detail page. Styles: `.review-needed` in
`ui/src/pages/ecn-detail/ecn-detail.css`.

## What saving does

1. The browser sends `POST /api/ecns/{id}/reviews` with:
   - `request_id`, `reviewer`, `outcome` (`ACCEPTED` or `CHANGES_REQUESTED`) and `comments`;
   - the displayed snapshot identity: `evidence_hash` and `snapshot_version`.
2. The API rebuilds the snapshot. If its identity no longer matches, it returns
   409 "Reload this snapshot". An `ACCEPTED` save on a blocked ECN returns 409
   `ECN_ACCEPT_BLOCKED`.
3. One row is appended to the HANA review table. The table has these columns:
   `REVIEW_ID, ECN_ROW_ID, EVIDENCE_HASH, SNAPSHOT_VERSION, REVIEWER, ACTOR_ID
   (sha256 of the API key), OUTCOME, COMMENTS, CREATED_AT`.
   - Rows are never overwritten.
   - Replaying the same `request_id` with the same content is a no-op.
   - Reusing it with different content returns 409.
4. The API returns the full history, and the card re-renders it.

A review belongs to one assessment (`ECN_ROW_ID`). A reassessment creates a new
assessment, which starts without reviews. Reviews never change source facts, AI
results or findings.

Older rows may carry the retired outcomes `REVIEWED`, `CLARIFICATION_REQUESTED`
and `CORRECTION_RECORDED`. They stay readable in the history and show as
"Reviewed" in the header and the list. They can no longer be submitted.

## Acceptance plan

`GET /api/ecns/{id}/acceptance-plan` returns the rows the card shows. It is
built by `build_acceptance_plan(snapshot)` (`api/app/ecn/evidence/acceptance_plan.py`)
from the snapshot's `bom_changes.materials` and `plant_impact`. The snapshot
cannot change once built (it is pinned by `evidence_hash`), so the plan is
recomputed on each request and not stored.

| Part | Material master line | Purchase price |
|---|---|---|
| New part, already set up in the plant | "Release in PL10 from 2026-03-13" | Has price / Missing |
| New part, not yet set up in the plant | "Create and release in PL10 from 2026-03-13" | Has price / Missing |
| Old part, fully phased out | "Phase out in all plants from ..." | Not needed |
| Old part, no remaining use found or timing unknown | "Phase out in PL10 from ..." or "... date not known" | Not needed |
| Old part, still used elsewhere | "No change: still used in Q938VV0080403" | Not needed |
| Phase-out declared but not confirmed | "No change: phase-out not confirmed" | Not needed |

**Price rule.** The data has no prices. A new part counts as "Has price" when
its material number appears in the purchasing info record dataset, in **any**
plant (`plant_impact.priced_new_parts`). A new part that does not appear there
is "Missing", and **any** missing price blocks the ECN. Old parts never need a
price.

The block is enforced in three places:

- the card hides Accept ECN;
- `record_review` refuses `ACCEPTED` with `ECN_ACCEPT_BLOCKED` (409) and the
  banner text as message;
- the assistant's `draft_review_decision` refuses an Accept draft.

BOM item changes are not material master data. The card only says
"15 BOM item changes are handled outside the material master."
Material status codes are never shown.

Example response (shortened):

```json
{
  "available": true, "blocked": true,
  "block_reason": "Can't accept: 1 new part has no price.",
  "missing_price": ["QKX3105210"],
  "parts": [
    {"number": "QKX0187890", "description": "STICKER DIESEL FUEL OIL", "role": "NEW",
     "master_data": "Release in PK01, PL10 from 2026-03-13", "price": "HAS_PRICE"},
    {"number": "QKX3105210", "description": "AUTOCOLLANT JB PS POINTS DE GRAISSAGE", "role": "NEW",
     "master_data": "Create and release in PL10 from 2026-03-13", "price": "MISSING"},
    {"number": "QKX2833570", "description": "LUBRIFICATION DECALCOMANIE", "role": "OLD",
     "master_data": "Phase out in PL10 from 2026-05-18", "price": "NOT_NEEDED"}
  ],
  "bom_item_count": 15
}
```

Version 1 snapshots have no BOM changes. Their plan has no parts and
`available: false`, so nothing blocks them.

## Files

| Area | File |
|---|---|
| Decision contract and save route | `api/app/routers/workspace_actions.py` (`ReviewSubmission`, `save_review`) |
| Append-only persistence | `api/app/persistence/workspace_repository.py`, `workspace_contracts.py` |
| Acceptance plan and price rule | `api/app/ecn/evidence/acceptance_plan.py`, `plant_impact.py` (`priced_new_parts`) |
| Plan route | `api/app/routers/snapshots.py` (`acceptance_plan`) |
| Block on save | `api/app/ecn/services/reviews.py` (`record_review`) |
| List Review column | `api/app/persistence/ecn_repository.py` (`_list_sql`, ranked review join), `codec.py`, `api_models.py` (`EcnListItem.review_outcome`), `ui/src/pages/ecns/` |
| Review card and "review needed" callout | `ui/src/components/engineering-review.js`, styles in `ui/src/pages/ecn-detail/ecn-detail.css` |
| Shared labels | `ui/src/modules/review-outcomes.js` |

The list query uses a window function because HANA rejects `ORDER BY` inside a
correlated subquery (error 309):

```sql
LEFT JOIN (SELECT R."ECN_ROW_ID", R."OUTCOME", ROW_NUMBER() OVER (
  PARTITION BY R."ECN_ROW_ID" ORDER BY R."CREATED_AT" DESC, R."REVIEW_ID" DESC) AS "REVIEW_RANK"
  FROM "ECN_IMPACT_REVIEW" R) RV ON RV."ECN_ROW_ID" = E."ECN_ROW_ID" AND RV."REVIEW_RANK" = 1
```

## Saving a review from the ECN assistant

The chat assistant can also record a decision. It turns the user's remarks into
a review comment, shows the draft, and saves it only after the user confirms in
chat. Both paths call `record_review` in `api/app/ecn/services/reviews.py`, so
the snapshot check, the stored columns and the hashed `ACTOR_ID` are identical.
After an assistant save the chat dispatches the window event
`ecn-review-saved`, and this card reloads its history. See
`docs/ecn-assistant-decision-tools.md`.

## How to test

```bash
PYTHONPATH=api .venv/bin/python -m pytest -q tests/unit/test_acceptance_plan.py tests/unit/test_plant_impact.py tests/unit/test_assistant_decision_tools.py
```

```bash
npm --prefix ui test -- snapshot-workspace
```

Manual check:

1. Open ECN346863 and open **Engineering review**. The banner says
   "Can't accept: 7 new parts have no price.", the seven rows show "Missing",
   and only Reject is offered.
2. Click Reject, and click Confirm reject without text. The form
   is blocked with "Say what must change."
3. On a ready ECN, Accept ECN without a name shows "Enter your name."; with a
   name it asks for confirmation, then shows the green banner and "Done" on
   every row.
4. After a save, the header pill and the list Review column show the decision;
   a rejected ECN shows a red "Rejected" pill.
5. On an ECN with no review, the amber "Your engineering review is needed"
   callout shows under the title. **Review and decide** opens the Engineering
   review section and scrolls to it. After saving a decision the callout
   disappears.
