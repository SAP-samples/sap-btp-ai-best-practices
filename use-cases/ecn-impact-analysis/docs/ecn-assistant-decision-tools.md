# ECN assistant decision tools

## What it is and why

The ECN assistant (chat panel on the ECN details page) used to answer "what
should I do next?" without knowing whether the ECN was accepted, and referred
to "the two new parts" without naming them. This feature gives the assistant:

- review status: whether the ECN is accepted, has changes requested, or is
  unreviewed, and what an acceptance covers in SAP;
- the stored AI impact analysis;
- data-quality items that need attention, with their exact part numbers;
- a draft, confirm and save flow for Accept or Request-changes reviews, which
  turns the user's remarks into the review comment.

The system prompt requires exact part numbers in every answer that refers to
parts, and a decision line before any list of next steps.

Chat answers follow a fixed format (the "Chat answer format" block in
`BASE_PROMPT`, which overrides the shared analysis `STYLE` for chat only):

```
Impact: Medium
What changed:
1. QKX4638810 is replaced by QKX463881A
2. QKX8914160 is replaced by QKX891416A
EBOM Release Date: 1/16/2026
MBOM Cutover Date: 6/30/2026
```

- The stored AI analysis, its currency and model are never mentioned.
- Facts that do not change (for example an unchanged quantity) are left out.
- Gap, caveat and coverage statements appear only when the user asks about
  gaps, data quality, readiness or next steps.

## Tools

All tools are closures in `api/app/ecn/assistant/tools.py` (`create_tools`).
They are bound to the server-selected ECN and pass through `_bounded()`.

| Tool | Input | Output |
|---|---|---|
| `get_review_status` | none | `current_decision` (`NONE`, `ACCEPTED`, `CHANGES_REQUESTED`, or a legacy outcome), `latest` and `history` (last five, newest first) with `applies_to_current_snapshot`, `total`, `acceptance` (`blocked`, `block_reason`, up to ten `parts` with `master_data` and `price`, `parts_total`, up to ten `missing_price` part numbers and their total, `sap_write: "simulated"`) |
| `get_impact_analysis` | none | `available: false`, or `is_current`, `overall_impact`, `completeness`, `model_name`, `created_at`, `impact_factors{severity, assessment}`, `sections{status, assessment}` (1500 characters max), `next_steps` |
| `get_data_quality_issues` | `page` (1+) | `readiness_questions` (code, question, why, resolution, affected_count, `parts`), `readiness_total`, `data_quality` and `missing_evidence` pages, `has_next`; five per section per page |
| `draft_review_decision` | `outcome`, `reviewer`, `comments` | `drafted`, the normalized draft, `current_decision`, `warnings`, `confirmation_required: true`; or `error_code` `REVIEWER_REQUIRED` / `COMMENTS_REQUIRED` / `ECN_ACCEPT_BLOCKED` (Accept while a new part has no price; `message` is the one-line reason). Writes nothing. |
| `submit_review_decision` | `outcome`, `reviewer`, `comments` | `saved: true`, `created_at`, `sap_updated: false`; or `saved: false` with `DRAFT_NOT_SHOWN`, `ALREADY_SAVED`, `REVIEWS_UNAVAILABLE`, `REVIEW_SNAPSHOT_MISMATCH`, `ECN_ACCEPT_BLOCKED`, ... |

Data sources:

- Reviews come from `WorkspaceRepository.reviews` (`ECN_IMPACT_REVIEW`).
- The acceptance plan comes from `build_acceptance_plan(snapshot overview)` (`api/app/ecn/evidence/acceptance_plan.py`). See [engineering-review.md](engineering-review.md).
- The analysis comes from `EcnRepository.get_ecn_detail(...).analysis` (`ECN_IMPACT_ANALYSIS`).
- Readiness questions come from the snapshot overview; the stored warnings come from `paginate_business_evidence`.

## Draft, confirm, save

The procedure lives in the on-demand agent skill
`api/app/ecn/assistant/agent_skills/review-decision/SKILL.md`. The base prompt
only tells the model to load it before accepting or requesting changes.

1. Read the status. Ask for the reviewer name if it is missing.
2. Write a professional comment from the user's remarks. Keep all their points, name the parts, VC groups and plants, and add nothing the user did not say.
3. Call `draft_review_decision` and show the draft verbatim: decision, reviewer, and the comment in a quote block.
4. Call `submit_review_decision` only after the user's next message confirms that exact draft.

The server enforces two guards, whatever the model does:

- `draft_was_shown` requires the previous assistant turn in the browser history to contain the comment (quote markers and whitespace normalized), the reviewer name and the decision wording. Otherwise it returns `DRAFT_NOT_SHOWN`.
- It saves at most one review per answer. The request ID is `uuid5(ecn_row_id | evidence_hash | outcome | reviewer | comments)`, so a replayed confirmation saves once.

```python
if not draft_was_shown(history, outcome, reviewer, comments):
    return _bounded({... "saved": False, "error_code": "DRAFT_NOT_SHOWN", ...})
reviews = await run_blocking(record_review, snapshots, workspace, ecn_row_id,
                             request_id=request_id, ..., actor=actor, ...)
state["saved"] = True
```

`record_review` (`api/app/ecn/services/reviews.py`) is the only code that
writes `ECN_IMPACT_REVIEW`. The review route `POST /api/ecns/{id}/reviews` and
the tool both use it. It checks the snapshot identity, refuses `ACCEPTED` with
`ECN_ACCEPT_BLOCKED` (409) while a new part has no price, stores the row, and
returns the history. The chat route hashes the caller's `X-API-Key` into
`ACTOR_ID`, exactly as the review route does.

The chat response carries `review_saved`. When it is true,
`ui/src/components/ecn-chat.js` dispatches `ecn-review-saved`, and the
Engineering review card (`engineering-review.js`) reloads its history and
header badge.

## Related files

- `api/app/ecn/assistant/tools.py`: the tools and the helpers `review_status`, `acceptance_summary`, `analysis_summary`, `readiness_question`, `review_problem` and `draft_was_shown`.
- `api/app/ecn/assistant/runtime.py`: the prompt rules; passes `workspace`, `history`, `actor_id` and per-turn `review_state`; returns `review_saved`.
- `api/app/ecn/assistant/agent_skills/review-decision/SKILL.md` and `inspect-ecn/SKILL.md`.
- `api/app/ecn/services/reviews.py`, `api/app/routers/workspace_actions.py`, `api/app/routers/assistant.py` and `api/app/main.py`.
- `ui/src/components/ecn-chat.js` and `ui/src/components/engineering-review.js`.
- Tests: `tests/unit/test_assistant_decision_tools.py`, `tests/unit/test_plain_language_prompts.py` and `ui/tests/snapshot-workspace.test.js`.

## How to test

```bash
PYTHONPATH=api .venv/bin/python -m pytest -q tests/unit/test_assistant_decision_tools.py tests/unit/test_plain_language_prompts.py
```

```bash
npm --prefix ui test -- snapshot-workspace
```

Manual check on an ECN with BOM changes:

1. Ask "what action items should I take next?". The first line states the review decision. Every part is named by number. No generic tooling or documentation checks appear unless the data supports them.
2. Say "request changes: the MBOM cut-over date at PL10 looks wrong, check with planning". The assistant asks for the reviewer name, then shows a draft. Nothing is saved yet.
3. Reply "yes". The assistant confirms the save, and the Engineering review card shows the new decision without a page reload.
