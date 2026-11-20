# ECN assistant panel

## Purpose

The **ECN Assistant** opens from the **Open ECN chat** action in the upper-right Shell Bar using UI5's public `assistant` slot. It uses the standard `discussion-2` chat icon and SAP Horizon blue theme tokens. The application remains plain JavaScript with UI5 Web Components; React documentation was used only to validate the shared theming approach.

The panel starts closed at every width. It floats over the page at 416px wide, with 30px corners, 16px right/bottom insets, and a maximum height of 768px. Narrow screens use 8px insets and constrain the panel to the viewport. Opening it leaves the page at its full width. Clicking the shell action toggles it; **Close chat** or Escape closes it and returns focus to the shell action. This is a nonmodal complementary panel, so the page stays interactive and keyboard Tab is not trapped. Nested source-inspection dialogs own Escape while open.

The scope guidance and rounded composer form the initial screen. Sending shows a right-aligned question and left-aligned answer, with the composer at the bottom. Enter sends, Shift+Enter inserts a new line, and the send action stays disabled for empty input, pending requests, or an unverified RDF projection. Errors preserve the unsent text. Closing and reopening preserves the current draft and conversation; reload clears browser-memory conversations.

The presentation implements only the launcher, focused composer, and conversation pattern captured in [the live reference](joule-ui-reference.md). It has no Joule branding, purple palette, conversation-history menu, fullscreen, debug panel, tool-trace display, attachments, or response feedback toolbar. Verified object and ECN source links remain available in answers, and confirmed engineering-review saves still refresh the review/audit views.

The palette uses `--sapButton_Emphasized_Background` (#0070f2), `--sapHighlightColor` (#0064d9), `--sapContent_FocusColor` (#0032a5), `--sapButton_Selected_Background` (#edf6ff), and neutral SAP surfaces. These are active-theme variables, so other SAP themes can supply their own values. See [UI5 Web Components React styling](https://ui5.github.io/webcomponents-react/v2/?path=/docs/knowledge-base-styling--docs), which recommends shared SAP CSS variables and links to [SAP theming-base-content](https://github.com/SAP/theming-base-content/tree/master/content/Base/baseLib).

Verified screenshots: [shell launcher](assets/ecn-assistant-blue/00-shell-launcher.png), [focused composer](assets/ecn-assistant-blue/03-focused-composer.png), [conversation](assets/ecn-assistant-blue/05-conversation.png), and [mobile conversation](assets/ecn-assistant-blue/mobile-conversation.png). Chat responses in these screenshots are synthetic. [Palette evidence](assets/ecn-assistant-blue/palette.json) records the active browser values and documentation sources; the installed SAP theming-base-content version is 11.36.4.

Outside an ECN page the same panel runs the global chat over every stored
ECN; see [Global ECN chat](ecn-global-chat.md).

## Inputs and outputs

- Input: the current ECN assessment ID, RDF graph readiness, and a question entered by the reviewer.
- Output: the answer, consulted-object/ECN links, and `review_saved`. Opening or closing the panel makes no API call. The assistant never changes source data; its only write is an engineering review the user confirmed in chat (see `docs/ecn-assistant-decision-tools.md`).
- State: panel visibility belongs to the mounted application shell. Conversation turns stay in browser memory per assessment; the draft remains in the mounted text area while the panel is closed.

## Related files

- `ui/index.html`: supplies the named upper-right Shell Bar chat action and retained navigation menu at all widths.
- `ui/src/modules/shell-chat.js`: binds the shell action, mounts the floating panel, open/close focus behavior, and Escape handling; `ui/src/components/snapshot-workspace.js` binds it to the open ECN.
- `ui/src/components/ecn-chat.js`: renders the scoped conversation and scope guidance.
- `ui/src/style.css`: positions the panel and entry points and constrains the message area.
- `ui/tests/oem-feedback-06-chat.test.js`, `ui/tests/snapshot-workspace.test.js` and `ui/tests/e2e/shell.spec.js`: check scope, draft retention, the single persistent action, focus restoration, public runtime slot assignment/button semantics, and the absence of mobile control collisions.

## Key behavior

The same chat element remains mounted as visibility changes, preserving the unsent draft and in-memory conversation. `shell-chat.js` uses the same Shell Bar action at every viewport width. Its public UI5 Button `accessibilityAttributes` forwards `controls` and `expanded` to the focusable button. Reset removes action/keyboard listeners, aborts the child binding, and removes the panel. Responses arriving while the panel is closed do not steal focus from the launcher.

## Verify

Checks from `ui/`: `npm test`, `npm run build`, and `npm run test:e2e -- tests/e2e/assistant.spec.js tests/e2e/shell.spec.js`. No typecheck script is defined (plain ESM). The isolated mocked browser fixture checks right-side slot assignment, actual button ARIA, close/reopen preservation, pending/answered/error states, Enter/Shift+Enter, safe Markdown, sources, SAP colours, Escape focus return, and mobile/short-screen containment.

## BOM-driven change and date questions

`list_bom_changes(number="", page=1)` reads five change lines per page from the
same assessed BOM projection as Overview. Optional `number` filters an exact
old/new part or assembly. Output includes total/has_next, complete change counts,
plant/model, old end date, separate engineering/manufacturing start dates,
pairing qualifiers, source rows and relevant new-part/phase-out flags. The ECN
identity remains server-owned; the tool cannot select another assessment.

Chat uses this tool before searching raw ECN fields for replacement or cut-over
questions. JSON date discovery remains appropriate for ECN header dates. The
bounded overview excludes full part/reference arrays and returns representative
BOM/plant context, preventing large lists from hiding the entire response.

Related files: `api/app/ecn/assistant/tools.py`, `runtime.py`, both application
assistant skills, and `tests/unit/test_chat_bom_changes.py`.

```bash
PYTHONPATH=api .venv/bin/python -m pytest -q tests/unit/test_chat_bom_changes.py
```

Live regression: ask ECN345274 which parts are replaced, where, and when. The
answer must distinguish engineering release on 16 January 2026 from manufacturing
cut-over on 30 June, after the old parts end on 29 June. An ECN JSON field search
alone is not enough to claim that those BOM dates are missing.

## Review status, impact analysis and data-quality questions

Three read tools give the assistant the context it needs for "what should I do
next?" questions. See `docs/ecn-assistant-decision-tools.md` for the full
contract.

- `get_review_status()`: current engineering decision (`NONE`, `ACCEPTED`,
  `CHANGES_REQUESTED`), whether it was made on the current snapshot, the last
  five reviews, and the acceptance plan: each part's material master change,
  its price state, and whether the ECN is blocked by a new part with no price.
- `get_impact_analysis()`: stored AI analysis (overall impact, factor
  severities, section status, next steps) with `is_current` against the
  snapshot evidence hash, or `available: false`.
- `get_data_quality_issues(page=1)`: readiness questions with the exact part
  numbers they concern, plus stored data-quality and missing-evidence records.

The system prompt now requires exact part numbers in every bullet that refers
to parts, and a decision line before any list of next steps.
