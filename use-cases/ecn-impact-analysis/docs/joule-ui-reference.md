# Joule UI integration reference

Captured from the live SAP Build Work Zone page on October 27, 2026. This is the original full reference capture. The implemented ECN version follows only captures 00, 03, and 05, with SAP blue theme tokens and a minimal control set; see [the current assistant documentation](ecn-assistant-panel.md).

Open the [interactive reference gallery](assets/joule-reference/index.html) to compare the captured states. Each numbered PNG has a matching `*-measurements.json` containing rendered element geometry, computed styles, and links to the captured stylesheet and theme-token data. Measurements are in CSS pixels. The desktop browser used a 1200 × 762 viewport with device pixel ratio 2, so a 416-pixel panel produces an 832-pixel screenshot.

## Actual integration architecture

This page uses the SAP DAS/Joule web client, embedded in an iframe, with a SAPUI5 shell plugin as its launcher. It is a floating overlay over the application. Opening it does not reserve a column or reduce the application width.

The public Work Zone `Component.js` loads `resources/public/webclient/bootstrap.js`, calls `window.sap.das.bootstrap.run()`, and registers a header item through `FrameBoundExtension.createHeaderItem` at `position: "end"`. Its fallback uses the Fiori renderer's `addHeaderEndItem`. The press handler calls `sap.das.webclient.toggle()`. The item is shown on the home page and in all apps. These are observed calls in the loaded public scripts; they are not a standalone setup recipe.

The launcher uses `sap-icon://da` (diamond with a sparkle); the plugin also declares `sap-icon://da-2`. The observed glyph is U+E2A3 from the `SAP-icons` font. Its accessible name and tooltip are both “Joule.”

The DOM hierarchy is:

```text
SAPUI5 shell header item: Joule
body
  #cai-webclient-main                 fixed-position overlay
    #cai-webclient-container          rounded, clipped, shadowed frame
      #cai-webclient-customDragDivHeader  15px cursor:move strip
      iframe#cai-webclient-iframe      title="Joule"
        .mainApp.dasApp.horizonTheme
          header
          welcome screen / conversation
          composer + disclaimer
          internal menu / debug panel
```

The iframe loads `resources/webclient/webclient.html` on the Joule tenant. It loads React and UI5 vendor bundles, plus `chat-components.js` and `webclient.js`. Observed controls include `ui5-button`, `ui5-icon`, `ui5-textarea`, `ui5-file-uploader`, `ui5-dialog`, `ui5-popover`, `ui5-bar`, `ui5-text`, and `ui5-checkbox`. The host and iframe communicate through an origin-targeted `postMessage` bridge. The bootstrap exposes methods including `show`, `hide`, `toggle`, `isOpen`, `expand`, `collapse`, and `toggleFullScreen`.

The host uses SAPUI5 1.152.0 with `sap_horizon`. The iframe uses its own theme styles and controls. See [integration evidence](assets/joule-reference/integration.json) for resource names and short public-source excerpts. The source component version of the iframe is not established by those bundle names.

## Panel geometry and surface

| Property | Observed value |
| --- | --- |
| Position | Fixed, right 16px, bottom 16px |
| Width | 416px; max-width 100% |
| Height | 90% of viewport; max-height 768px |
| Actual desktop bounds | x 768, y 60.203125, width 416, height 685.796875 |
| Outer/iframe corner radius | 30px |
| Overflow | Hidden on container and iframe |
| Layer | Host z-index 11 |
| Host shadow | `var(--sapContent_Shadow2)` |
| Resolved host shadow | `0 0 0 1px rgba(34,53,72,.48), 0 10px 30px rgba(34,53,72,.25)` |
| Main content | White surface; animated blurred colour mesh near the bottom of the welcome screen |
| Backdrop | No page-wide dimming in the normal panel state |
| Drag affordance | 15px-high strip at the top with `cursor: move`; actual dragging was not tested |

The inline host sizing is captured exactly in each measurement file. The normal panel's visible top is a consequence of height and bottom anchoring; it is not originally tied to the shell's 52px height.

## Header

The header content is 56px high with a 2px lower area. It uses horizontal padding of `0.6rem` (9.6px). Each circular toolbar control has a 44 × 44px UI5 host and a 42 × 42px rendered inner button. Icons have 16px layout boxes and an internal 1.25 scale (approximately 20px visual size). The right-side actions advance in 48px steps.

| Position | Label / function | Icon |
| --- | --- | --- |
| Left | View Main Menu / Close Main Menu | `screen-split-one` |
| Beside menu | “Joule” on welcome; truncated conversation title after sending | Text, 14px, weight 600 |
| Conversation-only | View Conversation Options | `overflow` |
| Right | Fullscreen / Exit Fullscreen | `full-screen` / `exit-full-screen` |
| Right | Debug | `indent` |
| Far right | Close | `decline` |

Toolbar foreground is `--sapAssistant_Color1` (#5d36ff). Button surfaces use `--sapAssistant_Answer_Background` (#eff1f2). Buttons use a 60px radius and a 0.15s linear transition. Their pressed/focused style adds a 2px assistant-colour border.

The `indent` control is explicitly a developer Debug action. It opens a fullscreen split view with request logs and payload inspection, and changes the toolbar to include “Save messages to file.” See capture 07. Its appearance in this tenant does not establish that every Joule production integration has this control.

## Welcome screen

Capture 01 is the first welcome screen supplied for inspection. Its greeting is “Let's finish strong. How can I help?” A new conversation later produced “Still with you! What's next?”, so greeting text varies.

| Element | Bounds in the 416 × 686 iframe |
| --- | --- |
| Welcome illustration | x 180, y 104.296875, 56 × 56px |
| Greeting area | y 216.296875, height 67.203125px |
| Greeting text container | x 32, width 352px |
| Composer | x 16, y 341.5, 384 × 108px |
| Disclaimer | y 662, 11px text, 12px line box |

The greeting uses SAP “72,” 28px text, normal weight, centred alignment, and a horizontal #2800cf → #8c00a9 gradient clipped to the text. Its salutation wrapper has `margin: 3.5rem auto` and a maximum width of 50rem. Text appears through a letter reveal animation (0.3s ease-out per letter). The composer fades in with a 0.4s ease-out transition. The mesh consists of animated image layers rather than a single static CSS gradient; stylesheet evidence records these rules.

The illustration is preserved as [reference SVG](assets/joule-reference/joule-logo-reference.svg). The replacement should use the ECN assistant's chosen illustration/icon and palette in the same space.

## Composer

| Property | Observed value |
| --- | --- |
| Outer width | Panel width minus 32px; fullscreen maximum 800px |
| Minimum height | 108px (`6.75rem`) |
| Corner radius | 30px (`1.875rem`) |
| Padding | 14px in the captured multiline welcome layout |
| Background | #ffffff |
| Resting border | 1px solid #cdc2ff |
| Resting shadow | 0 0 2px and 0 8px 16px in assistant colour at 16% opacity |
| Hover border | #2800cf; shadow changes to 0 4px 8px |
| Textarea | UI5 textarea with transparent surface, no visible inner border |
| Placeholder | “Message Joule...”, SAP “72,” 14px, italic, #556b82 |
| Bottom button row | 12px below the text area; 28px-high circular actions |
| Left button | `add`; #eff1f2 background, assistant-colour icon |
| Right button | `paper-plane`; assistant-colour background, white icon |
| Empty send | Same colour at 0.3 opacity |
| Populated send | Full opacity; a Clear control appears in the text area |

Captures 03 and 04 preserve focused and enabled states. Capture 03 also includes pointer hover, so its border is #2800cf. The purple focus appearance in the supplied screenshot is not equivalent to every pointer-hover state.

The plus button is an attachment uploader. Its observed accepted extensions are `.pdf,.doc,.docx,.txt,.rtf,.jpeg,.png,.webp,.gif,.json,.md`; multiple selection is disabled. No file was uploaded. The ECN backend currently accepts text-only chat, so attachment support would need its own implementation decision before presenting this as a working control.

## Conversation presentation

After sending “Hello!” and receiving a response, the welcome screen is replaced by a scrolling conversation and the composer moves to the bottom. The header changes to a conversation title and adds an overflow/options action.

The transcript uses 16px SAP “72” text with 24px line height. A centred day/time line uses 12px text. User messages align right on #eae5ff with 8px corners and a square bottom-right corner; text has 8px vertical and 16px horizontal padding. Answers use #eff1f2, span the 384px content width, and have 16px horizontal padding. See capture 05 and its measured nested elements for the exact answer/background geometry.

The answer action row contains Copy, Helpful, Not Helpful, Retry, and Response Insights; the user message has Edit. These are observed UI affordances. Only the greeting send/response was exercised; feedback, edit, retry, insights, and conversation option mutations were not tested.

In the captured conversation, the composer bounds are x 16, y 534, width 384, height 108. Its surrounding footer is 160px high and includes the persistent disclaimer “Joule uses AI. Verify results.”

## Menu, fullscreen, dismissal, and responsiveness

The conversation menu is a 256px-wide drawer inside the panel. It overlays the conversation in normal mode and contains New Conversation, conversation entries, and a bottom Preferences area. The active entry has a purple selection/focus treatment. Captures 02 and 08 show empty and populated menus. In the captured fullscreen welcome state, the menu is open and the main content occupies the remaining width, with the composer capped at 800px.

Fullscreen expands to almost the complete viewport and retains 30px corners. The measured host is x 2, y 1, width 1198, height 760 at a 1200 × 762 viewport. The host inline styles use `width: calc(100% - 2px)`, `height: calc(100% - 2px)`, `top: 1px`, and `bottom: 1px`; its container is also two pixels narrower, giving the iframe a 1196px viewport.

Verified normal-panel behavior:

- Close hides the rendered chat and disables pointer events on the host, while keeping the iframe mounted.
- The shell launcher reopens the same panel. A sent conversation survived close/reopen.
- Escape from the composer closes the panel.
- Clicking outside leaves the panel open; the underlying page remains available.
- Opening, closing, and using fullscreen leave the application viewport at its full width.
- New Conversation returns to the welcome presentation.

See [verified behavior](assets/joule-reference/verified-behavior.json). Keyboard focus trapping and Tab order were not exhaustively tested.

Responsive captures 09 and 10 were taken after using fullscreen and resizing. At 390 × 844, the host clamps to 390px while the inner minimum width remains 416px, clipping the right edge of controls. At 1440 × 900, height caps at 768px. The runtime retained an explicit top position and changed the right inset to 0% during those resize operations. These are observations of this session, not a universal breakpoint specification. A replica should contain its controls on narrow screens; the reference's clipping is a defect to avoid. Original desktop size, device pixel ratio, and normal-panel positioning were restored afterward.

## Colour tokens to replace together

| Token / role | Reference |
| --- | --- |
| `--sapAssistant_Color1`, border, primary controls | #5d36ff |
| `--sapAssistant_Color2` | #a100c2 |
| `--sapAssistant_BackgroundGradient` | `linear-gradient(#5d36ff, #a100c2)` |
| `--sapAssistant_Hover_Background`, hover border | #2800cf |
| `--sapAssistant_Question_Background`, question border | #eae5ff |
| `--sapAssistant_Answer_Background`, answer border | #eff1f2 |
| Question / answer text | #131e29 |
| Placeholder / disclaimer | #556b82 |
| Composer resting border | #cdc2ff |
| Greeting gradient | #2800cf → #8c00a9 |

The palette should change as a coordinated set, including welcome illustration, greeting gradient, shadows, focus rings, selection states, question tint, and send button. Layout and interaction should follow the captured reference. The exact ECN palette and differentiated icon are still to be chosen.

## Mapping to the existing ECN application

The repository already uses UI5 Web Components. Its installed `ShellBar.d.ts` declares the public `assistant: Slot<IButton>` API, available since 2.0.0. The appropriate shell integration is a `ui5-button` in `slot="assistant"` with our accessible name and icon. The actual Work Zone plugin API above applies to that SAPUI5 host; this app has a different shell implementation.

| Existing file | Replacement work |
| --- | --- |
| `ui/index.html` | Move `#assistant-shell-action` from `startButton` to the public `assistant` slot; keep navigation in its own start slot. |
| `ui/src/modules/shell-chat.js` | Render the round toolbar, floating inset panel, internal menu, fullscreen state, and explicit user-open behavior. Replace the wide-screen default-open and medium-screen backdrop rules with the reference presentation. |
| `ui/src/components/ecn-chat.js` | Render welcome and transcript states, bottom composer, disabled/ready/busy/error states, conversation controls, and disclaimer. Continue using the existing scoped/global chat services, safe Markdown, sources, review-save event, graph-readiness checks, and bounded history. |
| `ui/src/style.css` | Replace square edge-attached styling and reserved shell column with the measured dimensions, radii, spacing, typography, and coordinated assistant palette. |
| Chat unit and browser checks | Update expectations that currently require the upper-left launcher and reserved page column. Check launcher position, overlay geometry, state transitions, focus, viewport containment, and preservation of the existing chat workflow. |
| `docs/ecn-assistant-panel.md` | Update the documented presentation after implementation. |

Reproducing this presentation with the existing backend is a UI replacement. Embedding the actual SAP-hosted Joule web client would require its tenant/runtime configuration and conversation contract; the inspected bootstrap does not demonstrate that it can be connected directly to our custom ECN chat endpoints. The captured public-source excerpts document this distinction without assuming such compatibility.

## Capture coverage

The gallery includes launcher, welcome, menu, focused composer, enabled composer, answered conversation, fullscreen, debug, populated menu, mobile, and wide-screen captures. Styles and geometry were read from the live iframe, including UI5 shadow DOM; they were not inferred only from the supplied image. The original capture step did not change application files. Unverified behaviors are called out above so future implementation does not treat inferred interactions as confirmed.
