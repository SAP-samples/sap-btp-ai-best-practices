# OEM app feedback — plain implementation instructions

**Source:** `RE_ OEM App Feedback.eml` — original feedback dated 26 October 2026, with a table-width follow-up dated 27 October 2026. The original email is marked **High importance**.

**Scope:** This document translates the feedback into clear tasks and includes the relevant screenshots. It does not implement the requests or certify their current completion. Questions in the email remain clarification tasks; suggested interpretations below are not confirmed business rules.

The screenshots reflect the app when the email was written, not necessarily the current local version. Email headers, contact details, signatures and the signature logo are omitted. Browser chrome and the desktop/taskbar are cropped from the full-desktop screenshot; application evidence and annotations are retained. Keep this document and its screenshots local unless sharing is approved.

## Task checklist

The numbering preserves the original 11 items. Item 12 is the follow-up request.

- [ ] **1. Linked ECN:** Add a clickable Linked ECN field beside Release date in Overview; use an explicitly simulated relationship for the demo.
- [ ] **2. Feasibility result:** Restore the missing Feasible / Not feasible control or indicator beside the manufacturing-feasibility heading.
- [ ] **3. Measurement formatting:** Remove thousands-separator commas from millimetre values and ranges.
- [ ] **4. Geometry sections:** Collapse each geometry section by default; show its result in the collapsed header; display the plant description instead of the plant code in user-facing labels.
- [ ] **5. Currency and coverage:** Display the euro symbol for EUR prices; explain or rename the unclear Coverage field.
- [ ] **6. Chat placement:** Either keep ECN chat persistently open in a stable position, or expose it through a Joule/sparkle icon in the upper-left area. Confirm which option is wanted.
- [ ] **7. Duplicate navigation:** Remove only the three sidebar icons highlighted in the screenshot; keep their content accessible from the main detail page.
- [ ] **8. Acceptance rules:** Explain the actual conditions for accepting an ECN and why Accept ECN is disabled in the example.
- [ ] **9. Inbox statuses:** Explain what triggers In progress and Generation needs attention.
- [ ] **10. Plant names:** Show plant descriptions rather than plant codes in the change summary and other applicable user-facing text.
- [ ] **11. Engineering drawing:** Confirm whether a drawing-view field/link can be added, and provide it if an authorized drawing source is available.
- [ ] **12. Table layout:** Fix column widths and wrapping so table content never overlaps.

## 1. Add a Linked ECN field in Overview

**Instruction**

- Place a field labeled **Linked ECN** next to **Release date** in the Overview section.
- Make its value a clickable link that opens the related ECN.
- For the demo, simulate the relationship as explicitly requested in the email; do not present an invented relationship as source-backed fact.

**Done when:** The field is visible beside Release date and its demo link opens the intended related ECN.

**Clarify:** Which ECN should the demonstration link target? Should there be one related ECN or several? The email does not specify a relationship type or target.

**Screenshot:** No dedicated screenshot was attached to this item.

## 2. Restore the missing feasibility result beside the heading

**Instruction**

- Add the missing **Feasible / Not feasible** control or result indicator beside the manufacturing-feasibility heading, in the highlighted position.
- Make the overall result understandable without requiring the user to inspect every check.

**Done when:** A clearly labeled overall feasibility result appears at the top of the section.

**Clarify before implementation:** The email calls this a “button,” but does not say whether it is a read-only result or an action. Confirm its behavior and the rule that produces the overall result. Do not collapse missing or inconclusive evidence into an unsupported Feasible/Not feasible verdict.

![Highlighted empty area beside the manufacturing-feasibility heading where the Feasible / Not feasible control is requested.](assets/oem-app-feedback/02-11-feasibility-header-and-drawing.png)

## 3. Remove commas from millimetre measurements

**Instruction**

- Remove thousands-separator commas in displayed **mm** measurements, ranges and calculation requirements.
- Preserve the number, decimal precision and unit; this is a presentation change, not a recalculation.
- Apply the same formatting to summary findings and detailed geometry tables.

**Example:** `200–1,500 mm` → `200–1500 mm`. Keep a decimal measurement such as `629.6 mm` unchanged.

**Done when:** Millimetre values and ranges no longer contain thousands-separator commas, and their numeric meaning is unchanged.

![The highlighted measurement range currently reads 200–1,500 mm; remove the thousands-separator comma.](assets/oem-app-feedback/03-mm-number-format.png)

## 4. Collapse geometry sections and show results in their headers

**Instruction**

- Make individual geometry/check sections, such as **Material Designation**, **Tube Length** and **Bend Angles**, expandable and **collapsed by default**.
- Add a chevron/caret to make expansion discoverable.
- Show the section's **Pass / Fail** result beside its title while collapsed. Preserve a truthful unavailable/inconclusive state where no pass/fail result is supported.
- Replace the plant code **PS4** with the plant description **Plant A** in user-facing section headings and applicable labels, as requested.
- Keep measurements, requirements and calculation evidence available when expanded.

**Done when:** Users can scan section titles and results first, then expand only the geometry evidence they need; headings use the requested plant description.

**Clarify:** Confirm the section-level aggregation rules and which PS4 labels represent the plant. Do not alter immutable source text or technical specification identifiers merely to rename the presentation.

![Arrows identify the geometry subsection headings that should become collapsed disclosures with their results visible at the top.](assets/oem-app-feedback/04-collapsible-geometry-checks.png)

## 5. Use the euro symbol and explain Coverage

### 5a. Price display

**Instruction:** Use **€** for prices whose actual currency is EUR, while preserving the amount and price basis.

**Example:** `EUR 4.70 / 1 EA` → `€4.70 / 1 EA`.

**Done when:** EUR prices include the euro symbol. Missing prices and “Not allowed” states are not converted into amounts; non-EUR prices retain their actual currency.

### 5b. Coverage terminology

**Instruction:** Explain what **Coverage** means in **Observed BOM Coverage**, or replace the heading with a clearer label after confirming its meaning.

**Suggested wording, subject to confirmation:** If the field describes presence in engineering/manufacturing BOMs, use **BOM presence** and values such as **Found in both EBOM and MBOM**, **EBOM only**, or **MBOM only**. The screenshot's “Observed in both” should say which two sources it means.

**Done when:** A reviewer can understand the field without knowing internal terminology. Do not imply complete or approved coverage solely because a material appears in both source types.

![Red boxes highlight the unclear Coverage column and the source purchase price currently displayed with EUR instead of the euro symbol.](assets/oem-app-feedback/05-currency-and-bom-coverage.png)

## 6. Give ECN chat a stable, discoverable location

**Instruction:** Choose one of the two options offered in the email:

1. **Persistent panel:** Keep ECN chat open in a stable position.
2. **Icon entry point:** Put a Joule/sparkle chat icon in the **upper-left** area so users can open chat predictably.

**Suggested interpretation:** “Stay open stagnet” appears to mean “stay open/static,” rather than move or disappear. Confirm this interpretation and the preferred option.

**Done when:** Chat has a consistent entry point or persistent location, and opening it does not obscure essential page content or decision actions.

**Clarify:** The email does not establish conversation-retention behavior. Do not assume that “stay open” means retaining chat across reloads or changing data-retention policy.

![The screenshot shows ECN chat open on the right side; the request is for stable placement or an upper-left Joule/sparkle entry point.](assets/oem-app-feedback/06-chat-placement.png)

## 7. Remove the three duplicate sidebar icons

**Instruction**

- Remove the **three icons inside the red rectangle** from the sidebar because their content is already available on the main impact-analysis/detail page.
- Identify the actual navigation items represented by these icons before changing the menu; the email does not name their routes.
- Keep the other sidebar items, including the unhighlighted Inbox/current-detail items and Settings.
- Keep the corresponding content accessible through the main page's navigation.

**Done when:** The three highlighted shortcuts are gone, without removing their underlying content or unrelated navigation.

![The red rectangle marks exactly three sidebar shortcuts to remove; the surrounding icons are not part of this request.](assets/oem-app-feedback/07-duplicate-navigation-icons.png)

## 8. Explain why Accept ECN is disabled

**This is a clarification request, not permission to loosen acceptance rules.**

**Instruction**

- Document the actual conditions that enable **Accept ECN** and the condition preventing acceptance in the screenshot's example.
- When acceptance is disabled, show the specific reason and the next relevant evidence or action near the decision control.
- Keep the existing server-side acceptance and reviewer-authorization rules intact. Do not enable acceptance just to make the demo button clickable.

**Done when:** A reviewer can tell why acceptance is unavailable and what must happen next; the documented explanation matches the implemented rules.

**Current presentation constraint:** The separate **Review status / Decision readiness** card was subsequently removed at the user's request. Do not recreate that card to answer this email item; use the existing decision-footer explanation or another agreed presentation.

![Accept ECN is disabled beside Send back to engineering; the email asks which rules cause this state.](assets/oem-app-feedback/08-disabled-accept-ecn.png)

## 9. Explain the two Inbox status counters

**This is a clarification request.**

**Instruction**

- Define the exact conditions that contribute to **In progress**.
- Define the exact conditions that contribute to **Generation needs attention**.
- Explain when records enter and leave those states and what the user should do when attention is required.
- Distinguish AI-generation state from engineering-review state and source-data warnings.

**Done when:** Each counter has an accurate, plain-language explanation, such as help text or a tooltip, based on the actual application state mapping.

**Clarify:** The email does not define the job-state mappings. Verify them instead of inferring rules from the labels or the zero counts shown.

![The highlighted Inbox counters are In progress and Generation needs attention; the email asks what triggers them.](assets/oem-app-feedback/09-inbox-status-triggers.png)

## 10. Show plant descriptions instead of codes

**Instruction**

- Replace code-only plant references with plant descriptions in user-facing summaries, especially the **Where** line in **Change summary**.
- Apply consistent naming to applicable plant-related headings and displays.
- If a code is useful for traceability, keep it as secondary detail rather than the primary label.
- Use verified plant mappings. If no description is available, fall back honestly to the code instead of inventing a name.

**Example based on the screenshots:** `Plants PK01 and PL10` → `Plants Plant B and Plant A` (subject to confirming the source mapping).

**Done when:** The summary names the plants in understandable language and preserves their correct identity.

![The highlighted Where line uses PK01 and PL10; show their plant descriptions instead of code-only references.](assets/oem-app-feedback/10-plant-descriptions.png)

## 11. Add a way to view the engineering drawing

**This is a feasibility question pending confirmation of the drawing source.**

**Instruction**

- Check whether an engineering drawing can be linked to the selected material/ECN.
- If available and authorized, add a clearly labeled **View engineering drawing** field/link or action near the material/manufacturing-feasibility context.
- Open the correct diagram/document; do not link an unrelated drawing or silently substitute a generated image.
- If no drawing is available, show a clear unavailable state.

**Done when:** The user can view the intended drawing, or sees a truthful explanation that it is unavailable.

**Clarify:** Confirm the document location, material/revision association, access permissions, file type and viewing behavior. The email provides no drawing URL or attachment; the image below is a UI screenshot, not an engineering drawing.

![The email reuses the manufacturing-feasibility screenshot when asking for a field to view the engineering drawing.](assets/oem-app-feedback/02-11-feasibility-header-and-drawing.png)

## 12. Fix table widths so content does not overlap

**Follow-up request added on 27 October 2026.**

**Instruction**

- Fix the table layout in the material-detail view, particularly the **observed BOM occurrences** table.
- Give each column sufficient width or controlled wrapping. Long hierarchy paths must not intrude into item/position, quantity, effective-interval or change-reference cells.
- Ensure headers are readable rather than colliding or being cut off without an accessible alternative.
- Use contained scrolling or responsive row details when the table cannot fit; do not squeeze text into overlapping columns.
- Check the related coverage, procurement and geometry tables for the same issue.

**Done when:** Headers and cell contents stay within their columns with long real-world values, at normal desktop width and narrow layouts; evidence remains readable and accessible.

![The observed BOM occurrences table has overlapping hierarchy, item/position, effective-interval and change-reference content; correct the column layout.](assets/oem-app-feedback/12-table-width-overlap.png)

## Decisions needed before closing the feedback

- Which related ECN should item 1 link to in the demo?
- Is the item 2 feasibility element a result indicator or an action, and what determines its result?
- What are the item 4 section-result aggregation rules and the confirmed plant-name mapping?
- What does item 5 Coverage mean in the data model?
- Which of the two chat-placement options in item 6 is preferred?
- What are the actual acceptance rules and Inbox generation-status mappings in items 8 and 9?
- Is an authorized, revision-correct engineering drawing source available for item 11?

The email also invites additional UX feedback to the implementation contact; it does not specify further concrete changes.

## Screenshot traceability

All **10 distinct feedback screenshots** are included above. Items 2 and 11 share one image; item 1 has none. The signature logo is intentionally excluded.

| Original email image | Local image | Feedback item(s) |
|---|---|---|
| `image011.png` | `12-table-width-overlap.png` (application-only crop) | 12 |
| `image013.png` | `02-11-feasibility-header-and-drawing.png` | 2, 11 |
| `image014.png` | `03-mm-number-format.png` | 3 |
| `image015.png` | `04-collapsible-geometry-checks.png` | 4 |
| `image016.png` | `05-currency-and-bom-coverage.png` | 5 |
| `image017.png` | `06-chat-placement.png` | 6 |
| `image018.png` | `07-duplicate-navigation-icons.png` | 7 |
| `image019.png` | `08-disabled-accept-ecn.png` | 8 |
| `image020.png` | `09-inbox-status-triggers.png` | 9 |
| `image021.png` | `10-plant-descriptions.png` | 10 |
