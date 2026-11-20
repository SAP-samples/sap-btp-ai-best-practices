# UI/UX remediation delivery plan

**Status:** UI implementation and automated validation are complete on `feature/ui-ux-remediation-wp1-responsive-assistant`; this includes the later DynamicPage lifecycle P1 resolution. Manual screen-reader, high-contrast, and 200%-zoom checks remain release-readiness sign-off activities. No deployment, BTP resource change, live-service call, or customer-data operation has been performed.

## Final-review prototype remediation — parent-verified

Production exclusion means more than omitting prototype routes. `ui/src/modules/router.js` uses literal asset-loader entries for product pages and declares every `visual-prototype`, `ecn-worklist-prototype`, `ecn-decision-prototype`, and `material-traceability-prototype` import only inside an `import.meta.env.DEV` branch. Vite production replacement and dead-code elimination are therefore intended to omit each prototype module, raw HTML, CSS, transitive prototype-only UI5 import, and fixture text; production has neither a registered route nor a runtime loader for any prototype page. Parent verification previously confirmed this boundary for Floorplans 0–2; Floorplan 3 requires the production build and artifact scan recorded below before the same claim can be made for it.

The shell's `<main class="content">` remains the sole primary landmark. The Floorplan 2 DynamicPage content accessibility metadata is a named `region` (`Static decision detail`), not a nested `main`.

The parent verification ran the following from the repository root after a production build; the second check detects emitted prototype CSS selectors or fixture text even when Vite uses content-hashed filenames:

```bash
npm --prefix ui run build
if find ui/dist -type f \( -iname '*visual-prototype*' -o -iname '*ecn-worklist-prototype*' -o -iname '*ecn-decision-prototype*' \) -print -quit | grep -q .; then
  echo 'Prototype-named production asset emitted' >&2; exit 1
fi
if grep -R -n -E --include='*.js' --include='*.css' --include='*.html' \
  'floorplan-zero__header|floorplan-one__page|floorplan-two__page|ECN-FX-201|Static ECN worklist results|Floorplan 0' ui/dist; then
  echo 'Prototype fixture text or CSS emitted in production build' >&2; exit 1
fi
```

## Developer-only Floorplan 0 record — 2026-02-17

The prior custom visual-direction prototype style is rejected: its custom palette, typography, rounded surfaces, status pills, visual switcher, query-driven variants, and arrow shortcuts forced UI5 into a foreign aesthetic. The prototype route is rewritten as **Floorplan 0 — UI5 native fidelity baseline**, a development-only static component/state reference. It retains the existing application shell and uses installed UI5 Web Components with native rendering and token-driven spacing only.

Floorplan 0 visibly demonstrates static success, loading, empty, and error conventions through `ui5-message-strip`, `ui5-busy-indicator`, and `ui5-illustrated-message`; `NoData` is explicitly imported as a non-default illustration. Its loading and feedback regions carry explicit status/busy/error semantics. It makes no API, network, live-data, or customer-system call and uses only sanitized static fixture wording. The route remains registered only under `import.meta.env.DEV`; its production mounted behavior is therefore inert. It does not change production ECN or Settings pages, data/API contracts, packages, or deployment.

**Floorplan 1 implementation record — 2026-02-17:** `/prototype/ecn-worklist` is a second, separate DEV-only route (`import.meta.env.DEV`) for the static sanitized **ECN Worklist/List-Report-style** success fixture; it preserves Floorplan 0 and is not registered in production. The plain UI5 Web Components composition is a fixed-height `ui5-page` host with a native `ui5-bar` header, a native `ui5-form` (`accessible-mode="Edit"`, `layout="S1 M2 L2 XL2"`) with labelled readonly `ui5-input` search and disabled `ui5-select` fixture-state controls, then a separate named `ui5-bar` in page content. That result bar displays **Results** and holds the static Clear filters and emphasized Import ECNs `ui5-button` actions in its public `endContent` slot. The result bar replaces the removed `ui5-toolbar` composition; no clone, deferred measurement, or internal overflow workaround remains. The page retains an accessibly named responsive `ui5-table` (`ui5-table-header-row`/header cells/rows/cells, `overflow-mode="Popin"`), state-only native `ui5-tag`, and an informational `ui5-message-strip`. The table contains fictional fixture records only. Its labelled `Open (static)` row affordance and both result actions have their clicks prevented, so they cannot navigate or mutate state. The fixture makes no API/network/live/customer call and does not alter production ECN/Settings behavior, API/data contracts, packages, deployment, tests, selection, bulk processing, pagination, or query state. It deliberately adds no dashboard/KPI cards, custom table/status/action styling, visual switcher, Dialog, or FlexibleColumnLayout. The success state is visible; Floorplan 0 remains the explicit static reference for loading, empty, and error semantics because this slice has no fetch lifecycle.

**Floorplan 1 browser validation — 2026-02-17:** Parent validation rendered the local development fixture at `http://localhost:5173/prototype/ecn-worklist`. At `1280 × 720`, both labelled Form fields, the Results title, Clear and Import actions, and all six table headers were visible. At `390 × 844`, the two Form items were one column, Results and both actions remained visible/contained, the table used `Popin`, and `document.documentElement.scrollWidth === 390`. At both widths, activating a static action did not change the URL or the three fixture rows. Focused navigation tests passed (`3`), the full unit suite passed (`18` files / `128` tests), the Vite production build passed, and `git diff --check` plus `git diff --cached --quiet` passed. The Vite build retains the pre-existing >`500 kB` chunk advisory. This is local-fixture evidence only; it does not validate production authorization, APIs, or live data.

**Exact next bounded Worklist steps:** obtain approval for a distinct fixture-only interaction/state slice; if approved, define accessible loading, empty, and recoverable-error behavior without a data source; then separately decide whether any production-worklist discovery is authorized. Do not connect the prototype to ECN APIs or migrate the production Inbox by default.

**Lifecycle correction — 2026-02-17:** Independent review found that navigating from the prototype to an unmatched URL could leave `visual-prototype-active` on `document.body`, because the router’s 404 path did not execute the active page cleanup. `ui/src/modules/router.js` now atomically detaches a page cleanup and shares its in-flight teardown promise across competing valid-route and 404 navigation, so a cleanup runs exactly once and the 404 waits for it. `ui/tests/navigation.test.js` covers both sequential cleanup and the deterministic overlapping-navigation case. This bounded router fix is required to preserve the existing shell’s controls on the 404 route; it does not alter API/data behavior.

**Floorplan 2 implementation record — 2026-02-17:** `/prototype/ecn-decision` is the single approved development-only static sanitized ECN decision-detail fixture. The route is registered only under the existing `import.meta.env.DEV` gate, preserving Floorplans 0 and 1 and omitting the route from production. It does not use React or claim a nonexistent native ObjectPage. Instead, it composes the installed UI5 Web Components 2.26 DynamicPage family (`ui5-dynamic-page`, `ui5-dynamic-page-title`, and `ui5-dynamic-page-header`) with a fixed-height host and native `ui5-bar` floating footer. Its explicit public component imports also register `ui5-title`, `ui5-tag`, `ui5-message-strip`, `ui5-panel`, `ui5-list`/`ui5-li`, and `ui5-button`.

The fixture’s first reading path is a human-readable fictional change title, a single fictional source-snapshot-backed Decision readiness statement, and concise source/BOM/review facts. A collapsed native `ui5-panel` progressively reveals detailed `ECN-FX-*` fixture evidence. A visible informational MessageStrip and a disabled review button make the static/no-network boundary explicit; the footer neither executes nor implies a review decision. The DynamicPage’s public `accessibilityAttributes` property names root/header/content/footer areas. Its content area is a named `region`, not `main`, because the persistent shell `<main class="content">` is the only primary landmark. Ordered headings, status semantics for the fixture notice, named footer actions, and native UI5 focus behavior provide the static accessibility boundary. Page cleanup removes its body class and inert-action listeners. No dashboard/KPI cards, fake tab bar, generic card grid, giant Dialog, `ui5-flexible-column-layout`, production visual migration, data/API/backend contract, package, test, deployment, or artifact change is included.

Floorplan 2 is a static success fixture only: no request or result transition exists to simulate loading, empty, or error. Floorplan 0 remains the separate native static reference for those conventions. **Floorplan 2 parent validation — 2026-02-17:** local fixture inspection confirmed the named DynamicPage root/header/content/footer regions, the visible static-fixture status notice, collapsed evidence disclosure, and disabled footer action. At `390 × 844`, `document.documentElement.scrollWidth === 390`; the fixed DynamicPage host and floating footer remained contained. Expanding the local evidence Panel did not change the URL or six local fixture list items. Navigating to an unmatched local route removed `ecn-decision-prototype-active` and showed Page Not Found. `npm --prefix ui test -- --run` passed (`18` files / `128` tests); `npm --prefix ui run build` and `npm --prefix ui run test:e2e` passed (`15` tests); `git diff --check` and `git diff --cached --quiet` passed. Production artifact scans confirmed that all three development prototypes—not only `/prototype/ecn-decision`—have no named asset, route string, fixture marker, or prototype CSS selector in `ui/dist`. The pre-existing Vite >`500 kB` chunk advisory remains. A final independent read-only review of the loader-map and landmark remediations returned **No findings** with merge verdict **OK**. This is local static-fixture evidence only; it does not validate production authorization, APIs, live data, screen readers, high contrast, or 200% zoom. Further Floorplan 2 fixture interactions or states, and any production discovery or migration, require separate approval.

**Floorplan 3 implementation record — 2026-02-17:** `/prototype/material-traceability` is the separately approved development-only static sanitized material-traceability fixture. The route is added only within the existing `import.meta.env.DEV` route gate; its literal JS, raw-HTML, and CSS imports exist only inside the router's existing DEV branch. This preserves the product loader map, dependency injection, 404/cleanup behavior, and Floorplans 0–2 while making route and complete asset-graph exclusion the production design.

Its fixed-height/full-width host uses the installed native `ui5-page`, not DynamicPage, FlexibleColumnLayout, or Dialog. A named native `ui5-bar` header provides an inert Back-to-ECN affordance, a readable material-description H1, and quiet `MAT-FX-410` identity. The default answer first states the fictional ECN link and two BOM occurrences, then gives the deterministic absence state `No automatic feasibility report is attached to this fictional material.` and explicitly says no upload or analysis exists. Three collapsed, accessibly named H2 `ui5-panel` sections disclose occurrence evidence, procurement/source facts, and technical provenance via native `ui5-list` with inactive `ui5-li` items. The composition also explicitly registers `ui5-title`, `ui5-message-strip`, `ui5-button`, `ui5-bar`, `ui5-panel`, `ui5-list`, `ui5-li`, and `ui5-page` from the installed UI5 Web Components 2.26 packages. It uses plain ESM/Vite and no React. No tag is included because the content establishes no true status.

The fixture contains only fictional `ECN-FX-*`/`MAT-FX-*` records and makes no API, network, customer-system, upload, analysis, mutation, or navigation call. It changes no production UI, backend/API/data contract, package, deployment, or tests, and excludes nested panels, dashboard cards/KPIs, tables, custom UI5 skinning, AI-first text, and raw-ID hero treatment. The shell's existing `<main class="content">` remains the sole main landmark. Named regions, an H1→H2 heading order, public Panel accessibility properties, inactive evidence items, and an accessible inert Back action define the static accessibility boundary. Cleanup removes click listeners and the prototype body class; assistant UI is hidden only while that class is present.

**Floorplan 3 parent validation — 2026-02-17:** local fixture inspection passed at `1280 × 720` and `390 × 844`. The named native Page regions and H1→H2 hierarchy were exposed in the accessibility tree; all three evidence Panels were collapsed initially; every evidence row remained `Inactive`; opening Occurrence evidence and activating Back to ECN left the URL and eight fixture list items unchanged. At both widths the fixed Page matched its host height and document width matched viewport width (`1280` and `390` respectively), with no document-level horizontal overflow. Navigating to an unmatched route removed `material-traceability-prototype-active`, restored the narrow Shell assistant action, and rendered **Page Not Found**. Browser request inspection found no fetch/XHR/API call; the only non-local requests were the app's existing SAP 72 font assets from jsDelivr. `npm --prefix ui test -- --run` passed (`18` files / `128` tests); `npm --prefix ui run build` and `npm --prefix ui run test:e2e` passed (`15` tests); `git diff --check` and `git diff --cached --quiet` passed. The production scans below found no prototype-named asset, route, fixture marker, or CSS selector in `ui/dist`. The pre-existing Vite >`500 kB` chunk advisory remains. Final independent read-only review returned **No findings** with merge verdict **OK**; the reviewer relied on the supplied parent evidence and did not rerun checks. Manual screen-reader, high-contrast, and 200%-zoom sign-off remain open. Any additional fixture interaction/state, production discovery/migration, or live/product-data connection requires separate approval.

```bash
npm --prefix ui test -- --run
npm --prefix ui run build
npm --prefix ui run test:e2e
git diff --check
git diff --cached --quiet
if find ui/dist -type f -iname '*material-traceability-prototype*' -print -quit | grep -q .; then
  echo 'Material traceability prototype asset emitted in production' >&2; exit 1
fi
if grep -R -n -E --include='*.js' --include='*.css' --include='*.html' \
  'material-traceability-prototype|/prototype/material-traceability|floorplan-three|MAT-FX-410|No automatic feasibility report is attached to this fictional material' ui/dist; then
  echo 'Material traceability route, fixture text, or CSS emitted in production' >&2; exit 1
fi
```

**Implementation record — 2026-02-16:**

- Implemented the approved responsive assistant decision: non-modal docked panel at `>=1400px`; modal overlay at `601–1399px`; named Shell Bar action and full-width modal panel at `<=600px`.
- Removed the narrow-screen floating launcher, added scope text (`this ECN` or `all stored ECNs`), implemented Escape handling, modal Tab containment, and focus restoration to the opener.
- Added deterministic Vitest and Playwright coverage for the Shell Bar entry point, modal semantics, forward and reverse modal Tab containment at narrow and medium breakpoints, focus restoration, and the absence of Inbox/Accept-action collision at `320px`, `390px`, and `480px`.
- Validation completed after the DynamicPage lifecycle P1 resolution: focused adapter/detail tests passed (`22` tests); `npm --prefix ui test -- --run` passed (`126` tests across `18` files); `npm --prefix ui run build` passed; `npm --prefix ui run test:e2e` passed (`15` tests); and `git diff --check`, `git diff --cached --quiet`, plus the two new untracked-file whitespace checks passed. The production build retains the existing Vite advisory for chunks larger than `500 kB`; no dependency, deployment, or runtime behavior was changed to suppress it.
- Implemented the WP2 Inbox triage slice: at `<=600px`, Search, Analysis, Sort, and Go remain immediately available, while ECN number, Plant, Warnings, Order, and Rows are in a named native-button `Filters` disclosure with `aria-expanded`, `aria-controls`, an active secondary-filter count, and Clear filters. Desktop retains all criteria directly visible. Import reset now refreshes both the visible and accessible secondary-filter count.
- Implemented the first approved interaction-standardization increment in the Inbox: finite analysis, review, and impact states now render as explicit UI5 semantic tags in both desktop rows and mobile cards. Existing reset, filter, generation, loading, and list-error feedback is consolidated in UI5 message strips while retaining the established explicit `status`/`alert` live-region semantics. The existing table, cards, labels, focus restoration, filters, KPI action, selection behavior, and API contracts remain unchanged.
- Made Generation needs attention an explicit keyboard-accessible action that applies the existing `analysis_state=FAILED` list filter, resets pagination to page 1, and keeps the current query/sort/page-size state and existing server-side selection contract unchanged. Its count now reports only `failed_count`, matching the exact actionable result set; the default `uploaded_at desc` sort was not changed.
- Completed the final responsive-review findings: at `>=1400px`, an open complementary assistant now reserves a real Shell layout column rather than covering evidence or decision actions. At `<=600px`, Inbox rows now use source-backed ECN result cards showing ECN/description, analysis and review state, impact, warnings, plant state, selection, and View; the desktop table remains available above that breakpoint. List refresh restores keyboard focus to the currently visible representation, duplicate card/table selection controls remain synchronized across breakpoint changes, and unavailable modal chat focuses its enabled Close control instead of leaving focus on a hidden opener. Browser tests prove no wide-assistant overlap, the visible card containment, responsive selection behavior, modal focus fallback, and the responsive result representation. The E2E workflow tests were aligned with the documented `/references` audit route, the Advanced-administration reset disclosure, and fixture-size-independent import behavior.
- Final independent read-only review completed after the accessibility/focus fixes: **No findings**. Manual screen-reader, high-contrast, and 200%-zoom checks remain the documented release-readiness sign-off activities.

- Added the WP3 Decision readiness slice to ECN detail: source status, source-backed BOM evidence, AI-summary availability, and the existing acceptance-plan/review state now appear in a compact summary before the section navigation. A blocked Accept ECN shows the exact existing acceptance-plan reason and a scrolling path to BOM evidence; a failed or blocked AI summary explicitly leaves deterministic evidence available. A successfully fetched but unavailable acceptance plan is now labelled **Information unavailable**, not Ready, without adding a frontend acceptance blocker or changing the enabled state of Accept ECN. No backend field, acceptance rule, timestamp, confidence claim, or review-audit behavior was added or changed.

- Added the WP4 Settings clarity slice: the existing `/settings` route now visibly separates Provider readiness, Reference datasets, and progressively disclosed Advanced administration. Each source replacement follows choose file, describe scope, then validate-and-activate guidance; publication cards clarify active-scope, warning, and pinned-ECN consequences. The exact `RESET HISTORY` confirmation and fail-closed activity guard remain unchanged, and no source/provider API contract was changed.
- Contained the ECN detail `@ui5/webcomponents-fiori` DynamicPage 2.26.0 internal-DOM dependency in `ui/src/pages/ecn-detail/dynamic-page-scroll-adapter.js`. The adapter is the sole runtime boundary for the installed component's internal scroll-container and title-wrapper selectors, geometry, scroll subscription, pinned-title alignment, and section-target scrolling. It deliberately uses native `scrollIntoView` when those internals are absent or changed; no page code receives an internal node or selector. This is contained technical debt, not a claim of a public DynamicPage replacement: the approved shared scrolling page, route targets, pinned-title offset, section scroll spy, and `aria-current` behavior remain in place. Focused deterministic adapter absence/fallback, pinned-offset/scroll-subscription, and in-place route-target tests passed together with the full unit, build, E2E, and whitespace validation suite; revalidate its precision behavior whenever the installed DynamicPage template/version changes.

- **Reviewer P1 — DynamicPage scroll-observer lifecycle: resolved and validated.** Installed UI5 2.26.0 does not emit `ui5-after-render`, so the detail page no longer depends on that nonexistent event. The page schedules a refresh immediately before active-section calculation, and the adapter now owns MutationObserver-based detection of internal scroller rendering or replacement, including listener reattachment and active-navigation refresh. Deterministic adapter and page tests render and replace the internal scroller after initialization without synthetic render-event dispatch, asserting the replacement listener updates `aria-current`. The focused DynamicPage tests, complete UI unit suite, production build, full E2E suite, whitespace checks, and staged-diff check all passed after this follow-up.

**Purpose:** Provide a durable, agent-ready task packet for bringing the ECN Impact Analysis UI to a consistent SAP/Fiori standard while preserving the existing, evidence-led engineering-review workflow. This plan records the verified review findings, implementation decisions, boundaries, work packages, acceptance criteria, and handoff requirements. It is the durable source of truth for this UX remediation; chat history is not.

**Audience:** Delivery lead, UI implementers, QA/accessibility reviewers, and SAP PoC agents using `sap-poc-delivery` conventions.

**Related documents:**

- [ECN Impact user interface](ecn-impact-ui.md) — product behavior and current UI intent.
- [ECN snapshot review workspace](snapshot-review-redesign.md) — assessment, import, assistant, and review boundaries.
- [ECN detail review clarity](ecn-detail-review-clarity.md) — evidence/readiness presentation contract.
- [ECN assistant panel](ecn-assistant-panel.md) — current assistant behavior.
- [Engineering review](engineering-review.md) — accept/send-back business rules.

Where this plan conflicts with an older document on **presentation or interaction behavior**, this plan governs the UX remediation. It does not change backend evidence, review, authorization, or data-retention contracts without explicit approval.

---

## 1. Delivery contract

### 1.1 Business outcome

Enable an engineering reviewer to safely and quickly:

1. find the ECNs requiring attention;
2. understand what changed and which evidence supports the impact;
3. identify any missing or stale evidence that blocks a decision;
4. accept an ECN or send it back with a recorded rationale; and
5. ask an evidence-grounded assistant without losing context or misreading its scope.

The PoC remains **Autonomy Level 0**: AI may explain, retrieve, and draft, but a human makes the engineering decision. The UI must never imply that AI output is source evidence or that an ECN was approved automatically.

### 1.2 Primary users and workflows

| User | Primary task | Success measure |
|---|---|---|
| Engineering reviewer | Triage an Inbox, inspect one ECN, decide accept/send back | Can reach an ECN needing action quickly and understands the decision blocker or next action without interpreting raw technical data |
| Engineering manager/change coordinator | Monitor exception volume and review progress | Can identify pending/failed/high-impact items and navigate to the relevant queue |
| Data steward | Publish a new governed source snapshot | Can validate scope, warnings, coverage, and activation consequences before replacing a dataset |
| PoC administrator/developer | Check model/provider readiness and diagnose configuration safely | Can see safe status metadata without exposing credentials, endpoint URLs, prompts, or customer data |

### 1.3 In scope

- UI hierarchy, responsive behavior, component consistency, accessibility, feedback, and interaction design for the existing Inbox, ECN detail page, assistant, and Settings.
- UI-only test fixtures and Playwright/Vitest coverage needed to prove the changed behavior.
- Presentation of already available readiness, evidence, review, and assistant data.
- Small frontend-only API DTO consumption changes **only if an existing safe endpoint already provides the required field**.

### 1.4 Explicitly out of scope

- SAP BTP provisioning, Cloud Foundry deployment, destinations, service bindings, or credentials.
- Changes to customer source data, HANA records, AI Core settings, model deployments, or review decisions.
- New AI models, RAG, agent tools, provider calls, prompt changes, or wider assistant permissions.
- Changing backend business rules for acceptance, source readiness, audit retention, or authorization without a separately approved design.
- Replacing the entire UI framework or adding a second design system.

If an implementation needs a new backend field or changes a review/AI contract, pause and raise it as a delivery-lead decision. For SAP BTP AI work, invoke `sap-btp-ai` first and then the relevant specialist skill before making a proposal.

---

## 2. Verified baseline and review evidence

### 2.1 Technology baseline

- The UI is plain JavaScript built with Vite.
- Installed dependencies resolved locally include `@ui5/webcomponents@2.26.0`, `@ui5/webcomponents-fiori@2.26.0`, and `@playwright/test@1.63.0`. `ui/package.json` declares compatible `^2.12.0` ranges; do not alter dependencies merely for this remediation.
- The shell already uses `ui5-shellbar`, `ui5-side-navigation`, and `ui5-navigation-layout` in `ui/index.html`.
- ECN detail already uses `ui5-dynamic-page`, `ui5-dynamic-page-title`, `ui5-dynamic-page-header`, `ui5-bar`, `ui5-button`, `ui5-input`, and `ui5-textarea` in `ui/src/pages/ecn-detail/ecn-detail.html`.
- The application currently mixes these UI5 components with native buttons, inputs, selects, file inputs, tables, and dialogs. Native controls are not prohibited; the implementation issue is inconsistent behavior, feedback, focus, and responsive treatment in high-frequency workflows.

### 2.2 Confirmed mobile blocker

A local Playwright review with the repository mock API at **390 × 844** confirmed that the fixed chat launcher obstructs primary controls:

| Page | Launcher bounds | Obstructed control | Control bounds | Result |
|---|---:|---|---:|---|
| Inbox | `x=285.02, y=776, w=88.98, h=52` | `Go` | `x=321.63, y=808.33, w=39.38, h=36` | Bounds intersect |
| ECN detail | `x=285.02, y=776, w=88.98, h=52` | `Accept ECN` | `x=268.14, y=796, w=93.86, h=36` | Bounds intersect |

This is a P0 defect. It must be resolved before further visual polish or a customer mobile demonstration.

### 2.3 Confirmed UX observations

- At 390 px wide, Inbox KPI cards plus seven filter controls push the first result table below the first viewport. The table began around `y=934` on an `844`-pixel-high viewport.
- The desktop Inbox is visually coherent, but its many-column table is an inefficient primary small-screen experience.
- Inbox KPI counts are useful context but do not consistently provide a direct next action.
- The ECN detail page includes useful evidence sections and a floating decision bar, but decision readiness and disabled-action reasons are distributed rather than summarized.
- The assistant is browser-memory-only, per-assessment, and can expose consulted objects/ECNs. This is a positive boundary that must be retained.
- The settings page combines provider diagnostics, dataset publication, mapping, and destructive history administration into one dense experience.
- Existing UI behavior includes ARIA labels on many controls, accessible names/tooltips on Inbox icon actions, `role="log"`/`aria-live="polite"` in chat, and semantic page sections. Preserve these strengths.

### 2.4 Review limitations

- The visual review used local, sanitized/mocked API responses. It did not call live HANA, AI Core, SAP systems, or customer data.
- The mock did not fulfill every detail-page endpoint; observed local `Loading…`/unavailable states are not by themselves production defects.
- No automated axe-core scan or NVDA/VoiceOver evaluation has yet been run. The accessibility work package must add this manual and automated validation.

---

## 3. UX decisions that implementation must follow

### D1 — Assistant responsiveness and modality

**Wide screens (at least 1400 px):** retain a docked, non-modal side panel beside the evidence page. The page remains usable; no backdrop blocks it.

**Medium screens (601–1399 px):** use one clearly announced assistant entry point and open the assistant as a modal, focused overlay only if it blocks the page. Do not combine a blocking backdrop with an unnamed non-modal `<aside>`.

**Narrow screens (600 px or below):** remove the floating bottom-right launcher. Place the assistant action in the Shell Bar or another persistent, non-overlapping page action. Open a full-height modal dialog/drawer pattern. The assistant must not share the bottom safe area with the ECN decision footer.

**Rationale:** a fixed bottom launcher is incompatible with the page’s existing floating decision bar and filter actions. Wide-screen side-by-side evidence remains valuable, while narrow-screen users need a focused, non-obstructing interaction.

### D2 — Inbox is an exception queue, not a filter form

The Inbox must optimize for items needing human action.

- Present a compact initial filter set: search, action/status, and sort.
- Put secondary criteria in a progressive-disclosure filter dialog/panel.
- Use actionable KPI/exception summaries. A KPI that represents a queue must apply a visible filter or link to the queue.
- Preserve active filters and selected row context when navigating to ECN detail and back.
- Use a responsive list/pop-in representation at narrow widths: identifier/description, actionable state, severity/impact, plant count, and one primary action remain visible. Lower-priority fields may be expanded or shown in a pop-in.

### D3 — ECN decision readiness is explicit

The ECN detail header must state whether a reviewer can act and why. Display a compact, source-backed **Decision readiness** summary near the title or immediately before the evidence content. It must cover only facts already available in the assessment, for example:

- source/snapshot readiness;
- analysis generation state;
- BOM evidence readiness;
- plant-impact readiness;
- review preconditions such as reviewer identity, if applicable.

The decision footer must state the specific blocking reason whenever `Accept ECN` is unavailable and provide a link/scroll target to the relevant evidence. It must not infer a new rule or loosen existing acceptance rules.

### D4 — Preserve one section-navigation model

The existing detail navigation behaves as a scrolling section navigator. Keep that model rather than creating a hybrid tab-and-scroll experience:

- the active anchor changes as the user scrolls;
- the active control uses `aria-current="true"`;
- the label exactly matches the target section heading;
- keyboard activation scrolls the target into a visible reading position and moves focus only when that improves orientation.

Do not introduce `ui5-tabcontainer` unless the product owner explicitly wants distinct views where non-selected content is not in the scrolling page.

### D5 — AI output always has business-readable provenance

Where the UI shows an AI-generated summary or assistant response, show the user enough context to understand its boundary:

- scope: current ECN or all stored ECNs;
- source/evidence state and source snapshot freshness where available;
- generation/retrieval time where available;
- consulted business objects or ECNs as readable links;
- a plain-language limitation when evidence is incomplete.

Technical tool traces, raw hashes, prompt content, credentials, tenant identifiers, and backend paths remain diagnostic-only and are not displayed by default.

### D6 — Settings separates operational intent

Organize Settings into three user-facing areas, either separate routes or clearly disclosed sections:

1. **Provider readiness** — safe status and a `Check now` action.
2. **Reference datasets** — active publication, scope, warning count, source date, validation, replacement, and activation confirmation.
3. **Advanced administration** — plant mapping and history reset, with strong warning/confirmation treatments.

The data steward must understand the consequence of replacement before activation. Technical hashes are secondary traceability metadata, not the central proof of a dataset’s identity.

---

## 4. Work packages and implementation order

One work package should have one writer for overlapping frontend paths. Do not run parallel writers against `ui/src/style.css`, `ui/src/modules/shell-chat.js`, or `ui/src/pages/ecn-detail/`.

### WP0 — Establish reproducible UX test fixtures and a visual baseline

**Outcome:** The team can prove viewport-specific behavior without live services or customer data.

**Primary paths:**

- `ui/tests/e2e/shell.spec.js`
- `ui/tests/e2e/api-mock.js`
- `ui/tests/fixtures/api-responses.js`
- Relevant existing page test files in `ui/tests/`

**Tasks:**

1. Reuse existing sanitized fixtures; do not create live API dependencies.
2. Add shared helpers to obtain control bounding rectangles and test whether two rectangles intersect.
3. Capture baseline tests for the Inbox at 390 px and ECN detail at 390 px.
4. Ensure the fixture includes a valid list, detail envelope, snapshot, readiness states, and a blocked/disabled-accept state needed by later packages.
5. Keep tests outcome-oriented; do not hard-code incidental stylesheet class hierarchies if a user-visible locator is available.

**Acceptance criteria:**

- Tests can prove the chat entry control does not intersect `Go`, `Import new ECN`, `Send back to engineering`, or `Accept ECN` at 390 px.
- Tests run without contacting external services.
- Existing desktop containment tests remain valid.

**Do not:** change app behavior or backend DTOs in this package.

---

### WP1 — Resolve chat obstruction and establish an accessible responsive assistant

**Outcome:** Chat remains available without obstructing review/filter actions and follows a coherent modal/non-modal accessibility model.

**Likely paths:**

- `ui/index.html`
- `ui/src/main.js`
- `ui/src/modules/shell-chat.js`
- `ui/src/style.css`
- `ui/src/components/ecn-chat.js`
- `ui/src/pages/ecn-detail/ecn-detail.css`
- `ui/tests/e2e/shell.spec.js`
- `ui/tests/snapshot-workspace.test.js`

**Implementation requirements:**

1. Add a visible, named shell-level assistant action suitable for narrow layouts. It must use an accessible name and be available only when its current global/ECN scope is meaningful.
2. On screens at least 1400 px, preserve the current non-modal docked panel behavior where the content area reserves reading space.
3. At 600 px and below, eliminate `.assistant-launcher` as a fixed bottom-right control. Do not simply move it a few pixels; it must not compete with floating page actions.
4. For a modal assistant presentation, use an appropriate UI5 modal component supported by the locked dependency version, or implement equivalent dialog semantics: role/name, modal state, initial focus, Escape behavior, focus containment, and focus restoration to the invoker.
5. For non-modal presentation, remove the blocking backdrop and keep the main content operable.
6. Retain existing browser-memory conversation behavior, per-assessment isolation, draft retention while collapsed, and reload clearing.
7. Make assistant scope visible in the UI: `This ECN: <number>` or `All stored ECNs`.
8. Keep the current read-only data boundary: no assistant action may acquire new write permissions or broaden the available ECN scope.

**Acceptance criteria:**

- No overlapping interactive bounds at 320 px, 390 px, 480 px, 768 px, and 1440 px.
- The 390 px ECN detail footer remains visible and operable with chat closed.
- With chat open, keyboard focus behaves as the selected modal/non-modal design specifies; Escape closes modal chat and returns focus to its invoker.
- A screen-reader user can determine the assistant’s scope and close it.
- Existing assistant tests still prove draft retention and conversation isolation.

**Human decision required before implementation:** confirm whether medium-width chat (601–1399 px) should be a modal panel or a non-modal docked/resizable panel. The recommended default is modal at this range because the current overlay uses a backdrop.

---

### WP2 — Make the Inbox responsive and task-first

**Outcome:** A reviewer reaches the relevant ECN queue quickly on desktop and mobile, without sacrificing information needed for power users.

**Likely paths:**

- `ui/src/pages/ecns/ecns.html`
- `ui/src/pages/ecns/ecns.js`
- `ui/src/pages/ecns/ecns.css`
- `ui/src/components/ecn-kpi-header.js` only if shared KPI behavior is intentionally aligned
- `ui/src/style.css`
- `ui/tests/ecn-views.test.js`
- `ui/tests/e2e/shell.spec.js`

**Implementation requirements:**

1. Design the Inbox around a minimum viable triage set: text search, primary action/state filter, and sort.
2. Place secondary filters in an explicit `Filters` control with active-filter count and clear/reset behavior.
3. Make relevant KPI cards actionable. A click must apply a defined query state and visibly announce/reflect the changed result set.
4. On narrow screens, show results within the first viewport or after one short, intentional disclosure action. Do not require the user to traverse all desktop filters before seeing any ECN.
5. Define mobile row priority: ECN identifier/description, analysis/review state, impact/warning signal, plant count, and `View`.
6. Use a responsive UI5 table/pop-in pattern where supported by the current installed component set; otherwise implement an accessible list/card representation for narrow widths while preserving the desktop table.
7. Retain real table semantics for desktop. Preserve the current accessible full description behavior for visually clamped text.
8. Group bulk actions in a semantic toolbar and explain why actions are disabled with zero selection. Keep destructive actions visibly distinct.
9. Preserve query, sort, pagination, and selection semantics defined in the existing UI/API contract. Do not change server-side selection resolution.

**Acceptance criteria:**

- At 390 px, an ECN result is visible after loading without requiring the user to scroll past the entire filter form.
- At 390 px, the result representation has no document-level horizontal overflow.
- At desktop width, all currently required columns and bulk actions remain available.
- Applying an actionable KPI changes the visible filter/result state and remains keyboard accessible.
- Return navigation from detail retains Inbox context.

**Product decision required before implementation:** approve the default priority/sort. Recommendation: surface `needs attention` / pending review before chronological upload order.

---

### WP3 — Clarify ECN decision readiness and evidence-driven actions

**Outcome:** The reviewer can see the decision state, the blocker, and the next safe action without reconciling multiple cards or inferring rules.

**Likely paths:**

- `ui/src/pages/ecn-detail/ecn-detail.html`
- `ui/src/pages/ecn-detail/ecn-detail.js`
- `ui/src/pages/ecn-detail/ecn-detail.css`
- `ui/src/components/assessment-context.js`
- `ui/src/components/readiness-list.js`
- `ui/src/components/analysis-sections.js`
- `ui/tests/ecn-detail-page.test.js`
- `ui/tests/e2e/shell.spec.js`

**Implementation requirements:**

1. Add a compact Decision readiness presentation using existing snapshot/review/readiness facts. Do not calculate new backend business state in the browser.
2. Present readiness in business language: ready, blocked, needs investigation, or information unavailable. Preserve the system’s shared result-state vocabulary where it is applicable.
3. When the Accept action is disabled, display the exact existing reason and a path to inspect the relevant evidence.
4. Distinguish source readiness, AI-summary availability, and business acceptance. A failed summary generation cannot hide deterministic evidence; incomplete evidence cannot be presented as an approved decision.
5. Maintain the Dynamic Page and floating decision footer, but ensure the footer never obscures content and can coexist with responsive chat.
6. Retain the scrolling section navigation model and current scroll-spy/`aria-current` behavior. Use consistent labels between navigation and sections.
7. Add concise AI provenance beside the summary only when supported by data already available to the UI. Do not invent timestamps, confidence, data source names, or completeness claims.
8. Use semantic UI feedback for loading, retry, error, and success. The message must explain business impact and next action, not only technical failure.

**Acceptance criteria:**

- A reviewer can identify why an ECN is not acceptable from the first screen of detail without opening a technical dialog.
- Available deterministic evidence remains inspectable when AI summary generation fails.
- Each disabled decision action has visible explanatory text.
- Section navigation correctly updates `aria-current` on manual scrolling and keyboard activation.
- `Accept ECN` and `Send back to engineering` retain the current backend/audit behavior.

**Human decision required before implementation:** confirm the approved visual wording for the readiness states and whether reviewer identity is a mandatory precondition for acceptance in every tenant/workflow state.

---

### WP4 — Standardize settings and high-frequency form/dialog components

**Outcome:** High-frequency administrative interactions feel like one coherent Fiori application and communicate validation and risk clearly.

**Likely paths:**

- `ui/src/pages/settings/settings.html`
- `ui/src/pages/settings/settings.js`
- `ui/src/pages/settings/settings.css`
- `ui/src/pages/settings/plant-mapping.js`
- `ui/src/pages/ecns/ecns.html`
- `ui/src/pages/ecns/ecns.js`
- `ui/src/pages/ecn-detail/ecn-detail.html`
- `ui/src/pages/ecn-detail/ecn-detail.js`
- `ui/src/style.css`
- Corresponding `ui/tests/*settings*`, `ui/tests/*ecn*`, and E2E tests

**Component direction:**

| Interaction | Preferred direction |
|---|---|
| Filters and text fields | UI5 input/select controls or a consistently styled native form pattern; choose one per workflow and preserve labels/validation semantics |
| File selection | `ui5-file-uploader` or equivalent UI5-supported upload control, with validation and busy/error feedback |
| Confirmation and send-back flows | UI5 dialog or equivalent fully accessible modal implementation with labeled header/footer actions |
| Provider/source feedback | semantic message strip, busy indicator, and status/tag components where available |
| Dense data | responsive UI5 table/pop-in for primary lists; contained native table may remain for technical evidence where scrolling is intentionally required |
| Status treatment | UI5 semantic status/tag where it conveys standard status; custom pills may remain only if semantic state, contrast, and accessible text are preserved |

**Implementation requirements:**

1. Do not replace controls solely for cosmetic uniformity. Prioritize frequent user actions, modal flows, input validation, uploads, status feedback, and keyboard consistency.
2. Split Settings visibly into Provider readiness, Reference datasets, and Advanced administration. Route splits require product approval; progressive disclosure within `/settings` is acceptable.
3. Make source replacement a clear progression: choose file, specify scope/date, validate, review warnings/coverage, then activate.
4. Preserve the security boundary: status views never reveal credentials, endpoint URLs, tenant IDs, request/response bodies, prompts, or source payloads.
5. Keep the guarded history reset behavior (`RESET HISTORY` exact phrase and fail-closed status check) unchanged unless separately approved.

**Acceptance criteria:**

- Every high-frequency form control has a programmatic name, visible label, accessible validation/error state, and clear busy state.
- Dialogs have an explicit title, cancellable path, predictable initial focus, Escape behavior, and focus return.
- A user can distinguish provider readiness from dataset activation and destructive administration without scanning one long undifferentiated form.
- Existing reference-dataset and reset safety tests remain passing.

---

### WP5 — Accessibility and visual quality gate

**Outcome:** The completed UI meets an agreed, testable baseline before a customer-facing PoC demo.

**Primary paths:** All touched UI files and `ui/tests/`.

**Required checks:**

1. Keyboard-only walkthrough:
   - Inbox filters, filter disclosure, KPI actions, row selection, table navigation, bulk actions, and pagination.
   - ECN detail section navigation, evidence links, assistant, send-back, and accept actions.
   - Settings provider check, upload/validation, plant mapping, and guarded reset.
2. Screen-reader walkthrough using NVDA with Chrome or VoiceOver with Safari:
   - shell navigation, page landmarks, dialogs, assistant scope, dynamic statuses, table context, and error messages.
3. Automated scan using the project-approved accessibility tool. Introducing a new dependency needs delivery-lead approval; use a temporary local tool only if it does not change tracked dependencies.
4. Visual viewport matrix at 320, 390, 480, 768, 1024, 1440, and 1600 px widths.
5. Browser zoom at 200%, high-contrast/theme validation if supported by the active UI5 theme, and reduced-motion review for any introduced animation.
6. Verify no control collision, clipped primary action, document-level horizontal overflow, inaccessible dialog, or unannounced async error.

**Acceptance criteria:**

- All P0/P1 task flows can be completed with keyboard only.
- No critical or serious accessibility violation remains in changed flows without a recorded, approved exception.
- Every async workflow communicates loading, success, and recoverable failure.
- The visual test matrix passes with no overlapping interactive controls.

---

## 5. Data, AI, and security guardrails

These are non-negotiable for every work package:

- Use existing sanitized fixtures for UI tests. Do not paste customer records, credentials, tokens, connection strings, prompts, or raw source payloads into test data, screenshots, logs, or documentation.
- Do not call live provider, HANA, SAP, Windchill, or BTP endpoints for a UI-only change without explicit user approval and a bounded read-only scope.
- Do not expose provider diagnostics beyond the already approved safe metadata.
- Preserve the assistant’s browser-memory-only conversation handling and per-assessment isolation.
- Preserve review audit behavior: UI changes must not make a review decision, write source data, or change the accepted exact snapshot automatically.
- Evidence is source-backed; AI text is a labeled interpretation/assistance record. The UI must keep that distinction visible.

---

## 6. Verification commands and expected evidence

Run the smallest relevant checks during each package, then the full UI gate before handoff:

```sh
# Focused unit/component tests selected by the changed package
npm --prefix ui test -- --run <relevant-test-file>

# Complete UI unit/component suite
npm --prefix ui test

# Production UI build
npm --prefix ui run build

# Browser acceptance suite
E2E_PORT=4187 npm --prefix ui run test:e2e
```

The command output must be captured in the agent handoff with pass/fail status. Do not claim browser, accessibility, live integration, deployment, or device validation that was not run.

For WP1/WP2/WP3, add deterministic browser assertions for:

- no rectangle intersection between chat access and primary action controls;
- no document-level horizontal overflow at narrow widths;
- visible, reachable decision-footer controls;
- keyboard focus behavior on open/close for assistant and dialogs;
- active ECN navigation state after scrolling;
- preserved filter state after opening and returning from detail.

---

## 7. Definition of done

The UX remediation is complete only when all conditions below are satisfied:

1. **Mobile safety:** At 320, 390, and 480 px, the assistant never overlaps a primary action, decision footer, filter submission control, or system navigation.
2. **Triage efficiency:** At 390 px, a reviewer can see an ECN result without traversing the entire desktop filter form.
3. **Decision clarity:** On an ECN page, the current decision readiness, blocker, and next safe action are visible before a reviewer reaches the final decision footer.
4. **Evidence integrity:** Deterministic/source-backed evidence remains visible when AI generation is unavailable; AI content is labeled and scoped.
5. **Responsive data usability:** Inbox remains a usable, accessible exception queue on mobile and retains desktop information density for power users.
6. **Interaction consistency:** Updated forms, uploads, dialogs, statuses, and toolbars follow one coherent UI5/Fiori interaction standard.
7. **Accessibility:** Keyboard, screen-reader, async feedback, and visual viewport acceptance criteria in WP5 are met.
8. **No scope regression:** Existing import boundaries, server-side selection semantics, review audit behavior, source-data safety, assistant read-only scope, and no-live-call constraints are preserved.
9. **Validation evidence:** Exact test commands and outcomes are recorded. Any skipped test, known limitation, or product decision is explicitly documented.

---

## 8. Agent execution protocol and handoff packet

Before acting, an implementation agent must:

1. Read this document, the linked product/evidence documents, the active acceptance criteria, and all files named in its chosen work package.
2. Inspect the current code before proposing edits. The UI may have changed since this plan was written.
3. Record which findings are still reproducible versus superseded.
4. Limit its changes to one approved work package unless a delivery lead explicitly joins packages.
5. Keep facts and assumptions separate. Ask for a decision rather than inventing a business rule, an API field, or a component capability.
6. Use `sap-poc-delivery` conventions: smallest demonstrable vertical slice, sanitized fixtures, deterministic checks, no external writes/deployments, and durable documentation.
7. For any SAP BTP AI change, invoke `sap-btp-ai` first and then the appropriate specialized skill. UI remediation alone does not authorize AI-service changes.

### Required agent handoff

Every agent returns the following, and appends any durable decision/limitation to the relevant documentation:

```md
## UX remediation handoff

- Work package: WP<n> — <name>
- Outcome and acceptance criterion addressed:
- Owned/changed paths:
- Read-only paths consulted:
- Verified facts:
- Assumptions:
- UI/component decisions made and rationale:
- Accessibility behavior tested:
- Commands run and exact outcomes:
- Screens/viewports tested:
- Open risks or known limitations:
- Decisions requiring human approval:
- Next safe action:
```

### Suggested implementation-task prompt

Use this only after a delivery lead chooses a work package:

> Implement **WP<n>** from `docs/ui-ux-remediation-plan.md` as the smallest complete vertical slice. Read the named related documents and current code first. Preserve existing ECN evidence, review-audit, assistant-scope, and no-live-service boundaries. Use sanitized local fixtures only. Do not provision, deploy, alter BTP resources, call live services, or make backend business-rule changes. Add focused deterministic tests, run the listed validation commands, and return the required UX remediation handoff. If a required product, accessibility, component-capability, API, or authorization decision is missing, stop and request it rather than inventing it.

---

## 9. Open decisions for the delivery lead/product owner

These decisions are intentionally not invented by this plan:

1. **Medium-width chat:** modal overlay versus non-modal docked/resizable panel from 601–1399 px. Recommended: modal overlay with complete dialog semantics.
2. **Inbox default priority:** chronological upload order versus an exception-oriented sort. Recommended: needs-attention/pending-review items first.
3. **Decision-readiness wording:** approved labels for `ready`, `blocked`, `needs investigation`, and `information unavailable` in the customer’s engineering terminology.
4. **Acceptance preconditions:** whether reviewer identity is universally mandatory or tenant/workflow-dependent.
5. **Settings information architecture:** separate routes versus disclosed sections within the current `/settings` page.
6. **Accessibility quality bar:** chosen automated tool and the named browser/screen-reader combination for release evidence.

No agent should block all remediation on these decisions. Each agent should implement only the portions whose behavior is approved and unambiguous, then surface the open decision in its handoff.
