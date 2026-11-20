import { expect, test } from "@playwright/test";
import {
  businessEvidenceEnvelope,
  detailEnvelope,
  providerStatusEnvelope,
  referenceDatasetsEnvelope,
} from "../fixtures/api-responses.js";

/** Install mutually consistent list and progress responses for shell checks. */
async function mockList(page) {
  await page.route("**/api/ecns**", (route) => route.fulfill({ json: {
    items: [{
      ecn_row_id: "11111111-1111-4111-8111-111111111111",
      ecn_number: "ECN100",
      display_ecn_number: "ECN100-WIDE-IDENTIFIER-FOR-OVERFLOW",
      title: "Wide row",
      description_summary: "A deliberately wide description for responsive containment verification",
      source_ordinal: 0,
      uploaded_at: "2026-09-14T08:00:00Z",
      source_file_name: "deliberately-wide-source-file-name.json",
      counts: {
        affected_item_count: 3,
        raw_ecn_match_count: 4,
        ebom_match_count: 2,
        ebom_context_count: 1,
        mbom_match_count: 1,
        warning_count: 2,
      },
      analysis_state: "COMPLETED_WITH_GAPS",
      completeness: "COMPLETED_WITH_GAPS",
      overall_impact: "MEDIUM",
    }],
    page: 1,
    page_size: 25,
    total_items: 1,
    total_pages: 1,
  } }));
  await page.route("**/api/analysis-progress**", (route) => route.fulfill({ json: {
    upload_id: null,
    queued_count: 0,
    blocked_count: 0,
    running_count: 0,
    completed_count: 0,
    completed_with_gaps_count: 1,
    failed_count: 0,
    unfinished_count: 0,
    total_count: 1,
  } }));
}

function overviewSnapshot(rowId) {
  return {
    assessment_id: rowId,
    evidence_hash: "a".repeat(64),
    version: "ecn-snapshot-v2",
    counts: { technical_objects: 2, supporting_objects: 0, bom_change_objects: 1, located_objects: 2, unlocated_objects: 0 },
    questions: [], limitations: [],
    part_descriptions: [
      { number: "OLD1", description: "Metal shim ring" },
      { number: "NEW1", description: "Flexible shim ring" },
    ],
    ecn_context: {
      owner_name: "Dana Whitfield",
      business_driver: "Correct the released design data before model cutover.",
      financial_notes: "Estimated increase of EUR 0.85 per unit.",
      source: "SOURCE",
    },
    plant_impact: { plants: [{
      plant_code: "PL10", plant_name: "Plant A", cutover_from: "2026-05-19",
      materials: [{ number: "NEW1", description: "Flexible shim ring", procurement_types: ["External procurement"] }],
    }] },
    bom_changes: { lines: [{
      qualifiers: [], levels: ["EBOM", "MBOM"], plants: [{ plant_code: "PL10", plant_name: "Plant A" }],
      outgoing: { number: "OLD1", description: "Metal shim ring", date: "2026-05-19" },
      incoming: { number: "NEW1", description: "Flexible shim ring", date: "2026-05-19" },
    }], materials: [] },
    // MOCK_DATA_RETIREMENT_VALIDATION_DISABLED_START: overlay-only snapshot fields.
    // phase_schedule: [{
    //   material_number: "NEW1", plant_code: "PL10", description: "Flexible shim ring",
    //   from_label: "Eng. Approved", to_label: "Phase-In", effective_date: "2026-05-19", planner_confirmed: false,
    // }],
    // purchase_prices: [{ material_number: "NEW1", plant_code: "PL10", status: "MISSING" }],
    // demo_assumptions: ["Cost, sourcing, and manufacturing timing values are illustrative demonstration data."],
    // MOCK_DATA_RETIREMENT_VALIDATION_DISABLED_END: overlay-only snapshot fields.
  };
}

function longBomSnapshot(rowId) {
  const snapshot = overviewSnapshot(rowId);
  snapshot.bom_changes.lines = Array.from({ length: 10 }, (_, index) => ({
    qualifiers: [], levels: ["EBOM", "MBOM"], plants: [{ plant_code: "PL10", plant_name: "Plant A" }],
    outgoing: { number: `OLD${index + 1}`, description: `Metal shim ring ${index + 1}`, date: "2026-05-19" },
    incoming: { number: `NEW${index + 1}`, description: `Flexible shim ring ${index + 1}`, date: "2026-05-19" },
  }));
  return snapshot;
}

test("root renders the ECN product route", async ({ page }) => {
  await mockList(page);
  await page.goto("/");
  await expect(page.getByRole("heading", { name: "ECN Inbox" })).toBeVisible();
  await expect(page.locator("#table-scroll .ecn-link", { hasText: /^ECN100-WIDE/ })).toBeVisible();
});

test("a deliberately wide row remains contained at 390px", async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await mockList(page);
  await page.goto("/ecns");
  const dimensions = await page.locator("#ecn-card-list .ecn-result-card").evaluate((node) => ({
    right: node.getBoundingClientRect().right,
    viewportWidth: window.innerWidth,
    rootScrollWidth: document.documentElement.scrollWidth,
    rootClientWidth: document.documentElement.clientWidth,
  }));
  expect(dimensions.right).toBeLessThanOrEqual(dimensions.viewportWidth);
  expect(dimensions.rootScrollWidth).toBe(dimensions.rootClientWidth);
  await expect(page.locator("#ecn-card-list .ecn-result-card-description")).toContainText("deliberately wide description");
});

test("all four overview cards filter by keyboard and mouse with grouped statuses", async ({ page }) => {
  await mockList(page);
  const states = ["COMPLETED", "COMPLETED_WITH_GAPS", "QUEUED", "RUNNING", "BLOCKED", "FAILED", "READY"];
  const requests = [];
  await page.route("**/api/ecns?**", (route) => {
    const params = new URL(route.request().url()).searchParams;
    requests.push(Object.fromEntries(params));
    const filter = params.get("analysis_state");
    const selected = filter === "SUMMARIES_AVAILABLE" ? ["COMPLETED", "COMPLETED_WITH_GAPS"]
      : filter === "IN_PROGRESS" ? ["QUEUED", "RUNNING"] : filter ? [filter] : states;
    const items = selected.map((state) => ({
      ecn_row_id: `11111111-1111-4111-8111-${String(states.indexOf(state) + 1).padStart(12, "0")}`,
      display_ecn_number: `ECN-${state}`, ecn_number: `ECN-${state}`,
      description_summary: "Synthetic shaft change", uploaded_at: "2026-09-14T08:00:00Z",
      counts: { affected_item_count: 1, warning_count: 1 }, analysis_state: state,
    }));
    return route.fulfill({ json: { items, page: 1, page_size: 50, total_items: items.length, total_pages: 1 } });
  });
  await page.goto("/ecns?query=shaft&plant=PL10&has_warnings=true&analysis_state=BLOCKED&sort=warning_count&order=asc&page=2&page_size=50");
  await expect(page.getByText("Show needs attention", { exact: true })).toHaveCount(0);
  for (const [id, filter, expected, key] of [
    ["kpi-complete-filter", "SUMMARIES_AVAILABLE", ["COMPLETED", "COMPLETED_WITH_GAPS"], "Enter"],
    ["kpi-progress-filter", "IN_PROGRESS", ["QUEUED", "RUNNING"], "Space"],
    ["kpi-attention", "FAILED", ["FAILED"], null],
    ["kpi-total-filter", "", states, "Enter"],
  ]) {
    await page.locator("#table-scroll .summary-selection").first().check();
    const card = page.locator(`#${id}`);
    if (key) {
      await card.focus();
      await page.keyboard.press(key);
    } else await card.click();
    await expect(page.locator('[name="analysis_state"]')).toHaveValue(filter);
    await expect(card).toHaveAttribute("aria-pressed", "true");
    await expect(card).toBeFocused();
    await expect(page.locator("#summary-selection-status")).toHaveText("0 selected");
    await expect(page.locator("#table-scroll .ecn-link")).toHaveText(expected.map((state) => `ECN-${state}`));
    expect(requests.at(-1)).toMatchObject({
      query: "shaft", plant: "PL10", has_warnings: "true", sort: "warning_count",
      order: "asc", page: "1", page_size: "50",
    });
    expect(requests.at(-1).analysis_state || "").toBe(filter);
    expect(new URL(page.url()).searchParams.get("analysis_state") || "").toBe(filter);
  }
  await page.reload();
  await expect(page.locator("#kpi-total-filter")).toHaveAttribute("aria-pressed", "true");
});

test("narrow Inbox keeps triage controls first and discloses active secondary filters by keyboard", async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await mockList(page);
  await page.goto("/ecns?query=shaft&sort=warning_count&order=asc&page=2&page_size=50");

  const disclosure = page.getByRole("button", { name: /^Filters,/ });
  const secondary = page.locator("#secondary-filters");
  await expect(disclosure).toHaveAttribute("aria-controls", "secondary-filters");
  await expect(disclosure).toHaveAttribute("aria-expanded", "false");
  await expect(disclosure).toContainText("2 active");
  await expect(secondary).toBeHidden();
  for (const name of ["query", "analysis_state", "sort"]) {
    await expect(page.locator(`[name="${name}"]`)).toBeVisible();
  }
  await expect(page.locator("#apply-filters")).toBeVisible();
  const resultBounds = await page.locator("#ecn-card-list").boundingBox();
  expect(resultBounds.y).toBeLessThan(844);

  await disclosure.focus();
  await page.keyboard.press("Enter");
  await expect(disclosure).toHaveAttribute("aria-expanded", "true");
  await expect(secondary).toBeVisible();
  await page.getByRole("button", { name: "Clear filters" }).click();
  await expect(disclosure).toContainText("0 active");

  await page.getByRole("button", { name: /Generation needs attention/ }).click();
  await expect(page.locator('[name="analysis_state"]')).toHaveValue("FAILED");
  await expect(page.locator("#filter-result-status")).toContainText("needs attention");
  await expect(page).toHaveURL(/analysis_state=FAILED/);
  await expect(page).toHaveURL(/page=1/);
});


test("narrow Inbox exposes chat in the shell without covering its primary filter action", async ({ page }) => {
  for (const width of [320, 390, 480]) {
    await page.setViewportSize({ width, height: 844 });
    await mockList(page);
    await page.goto("/ecns");

    const shellAction = page.locator("#assistant-shell-action");
    const launcher = page.locator(".assistant-launcher");
    await expect(shellAction).toBeVisible();
    await expect(launcher).toHaveCount(0);
    await expect(page.locator("#table-scroll")).toBeHidden();
    await expect(page.locator("#ecn-card-list .ecn-result-card")).toHaveCount(1);
    const overlaps = await page.evaluate(() => {
      const entry = document.querySelector("#assistant-shell-action").getBoundingClientRect();
      const action = document.querySelector("#apply-filters").getBoundingClientRect();
      return entry.left < action.right && entry.right > action.left && entry.top < action.bottom && entry.bottom > action.top;
    });
    expect(overlaps).toBe(false);
  }

  const cardSelection = page.locator("#ecn-card-list .summary-selection").first();
  await cardSelection.check();
  await page.setViewportSize({ width: 1024, height: 844 });
  const tableSelection = page.locator("#table-scroll .summary-selection").first();
  await expect(tableSelection).toBeChecked();
  await tableSelection.uncheck();
  await page.setViewportSize({ width: 390, height: 844 });
  await expect(cardSelection).not.toBeChecked();

  const shellAction = page.locator("#assistant-shell-action");
  await shellAction.click();
  await expect(page.locator("#ecn-assistant-panel")).toHaveAttribute("role", "complementary");
  await expect(page.locator("#ecn-assistant-panel")).not.toHaveAttribute("aria-modal", "true");
  await page.keyboard.press("Escape");
  await expect(shellAction).toBeFocused();
});

test("chat uses the public right-side assistant slot with real button semantics", async ({ page }) => {
  // All API traffic is synthetic; even an unexpected request must not reach a customer service.
  await page.route("**/api/**", route => route.fulfill({ status: 503, json: { detail: "Synthetic fixture: request not supplied" } }));
  await mockList(page);
  let chatRequests = 0;
  await page.route("**/api/chat", route => { chatRequests++; return route.fulfill({ status: 503, json: { detail: "No chat request expected" } }); });
  for (const width of [320, 390, 600, 601, 1024, 1399, 1400, 1512]) {
    await page.setViewportSize({ width, height: 844 });
    await page.goto("/ecns");
    const action = page.locator("#assistant-shell-action");
    const realButton = action.getByRole("button", { name: "Open ECN chat", exact: true });
    const menu = page.getByRole("button", { name: "Toggle navigation", exact: true });
    await expect(menu).toBeVisible();
    await expect(realButton).toBeVisible();
    await expect(page.locator(".assistant-launcher")).toHaveCount(0);
    await expect(realButton).toHaveAttribute("aria-controls", "ecn-assistant-panel");
    await expect(realButton).toHaveAttribute("aria-expanded", "false");
    // assignedSlot is standard platform API, not private UI5 shadow styling/mutation.
    expect(await action.evaluate(node => node.assignedSlot?.name)).toBe("assistant");
    const menuBox = await menu.boundingBox();
    const actionBox = await realButton.boundingBox();
    const brandingBox = await page.locator("ui5-shellbar-branding").boundingBox();
    expect(menuBox.x + menuBox.width).toBeLessThanOrEqual(actionBox.x);
    expect(actionBox.x).toBeGreaterThanOrEqual(brandingBox.x + brandingBox.width);
    expect(actionBox.x + actionBox.width).toBeLessThanOrEqual(width);
    await menu.click();
    await expect(page.locator("#nl1")).toHaveAttribute("mode", "Expanded");
    await menu.click();
    await expect(page.locator("#nl1")).toHaveAttribute("mode", "Collapsed");
    await realButton.focus();
    await page.keyboard.press("Enter");
    await expect(page.locator("#ecn-assistant-panel")).toBeVisible();
    await expect(realButton).toBeVisible();
    await expect(realButton).toHaveAttribute("aria-expanded", "true");
    await expect(realButton).not.toHaveAttribute("aria-haspopup", "dialog");
    await page.keyboard.press("Escape");
    await expect(page.locator("#ecn-assistant-panel")).toBeHidden();
    await expect(realButton).toBeFocused();
    await expect(realButton).toHaveAttribute("aria-expanded", "false");
    expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBe(width);
  }
  expect(chatRequests).toBe(0);
});

test("BOM evidence table scrolls inside a contained card at 390px", async ({ page }) => {
  const rowId = "11111111-1111-4111-8111-111111111111";
  await page.setViewportSize({ width: 390, height: 844 });
  await page.route(`**/api/ecns/${rowId}`, (route) => route.fulfill({ json: detailEnvelope() }));
  await page.route(`**/api/ecns/${rowId}/snapshot`, (route) => route.fulfill({ json: overviewSnapshot(rowId) }));
  await page.goto(`/ecns/${rowId}`);

  const cards = page.locator("#bom-changes");
  await expect(cards.locator(".table-scroll")).toHaveCount(1);
  const dimensions = await cards.evaluateAll((nodes) => nodes.map((card) => {
    const tableScroll = card.querySelector(".table-scroll");
    return {
      cardRight: card.getBoundingClientRect().right,
      tableClientWidth: tableScroll.clientWidth,
      tableScrollWidth: tableScroll.scrollWidth,
      viewportWidth: window.innerWidth,
    };
  }));

  for (const card of dimensions) {
    expect(card.cardRight).toBeLessThanOrEqual(card.viewportWidth);
    expect(card.tableScrollWidth).toBeGreaterThan(card.tableClientWidth);
  }
  expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBe(390);
});

test("long BOM evidence stays within a vertically scrollable card", async ({ page }) => {
  const rowId = "11111111-1111-4111-8111-111111111111";
  await page.setViewportSize({ width: 1280, height: 1000 });
  await page.route(`**/api/ecns/${rowId}`, (route) => route.fulfill({ json: detailEnvelope() }));
  await page.route(`**/api/ecns/${rowId}/snapshot`, (route) => route.fulfill({ json: longBomSnapshot(rowId) }));
  await page.goto(`/ecns/${rowId}`);

  const viewport = page.locator("#bom-changes .table-scroll");
  await expect(viewport).toBeAttached();
  const metrics = await viewport.evaluate((node) => {
    const styles = getComputedStyle(node);
    node.scrollTop = node.scrollHeight;
    return {
      clientHeight: node.clientHeight,
      scrollHeight: node.scrollHeight,
      scrollTop: node.scrollTop,
      overflowY: styles.overflowY,
      headerPosition: getComputedStyle(node.querySelector("th")).position,
    };
  });

  expect(metrics.clientHeight).toBeLessThanOrEqual(512);
  expect(metrics.scrollHeight).toBeGreaterThan(metrics.clientHeight);
  expect(metrics.scrollTop).toBeGreaterThan(0);
  expect(metrics.overflowY).toBe("auto");
  expect(metrics.headerPosition).toBe("sticky");
});

test("manual detail scrolling updates the active section", async ({ page }) => {
  const rowId = "11111111-1111-4111-8111-111111111111";
  await page.route(`**/api/ecns/${rowId}`, (route) => route.fulfill({ json: detailEnvelope() }));
  await page.route(`**/api/ecns/${rowId}/snapshot`, (route) => route.fulfill({ json: overviewSnapshot(rowId) }));
  await page.goto(`/ecns/${rowId}`);
  await expect(page.locator("#bom-changes .table-scroll")).toBeAttached();

  await page.locator("#ecn-page").evaluate((dynamicPage) => {
    const scrollContainer = dynamicPage.shadowRoot.querySelector(".ui5-dynamic-page-scroll-container");
    const titleWrapper = dynamicPage.shadowRoot.querySelector(".ui5-dynamic-page-title-header-wrapper");
    const navigation = document.querySelector(".ecn-section-navigation");
    const bomChanges = document.querySelector("#bom-changes");
    const visibleTop = Math.max(0, titleWrapper.getBoundingClientRect().bottom - scrollContainer.getBoundingClientRect().top)
      + navigation.getBoundingClientRect().height + 16;
    scrollContainer.scrollTop += bomChanges.getBoundingClientRect().top - scrollContainer.getBoundingClientRect().top - visibleTop;
    scrollContainer.dispatchEvent(new Event("scroll"));
  });

  await expect.poll(() => page.getByRole("button", { name: "BOM Change Evidence" }).getAttribute("aria-current")).toBe("true");
});

test("detail renders stored sections and controlled evidence safely", async ({ page }) => {
  const rowId = "11111111-1111-4111-8111-111111111111";
  await page.route(`**/api/ecns/${rowId}`, (route) => route.fulfill({ json: detailEnvelope() }));
  await page.route(`**/api/ecns/${rowId}/snapshot`, (route) => route.fulfill({ json: overviewSnapshot(rowId) }));
  await page.route(
    `**/api/ecns/${rowId}/evidence**`,
    (route) => route.fulfill({ json: businessEvidenceEnvelope("modified_materials") }),
  );
  await page.goto(`/ecns/${rowId}`);

  await expect(page.getByRole("heading", { name: /ECN100/ })).toBeVisible();
  await expect(page.locator("#affected-sites-list")).toContainText("Plant A");
  await expect(page.locator("#overview-impact")).toHaveCount(0);
  await expect(page.locator("#overview-decision")).toHaveCount(0);
  await expect(page.locator("#attention")).toHaveCount(0);
  await page.goto(`/ecns/${rowId}/impact`);
  await expect(page.locator("#source-record")).toBeVisible();
  await page.goto(`/ecns/${rowId}/references`);
  await expect(page.locator("#audit-log-card")).toBeVisible();
  await expect(page.getByText("E-0123456789abcdef")).toHaveCount(0);
});
test("detail omits review status and keeps blocked acceptance and evidence reachable at 390px", async ({ page }) => {
  const rowId = "11111111-1111-4111-8111-111111111111";
  await page.setViewportSize({ width: 390, height: 844 });
  await page.route(`**/api/ecns/${rowId}`, (route) => route.fulfill({ json: detailEnvelope({
    analysis: null,
    job: { job_id: "job-1", state: "FAILED", attempt_count: 1, error_code: "PROVIDER_TIMEOUT", error_message: "Generation timed out" },
  }) }));
  await page.route(`**/api/ecns/${rowId}/snapshot`, (route) => route.fulfill({ json: overviewSnapshot(rowId) }));
  await page.route(`**/api/ecns/${rowId}/acceptance-plan`, (route) => route.fulfill({ json: {
    available: true,
    blocked: true,
    block_reason: "Can't accept: 1 new part has no price.",
  } }));
  await page.route(`**/api/ecns/${rowId}/versions`, (route) => route.fulfill({ json: [] }));
  await page.route(`**/api/ecns/${rowId}/reviews`, (route) => route.fulfill({ json: [] }));

  await page.goto(`/ecns/${rowId}`);
  await expect(page.locator("#source-record")).toBeVisible();
  await expect(page.locator("#decision-readiness")).toHaveCount(0);
  await expect(page.getByRole("heading", { name: "Decision readiness" })).toHaveCount(0);
  await expect(page.getByText("Review status", { exact: true })).toHaveCount(0);
  await expect(page.locator("#accept-ecn-explanation")).toHaveCount(0);
  await expect(page.locator("#bom-changes .bom-acceptance-summary")).toContainText("Can't accept: 1 new part has no price.");
  expect(await page.locator("#accept-ecn-footer").evaluate((node) => node.disabled)).toBe(true);
  await page.getByRole("button", { name: "BOM Change Evidence", exact: true }).click();
  await expect(page.getByRole("button", { name: "BOM Change Evidence" })).toHaveAttribute("aria-current", "true");
  expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBe(390);
});

test("synthetic detail preserves section order, chat overlay, and responsive containment", async ({ page }) => {
  await page.setViewportSize({ width: 1512, height: 982 });
  const rowId = "11111111-1111-4111-8111-111111111111";
  await page.route(`**/api/ecns/${rowId}`, (route) => route.fulfill({ json: detailEnvelope() }));
  await page.route(`**/api/ecns/${rowId}/snapshot`, (route) => route.fulfill({ json: overviewSnapshot(rowId) }));
  await page.route(`**/api/ecns/${rowId}/changes?**`, (route) => route.fulfill({ json: {
    total: 1, items: [{ number: "SIM1000", action: "PHASE_OUT", counterparts: ["SIM100A"],
      qualifiers: [], vc_groups: ["SIM-VC"], pairing_label: "Number-pattern match", usage: { label: "Still used elsewhere" } }],
  } }));
  await page.route(`**/api/ecns/${rowId}/versions`, (route) => route.fulfill({ json: [] }));
  await page.route(`**/api/ecns/${rowId}/reviews`, (route) => route.fulfill({ json: [] }));
  await page.route(`**/api/ecns/${rowId}/objects?**`, (route) => route.fulfill({ json: {
    total: 1,
    items: [{ number: "Q900VV0080463", description: "PLATES_DECALS_SAFET_ACCU_EPL_HAZ_CVT_JB", technical: true, types: ["OEM VC Group"], roles: ["RESULTING"], observed_levels: ["3"], occurrence_count: 1 }],
  } }));
  await page.route(`**/api/ecns/${rowId}/object?**`, (route) => route.fulfill({ json: {
    number: "Q900VV0080463", description: "PLATES_DECALS_SAFET_ACCU_EPL_HAZ_CVT_JB", types: ["OEM VC Group"], coverage: "Located in supplied BOM",
    location_total: 0, reference_total: 0, source_reference_total: 1, locations: [], reference_records: [],
    source_references: [{ role: "RESULTING", path: "$[0].ImplementationPlan[1].ResultingLinks[2].ResultingObjects[0]", revision: "9", state: "APPROVEDFORSERIES" }],
  } }));

  await page.goto(`/ecns/${rowId}?sort=uploaded_at&order=desc&page=1&page_size=25`);
  const sections = [page.locator("#source-record"), page.locator("#affected-sites")];
  const boxes = await Promise.all(sections.map(section => section.boundingBox()));
  for (let index = 1; index < boxes.length; index++) expect(boxes[index].y).toBeGreaterThan(boxes[index - 1].y);
  await expect(page.locator("#overview-impact")).toHaveCount(0);
  const detail = await page.locator(".ecn-detail").boundingBox();
  await expect(page.locator(".ecn-assistant")).toBeHidden();
  await page.getByRole("button", { name: "Open ECN chat" }).click();
  const chat = await page.locator(".ecn-assistant").boundingBox();
  expect(chat.x).toBeLessThan(detail.x + detail.width);
  expect(chat.width).toBe(416);
  expect(chat.x + chat.width).toBe(1496);
  expect(await page.locator(".ecn-assistant").evaluate(node => getComputedStyle(node).position)).toBe("fixed");

  await page.getByRole("button", { name: "Close chat", exact: true }).click();
  const closedDetail = await page.locator(".ecn-detail").boundingBox();
  expect(closedDetail.width).toBe(detail.width);
  await expect(page.locator(".assistant-launcher")).toHaveCount(0);
  await expect(page.locator("#assistant-shell-action")).toBeVisible();
  await page.getByRole("button", { name: "Open ECN chat" }).click();
  await expect(page.locator("#assistant-shell-action")).toBeVisible();
  await expect(page.getByRole("button", { name: "Close chat", exact: true })).toBeVisible();
  await page.goto(`/ecns/${rowId}/impact`);
  await expect(page.locator("#source-record")).toBeVisible();
  await page.goto(`/ecns/${rowId}/references`);
  await page.goto(`/ecns/${rowId}`);
  await page.setViewportSize({ width: 390, height: 844 });
  await expect(page.locator("#ecn-assistant-panel")).toBeHidden();
  const shellAction = page.locator("#assistant-shell-action");
  for (const width of [320, 390, 480]) {
    await page.setViewportSize({ width, height: 844 });
    await expect(shellAction).toBeVisible();
    await expect(page.locator(".assistant-launcher")).toHaveCount(0);
    const overlapsAccept = await page.evaluate(() => {
      const entry = document.querySelector("#assistant-shell-action").getBoundingClientRect();
      const action = document.querySelector("#accept-ecn-footer").getBoundingClientRect();
      return entry.left < action.right && entry.right > action.left && entry.top < action.bottom && entry.bottom > action.top;
    });
    expect(overlapsAccept).toBe(false);
  }
  await page.setViewportSize({ width: 390, height: 844 });
  await shellAction.click();
  await expect(page.locator("#ecn-assistant-panel")).toHaveAttribute("role", "complementary");
  await page.keyboard.press("Escape");
  await expect(shellAction).toBeFocused();
  expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBe(390);
});

test("Settings requires an idle state and exact reset confirmation", async ({ page }) => {
  let resetBody = null;
  await page.route("**/api/analysis-progress**", (route) => route.fulfill({ json: {
    upload_id: null,
    queued_count: 0,
    blocked_count: 0,
    running_count: 0,
    completed_count: 2,
    completed_with_gaps_count: 0,
    failed_count: 0,
    unfinished_count: 0,
    total_count: 2,
  } }));
  await page.route("**/api/history", async (route) => {
    resetBody = route.request().postDataJSON();
    await route.fulfill({ json: {
      deleted_attempt_count: 2,
      deleted_analysis_count: 2,
      deleted_job_count: 2,
      deleted_ecn_count: 2,
      deleted_upload_count: 1,
      deleted_bom_row_count: 99,
      deleted_source_artifact_count: 2,
    } });
  });
  await page.route("**/api/ecns**", (route) => route.fulfill({ json: {
    items: [], page: 1, page_size: 25, total_items: 0, total_pages: 0,
  } }));
  await page.route("**/api/provider-status", (route) => route.fulfill({ json: providerStatusEnvelope() }));
  await page.route("**/api/reference-datasets", (route) => route.fulfill({ json: referenceDatasetsEnvelope() }));

  await page.goto("/settings");
  await page.locator("#history-reset-disclosure").click();
  const confirmation = page.getByLabel(/Type RESET HISTORY/);
  const reset = page.getByRole("button", { name: "Reset analysis history" });
  await confirmation.fill("reset history");
  await expect(reset).toBeDisabled();
  await confirmation.fill("RESET HISTORY");
  await expect(reset).toBeEnabled();
  await reset.click();
  await expect(page).toHaveURL(/\/ecns$/);
  await expect(page.getByText(/History reset:/)).toBeVisible();
  await expect(page.getByText("No ECNs have been uploaded.")).toBeVisible();
  expect(resetBody).toEqual({ confirmation: "RESET HISTORY" });
});
