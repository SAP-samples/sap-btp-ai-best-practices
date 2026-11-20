import { expect, test } from "@playwright/test";
import { installApiMock } from "./api-mock.js";
import { rowOne } from "../fixtures/api-responses.js";

const longPath = "SYNTHETIC-HIERARCHY-" + "A".repeat(220);
const longReference = "SYNTHETIC-CHANGE-" + "R".repeat(180);
const position = "SYNTHETIC-POSITION-" + "P".repeat(100);
const formula = "SYNTHETIC-LOOKUP-" + "F".repeat(220);
function objectFixture() {
  const occurrence = { source: "Enriched EBOM", kind: "ENRICHED_EBOM", path: [{ number: "SIM-ROOT", description: longPath }, { number: "SIM-CHILD", description: longPath }], item_or_position: position, quantity: "123456789.125", unit: "EA", valid_from_date: "2026-03-13", valid_to_date: "9999-12-31", change_number: longReference };
  return { number: "SIM-PIPE", description: "Synthetic pipe with a deliberately detailed source-backed material description for coverage layout", types: [], material_brief: {
    bom_coverage: { engineering_occurrences: 12, manufacturing_occurrences: 0, coverage: "ENGINEERING_ONLY", effective_interval_count: 1, plants: ["SIM"], occurrences: Array.from({ length: 12 }, () => ({ ...occurrence })) },
    sourcing: [{ plant_code: "SIM", plant_name: "Synthetic facility " + "N".repeat(160), procurement_type_labels: ["External procurement"], source_lead_time: { status: "SOURCE_LEAD_TIME_AVAILABLE", display: "30 days" }, source_purchase_price: { status: "SOURCE_PRICE_AVAILABLE", net_price: "12.50", currency: "EUR", price_per: "1", order_price_unit: "EA" } }],
  }, manufacturing_feasibility: { overall_result: "Feasible against the assessed PS4 requirements", candidate_reports: [{ id: "synthetic", label: "Synthetic primary", completeness: "complete", result: "Feasible against the assessed PS4 requirements", limitations: [], criteria: [
    { id: "raw_material", title: "Raw material", status: "Pass", calculator: { raw_material: { designation: longPath, guidelineCoverage: "listed" } } },
    { id: "tube_length", title: "Tube length", status: "Pass", calculator: { tube_length: { diameterMm: "10", wallThicknessMm: "1.00", totalLengthMm: "629.6", minimumMm: "200", maximumMm: "1500", status: "Pass" } } },
    { id: "bend_angles", title: "Bend angles", status: "Pass", calculator: { bend_angles: { results: [{ angleDeg: "90", status: "Pass" }] } } },
    { id: "automatic_crimp_dimensions", title: "Automatic crimp", status: "Pass", calculator: { schedule_mapping: { l1: { sourcePoint: position, sourceField: "STRAIGHT PART" } }, auto_crimp: { overallStatus: "Pass", results: { l1: { designValueMm: "1000.00", requiredMinimumMm: "200.00", formulaOrLookup: formula, status: "Pass" } } } } },
    { id: "robot_pickup", title: "Robot pickup", status: "Met", calculator: { robot_pickup: { status: "Met", l1Branch: { designValueMm: "1000", requiredMinimumMm: "200", status: "Pass" } } } },
  ] }] } };
}
function snapshotFixture() {
  return { assessment_id: rowOne.ecn_row_id, version: "ecn-snapshot-v2", evidence_hash: "a".repeat(64), counts: { technical_objects: 1, supporting_objects: 0, bom_change_objects: 1, located_objects: 1, unlocated_objects: 0 }, questions: [], limitations: [], part_descriptions: [{ number: "SIM-PIPE", description: "Synthetic pipe" }], plant_impact: { plants: [] }, bom_changes: { lines: [{ qualifiers: [], levels: ["EBOM"], plants: [], incoming: { number: "SIM-PIPE", description: "Synthetic pipe", date: "2026-03-13" } }], materials: [] } };
}

/** Check text fragments (not just element boxes) so intrinsic flex content cannot
 * silently paint over the neighbouring item/quantity/date/reference cells. */
async function assertContainedText(region) {
  const result = await region.evaluate(node => {
    const failures = [];
    for (const cell of node.querySelectorAll("th, td")) {
      const bounds = cell.getBoundingClientRect();
      const walker = document.createTreeWalker(cell, NodeFilter.SHOW_TEXT);
      while (walker.nextNode()) {
        const text = walker.currentNode;
        if (!text.textContent.trim()) continue;
        const range = document.createRange(); range.selectNodeContents(text);
        for (const rect of range.getClientRects()) {
          if (rect.width && (rect.left < bounds.left - 1 || rect.right > bounds.right + 1 || rect.top < bounds.top - 1 || rect.bottom > bounds.bottom + 1)) failures.push(text.textContent.slice(0, 60));
        }
      }
    }
    const bounds = node.getBoundingClientRect(); const body = node.closest(".object-dialog-body").getBoundingClientRect();
    return { failures, width: node.clientWidth, scrollWidth: node.scrollWidth, bounded: bounds.left >= body.left - 1 && bounds.right <= body.right + 1, layout: getComputedStyle(node.querySelector("table")).tableLayout, overflow: getComputedStyle(node).overflowX };
  });
  expect(result.failures).toEqual([]); expect(result.bounded).toBe(true);
  expect(result.layout).toBe("fixed"); expect(result.overflow).toBe("auto");
  expect(result.width).toBeGreaterThan(0);
}
for (const viewport of [{ width: 1280, height: 720 }, { width: 390, height: 844 }]) {
  test(`material evidence contains long source values at ${viewport.width}x${viewport.height}`, async ({ page }) => {
    await page.setViewportSize(viewport);
    const mock = await installApiMock(page, { seeded: true, completedImmediately: true });
    const object = objectFixture(); const frozenText = JSON.stringify(object); let objectReads = 0;
    await page.route(`**/api/ecns/${rowOne.ecn_row_id}/snapshot`, route => route.fulfill({ json: snapshotFixture() }));
    await page.route(`**/api/ecns/${rowOne.ecn_row_id}/object?**`, route => { objectReads++; return route.fulfill({ json: object }); });
    await page.goto(`/ecns/${rowOne.ecn_row_id}`);
    const opener = page.locator("#bom-changes .bom-part-link", { hasText: "SIM-PIPE" });
    await opener.click(); const dialog = page.locator("dialog.object-dialog"); await expect(dialog).toBeVisible();
    const occurrences = dialog.locator(".observed-occurrences-details");
    const outer = dialog.locator(".drawing-feasibility-calculations");
    const groups = dialog.locator(".drawing-feasibility-calculation-group");
    await expect(groups).toHaveCount(5);
    for (const disclosure of [occurrences, outer]) await expect(disclosure).not.toHaveAttribute("open", "");
    for (let i = 0; i < 5; i++) await expect(groups.nth(i)).not.toHaveAttribute("open", "");
    await occurrences.locator(":scope > summary").focus(); await page.keyboard.press("Enter"); await expect(occurrences).toHaveAttribute("open", "");
    await outer.locator(":scope > summary").focus(); await page.keyboard.press("Space"); await expect(outer).toHaveAttribute("open", "");
    for (let i = 0; i < 5; i++) {
      const group = groups.nth(i); await group.locator(":scope > summary").focus(); await page.keyboard.press("Enter"); await expect(group).toHaveAttribute("open", "");
    }
    const regions = dialog.locator(".material-evidence-table"); await expect(regions).toHaveCount(8);
    for (let i = 0; i < await regions.count(); i++) {
      const region = regions.nth(i); await region.scrollIntoViewIfNeeded(); await assertContainedText(region);
      await region.focus(); await expect(region).toBeFocused(); await expect(region).toHaveAttribute("aria-label", /.+/);
      // Every horizontal track can be reached through keyboard scrolling.
      await page.keyboard.press("ArrowRight");
      await expect.poll(() => region.evaluate(node => node.scrollWidth <= node.clientWidth || node.scrollLeft > 0)).toBe(true);
      const reachable = await region.evaluate(node => {
        const failures = [];
        for (const cell of node.querySelectorAll("tbody tr:first-child td")) {
          // Wide cells need not fit all at once, but both ends of every value's
          // column must be reachable without document-level scrolling.
          node.scrollLeft = 0;
          const bounds = node.getBoundingClientRect();
          node.scrollLeft += cell.getBoundingClientRect().left - bounds.left;
          if (cell.getBoundingClientRect().left < bounds.left - 1 || cell.getBoundingClientRect().left > bounds.right) failures.push(cell.dataset.label);
          node.scrollLeft += cell.getBoundingClientRect().right - bounds.right;
          if (cell.getBoundingClientRect().right > bounds.right + 1 || cell.getBoundingClientRect().right < bounds.left) failures.push(cell.dataset.label);
        }
        return failures;
      });
      expect(reachable).toEqual([]);
    }
    const occurrenceRegion = dialog.locator(".observed-occurrences-table");
    const metrics = await occurrenceRegion.evaluate(node => { node.scrollTop = node.scrollHeight; return { height: node.clientHeight, scrollHeight: node.scrollHeight, scrollTop: node.scrollTop, sticky: getComputedStyle(node.querySelector("th")).position, limit: Math.min(28 * parseFloat(getComputedStyle(document.documentElement).fontSize), innerHeight * .45) }; });
    expect(metrics.height).toBeLessThanOrEqual(metrics.limit + 1); expect(metrics.scrollHeight).toBeGreaterThan(metrics.height); expect(metrics.scrollTop).toBeGreaterThan(0); expect(metrics.sticky).toBe("sticky");
    await expect(occurrenceRegion).toContainText(position); await expect(occurrenceRegion).toContainText(longPath); await expect(occurrenceRegion).toContainText(longReference);
    await expect(occurrenceRegion).toContainText("123456789.125 EA"); await expect(occurrenceRegion).toContainText("13 Mar 2026 – No planned end");
    await expect(dialog.locator(".bom-coverage-table")).toContainText("Enriched EBOM: 12");
    await expect(dialog.locator(".material-procurement-table")).toContainText("30 days");
    await expect(dialog.locator(".material-procurement-table")).toContainText("€12.50 / 1 EA");
    await expect(dialog.locator(".drawing-feasibility-calculation-table").nth(3)).toContainText(formula);
    expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBe(viewport.width);
    expect(objectReads).toBe(1); expect(JSON.stringify(object)).toBe(frozenText);
    expect(mock.requests.filter(request => request.method !== "GET")).toEqual([]);
    await dialog.getByRole("button", { name: "Close", exact: true }).click(); await expect(opener).toBeFocused(); await expect(dialog).not.toBeVisible();
  });
}
