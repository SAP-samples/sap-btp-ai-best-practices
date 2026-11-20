import { expect, test } from "@playwright/test";
import { installApiMock } from "./api-mock.js";
import { rowOne } from "../fixtures/api-responses.js";

const id = rowOne.ecn_row_id;
const numbers = ["NEW-PIPE", "NEW-SCREW", "NEW-BUSH", "UNPLACED-PART"];
const snapshot = {
  assessment_id: id, version: "ecn-snapshot-v2", evidence_hash: "a".repeat(64),
  counts: {}, questions: [], limitations: [],
  plant_impact: { plants: [{ plant_code: "PL10", plant_name: "Plant A", materials: numbers.slice(0, 3).map(number => ({ number })) }] },
  bom_changes: { materials: [], lines: numbers.slice(0, 3).map(number => ({
    assembly: "A1", assembly_description: "Hydraulic supply", plants: [{ plant_code: "PL10", plant_name: "Plant A" }],
    incoming: { number, description: `${number} description` }, levels: ["EBOM"], qualifiers: [],
  })) },
};
const plan = {
  available: true, blocked: true, block_reason: "Can't accept: 4 new parts have no price.",
  missing_price: numbers,
  parts: numbers.map(number => ({ number, description: `${number} description`, role: "NEW", price: "MISSING", master_data: "Create and release in PL10 date not known" })),
};

for (const viewport of [{ width: 1280, height: 720 }, { width: 390, height: 844 }]) {
  test(`blocked-material actions open the exact part and required source at ${viewport.width}px`, async ({ page }) => {
    await page.setViewportSize(viewport);
    const api = await installApiMock(page, { seeded: true, completedImmediately: true });
    const objects = [];
    const errors = [];
    page.on("pageerror", error => errors.push(error.message));
    await page.route(`**/api/ecns/${id}/snapshot`, route => route.fulfill({ json: snapshot }));
    await page.route(`**/api/ecns/${id}/acceptance-plan`, route => route.fulfill({ json: plan }));
    await page.route(`**/api/ecns/${id}/object?**`, route => {
      const number = new URL(route.request().url()).searchParams.get("number");
      objects.push(number);
      return route.fulfill({ json: { number, description: `${number} description`, types: [], material_brief: { sourcing: [] } } });
    });
    await page.goto(`/ecns/${id}`);
    await expect(page.locator("#bom-changes .bom-acceptance-summary")).toHaveText(plan.block_reason);
    await expect(page.locator("#accept-ecn-explanation")).toHaveCount(0);
    await expect(page.locator("#review-blocked-materials")).toHaveCount(0);
    await page.getByRole("button", { name: "BOM Change Evidence", exact: true }).click();
    const table = page.locator("#bom-changes");
    await expect(table.locator("thead th")).toHaveText([
      "Material", "Description", "ECN change status", "Parent assembly", "Affected plants", "Blocked status",
    ]);
    await expect(table.locator(".bom-change-row")).toHaveCount(4);
    await expect(table.locator(".bom-blocked-status .semantic-pill")).toHaveText(["Blocked", "Blocked", "Blocked", "Blocked"]);
    await expect(table.locator(".bom-blocked-status small")).toHaveText(Array(4).fill("Missing purchasing entry"));
    await expect(table.locator(".bom-blocked-status button, .bom-blocked-status a")).toHaveCount(0);
    await expect(table.locator(".bom-change-row").first()).toContainText("Plant A (PL10)");
    await expect(table.locator(".bom-assembly-group").last()).toContainText("Materials without BOM placement");
    await expect(table.locator(".bom-change-row").last()).toContainText("BOM placement not supplied");
    await expect(table.locator(".bom-change-row").last()).toContainText("Plant not determined");
    const scroll = table.locator(".table-scroll");
    const geometry = await scroll.evaluate(element => {
      element.scrollLeft = element.scrollWidth;
      return { pageWidth: document.documentElement.scrollWidth, viewport: innerWidth,
        width: element.clientWidth, fullWidth: element.scrollWidth, scrolled: element.scrollLeft };
    });
    expect(geometry.pageWidth).toBe(geometry.viewport);
    if (viewport.width < 600) {
      expect(geometry.fullWidth).toBeGreaterThan(geometry.width);
      expect(geometry.scrolled).toBeGreaterThan(0);
    }
    const open = table.getByRole("button", { name: "NEW-BUSH", exact: true });
    await open.focus();
    await page.keyboard.press("Enter");
    const material = page.locator("dialog.object-dialog");
    await expect(material).toBeVisible();
    await expect(material.locator(".material-identifier")).toHaveText("Material: NEW-BUSH");
    const issue = material.getByRole("region", { name: "Required before ECN acceptance", exact: true });
    // A labelled section is an accessible region; the task is visible immediately.
    await expect(issue).toBeVisible();
    await expect(issue).toContainText("Add a purchase-info-record entry for NEW-BUSH");
    await expect(issue).toContainText("Create and release in PL10");
    await expect(issue).toContainText("Affected plants: Plant A (PL10)");
    await expect(issue).toContainText("Parent assemblies: Hydraulic supply");
    expect(objects).toEqual(["NEW-BUSH"]);
    const bounds = await issue.boundingBox();
    expect(bounds.y).toBeGreaterThanOrEqual(0);
    expect(bounds.y).toBeLessThan(viewport.height);
    await page.keyboard.press("Escape");
    await expect(material).not.toBeVisible();
    await expect(open).toBeFocused();
    await open.click();
    await expect(material).toBeVisible();
    await issue.getByRole("link", { name: "Open Purchase Info Records", exact: true }).click();
    await expect(page).toHaveURL(/\/settings\?source=PURCHASE_INFO_RECORDS$/);
    const source = page.locator('.source-card[data-dataset-type="PURCHASE_INFO_RECORDS"]');
    await expect(source).toBeVisible();
    await expect(source).toBeFocused();
    await expect(page.locator(".acceptance-materials-dialog")).toHaveCount(0);
    await expect(page.locator("dialog.object-dialog")).toHaveCount(0);
    expect(api.requests.filter(request => request.method !== "GET")).toEqual([]);
    expect(errors).toEqual([]);
  });
}
