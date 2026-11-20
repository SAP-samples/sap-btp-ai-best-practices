import { expect, test } from "@playwright/test";
import { installApiMock } from "./api-mock.js";
import { rowOne } from "../fixtures/api-responses.js";

const rowId = rowOne.ecn_row_id;
const snapshot = {
  assessment_id: rowId, version: "ecn-snapshot-v2", evidence_hash: "a".repeat(64),
  counts: { technical_objects: 1, supporting_objects: 0, bom_change_objects: 1, located_objects: 1, unlocated_objects: 0 },
  questions: [], limitations: [], part_descriptions: [{ number: "SIM-PIPE", description: "Synthetic tube" }],
  plant_impact: { plants: [] },
  bom_changes: { materials: [], lines: [{ qualifiers: [], levels: ["EBOM"], plants: [],
    incoming: { number: "SIM-PIPE", description: "Synthetic tube", date: "2030-06-14" } }] },
};

async function installDetail(page) {
  const api = await installApiMock(page, { seeded: true, completedImmediately: true });
  await page.route(`**/api/ecns/${rowId}/snapshot`, route => route.fulfill({ json: snapshot }));
  return api;
}

async function expectFooterContained(page) {
  const measurements = await page.locator(".decision-footer").evaluate(footer => {
    const bounds = footer.getBoundingClientRect();
    const failures = [];
    for (const selector of ["#decision-bar", "#acceptance-rules-open", "#send-back-footer", "#accept-ecn-footer", "#reviewer-name-footer"]) {
      const element = footer.querySelector(selector);
      if (element.hidden) continue;
      const r = element.getBoundingClientRect();
      if (r.left < bounds.left - 1 || r.right > bounds.right + 1 || r.top < bounds.top - 1 || r.bottom > bounds.bottom + 1 || r.bottom > innerHeight + 1) failures.push(selector);
    }
    return { failures, rootWidth: document.documentElement.scrollWidth, viewport: innerWidth };
  });
  expect(measurements.failures).toEqual([]);
  expect(measurements.rootWidth).toBe(measurements.viewport);
}

for (const viewport of [{ width: 1280, height: 720 }, { width: 390, height: 844 }]) {
  test(`acceptance rules stay above navigation and footer actions fit at ${viewport.width}x${viewport.height}`, async ({ page }) => {
    await page.setViewportSize(viewport);
    const api = await installDetail(page);
    let blocked = true;
    const reason = "Can't accept: 2 new parts have no price. " + "Synthetic source evidence must be reviewed. ".repeat(6);
    await page.route(`**/api/ecns/${rowId}/acceptance-plan`, route => route.fulfill({ json: { available: true, blocked, block_reason: blocked ? reason : null } }));
    const errors = []; page.on("pageerror", error => errors.push(error.message));
    await page.goto(`/ecns/${rowId}`);
    await expect(page.locator("#bom-changes .bom-acceptance-summary")).toHaveText(reason);
    await expect(page.locator("#accept-ecn-explanation")).toHaveCount(0);
    await expect.poll(() => page.locator("#accept-ecn-footer").evaluate(button => button.disabled)).toBe(true);
    await expect(page.locator("#decision-readiness")).toHaveCount(0);
    await expect(page.getByRole("textbox", { name: "Reviewer name", exact: true })).toBeVisible();
    await expectFooterContained(page);
    const trigger = page.getByRole("button", { name: "Acceptance rules", exact: true });
    await trigger.focus(); await page.keyboard.press("Enter");
    const dialog = page.getByRole("dialog", { name: "Acceptance rules", exact: true });
    await expect(dialog).toBeVisible();
    await expect(trigger).toHaveAttribute("aria-expanded", "true");
    const region = page.getByRole("region", { name: "Acceptance rules explanation", exact: true });
    await region.focus(); await expect(region).toBeFocused();
    const hit = await region.evaluate(element => {
      const bounds = element.getBoundingClientRect();
      const x = bounds.left + bounds.width / 2, y = bounds.top + 12;
      let target = document.elementFromPoint(x, y);
      while (target?.shadowRoot) {
        const child = target.shadowRoot.elementFromPoint(x, y);
        if (!child || child === target) break;
        target = child;
      }
      element.scrollTop = element.scrollHeight;
      return { reachable: element.contains(target), bounded: bounds.left >= 0 && bounds.right <= innerWidth && bounds.top >= 0 && bounds.bottom <= innerHeight,
        scrolled: element.scrollHeight <= element.clientHeight || element.scrollTop > 0 };
    });
    expect(hit).toEqual({ reachable: true, bounded: true, scrolled: true });
    await page.keyboard.press("Escape"); await expect(dialog).not.toBeVisible();
    await expect(trigger).toBeFocused(); await expect(trigger).toHaveAttribute("aria-expanded", "false");
    await trigger.click();
    // The public footer is slotted outside the internal shadow-root dialog node.
    await page.locator("#acceptance-rules-close").getByRole("button", { name: "Close", exact: true }).click();
    await expect(dialog).not.toBeVisible(); await expect(trigger).toBeFocused();
    await page.getByRole("button", { name: "BOM Change Evidence", exact: true }).click();
    await expect(page.getByRole("button", { name: "BOM Change Evidence", exact: true })).toHaveAttribute("aria-current", "true");
    blocked = false; await page.reload();
    await expect.poll(() => page.locator("#accept-ecn-footer").evaluate(button => button.disabled)).toBe(false);
    await page.locator("#accept-ecn-footer").click(); // Focus the empty name field; do not confirm or save.
    await expect(page.getByRole("textbox", { name: "Reviewer name", exact: true })).toBeVisible();
    await expectFooterContained(page);
    // Exercise SPA route departure while the native top-layer dialog is open.
    await page.getByRole("button", { name: "Acceptance rules", exact: true }).click();
    await expect(dialog).toBeVisible();
    await page.evaluate(() => {
      const state = { path: "/ecns" };
      history.pushState(state, "", state.path);
      dispatchEvent(new PopStateEvent("popstate", { state }));
    });
    await expect(page.getByRole("heading", { name: "ECN Inbox", exact: true })).toBeVisible();
    await expect(page.locator("#acceptance-rules")).toHaveCount(0);
    expect(api.requests.filter(request => request.method !== "GET")).toEqual([]);
    expect(errors).toEqual([]);
  });
}

test("local drawing preview has no upload and Escape revokes its URL without closing the material", async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  const api = await installDetail(page);
  await page.route(`**/api/ecns/${rowId}/object?**`, route => route.fulfill({ json: {
    number: "SIM-PIPE", description: "Synthetic tube", types: [], material_brief: {}, manufacturing_feasibility: null,
  } }));
  await page.addInitScript(() => {
    window.previewUrls = { created: [], revoked: [] };
    const create = URL.createObjectURL.bind(URL), revoke = URL.revokeObjectURL.bind(URL);
    URL.createObjectURL = file => { const value = create(file); window.previewUrls.created.push(value); return value; };
    URL.revokeObjectURL = value => { window.previewUrls.revoked.push(value); revoke(value); };
  });
  await page.goto(`/ecns/${rowId}`);
  await page.locator("#bom-changes .bom-part-link", { hasText: "SIM-PIPE" }).click();
  const dialog = page.locator("dialog.object-dialog"); await expect(dialog).toBeVisible();
  const preview = dialog.locator(".engineering-drawing-preview");
  await expect(preview).not.toContainText("No viewable drawing");
  await expect(preview).not.toContainText("Local preview");
  await expect(dialog.locator('input[type="file"]')).toHaveCount(1);
  await preview.getByLabel("Choose drawing", { exact: true }).setInputFiles({
    name: "synthetic.png", mimeType: "image/png",
    buffer: Buffer.from("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aG1cAAAAASUVORK5CYII=", "base64"),
  });
  expect(await page.evaluate(() => window.previewUrls.created)).toEqual([]);
  expect(await preview.locator('input[type="file"]').evaluate(input => input.files.length)).toBe(1);
  await preview.getByRole("button", { name: "View engineering drawing", exact: true }).click();
  await expect(preview.locator("img")).toBeVisible();
  await expect(preview.locator("img")).toHaveAttribute("src", /^blob:/);
  await page.keyboard.press("Escape");
  await expect(dialog).toBeVisible(); await expect(preview.locator(".drawing-preview-viewer")).not.toBeVisible();
  await expect(preview.getByRole("button", { name: "View engineering drawing", exact: true })).toBeFocused();
  const urls = await page.evaluate(() => window.previewUrls);
  expect(urls.created).toHaveLength(1); expect(urls.revoked).toEqual(urls.created);
  expect(api.requests.filter(request => request.method !== "GET")).toEqual([]);
  expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBe(390);
  await dialog.locator(".object-dialog-footer").getByRole("button", { name: "Close", exact: true }).click();
});
