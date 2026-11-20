import { expect, test } from "@playwright/test";
import { installApiMock } from "./api-mock.js";
import { rowOne } from "../fixtures/api-responses.js";

const id = rowOne.ecn_row_id;
const object = number => ({ number, description: `${number} description`, material_brief: {},
  change: { action: "PHASE_IN", counterparts: [number === "PART-A" ? "PART-B" : "PART-A"] } });
async function mount(page) {
  await installApiMock(page, { seeded: true, completedImmediately: true });
  await page.route(`**/api/ecns/${id}/snapshot`, route => route.fulfill({ json: {
    version: "ecn-snapshot-v2", counts: {}, questions: [], plant_impact: { plants: [] },
    bom_changes: { lines: ["PART-A", "PART-B"].map(number => ({ assembly: "A1", plants: [], incoming: { number }, qualifiers: [] })) },
  } }));
  await page.goto(`/ecns/${id}`);
  await page.getByRole("button", { name: "BOM Change Evidence", exact: true }).click();
}

for (const width of [1280, 390]) {
  test(`slow reads open immediately and closing prevents late reopening at ${width}px`, async ({ page }, testInfo) => {
    await page.setViewportSize({ width, height: 844 });
    let release;
    const held = new Promise(resolve => { release = resolve; });
    let finishHeld;
    const finished = new Promise(resolve => { finishHeld = resolve; });
    let requests = 0;
    await mount(page);
    await page.route(`**/api/ecns/${id}/object?**`, async route => {
      const first = ++requests === 1;
      if (first) await held;
      // Fulfill can race a real fetch abort after the overlay closes.
      await route.fulfill({ json: object("PART-A") }).catch(() => {});
      if (first) finishHeld();
    });
    await page.evaluate(() => {
      window.materialOpenTiming = new Promise(resolve => {
        document.addEventListener("click", function observe(event) {
          if (event.target.textContent !== "PART-A") return;
          document.removeEventListener("click", observe, true);
          const start = performance.now();
          requestAnimationFrame(() => resolve({ elapsedMs: performance.now() - start,
            open: document.querySelector("dialog.object-dialog").open }));
        }, true);
      });
    });
    const opener = page.locator("#bom-changes").getByRole("button", { name: "PART-A", exact: true });
    await opener.click();
    const dialog = page.locator("dialog.object-dialog");
    await expect(dialog).toBeVisible();
    await expect(dialog.locator('[role="status"]')).toHaveText("Loading material details…");
    await expect(dialog.locator(".material-identifier")).toHaveText("Material: PART-A");
    await expect(dialog.locator("table, input")).toHaveCount(0);
    const timing = await page.evaluate(() => window.materialOpenTiming);
    expect(timing.open).toBe(true); expect(timing.elapsedMs).toBeLessThan(250);
    await testInfo.attach("click-to-loading-overlay", { body: JSON.stringify(timing), contentType: "application/json" });
    await expect(dialog.locator("h2")).toBeFocused();
    expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBe(width);
    await page.keyboard.press("Tab");
    await expect(dialog.getByRole("button", { name: "Close", exact: true })).toBeFocused();
    await page.keyboard.press("Escape");
    await expect(dialog).not.toBeVisible(); await expect(opener).toBeFocused();
    release();
    await finished;
    // Give the browser a rendering turn to handle any late network delivery.
    await page.evaluate(() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve))));
    await expect(dialog).not.toBeVisible();
    // Barrier: a fresh request succeeds after the held one is released.
    await opener.click();
    await expect(dialog.locator(".engineering-drawing-preview")).toBeVisible();
    await expect(dialog.locator("h2")).toHaveText("PART-A description");
    await dialog.getByRole("button", { name: "Close", exact: true }).click();
    await expect(dialog).not.toBeVisible(); await expect(opener).toBeFocused();
  });
}

test("a failed read retries in the overlay and counterpart navigation preserves the original opener", async ({ page }) => {
  await mount(page);
  let first = true;
  await page.route(`**/api/ecns/${id}/object?**`, route => {
    if (first) {
      first = false;
      return route.fulfill({ status: 503, json: { code: "TEMPORARY", message: "Material service unavailable" } });
    }
    const number = new URL(route.request().url()).searchParams.get("number");
    return route.fulfill({ json: object(number) });
  });
  const opener = page.locator("#bom-changes").getByRole("button", { name: "PART-A", exact: true });
  await opener.click();
  const dialog = page.locator("dialog.object-dialog");
  await expect(dialog.locator('[role="alert"]')).toContainText("Material service unavailable");
  await expect(dialog.locator("table, input")).toHaveCount(0);
  await dialog.getByRole("button", { name: "Retry", exact: true }).click();
  await expect(dialog.locator(".engineering-drawing-preview")).toBeVisible();
  await dialog.getByRole("button", { name: "PART-B", exact: true }).click();
  await expect(dialog.locator(".material-identifier")).toHaveText("Material: PART-B");
  await expect(dialog.locator("h2")).toHaveText("PART-B description");
  await page.keyboard.press("Escape");
  await expect(dialog).not.toBeVisible(); await expect(opener).toBeFocused();
});

test("completing a slow read keeps keyboard focus on Close", async ({ page }) => {
  await mount(page);
  let release;
  const held = new Promise(resolve => { release = resolve; });
  await page.route(`**/api/ecns/${id}/object?**`, async route => {
    await held; await route.fulfill({ json: object("PART-A") });
  });
  const opener = page.locator("#bom-changes").getByRole("button", { name: "PART-A", exact: true });
  await opener.click();
  const dialog = page.locator("dialog.object-dialog");
  await expect(dialog.locator('[role="status"]')).toHaveText("Loading material details…");
  const close = dialog.getByRole("button", { name: "Close", exact: true });
  await close.focus(); release();
  await expect(dialog.locator("h2")).toHaveText("PART-A description");
  await expect(close).toBeFocused();
  await page.keyboard.press("Enter"); await expect(opener).toBeFocused();
});
