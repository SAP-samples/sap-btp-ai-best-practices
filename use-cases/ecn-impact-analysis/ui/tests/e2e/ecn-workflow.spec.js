import { expect, test } from "@playwright/test";
import { existsSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { installApiMock } from "./api-mock.js";

const fullEcnFile = fileURLToPath(new URL("../../../data/new_16-09/ECN_full.json", import.meta.url));

test("real JSON selection, direct generation, evidence, retry, and reset work end to end", async ({ page }) => {
  test.skip(!existsSync(fullEcnFile), "The optional ECN acceptance fixture is not available in this checkout.");
  test.setTimeout(120_000);
  const api = await installApiMock(page);
  await page.goto("/ecns?query=shaft&sort=warning_count&order=desc&page=1&page_size=25");
  await expect(page.getByText("No ECNs have been uploaded.")).toBeVisible();

  await page.getByRole("button", { name: "Import new ECN" }).click();
  await page.locator("#import-file").setInputFiles(fullEcnFile);
  await expect(page.locator("#selection-summary")).toHaveText(/0 selected of [1-9]\d*/);
  await expect(page.locator("#import-rows tr").first()).toContainText("Resolved");
  await expect(page.locator("#import-rows")).not.toContainText("[object Object]");
  await page.locator("#import-rows input[type=checkbox]").nth(0).check();
  await page.locator("#import-rows input[type=checkbox]").nth(1).check();
  await page.getByRole("button", { name: "Import selected ECNs" }).click();
  await expect(page.getByRole("link", { name: "ECN100" }).first()).toBeVisible();
  expect(api.uploadBodies[0]).toContain('name="request_id"');
  expect(api.uploadBodies[0]).toContain('name="ecn_file"');
  expect(api.uploadBodies[0]).toContain('filename="ECN_full.json"');
  expect(api.uploadBodies[0]).toContain("ECN107240");
  expect(api.uploadBodies[0]).toContain("ECN117440");
  expect(api.uploadBodies[0]).not.toContain("ECN122869");

  await page.getByRole("link", { name: "ECN100" }).first().click();
  await expect(page.locator("#source-record")).toBeVisible();
  await expect(page.locator("#decision-readiness")).toHaveCount(0);
  const detailPath = new URL(page.url()).pathname;
   await page.getByRole("button", { name: "Generate Summary" }).click();
  await expect(page.locator("#detail-generation-status")).toContainText("1 scheduled");
  await page.goto(`${detailPath}/impact`);
  await expect(page.locator("#source-record")).toBeVisible();
  await page.getByRole("button", { name: "Audit Log" }).click();
  await expect(page.locator("#audit-log-card")).toBeVisible();
  await page.goto(`${detailPath}/references`);
  await expect(page.locator("#audit-log-card")).toBeVisible();
  await page.getByRole("link", { name: "Back to ECNs" }).click();
  await expect(page).toHaveURL(/sort=uploaded_at/);
  await expect(page).not.toHaveURL(/query=shaft/);

  await page.locator("#table-scroll .ecn-link", { hasText: /^ECN-FAILED$/ }).click();
   await page.getByRole("button", { name: "Retry Analysis" }).click();
  await expect.poll(() => api.requests.filter(
    (request) => request.method === "POST" && request.path.endsWith("/analysis/retry"),
  ).length).toBe(1);
  await expect(page.locator("#detail-status")).toHaveCount(0);
   await expect(page.getByRole("button", { name: "Retry Analysis" })).toBeHidden();
  api.completeAnalysis("77777777-7777-4777-8777-777777777777");

  api.setResetActive(true);
  await page.goto("/settings");
  await page.locator("#history-reset-disclosure").click();
  await expect(page.getByRole("button", { name: "Reset analysis history" })).toBeDisabled();
  api.setResetActive(false);
  await page.getByLabel(/Type RESET HISTORY/).fill("reset history");
  await expect(page.getByRole("button", { name: "Reset analysis history" })).toBeDisabled();
  await page.getByLabel(/Type RESET HISTORY/).fill("RESET HISTORY");
  await expect(page.getByRole("button", { name: "Reset analysis history" })).toBeEnabled();
  await page.getByRole("button", { name: "Reset analysis history" }).click();
  await expect(page).toHaveURL(/\/ecns$/);
  await expect(page.getByText(/History reset:/)).toBeVisible();
  await expect(page.getByText("No ECNs have been uploaded.")).toBeVisible();
});

test("duplicate and missing business identifiers remain separate routes", async ({ page }) => {
  const api = await installApiMock(page, { seeded: true });
  await page.goto("/ecns");
  const duplicateLinks = page.locator("#table-scroll .ecn-link", { hasText: /^ECN100$/ });
  await expect(duplicateLinks).toHaveCount(2);
  expect(await duplicateLinks.nth(0).getAttribute("href")).not.toBe(
    await duplicateLinks.nth(1).getAttribute("href"),
  );
  await expect(page.locator("#table-scroll .ecn-link", { hasText: /^Missing ECN ID$/ })).toHaveCount(1);
  expect(api.requests.filter((request) => request.method !== "GET")).toHaveLength(0);
});

test("reading a completed detail never issues a generation mutation", async ({ page }) => {
    const api = await installApiMock(page, { seeded: true, completedImmediately: true });
    await page.goto("/ecns/11111111-1111-4111-8111-111111111111");
     await page.getByRole("button", { name: "Audit Log" }).click();
     await expect.poll(() => page.locator("#ecn-page").evaluate((dynamicPage) => (
       dynamicPage.shadowRoot.querySelector(".ui5-dynamic-page-scroll-container").scrollTop
     ))).toBeGreaterThan(0);
     const auditTab = page.getByRole("button", { name: "Audit Log" });
     await expect.poll(() => auditTab.evaluate((tab) => {
       const bounds = tab.getBoundingClientRect();
       return bounds.top >= 0 && bounds.bottom <= window.innerHeight;
     })).toBe(true);
     await page.locator("#ecn-page").evaluate((dynamicPage) => {
       const scroller = dynamicPage.shadowRoot.querySelector(".ui5-dynamic-page-scroll-container");
       scroller.scrollTop = scroller.scrollHeight;
       scroller.dispatchEvent(new Event("scroll"));
     });
     await expect.poll(() => auditTab.evaluate((tab) => {
       const bounds = tab.getBoundingClientRect();
       return bounds.top >= 0 && bounds.bottom <= window.innerHeight;
     })).toBe(true);
     await page.getByRole("button", { name: "Plants" }).click();
     await expect(page.locator("#affected-sites-title")).toBeVisible();
     const plantTab = page.getByRole("button", { name: "Plants" });
     await expect.poll(async () => {
       const [tab, card] = await Promise.all([plantTab.boundingBox(), page.locator("#affected-sites").boundingBox()]);
       return !!tab && !!card && card.y >= tab.y + tab.height;
     }).toBe(true);
     await expect.poll(() => page.locator("#detail-title, #detail-title-snapped").evaluateAll((titles) => (
       titles.some((title) => {
         const bounds = title.getBoundingClientRect();
         return bounds.width > 0 && bounds.height > 0 && getComputedStyle(title).visibility !== "hidden";
       })
     ))).toBe(true);
     await page.locator("#ecn-page").evaluate((dynamicPage) => {
      const scroller = dynamicPage.shadowRoot.querySelector(".ui5-dynamic-page-scroll-container");
      scroller.scrollTop = 0;
      scroller.dispatchEvent(new Event("scroll"));
    });
    await expect.poll(() => page.locator("#detail-title, #detail-title-snapped").evaluateAll((titles) => (
      titles.some((title) => {
        const bounds = title.getBoundingClientRect();
        return bounds.width > 0 && bounds.height > 0 && getComputedStyle(title).visibility !== "hidden";
      })
    ))).toBe(true);
   await expect(page.locator("#audit-log-card")).toBeVisible();
   await page.goto("/ecns/11111111-1111-4111-8111-111111111111/impact");
     await expect(page.locator("#source-record")).toBeVisible();
     await page.getByRole("button", { name: "Audit Log" }).click();
   await expect(page.locator("#audit-log-card")).toBeVisible();
   await page.reload();
     await expect(page.locator("#source-record")).toBeVisible();
     await page.getByRole("button", { name: "Audit Log" }).click();
   await expect(page.locator("#audit-log-card")).toBeVisible();
  expect(api.requests.filter((request) => request.method === "POST")).toHaveLength(0);
});

test("main ECN sections replace duplicate sidebar shortcuts without reloading assessment or clearing chat", async ({ page }) => {
  await page.setViewportSize({ width: 1512, height: 982 });
  const api = await installApiMock(page, { seeded: true, completedImmediately: true });
  const rowId = "11111111-1111-4111-8111-111111111111";
  await page.goto(`/ecns/${rowId}`);
  await expect(page.locator("#detail-title")).toContainText("ECN100");
  await page.getByRole("button", { name: "Toggle navigation", exact: true }).click();
  await page.getByRole("button", { name: "Open ECN chat", exact: true }).click();
  const input = page.getByRole("textbox", { name: "Question about this ECN", exact: true });
  await input.fill("Keep this draft while I inspect the BOM");
  const before = api.requests.filter(request => request.path === `/api/ecns/${rowId}`).length;
  for (const name of ["Impact Analysis", "What Needs Attention", "Reference Documents"]) {
    await expect(page.getByRole("treeitem", { name, exact: true })).toHaveCount(0);
  }
  await page.getByRole("button", { name: "BOM Change Evidence", exact: true }).click();
  await expect(page.locator("#bom-changes")).toBeVisible();
  await page.getByRole("button", { name: "Overview", exact: true }).click();
  await expect(page.locator("#source-record")).toBeVisible();
  await expect(input).toHaveValue("Keep this draft while I inspect the BOM");
  expect(api.requests.filter(request => request.path === `/api/ecns/${rowId}`).length).toBe(before);
});
