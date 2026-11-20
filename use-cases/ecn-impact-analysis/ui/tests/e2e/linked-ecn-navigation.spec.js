import { expect, test } from "@playwright/test";
import { installApiMock } from "./api-mock.js";
import { detailEnvelope } from "../fixtures/api-responses.js";

const original = "11111111-1111-4111-8111-111111111111";
const followup = "66666666-6666-4666-8666-666666666666";

test("a follow-up opens its original ECN and keeps missing links disabled", async ({ page }) => {
  const api = await installApiMock(page, { seeded: true, completedImmediately: true });
  await page.route(/\/api\/ecns\/[^/?]+$/, async (route) => {
    const rowId = new URL(route.request().url()).pathname.split("/").pop();
    await route.fulfill({ json: detailEnvelope({
      ecn_row_id: rowId,
      ecn_number: rowId === original ? "ECN12345" : "ECN67890",
      display_ecn_number: rowId === original ? "ECN12345" : "ECN67890",
      description_text: rowId === original ? "Original hydraulic update." : "Linked to ecn 12345 and ECN-888.",
      linked_ecns: rowId === original ? [] : [
        { ecn_number: "ECN12345", ecn_row_id: original },
        { ecn_number: "ECN888", ecn_row_id: null },
      ],
    }) });
  });
  await page.goto(`/ecns/${followup}?query=shaft&sort=warning_count&page=2&page_size=25`);
  await expect(page.locator("#detail-title")).toHaveText("ECN67890");
  const link = page.getByRole("link", { name: "Open linked ECN ECN12345", exact: true });
  await expect(link).toBeVisible();
  const unavailable = page.getByRole("button", { name: "ECN888 unavailable", exact: true });
  await expect(unavailable).toBeDisabled();
  const before = page.url();
  await unavailable.evaluate((button) => button.click());
  await expect(page).toHaveURL(before);
  await link.focus();
  await page.keyboard.press("Enter");
  await expect(page).toHaveURL(new RegExp(`/ecns/${original}\\?`));
  await expect(page.locator("#detail-title")).toHaveText("ECN12345");
  const disabled = page.getByRole("button", { name: "No linked ECN", exact: true });
  await expect(disabled).toBeDisabled();
  await expect(page.locator("#linked-ecn-field a")).toHaveCount(0);
  await expect.poll(() => disabled.evaluate((button) => Number(getComputedStyle(button).opacity))).toBeLessThan(1);
  await page.goBack();
  await expect(page.locator("#detail-title")).toHaveText("ECN67890");
  await expect(link).toBeVisible();
  await page.locator("#back-to-list").click();
  await expect(page.locator("#table-scroll")).toBeVisible();
  const query = new URL(page.url()).searchParams;
  expect(query.get("query")).toBe("shaft");
  expect(query.get("page")).toBe("2");
  expect(query.get("sort")).toBe("warning_count");
  expect(api.requests.filter((request) => request.method !== "GET")).toHaveLength(0);
});
