import { expect, test } from "@playwright/test";
import { installApiMock } from "./api-mock.js";

/** Capture the inbox geometry and styles affected by other routes' CSS. */
async function inboxLayout(page) {
  await expect(page.locator("#table-scroll tbody tr")).toHaveCount(4);
  await expect(page).toHaveTitle(/^ECNs \|/);
  return page.locator("#table-scroll").evaluate((host) => {
    const table = host.querySelector("table");
    const header = table.querySelector("th");
    const cell = table.querySelector("td");
    const surface = host.closest(".surface");
    const properties = (node, names) => Object.fromEntries(names.map((name) => [name, getComputedStyle(node)[name]]));
    return {
      table: properties(table, ["tableLayout", "width"]),
      header: properties(header, ["padding", "whiteSpace", "backgroundColor", "fontSize", "verticalAlign"]),
      cell: properties(cell, ["padding", "overflowWrap", "fontSize", "verticalAlign"]),
      surface: properties(surface, ["padding", "marginBottom", "fontWeight"]),
      pagination: properties(document.querySelector('.pagination[aria-label="ECN pages"]'), ["marginTop", "fontSize"]),
      columns: [...table.querySelectorAll("thead th")].map((node) => Math.round(node.getBoundingClientRect().width * 100) / 100),
    };
  });
}

for (const width of [1280, 1920]) {
  test(`inbox columns stay stable after ECN and Settings navigation at ${width}px`, async ({ page }) => {
    await page.setViewportSize({ width, height: 1000 });
    await installApiMock(page, { seeded: true, completedImmediately: true });
    await page.goto("/ecns");
    const initial = await inboxLayout(page);
    expect(initial.table.tableLayout).toBe("fixed");
    await page.evaluate(() => { window.layoutNavigationMarker = true; });

    for (const returnVia of ["back link", "browser back", "side navigation"]) {
      await test.step(`return through ${returnVia}`, async () => {
        await page.locator("#table-scroll .ecn-link").first().click();
        await expect(page.locator("#source-record")).toBeVisible();
        await expect(page).toHaveTitle(/^ECN Overview/);
        if (returnVia === "back link") await page.locator("#back-to-list").click();
        else if (returnVia === "browser back") await page.goBack();
        else {
          // UI5 temporarily suppresses item activation during the shell animation.
          await expect(page.locator('#sn1 [side-nav-animating]')).toHaveCount(0);
          await page.getByRole("menuitemradio", { name: "ECNs", exact: true }).click();
        }
        expect(await inboxLayout(page)).toEqual(initial);
        expect(await page.evaluate(() => window.layoutNavigationMarker)).toBe(true);
      });
    }

    await expect(page.locator('#sn1 [side-nav-animating]')).toHaveCount(0);
    await page.getByRole("menuitemradio", { name: "Settings", exact: true }).click();
    await expect(page.locator("#settings-title")).toBeVisible();
    await expect(page).toHaveTitle(/^Settings/);
    await expect(page.locator('#sn1 [side-nav-animating]')).toHaveCount(0);
    await page.getByRole("menuitemradio", { name: "ECNs", exact: true }).click();
    expect(await inboxLayout(page)).toEqual(initial);
  });
}

test("opening an ECN directly and then returning uses the initial inbox layout", async ({ page }) => {
  await page.setViewportSize({ width: 1920, height: 1000 });
  await installApiMock(page, { seeded: true, completedImmediately: true });
  await page.goto("/ecns");
  const initial = await inboxLayout(page);
  const detailHref = await page.locator("#table-scroll .ecn-link").first().getAttribute("href");
  // A fresh document reverses the order in which the two page stylesheets load.
  await page.goto(detailHref);
  await expect(page.locator("#source-record")).toBeVisible();
  await page.locator("#back-to-list").click();
  expect(await inboxLayout(page)).toEqual(initial);
});
