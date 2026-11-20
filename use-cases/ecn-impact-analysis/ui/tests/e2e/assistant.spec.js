import { expect, test } from "@playwright/test";

async function mockInbox(page) {
  // No real API traffic or provider calls in these interaction checks.
  await page.route("**/api/**", route => route.fulfill({ status: 503, json: { detail: "Synthetic fixture only" } }));
  await page.route("**/api/ecns?**", route => route.fulfill({ json: { items: [], page: 1, page_size: 25, total_items: 0, total_pages: 1 } }));
  await page.route("**/api/analysis-progress**", route => route.fulfill({ json: {
    queued_count: 0, blocked_count: 0, running_count: 0, completed_count: 0,
    completed_with_gaps_count: 0, failed_count: 0, unfinished_count: 0, total_count: 0,
  } }));
}

test("minimal assistant preserves sending, errors, sources, and SAP theme colours", async ({ page }) => {
  await mockInbox(page);
  await page.setViewportSize({ width: 1200, height: 762 });
  let requestCount = 0;
  let release;
  let fail = false;
  await page.route("**/api/chat", async route => {
    requestCount++;
    if (fail) return route.fulfill({ status: 503, json: { message: "Synthetic temporary failure" } });
    expect(route.request().postDataJSON()).toEqual({ message: "Which ECNs need review?", history: [] });
    await new Promise(resolve => { release = resolve; });
    await route.fulfill({ json: {
      answer: "Review **ECN100** first. <script>alert('unsafe')</script>",
      consulted_ecns: [{ ecn_number: "ECN100", assessment_id: "fixture-100" }],
      tool_trace: [{ sequence: 1, tool: "internal_tool", arguments: {} }],
    } });
  });
  await page.goto("/ecns");
  const opener = page.getByRole("button", { name: "Open ECN chat", exact: true });
  const panel = page.locator("#ecn-assistant-panel");
  const input = panel.getByRole("textbox");
  const send = panel.getByRole("button", { name: "Send", exact: true });
  const pageWidth = await page.locator("ui5-navigation-layout").evaluate(e => e.getBoundingClientRect().width);
  await expect(panel).toBeHidden();
  await opener.click();
  await expect(input).toBeFocused();
  await expect(send).toBeDisabled();
  expect(requestCount).toBe(0);
  await expect(panel.getByRole("button")).toHaveCount(2); // close and send
  await expect(panel).not.toHaveAttribute("aria-modal", "true");
  expect(await page.locator("ui5-navigation-layout").evaluate(e => e.getBoundingClientRect().width)).toBe(pageWidth);
  const style = await panel.evaluate(e => {
    const s = getComputedStyle(e);
    return { borderRadius: s.borderRadius, focus: s.getPropertyValue("--chat-focus").trim(), primary: s.getPropertyValue("--chat-primary").trim() };
  });
  expect(style).toEqual({ borderRadius: "30px", focus: "#0032a5", primary: "#0070f2" });
  expect((await panel.locator(".chat-composer").boundingBox()).height).toBe(108);
  await input.fill("Which ECNs need review?");
  await expect(send).toBeEnabled();
  await input.press("Enter");
  await expect(input).toBeDisabled();
  await expect(panel.locator(".chat-pending")).toHaveText("Checking stored ECNs…");
  await expect(panel.locator(".chat-user")).toHaveText("Which ECNs need review?");
  await expect(panel.locator(".chat-welcome")).toBeHidden();
  // A response that arrives while closed must preserve focus on the launcher.
  await panel.getByRole("button", { name: "Close chat", exact: true }).click();
  await expect(opener).toBeFocused();
  release();
  await expect(panel.locator("textarea")).toBeEnabled();
  await expect(opener).toBeFocused();
  await opener.click();
  await expect(panel.locator(".chat-assistant strong")).toHaveText("ECN100");
  await expect(panel.getByRole("link", { name: "ECN100" })).toHaveAttribute("href", "/ecns/fixture-100");
  await expect(panel.locator("script, .chat-tool-trace")).toHaveCount(0);
  await expect(input).toHaveValue("");
  await expect(send).toBeDisabled();
  expect(await panel.locator(".chat-user").evaluate(e => getComputedStyle(e).backgroundColor)).toBe("rgb(237, 246, 255)");
  expect(await panel.locator(".chat-assistant").evaluate(e => getComputedStyle(e).backgroundColor)).toBe("rgb(239, 241, 242)");
  fail = true;
  await input.fill("Retry later");
  await send.click();
  await expect(panel.getByRole("status")).toContainText("Synthetic temporary failure");
  await expect(input).toHaveValue("Retry later");
  await expect(send).toBeEnabled();
  expect(requestCount).toBe(2);
  await input.press("Shift+Enter");
  await expect(input).toHaveValue("Retry later\n");
  expect(requestCount).toBe(2);
  await page.getByRole("heading", { name: "ECN Inbox", exact: true }).click();
  await expect(panel).toBeVisible();
  await input.focus();
  await input.press("Escape");
  await expect(panel).toBeHidden();
  await expect(opener).toBeFocused();
});

test("assistant remains contained on narrow and short screens", async ({ page }) => {
  await mockInbox(page);
  await page.goto("/ecns");
  await page.getByRole("button", { name: "Open ECN chat", exact: true }).click();
  for (const viewport of [{ width: 320, height: 568 }, { width: 390, height: 844 }, { width: 1024, height: 500 }]) {
    await page.setViewportSize(viewport);
    const panel = page.locator("#ecn-assistant-panel");
    const bounds = await panel.boundingBox();
    expect(bounds.x).toBeGreaterThanOrEqual(0);
    expect(bounds.x + bounds.width).toBeLessThanOrEqual(viewport.width);
    expect(bounds.y).toBeGreaterThanOrEqual(52);
    expect(bounds.y + bounds.height).toBeLessThanOrEqual(viewport.height);
    for (const control of [panel.getByRole("textbox"), panel.getByRole("button", { name: "Close chat" }), panel.getByRole("button", { name: "Send" })]) {
      const b = await control.boundingBox();
      expect(b.x + b.width).toBeLessThanOrEqual(bounds.x + bounds.width);
      expect(b.y + b.height).toBeLessThanOrEqual(bounds.y + bounds.height);
    }
    expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBe(viewport.width);
  }
});
