import { expect, test } from "@playwright/test";
import { readFileSync } from "node:fs";
import { installApiMock } from "./api-mock.js";
import { rowOne } from "../fixtures/api-responses.js";

const rowId = rowOne.ecn_row_id;
const samples = ["QKX5632660", "QKX5525860"];
const imageBytes = Buffer.from("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aG1cAAAAASUVORK5CYII=", "base64");

for (const width of [1280, 390]) {
  test(`one drawing chooser supplies preview and analysis at ${width}px`, async ({ page }) => {
    await page.setViewportSize({ width, height: 844 });
    await installApiMock(page, { seeded: true, completedImmediately: true });
    await page.route(`**/api/ecns/${rowId}/snapshot`, route => route.fulfill({ json: {
      version: "ecn-snapshot-v2", evidence_hash: "a".repeat(64), counts: {}, questions: [], plant_impact: { plants: [] },
      bom_changes: { materials: [], lines: [{ assembly: "A1", plants: [], incoming: { number: "SIM-PIPE" }, qualifiers: [] }] },
    } }));
    await page.route(`**/api/ecns/${rowId}/object?**`, route => route.fulfill({ json: {
      number: "SIM-PIPE", description: "Synthetic pipe", types: [], material_brief: {},
    } }));
    const analyses = [];
    await page.route("**/api/drawing-feasibility/assessments", route => {
      expect(route.request().method()).toBe("POST");
      analyses.push(route.request().postDataBuffer().toString("latin1"));
      return route.fulfill({ json: { candidate_reports: [], overall_result: "Inconclusive" } });
    });
    await page.goto(`/ecns/${rowId}`);
    await page.getByRole("button", { name: "BOM Change Evidence", exact: true }).click();
    await page.getByRole("button", { name: "SIM-PIPE", exact: true }).click();
    const material = page.locator("dialog.object-dialog");
    await expect(material.locator('input[type="file"]')).toHaveCount(1);
    const chooser = material.getByLabel("Choose drawing", { exact: true });
    await chooser.setInputFiles({ name: "selected.png", mimeType: "image/png", buffer: imageBytes });
    const view = material.getByRole("button", { name: "View engineering drawing", exact: true });
    await view.click();
    await expect(material.locator(".drawing-preview-media")).toBeVisible();
    expect(analyses).toEqual([]);
    await page.keyboard.press("Escape");
    await expect(material).toBeVisible();
    await material.getByRole("button", { name: "Analyze Drawing", exact: true }).click();
    await expect(material.locator(".drawing-feasibility-report-host")).toContainText("No drawing configuration report is available");
    expect(analyses).toHaveLength(1);
    expect(analyses[0]).toContain('filename="selected.png"');
    expect(analyses[0].split('name="drawing_file"')).toHaveLength(2);
    await chooser.setInputFiles({ name: "replacement.png", mimeType: "image/png", buffer: imageBytes });
    await expect(material.locator(".drawing-feasibility-report-host")).toBeEmpty();
    await expect(material.locator(".drawing-feasibility-overall-tag")).toHaveAttribute("aria-label", "Overall manufacturing feasibility: Not assessed");
    await expect(material.locator('input[type="file"]')).toHaveCount(1);
    expect(analyses).toHaveLength(1);
    expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBe(width);
  });

  test(`matching sample drawings open without file choosers at ${width}px`, async ({ page }, testInfo) => {
    await page.setViewportSize({ width, height: 844 });
    const api = await installApiMock(page, { seeded: true, completedImmediately: true });
    const drawingReads = [];
    const errors = [];
    const workers = [];
    const workerWarnings = [];
    page.on("pageerror", error => errors.push(error.message));
    page.on("worker", worker => workers.push(worker));
    page.on("console", message => {
      if (/fake worker|Failed to load module script/i.test(message.text())) workerWarnings.push(message.text());
    });
    // Reproduce Staticfile servers that recognize .js but serve .mjs as binary.
    // Vite preview normally supplies a JS MIME type for both and hides this bug.
    await page.route(/\/assets\/[^/?]+\.mjs(?:\?.*)?$/, async route => {
      const response = await route.fetch();
      await route.fulfill({ response, headers: { ...response.headers(), "content-type": "application/octet-stream" } });
    });
    await page.route(`**/api/ecns/${rowId}/snapshot`, route => route.fulfill({ json: {
      version: "ecn-snapshot-v2", evidence_hash: "a".repeat(64), counts: {}, questions: [], plant_impact: { plants: [] },
      bom_changes: { materials: [], lines: samples.map(number => ({ assembly: "A1", plants: [], incoming: { number }, qualifiers: [] })) },
    } }));
    await page.route(`**/api/ecns/${rowId}/object?**`, route => {
      const number = new URL(route.request().url()).searchParams.get("number");
      return route.fulfill({ json: { number, description: "Sample pipe", types: [], material_brief: {},
        engineering_drawing: { file_name: `${number}_2_AFS.pdf`, media_type: "application/pdf" } } });
    });
    await page.route(`**/api/ecns/${rowId}/drawing?**`, route => {
      expect(route.request().headers()["x-api-key"]).toBe("e2e-test-key");
      const number = new URL(route.request().url()).searchParams.get("number");
      expect(samples).toContain(number);
      drawingReads.push(number);
      return route.fulfill({ contentType: "application/pdf", body: readFileSync(`../api/app/ecn/fixtures/drawings/${number}_2_AFS.pdf`) });
    });
    await page.goto(`/ecns/${rowId}`);
    await page.getByRole("button", { name: "BOM Change Evidence", exact: true }).click();
    for (const number of samples) {
      const opener = page.getByRole("button", { name: number, exact: true });
      await opener.click();
      const material = page.locator("dialog.object-dialog");
      await expect(material).toBeVisible();
      await expect(material.locator('input[type="file"]')).toHaveCount(0);
      const preview = material.locator(".engineering-drawing-preview");
      await expect(preview).toContainText(`${number}_2_AFS.pdf`);
      await expect(preview).not.toContainText("No viewable drawing");
      await expect(preview).not.toContainText("Local preview");
      expect(drawingReads).not.toContain(number);
      const view = preview.getByRole("button", { name: "View engineering drawing", exact: true });
      await view.click();
      const canvas = preview.locator("canvas.pdf-drawing-page");
      await expect(canvas).toBeVisible();
      await expect(canvas).toHaveAttribute("data-rendered", "true");
      const pixels = await canvas.evaluate(canvas => {
        const data = canvas.getContext("2d").getImageData(0, 0, canvas.width, canvas.height).data;
        let dark = 0;
        for (let index = 0; index < data.length; index += 4) if (data[index] < 150 && data[index + 3] > 0) dark++;
        return dark;
      });
      expect(pixels).toBeGreaterThan(1000);
      const bytes = await preview.locator("a[download]").evaluate(async link => {
        const response = await fetch(link.href);
        return { type: response.headers.get("content-type"), start: (await response.text()).slice(0, 4) };
      });
      expect(bytes).toEqual({ type: "application/pdf", start: "%PDF" });
      await preview.getByRole("button", { name: "Zoom in", exact: true }).click();
      await expect(preview.locator(".pdf-drawing-controls")).toContainText("125%");
      if (number === samples[0]) await page.screenshot({ path: testInfo.outputPath("drawing-viewer.png") });
      await page.keyboard.press("Escape");
      await expect(preview.locator(".drawing-preview-viewer")).not.toBeVisible();
      await expect(view).toBeFocused();
      await material.locator(".object-dialog-footer").getByRole("button", { name: "Close", exact: true }).click();
      await expect(opener).toBeFocused();
    }
    expect(drawingReads).toEqual(samples);
    expect(workers.length).toBeGreaterThan(0);
    for (const worker of workers) {
      expect(new URL(worker.url()).pathname).toMatch(/\/assets\/pdf\.worker-[\w-]+\.js$/);
      const response = await page.request.get(worker.url());
      expect(response.ok()).toBe(true);
      expect(response.headers()["content-type"]).toMatch(/^(?:text|application)\/javascript\b/);
    }
    expect(workerWarnings).toEqual([]);
    expect(api.requests.filter(request => request.method !== "GET")).toEqual([]);
    expect(errors).toEqual([]);
    expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBe(width);
  });
}
