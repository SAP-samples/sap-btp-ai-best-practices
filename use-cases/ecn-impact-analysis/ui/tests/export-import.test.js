import { expect, it, vi } from "vitest";
import {
  importPreview,
  lifecycleLabel,
  parseImportText,
  serializeSelection,
} from "../src/modules/ecn-import-data.js";
import { bindEcnImportDialog } from "../src/components/export-import.js";
import html from "../src/pages/ecns/ecns.html?raw";

/** Read one jsdom file through the browser FileReader contract. */
const readFile = (file) => new Promise((resolve, reject) => {
  const reader = new FileReader();
  reader.addEventListener("load", () => resolve(reader.result));
  reader.addEventListener("error", () => reject(reader.error));
  reader.readAsText(file);
});

it("validates the root, resolves lifecycle objects, and serializes only selected records", () => {
  expect(() => parseImportText('{"Number":"ECN1"}')).toThrow("JSON array");
  const records = parseImportText(JSON.stringify([
    { Number: "ECN1", DescriptionSummary: "First", State: { Display: "Resolved", Value: "RESOLVED" } },
    { Number: "ECN2", Name: "Second", State: { Value: "INWORK" } },
    { Number: "ECN3", State: "Cancelled" },
  ]));

  expect(lifecycleLabel(records[0].State)).toBe("Resolved");
  expect(lifecycleLabel(records[1].State)).toBe("INWORK");
  expect(lifecycleLabel(records[2].State)).toBe("Cancelled");
  expect(importPreview(records, { query: "second", page: 1, pageSize: 25 })).toMatchObject({
    matching: 1,
    entries: [{ position: 2, ecnNumber: "ECN2", lifecycle: "INWORK" }],
  });
  expect(JSON.parse(serializeSelection(records, [3, 1]))).toEqual([records[0], records[2]]);
});

it("selects the whole file, retries remaining batches with stable IDs, and resets on reopen", async () => {
  document.body.innerHTML = `<main>${html}</main>`;
  const records = Array.from({ length: 30 }, (_, index) => ({ Number: `ECN${index + 1}` }));
  const clients = [];
  const createWorkerClient = vi.fn(() => {
    const client = {
      parse: vi.fn().mockResolvedValue({ total: records.length }),
      preview: vi.fn(async ({ query = "", page = 1, pageSize = 25 }) => importPreview(records, { query, page, pageSize })),
      serialize: vi.fn(async (positions) => serializeSelection(records, positions)),
      terminate: vi.fn(),
    };
    clients.push(client);
    return client;
  });
  const requestIds = ["batch-1", "batch-2", "unused"];
  const uploads = [];
  let failSecondBatch = true;
  const uploadSources = vi.fn(async ({ requestId, ecnFile }) => {
    uploads.push({ requestId, records: JSON.parse(await readFile(ecnFile)) });
    if (requestId === "batch-2" && failSecondBatch) {
      failSecondBatch = false;
      throw new Error("temporary failure");
    }
  });
  const imported = vi.fn();
  const dispose = bindEcnImportDialog({
    root: document.querySelector("main"),
    signal: new AbortController().signal,
    uploadSources,
    createWorkerClient,
    uuid: () => requestIds.shift(),
    imported,
  });

  document.querySelector("#open-import").click();
  const input = document.querySelector("#import-file");
  Object.defineProperty(input, "files", { configurable: true, value: [new File(["[]"], "full.json", { type: "application/json" })] });
  input.dispatchEvent(new Event("change", { bubbles: true }));
  await vi.waitFor(() => expect(document.querySelector("#select-all-import").disabled).toBe(false));
  document.querySelector("#select-all-import").click();
  document.querySelector("#import-selected").click();
  await vi.waitFor(() => expect(uploadSources).toHaveBeenCalledTimes(2));

  expect(uploads.map((upload) => upload.records.length)).toEqual([25, 5]);
  expect(document.querySelector("#import-status").textContent).toContain("Retry");
  document.querySelector("#import-selected").click();
  await vi.waitFor(() => expect(uploadSources).toHaveBeenCalledTimes(3));
  expect(uploads.map((upload) => upload.requestId)).toEqual(["batch-1", "batch-2", "batch-2"]);
  expect(imported).toHaveBeenCalledOnce();
  expect(clients[0].terminate).toHaveBeenCalledOnce();

  document.querySelector("#open-import").click();
  expect(document.querySelector("#selection-summary").textContent).toContain("0 selected");
  expect(clients).toHaveLength(2);
  document.querySelector("#cancel-import").click();
  expect(clients[1].terminate).toHaveBeenCalledOnce();
  dispose();
});
