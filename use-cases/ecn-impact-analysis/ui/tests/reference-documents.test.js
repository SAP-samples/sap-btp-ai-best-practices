import { expect, it, vi } from "vitest";
import { referenceLine, mountReferenceDocuments } from "../src/components/reference-documents.js";

it("keeps document type/state and exact BOM file, row, position and dates in compact lines", () => {
  expect(referenceLine({ number: "D1", types: ["Drawing"], revision: "B", state: "Released" })).toBe("D1 · Drawing · revision B · Released");
  const line = referenceLine({ number: "P1", file_name: "bom.xlsx", sheet: "MBOM", row: 42, parent: "A1", position: "0010", valid_from: "2026-01-02", valid_to: "2026-06-30" });
  for (const value of ["P1", "bom.xlsx", "MBOM", "row 42", "position 0010", "2 Jan 2026", "30 Jun 2026"]) expect(line).toContain(value);
});

it("uses complete snapshot references including BOM-only removed parts", async () => {
  const host = document.createElement("div");
  const fetch = vi.fn();
  const snapshot = { reference_documents: [{ number: "D1", types: ["Drawing"] }], programs: [{ ebom: [], mbom: [{ number: "OLD", sheet: "MBOM", row: 2 }] }] };
  mountReferenceDocuments(host, fetch, "row", new AbortController().signal, Promise.resolve(snapshot));
  const panel = host.querySelector('[data-evidence-section="mbom_matches"]');
  panel.open = true;
  panel.dispatchEvent(new Event("toggle"));
  await vi.waitFor(() => expect(panel.textContent).toContain("OLD"));
  expect(fetch).not.toHaveBeenCalled();
  expect(host.querySelectorAll("details")).toHaveLength(3);
  expect(host.querySelector(".section-count")).toBeNull();
});
