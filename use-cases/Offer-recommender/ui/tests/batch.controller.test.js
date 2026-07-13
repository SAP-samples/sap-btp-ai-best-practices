import { beforeEach, describe, expect, it, vi } from "vitest";

import { initBatchPage } from "../src/pages/batch/batch.controller.js";

describe("batch controller", () => {
  beforeEach(() => {
    global.URL.createObjectURL = vi.fn().mockReturnValue("blob:mock");
    global.URL.revokeObjectURL = vi.fn();
    HTMLAnchorElement.prototype.click = vi.fn();
    document.body.innerHTML = `
      <button id="batch-run"></button>
      <div id="batch-summary"></div>
      <div id="batch-artifacts"></div>
      <div id="batch-status"></div>
    `;
  });

  it("runs a batch and renders summary metrics", async () => {
    const api = {
      runBatch: vi.fn().mockResolvedValue({
        run_id: "run-1",
        summary: {
          total_accounts: 600,
          residential_accounts: 301,
          commercial_accounts: 299,
          accounts_with_final_offer: 412
        },
        artifacts: {
          excel_path: "/tmp/nbo.xlsx",
          json_path: "/tmp/nbo.json"
        }
      }),
      downloadBatchArtifact: vi.fn().mockResolvedValue(new Blob(["test"]))
    };

    await initBatchPage(api);
    expect(document.getElementById("batch-summary").hidden).toBe(true);
    expect(document.getElementById("batch-artifacts").hidden).toBe(true);

    document.getElementById("batch-run").click();
    await Promise.resolve();

    expect(api.runBatch).toHaveBeenCalledTimes(1);
    expect(document.getElementById("batch-status").textContent).toContain("Analysis completed");
    expect(document.getElementById("batch-summary").textContent).toContain("600");
    expect(document.getElementById("batch-artifacts").textContent).toContain("Download Excel");
    expect(document.getElementById("batch-artifacts").textContent).toContain("Download JSON");
    expect(document.getElementById("batch-artifacts").textContent).not.toContain("/tmp/nbo.xlsx");
    expect(document.getElementById("batch-artifacts").textContent).not.toContain("/tmp/nbo.json");
    expect(document.getElementById("batch-summary").hidden).toBe(false);
    expect(document.getElementById("batch-artifacts").hidden).toBe(false);

    document.querySelector('[data-artifact="nbo_recommendations.xlsx"]').click();
    await Promise.resolve();
    expect(api.downloadBatchArtifact).toHaveBeenCalledWith("run-1", "nbo_recommendations.xlsx");
  });
});
