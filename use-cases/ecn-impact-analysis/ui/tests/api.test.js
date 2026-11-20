import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

/** Import fresh API modules under one deterministic Vite environment. */
async function loadModules(apiKey = "") {
  vi.resetModules();
  vi.stubEnv("VITE_API_BASE_URL", "https://ui.test");
  vi.stubEnv("VITE_API_KEY", apiKey);
  return Promise.all([
    import("../src/services/api.js"),
    import("../src/services/ecn-api.js"),
  ]);
}

describe("request", () => {
  beforeEach(() => {
    globalThis.fetch = vi.fn().mockResolvedValue(
      new Response(JSON.stringify({ ok: true }), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      }),
    );
  });

  afterEach(() => vi.unstubAllEnvs());

  it("reads drawing bytes with authentication and preserves JSON errors", async () => {
    const [{ request, ApiError }] = await loadModules("secret-key");
    fetch.mockResolvedValueOnce(new Response("%PDF-1.7 sample", { headers: { "Content-Type": "application/pdf" } }));
    const drawing = await request("/api/ecns/row/drawing?number=PART", { responseType: "blob" });
    expect(drawing.type).toBe("application/pdf");
    expect(await drawing.text()).toBe("%PDF-1.7 sample");
    expect(new Headers(fetch.mock.calls[0][1].headers).get("X-API-Key")).toBe("secret-key");
    fetch.mockResolvedValueOnce(new Response(JSON.stringify({ code: "DRAWING_NOT_FOUND", message: "Drawing unavailable" }), { status: 404 }));
    await expect(request("/api/ecns/row/drawing?number=OTHER", { responseType: "blob" }))
      .rejects.toEqual(new ApiError(404, "DRAWING_NOT_FOUND", "Drawing unavailable"));
  });

  it("adds a configured API key and serializes JSON", async () => {
    const [{ request }] = await loadModules("secret-key");
    await request("/api/history", {
      method: "DELETE",
      body: { confirmation: "RESET HISTORY" },
    });
    const [, init] = fetch.mock.calls[0];
    expect(new Headers(init.headers).get("X-API-Key")).toBe("secret-key");
    expect(new Headers(init.headers).get("Content-Type")).toBe("application/json");
    expect(init.body).toBe('{"confirmation":"RESET HISTORY"}');
  });

  it("omits an empty API key and lets the browser set a multipart boundary", async () => {
    const [{ request }] = await loadModules("");
    const body = new FormData();
    body.set("request_id", "11111111-1111-4111-8111-111111111111");
    await request("/api/uploads", { method: "POST", body });
    const [, init] = fetch.mock.calls[0];
    expect(new Headers(init.headers).has("X-API-Key")).toBe(false);
    expect(new Headers(init.headers).has("Content-Type")).toBe(false);
    expect(init.body).toBe(body);
  });

  it("preserves false and zero query values and drops only empty values", async () => {
    const [, { listEcns }] = await loadModules();
    await listEcns({ page: 0, has_warnings: false, query: "", analysis_state: null });
    const url = new URL(fetch.mock.calls[0][0]);
    expect(Object.fromEntries(url.searchParams)).toEqual({ page: "0", has_warnings: "false" });
  });

  it("uploads ECN JSON without legacy workbook fields", async () => {
    const [, { uploadSources }] = await loadModules();
    const ecnFile = new File(["[]"], "ecn.json", { type: "application/json" });
    await uploadSources({ requestId: "request-1", ecnFile });
    const [, init] = fetch.mock.calls[0];
    expect([...init.body.keys()]).toEqual(["request_id", "ecn_file", "data_group"]);
    expect(init.body.get("request_id")).toBe("request-1");
    expect(init.body.get("ecn_file")).toBe(ecnFile);
  });

  it("encodes bounded business evidence pagination and technical opt-in", async () => {
    const [, { listBusinessEvidence }] = await loadModules();
    await listBusinessEvidence("row/id", "data_quality", {
      page: 2,
      pageSize: 10,
      includeTechnical: true,
    });
    const url = new URL(fetch.mock.calls[0][0]);
    expect(url.pathname).toBe("/api/ecns/row%2Fid/evidence");
    expect(Object.fromEntries(url.searchParams)).toEqual({
      section: "data_quality",
      page: "2",
      page_size: "10",
      include_technical: "true",
    });
  });

  it("keeps RDF projection diagnostics and retry scoped to one encoded ECN identity", async () => {
    fetch.mockImplementation(() => Promise.resolve(new Response(JSON.stringify({ ok: true }), {
      status: 200,
      headers: { "Content-Type": "application/json" },
    })));
    const [, { getRdfProjection, retryRdfProjection }] = await loadModules();
    await getRdfProjection("row/id");
    await retryRdfProjection("row/id");
    expect(fetch.mock.calls[0][0]).toBe("https://ui.test/api/ecns/row%2Fid/rdf-projection");
    expect(fetch.mock.calls[0][1].method).toBe("GET");
    expect(fetch.mock.calls[1][0]).toBe("https://ui.test/api/ecns/row%2Fid/rdf-projection/retry");
    expect(fetch.mock.calls[1][1].method).toBe("POST");
  });

  it("publishes reference workbooks using a supported maintained dataset type", async () => {
    const [, { publishReferenceDataset }] = await loadModules();
    const sourceFile = new File(["xlsx"], "materials.xlsx");
    await publishReferenceDataset({
      datasetType: "MATERIAL_MASTER",
      sourceFile,
    });
    const [url, init] = fetch.mock.calls[0];
    expect(url).toBe("https://ui.test/api/reference-datasets");
    expect(init.method).toBe("POST");
    expect(Object.fromEntries(init.body)).toEqual({
      dataset_type: "MATERIAL_MASTER",
      context_json: "{}",
      source_file: sourceFile,
    });
  });

  it("keeps cached provider reads separate from explicit preflight checks", async () => {
    fetch.mockImplementation(() => Promise.resolve(new Response(JSON.stringify({ ok: true }), {
      status: 200,
      headers: { "Content-Type": "application/json" },
    })));
    const [, { getProviderStatus, checkProviderStatus }] = await loadModules();
    await getProviderStatus();
    await checkProviderStatus();
    expect(fetch.mock.calls[0][0]).toBe("https://ui.test/api/provider-status");
    expect(fetch.mock.calls[0][1].method).toBe("GET");
    expect(fetch.mock.calls[1][0]).toBe("https://ui.test/api/provider-status/check");
    expect(fetch.mock.calls[1][1].method).toBe("POST");
  });

  it("throws the stable server error envelope", async () => {
    fetch.mockResolvedValueOnce(new Response(
      JSON.stringify({ code: "HISTORY_RESET_ACTIVE_JOBS", message: "Analysis is active", details: null }),
      { status: 409, headers: { "Content-Type": "application/json" } },
    ));
    const [{ ApiError, request }] = await loadModules();
    await expect(request("/api/history", { method: "DELETE", body: {} })).rejects.toEqual(
      expect.objectContaining({ status: 409, code: "HISTORY_RESET_ACTIVE_JOBS", message: "Analysis is active" }),
    );
    expect(ApiError.prototype).toBeInstanceOf(Error);
  });
});
