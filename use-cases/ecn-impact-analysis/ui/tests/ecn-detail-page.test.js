import { afterEach, beforeEach, expect, it, vi } from "vitest";
import html from "../src/pages/ecn-detail/ecn-detail.html?raw";
import { createEcnDetailPage } from "../src/pages/ecn-detail/ecn-detail.js";
import * as workspaceApi from "../src/services/workspace-api.js";
import { detailEnvelope } from "./fixtures/api-responses.js";

vi.mock("../src/services/workspace-api.js", () => ({
  generateSummaries: vi.fn(),
  getSnapshot: vi.fn(),
  getAnalysisStages: vi.fn(),
  listReviews: vi.fn(),
  saveReview: vi.fn(),
  getAcceptancePlan: vi.fn(),
  versions: vi.fn(),
}));

beforeEach(() => {
  vi.clearAllMocks();
  workspaceApi.listReviews.mockResolvedValue([]);
  workspaceApi.getAcceptancePlan.mockResolvedValue({ available: true, blocked: false });
  workspaceApi.versions.mockResolvedValue([]);
});

afterEach(() => {
  vi.unstubAllGlobals();
});

/** Mount the detail template and return a route context with list state. */
function detailContext() {
  document.body.innerHTML = `<main>${html}</main>`;
  return {
    container: document.querySelector("main"),
    params: { ecn_row_id: "11111111-1111-4111-8111-111111111111" },
    query: new URLSearchParams("query=shaft&sort=warning_count&page=2&page_size=25"),
    signal: new AbortController().signal,
    navigate: vi.fn(),
  };
}

it("loads source metadata and renders source-backed Audit Log events without generating on read", async () => {
  const detail = detailEnvelope();
  detail.analysis.impact_factors.manufacturing.assessment += " [E-0123456789abcdef]";
  const api = {
    getEcn: vi.fn().mockResolvedValue(detail),
    getSnapshot: vi.fn().mockResolvedValue({ assessment_date: "2026-09-14" }),
    retryAnalysis: vi.fn(),
  };
  workspaceApi.versions.mockResolvedValue([{
    ecn_row_id: detail.ecn_row_id,
    created_at: "2026-09-14T08:00:30Z",
  }]);
  workspaceApi.listReviews.mockResolvedValue([{
    reviewer: "Dana", outcome: "ACCEPTED", comments: "Ready for release", created_at: "2026-09-14T08:02:00Z",
  }]);
  await createEcnDetailPage({ api, startPolling: () => vi.fn() })(detailContext());
  expect(api.getEcn).toHaveBeenCalledWith(
    "11111111-1111-4111-8111-111111111111",
    expect.objectContaining({ signal: expect.any(AbortSignal) }),
  );
  expect(document.querySelector("#detail-title").textContent).toContain("ECN100");
  expect(document.querySelector("#detail-title-snapped").textContent).toContain("ECN100");
  expect(document.querySelector("#detail-title-snapped").getAttribute("slot")).toBe("snappedHeading");
  expect(document.querySelector("#detail-status")).toBeNull();
  expect(document.querySelector("#detail-impact")).toBeNull();
  expect(document.querySelector("#back-to-list").href).toContain("query=shaft");
  expect(document.querySelector("#source-record-fields").textContent).toContain("Released");
  expect(document.querySelector("#source-record-fields").textContent).toContain("Dana Whitfield");
  expect(document.querySelector("#source-record-fields").textContent).toContain("Q900V, M1V");

  expect(api.retryAnalysis).not.toHaveBeenCalled();
  expect(document.body.textContent).not.toContain("E-4");
  expect(document.body.textContent).not.toContain("E-0123456789abcdef");
  await vi.waitFor(() => expect(document.querySelector("#audit-log").textContent).toContain("Ready for release"));
  const audit = document.querySelector("#audit-log");
  expect(audit.textContent).toContain("Source import");
  expect(audit.textContent).toContain("ECN source imported");
  expect(audit.textContent).toContain("ecn.json");
  expect(audit.textContent).toContain("Generated analysis");
  expect(audit.textContent).toContain("Assessment status: Completed With Gaps");
  expect(audit.textContent).toContain("Assessment version created");
  expect(audit.textContent).toContain("Assessed as of 2026-09-14");
  expect(audit.textContent).toContain("Human decision");
  expect(audit.textContent).toContain("Accepted");
  expect(audit.textContent).toContain("Dana");
});

it("renders the single Audit Log card without legacy audit sections", async () => {
  const api = {
    getEcn: vi.fn().mockResolvedValue(detailEnvelope()),
    getSnapshot: vi.fn().mockResolvedValue({}),
    retryAnalysis: vi.fn(),
  };

  await createEcnDetailPage({ api, startPolling: () => vi.fn() })(detailContext());

  const card = document.querySelector("#tab-audit #audit-log-card");
  expect(card).not.toBeNull();
  expect(card.querySelector("h2").textContent).toBe("Audit Log");
  expect(card.querySelector(".eyebrow")).toBeNull();
  expect(document.querySelector("#tab-audit #audit-log")).not.toBeNull();
  expect(document.querySelector("#tab-plants #object-browser")).toBeNull();
  expect(document.querySelector("#tab-audit #object-browser")).toBeNull();
  expect(document.querySelector("#tab-audit #analysis-root")).toBeNull();
  expect(document.querySelector("#tab-audit #reference-documents")).toBeNull();
  expect(document.querySelector("#tab-audit #assessment-context")).toBeNull();
});

it("renders verified linked ECN actions and disables unavailable targets", async () => {
  const target = "66666666-6666-4666-8666-666666666666";
  const api = { getEcn: vi.fn().mockResolvedValue(detailEnvelope({ linked_ecns: [
    { ecn_number: "ECN12345", ecn_row_id: target },
    { ecn_number: "ECN888", ecn_row_id: null },
  ] })) };
  const cleanup = await createEcnDetailPage({ api, startPolling: () => vi.fn() })(detailContext());
  const link = document.querySelector("#linked-ecn-field a");
  expect(link.textContent).toBe("ECN12345");
  expect(link.getAttribute("aria-label")).toBe("Open linked ECN ECN12345");
  const url = new URL(link.href);
  expect(url.pathname).toBe(`/ecns/${target}`);
  expect(url.searchParams.get("query")).toBe("shaft");
  expect(url.searchParams.get("page")).toBe("2");
  const missing = document.querySelector("#linked-ecn-field button");
  expect(missing.textContent).toBe("ECN888");
  expect(missing.disabled).toBe(true);
  expect(missing.title).toContain("not available");
  cleanup();
});

it("keeps the original ECN disabled when no outgoing references exist", async () => {
  const api = { getEcn: vi.fn().mockResolvedValue(detailEnvelope({ linked_ecns: [] })) };
  const cleanup = await createEcnDetailPage({ api, startPolling: () => vi.fn() })(detailContext());
  expect(document.querySelector("#linked-ecn-field button").disabled).toBe(true);
  expect(document.querySelector("#linked-ecn-field a")).toBeNull();
  cleanup();
});

it("renders source-backed overview and plant cards without impact or decision cards", async () => {
  const snapshot = {
    counts: { technical_objects: 1, attention_items: 1 }, questions: [], parts: [], programs: [],
    part_descriptions: [
      { number: "OLD1", description: "Metal shim ring" },
      { number: "NEW1", description: "Flexible shim ring" },
    ],
    plant_impact: {
      plants: [{
        plant_code: "PL10", plant_name: "Plant A", status: "CHECK",
        new_parts_with_purchasing: [], new_parts_without_purchasing: ["NEW1"],
        materials: [{ number: "NEW1", description: "Flexible shim ring", procurement_types: ["External procurement"] }],
        cutover_from: "2026-05-19",
      }],
    },
    bom_changes: {
      lines: [{
        qualifiers: [], levels: ["EBOM", "MBOM"],
        plants: [{ plant_code: "PL10", plant_name: "Plant A" }],
        outgoing: { number: "OLD1", description: "Metal shim ring", date: "2026-05-19" },
        incoming: { number: "NEW1", description: "Flexible shim ring", date: "2026-05-19" },
      }], materials: [],
    },
    ecn_context: {
      owner_name: "Dana Whitfield", business_driver: "Replace the old part before the model cutover.",
      financial_impact: "INCREASE", financial_notes: "Estimated increase of EUR 0.85 per unit.",
    },
    // MOCK_DATA_RETIREMENT_VALIDATION_DISABLED_START: overlay-only snapshot fields.
    // phase_schedule: [{ material_number: "NEW1", description: "Flexible shim ring", from_label: "Eng. Approved", to_label: "Phase-In", planner_confirmed: false, effective_date: "2026-05-19" }],
    // purchase_prices: [{ material_number: "NEW1", plant_code: "PL10", status: "MISSING" }],
    // make_buy_recommendations: [{
    //   material_number: "NEW1", material_description: "Flexible shim ring", plant_code: "PL10", recommendation: "BUY",
    //   recommendation_summary: "External supply is lower cost for the planned volume.", owner: "Procurement",
    // }],
    // demo_assumptions: ["Cost and sourcing values are illustrative."],
    // MOCK_DATA_RETIREMENT_VALIDATION_DISABLED_END: overlay-only snapshot fields.
  };
  const api = {
    getEcn: vi.fn().mockResolvedValue(detailEnvelope({
      description_summary: "Replace steel shim with flexible plastic shim",
      analysis: { ...detailEnvelope().analysis, change_brief: "Convert the shim ring assembly from rigid to flexible components." },
    })),
    getSnapshot: vi.fn().mockResolvedValue(snapshot),
    listBusinessEvidence: vi.fn(),
    retryAnalysis: vi.fn(),
  };

  await createEcnDetailPage({ api, startPolling: () => vi.fn() })(detailContext());
  await vi.waitFor(() => expect(document.querySelector("#affected-sites-list").textContent).toContain("Plant A"));

  expect(document.querySelector("#decision-story").textContent).toBe("Convert the shim ring assembly from rigid to flexible components.");
  const briefIcon = document.querySelector(".decision-story-ai-icon");
  expect(briefIcon?.getAttribute("name")).toBe("ai");
  expect(briefIcon?.getAttribute("mode")).toBe("Image");
  expect(briefIcon?.getAttribute("accessible-name")).toBe("AI-generated change brief");
  expect(document.querySelector("#overview-change-reason")).toBeNull();
  const sourceRecord = document.querySelector("#source-record-fields");
  expect(sourceRecord.querySelector(".ai-change-summary")).not.toBeNull();
  const summaryIcon = sourceRecord.querySelector(".ai-change-summary-label ui5-icon");
  expect(summaryIcon?.getAttribute("name")).toBe("ai");
  expect(summaryIcon?.getAttribute("aria-hidden")).toBe("true");
  expect(sourceRecord.textContent).not.toContain("Replace the old part before the model cutover.");
  expect(document.querySelector("#overview-change-facts").textContent).toContain("Dana Whitfield");
  expect(document.querySelector("#overview-change-facts").textContent).toContain("2026");
  expect(document.querySelector("#source-record-fields").textContent).toContain("1 BOM change found");
  expect(document.querySelector("#affected-sites-list").textContent).toContain("Plant A");
  expect(document.querySelector("#affected-sites-list").textContent).toContain("PL10");
  expect(document.querySelector("#affected-sites-list").textContent).toContain("BOM evidence needs review");
  expect(document.querySelector("#affected-sites-list .affected-site-status").className).toContain("tone-warning");
  expect(document.querySelector("#decision-readiness")).toBeNull();
  expect(document.querySelector(".ecn-sections").firstElementChild.matches(".ecn-section-navigation")).toBe(true);
  expect(document.body.textContent).not.toContain("Review status");
  expect(document.body.textContent).not.toContain("Decision readiness");
  expect(document.querySelector("#overview-impact")).toBeNull();
  expect(document.querySelector("#overview-decision")).toBeNull();
  expect(document.querySelector("#plant-impact")).toBeNull();
  expect(document.querySelector("#attention")).toBeNull();
  expect(document.querySelector("#impact-analysis-details")).toBeNull();
  expect(document.querySelector("#parts-overview")).toBeNull();
  expect(document.querySelector("#tab-summary #executive-summary")).toBeNull();
});

it("uses the stored AI executive summary for the change summary and one-line story", async () => {
  const detail = detailEnvelope({
    description_summary: "Long source text that must not become the decision headline.",
  });
  delete detail.analysis.change_brief;
  detail.analysis.sections.executive_summary.assessment_markdown = "- **What changes:** Replace the rigid shim ring with a flexible component.\n- **What is impacted:** Shim assemblies.\n- **Where:** PL10.";
  const api = { getEcn: vi.fn().mockResolvedValue(detail), retryAnalysis: vi.fn() };

  await createEcnDetailPage({ api, startPolling: () => vi.fn() })(detailContext());

  const changeSummary = document.querySelector("#source-record-fields .ai-change-summary");
  expect(changeSummary.textContent).toContain("What changes: Replace the rigid shim ring with a flexible component.");
  expect(changeSummary.textContent).toContain("What is impacted: Shim assemblies.");
  expect(changeSummary.textContent).toContain("Where: PL10.");
  expect(changeSummary.querySelectorAll("li")).toHaveLength(3);
  expect(changeSummary.textContent).not.toContain("Long source text");
  expect(document.querySelector("#decision-story").textContent).toBe("Replace the rigid shim ring with a flexible component.");
  expect(document.querySelector("#decision-story").textContent).not.toContain("Long source text");
});

it("condenses a legacy identifier-heavy brief into a common-language change headline", async () => {
  const detail = detailEnvelope({
    analysis: {
      ...detailEnvelope().analysis,
      change_brief: "4 new assemblies PLATES_DECALS_GREASE_INSTRUCTIONS_PS_JB and PLATES_DECALS_SAFET_ACCU_EPL_HAZ_CVT_JB are added under M1V_PLATFORM.",
    },
  });
  const api = { getEcn: vi.fn().mockResolvedValue(detail), retryAnalysis: vi.fn() };

  await createEcnDetailPage({ api, startPolling: () => vi.fn() })(detailContext());

  expect(document.querySelector("#decision-story").textContent).toBe("Adds 4 new assemblies to the M1V platform.");
  expect(document.querySelector("#decision-story").textContent).not.toContain("PLATES_DECALS");
});

it("replaces a known part number in the AI brief with its supplied description", async () => {
  const detail = detailEnvelope({
    analysis: { ...detailEnvelope().analysis, change_brief: "New part 916100410830 is added to the engineering BOM." },
  });
  const snapshot = {
    counts: { technical_objects: 1, attention_items: 0 }, questions: [], parts: [], programs: [],
    bom_changes: { lines: [], materials: [{ number: "916100410830", description: "Flexible shim ring" }] },
    plant_impact: { plants: [] },
  };
  const api = {
    getEcn: vi.fn().mockResolvedValue(detail), getSnapshot: vi.fn().mockResolvedValue(snapshot),
    listChanges: vi.fn().mockResolvedValue({ items: [] }), listBusinessEvidence: vi.fn(), retryAnalysis: vi.fn(),
  };

  await createEcnDetailPage({ api, startPolling: () => vi.fn() })(detailContext());
  await vi.waitFor(() => expect(document.querySelector("#decision-story").textContent).toContain("Flexible shim ring"));
  expect(document.querySelector("#decision-story").textContent).not.toContain("916100410830");
});

it("shows independent RDF status and permits retry without hiding relational evidence", async () => {
  const projection = {
    ecn_row_id: "11111111-1111-4111-8111-111111111111",
    status: "FAILED",
    source_sha256: "d".repeat(64),
    projection_version: "1.0.0",
    verified_node_count: null,
    verified_triple_count: null,
    error_code: "RDF_PROJECTION_FAILED",
    error_message: "The RDF projection could not be rebuilt and verified.",
    last_attempt_at: null,
    current_at: null,
  };
  const api = {
    getEcn: vi.fn().mockResolvedValue(detailEnvelope()),
    getRdfProjection: vi.fn().mockResolvedValue(projection),
    retryRdfProjection: vi.fn().mockResolvedValue({ ...projection, status: "PENDING" }),
    listBusinessEvidence: vi.fn(),
    retryAnalysis: vi.fn(),
  };

  await createEcnDetailPage({ api, startPolling: () => vi.fn() })(detailContext());

  expect(api.getRdfProjection).toHaveBeenCalledWith(
    "11111111-1111-4111-8111-111111111111",
    expect.objectContaining({ signal: expect.any(AbortSignal) }),
  );
  expect(document.querySelector("#rdf-projection-status").textContent).toContain("Failed");
  expect(document.querySelector("#source-preparation").hidden).toBe(false);
  expect(document.querySelector("#rdf-projection-retry").hidden).toBe(false);
  expect(document.querySelector("#generate-summary").disabled).toBe(true);
  expect(document.querySelector("#source-record-fields").textContent).toContain("Released");

  document.querySelector("#rdf-projection-retry").click();
  await vi.waitFor(() => expect(api.retryRdfProjection).toHaveBeenCalledOnce());
  expect(document.querySelector("#rdf-projection-status").textContent).toContain("Pending");
});

it("scrolls route views in place without reloading the ECN", async () => {
  const context = detailContext();
  let switchView;
  context.view = "references";
  context.onViewChange = (handler) => { switchView = handler; };
  const api = { getEcn: vi.fn().mockResolvedValue(detailEnvelope()), retryAnalysis: vi.fn() };

  await createEcnDetailPage({ api, startPolling: () => vi.fn() })(context);

  expect(document.querySelector('[data-section-target="audit"]').getAttribute("aria-current")).toBe("true");
  expect(document.querySelector("#audit-log-card")).not.toBeNull();
  switchView("impact");
  expect(document.querySelector('[data-section-target="overview"]').getAttribute("aria-current")).toBe("true");
  expect(document.querySelector("#source-record")).not.toBeNull();
  expect(api.getEcn).toHaveBeenCalledTimes(1);
});

it("refreshes contained DynamicPage scrolling after internal rendering and replacement without a render event", async () => {
  const context = detailContext();
  let switchView;
  context.onViewChange = (handler) => { switchView = handler; };
  const dynamicPage = context.container.querySelector("#ecn-page");
  const shadowRoot = dynamicPage.shadowRoot || dynamicPage.attachShadow({ mode: "open" });
  const api = { getEcn: vi.fn().mockResolvedValue(detailEnvelope()), retryAnalysis: vi.fn() };

  await createEcnDetailPage({ api, startPolling: () => vi.fn() })(context);
  // Let UI5 finish its own deferred initial render before replacing the bounded
  // internal test fixture. This avoids racing the component's first template render.
  await vi.waitFor(() => expect(shadowRoot.querySelector(".ui5-dynamic-page-scroll-container")).not.toBeNull());
  shadowRoot.replaceChildren();

  const createScrollContainer = () => {
    const scrollContainer = document.createElement("div");
    scrollContainer.className = "ui5-dynamic-page-scroll-container";
    scrollContainer.getBoundingClientRect = () => ({ top: 10, bottom: 410 });
    Object.defineProperty(scrollContainer, "scrollTop", { configurable: true, value: 20, writable: true });
    scrollContainer.scrollTo = vi.fn();
    return scrollContainer;
  };
  const initialScrollContainer = createScrollContainer();
  const titleWrapper = document.createElement("div");
  titleWrapper.className = "ui5-dynamic-page-title-header-wrapper";
  titleWrapper.getBoundingClientRect = () => ({ top: 10, bottom: 60 });
  const navigation = document.querySelector(".ecn-section-navigation");
  navigation.getBoundingClientRect = () => ({ top: 60, bottom: 84, height: 24 });
  const sectionTops = { overview: 20, plants: 80, bom: 140, audit: 220 };
  document.querySelector("#source-record").getBoundingClientRect = () => ({ top: sectionTops.overview, bottom: sectionTops.overview + 50 });
  document.querySelector("#affected-sites").getBoundingClientRect = () => ({ top: sectionTops.plants, bottom: sectionTops.plants + 50 });
  document.querySelector("#bom-changes").getBoundingClientRect = () => ({ top: sectionTops.bom, bottom: sectionTops.bom + 50 });
  document.querySelector("#audit-log-card").getBoundingClientRect = () => ({ top: sectionTops.audit, bottom: sectionTops.audit + 100 });
  shadowRoot.append(initialScrollContainer, titleWrapper);

  await vi.waitFor(() => expect(document.querySelector('[data-section-target="plants"]').getAttribute("aria-current")).toBe("true"));

  const replacementScrollContainer = createScrollContainer();
  shadowRoot.replaceChildren(replacementScrollContainer, titleWrapper);
  await vi.waitFor(() => expect(document.querySelector('[data-section-target="plants"]').getAttribute("aria-current")).toBe("true"));

  sectionTops.plants = 120;
  sectionTops.bom = 90;
  replacementScrollContainer.dispatchEvent(new Event("scroll"));
  expect(document.querySelector('[data-section-target="bom"]').getAttribute("aria-current")).toBe("true");

  vi.stubGlobal("requestAnimationFrame", (callback) => {
    callback();
    return 1;
  });
  switchView("references");

  expect(navigation.style.getPropertyValue("--ecn-navigation-top")).toBe("50px");
  expect(replacementScrollContainer.scrollTo).toHaveBeenCalledWith({ top: 140, behavior: "auto" });
  expect(document.querySelector('[data-section-target="audit"]').getAttribute("aria-current")).toBe("true");
  expect(api.getEcn).toHaveBeenCalledTimes(1);
  vi.unstubAllGlobals();
});

it("uses text-only section navigation to scroll without hiding other sections", async () => {
  const context = detailContext();
  const cards = [
    document.querySelector("#source-record"),
    document.querySelector("#affected-sites"),
    document.querySelector("#bom-changes"),
    document.querySelector("#audit-log-card"),
  ];
  cards.forEach((card) => { card.scrollIntoView = vi.fn(); });
  const api = { getEcn: vi.fn().mockResolvedValue(detailEnvelope()), retryAnalysis: vi.fn() };

  await createEcnDetailPage({ api, startPolling: () => vi.fn() })(context);

  const navigation = [...document.querySelector(".ecn-section-navigation").querySelectorAll("button")];
  expect(navigation.map((button) => button.textContent)).toEqual(["Overview", "Plants", "BOM Change Evidence", "Audit Log"]);
  expect(document.querySelector("ui5-tabcontainer")).toBeNull();
  for (const [index, card] of cards.entries()) {
    navigation[index].click();
    await vi.waitFor(() => expect(card.scrollIntoView).toHaveBeenCalledWith({ block: "start", behavior: "auto" }));
    expect(navigation[index].getAttribute("aria-current")).toBe("true");
  }
  expect(document.querySelector("#tab-audit")).not.toBeNull();
});

it("a detail poll cannot abort a retry mutation", async () => {
  let pollLoad;
  let releaseRetry;
  let retrySignal;
  const api = {
    getEcn: vi.fn().mockResolvedValue(detailEnvelope({
      analysis: null,
      job: {
        job_id: "cccccccc-cccc-4ccc-8ccc-cccccccccccc",
        state: "RUNNING",
        attempt_count: 1,
        error_code: null,
        error_message: null,
      },
    })),
    retryAnalysis: vi.fn((rowId, { signal }) => {
      retrySignal = signal;
      return new Promise((resolve) => {
        releaseRetry = () => resolve({
          ecn_row_id: rowId,
          job_id: "cccccccc-cccc-4ccc-8ccc-cccccccccccc",
          state: "QUEUED",
        });
      });
    }),
  };
  await createEcnDetailPage({
    api,
    startPolling: ({ load }) => { pollLoad = load; return vi.fn(); },
  })(detailContext());
  document.querySelector("#retry-analysis").click();
  await Promise.resolve();
  await pollLoad();
  expect(retrySignal.aborted).toBe(false);
  releaseRetry();
});

it("refreshes canonical state and resumes polling when retry returns 409", async () => {
  const failed = detailEnvelope({
    analysis: null,
    job: {
      job_id: "cccccccc-cccc-4ccc-8ccc-cccccccccccc",
      state: "FAILED",
      attempt_count: 1,
      error_code: "PROVIDER_TIMEOUT",
      error_message: "Generation timed out",
    },
  });
  const queued = detailEnvelope({
    analysis: null,
    job: {
      job_id: "cccccccc-cccc-4ccc-8ccc-cccccccccccc",
      state: "QUEUED",
      attempt_count: 1,
      error_code: null,
      error_message: null,
    },
  });
  const startPolling = vi.fn(() => vi.fn());
  const api = {
    getEcn: vi.fn().mockResolvedValueOnce(failed).mockResolvedValueOnce(queued),
    listBusinessEvidence: vi.fn(),
    retryAnalysis: vi.fn().mockRejectedValue(
      Object.assign(new Error("Already queued"), { status: 409 }),
    ),
  };
  await createEcnDetailPage({ api, startPolling })(detailContext());
  document.querySelector("#retry-analysis").click();
  await vi.waitFor(() => expect(api.getEcn).toHaveBeenCalledTimes(2));
  expect(document.querySelector("#detail-status")).toBeNull();
  expect(startPolling).toHaveBeenCalledTimes(1);
});

it("regenerates an existing summary directly and starts polling", async () => {
  const queued = detailEnvelope({
    analysis: null,
    job: {
      job_id: "cccccccc-cccc-4ccc-8ccc-cccccccccccc",
      state: "QUEUED",
      attempt_count: 1,
      error_code: null,
      error_message: null,
    },
  });
  const startPolling = vi.fn(() => vi.fn());
  const api = {
    getEcn: vi.fn().mockResolvedValueOnce(detailEnvelope()).mockResolvedValue(queued),
    listBusinessEvidence: vi.fn(),
    retryAnalysis: vi.fn(),
    generateSummaries: vi.fn().mockResolvedValue({
      selection: { scheduled_job_ids: ["job-1"] },
      jobs: [{ state: "QUEUED" }],
    }),
  };
  await createEcnDetailPage({ api, startPolling, uuid: () => "request-1" })(detailContext());
  expect(document.querySelector("#generate-summary").disabled).toBe(false);
  expect(document.querySelector("#generate-summary").textContent).toBe("Regenerate Summary");
  document.querySelector("#generate-summary").click();
  await vi.waitFor(() => expect(api.generateSummaries).toHaveBeenCalledOnce());

  expect(api.generateSummaries.mock.calls[0][0]).toEqual({
    mode: "explicit",
    ecn_ids: ["11111111-1111-4111-8111-111111111111"],
  });
  expect(document.querySelector("#detail-generation-status").textContent).toContain("1 scheduled");
  expect(document.querySelector(".summary-dialog")).toBeNull();
  await vi.waitFor(() => expect(startPolling).toHaveBeenCalledOnce());
});

it("keeps operational diagnostics hidden while the Audit Log retains generated-analysis events", async () => {
  const detail=detailEnvelope({description_text:"Synthetic engineering impact demonstration; no source-system write-back."});
  const api={getEcn:vi.fn().mockResolvedValue(detail),getRdfProjection:vi.fn().mockResolvedValue({status:"CURRENT"}),getSnapshot:vi.fn().mockResolvedValue({})};
  const cleanup=await createEcnDetailPage({api,startPolling:()=>vi.fn()})(detailContext());
  expect(document.querySelector("#source-preparation").hidden).toBe(true);
  expect(document.querySelector("#tab-summary .factor-card")).toBeNull();
  expect(document.querySelector("#tab-summary #analysis-root")).toBeNull();
  expect(document.querySelector("#tab-audit #analysis-root")).toBeNull();
  expect(document.querySelector("#audit-log").textContent).toContain("Generated analysis");
  cleanup();
});

it("does not render staged factor next steps in the Overview", async () => {
  const detail=detailEnvelope();
  detail.analysis.prompt_version="ecn-snapshot-v3";
  detail.analysis.additional_next_steps={
    manufacturing:["Approve the dated manufacturing cutover for SIM1100."],
    procurement:["Confirm the stop-buy decision for SIM1100."],
    configuration:[],
  };
  const api={getEcn:vi.fn().mockResolvedValue(detail),getAnalysisStages:vi.fn().mockResolvedValue({})};
  await createEcnDetailPage({api,startPolling:()=>vi.fn()})(detailContext());
  expect(document.querySelector("#tab-summary").textContent).not.toContain("Approve the dated manufacturing cutover");
  expect(document.querySelector("#tab-summary").textContent).not.toContain("Confirm the stop-buy decision");
  expect(document.querySelector("#tab-summary .factor-card")).toBeNull();
});

it("keeps blocked acceptance in the footer and explains it in BOM evidence without an extra bar", async () => {
  const snapshot = {
    counts: { technical_objects: 1, attention_items: 0 },
    questions: [],
    bom_changes: { lines: [{ qualifiers: [], plants: [] }] },
    plant_impact: { plants: [] },
  };
  const api = {
    getEcn: vi.fn().mockResolvedValue(detailEnvelope({
      analysis: null,
      job: { job_id: "job-1", state: "FAILED", attempt_count: 1, error_code: "PROVIDER_TIMEOUT", error_message: "Generation timed out" },
    })),
    getSnapshot: vi.fn().mockResolvedValue(snapshot),
    retryAnalysis: vi.fn(),
  };
  workspaceApi.getAcceptancePlan.mockResolvedValue({
    available: true,
    blocked: true,
    block_reason: "Can't accept: 1 new part has no price.",
  });

  await createEcnDetailPage({ api, startPolling: () => vi.fn() })(detailContext());
  await vi.waitFor(() => expect(document.querySelector("#accept-ecn-reason").textContent).toBe("Can't accept: 1 new part has no price."));

  expect(document.querySelector("#decision-readiness")).toBeNull();
  expect(document.querySelector("#source-record-fields").textContent).toContain("1 BOM change found");
  expect(document.querySelector("#bom-changes h2").textContent).toBe("BOM Change Evidence");
  expect(document.querySelector("#accept-ecn-explanation")).toBeNull();
  expect(document.querySelector("#accept-ecn-footer").disabled).toBe(true);
  expect(document.querySelector("#accept-ecn-reason").textContent).toBe("Can't accept: 1 new part has no price.");
  expect(document.querySelector("#bom-changes .bom-acceptance-summary").textContent).toBe("Can't accept: 1 new part has no price.");
});

it("does not add a frontend blocker for an unavailable acceptance plan after removing review status", async () => {
  const snapshot = {
    counts: { technical_objects: 1, attention_items: 0 },
    questions: [],
    bom_changes: {},
    plant_impact: { plants: [] },
  };
  const api = {
    getEcn: vi.fn().mockResolvedValue(detailEnvelope()),
    getSnapshot: vi.fn().mockResolvedValue(snapshot),
    retryAnalysis: vi.fn(),
  };
  workspaceApi.getAcceptancePlan.mockResolvedValue({ available: false, blocked: false });

  await createEcnDetailPage({ api, startPolling: () => vi.fn() })(detailContext());
  await vi.waitFor(() => expect(document.querySelector("#accept-ecn-footer").disabled).toBe(false));

  expect(document.querySelector("#decision-readiness")).toBeNull();
  expect(document.querySelector("#accept-ecn-explanation")).toBeNull();
});

it("keeps source-backed follow-ups from becoming an acceptance rule without a review status card", async () => {
  const snapshot = {
    counts: { technical_objects: 1, attention_items: 1 },
    questions: [{ code: "MBOM_NOT_UPDATED" }],
    bom_changes: { lines: [{ qualifiers: [], plants: [] }] },
    plant_impact: { plants: [] },
  };
  const api = {
    getEcn: vi.fn().mockResolvedValue(detailEnvelope()),
    getSnapshot: vi.fn().mockResolvedValue(snapshot),
    retryAnalysis: vi.fn(),
  };
  workspaceApi.getAcceptancePlan.mockResolvedValue({ available: true, blocked: false });

  await createEcnDetailPage({ api, startPolling: () => vi.fn() })(detailContext());
  await vi.waitFor(() => expect(document.querySelector("#accept-ecn-footer").disabled).toBe(false));
  expect(document.querySelector("#decision-readiness")).toBeNull();
  expect(document.querySelector("#accept-ecn-explanation")).toBeNull();
});
