import { beforeEach, expect, it, vi } from "vitest";
import { PageRouter } from "../src/modules/router.js";
import { syncEcnNavigation } from "../src/modules/navigation.js";
import { renderChangeLists } from "../src/components/change-summary.js";
import { renderBomChanges } from "../src/components/bom-changes-table.js";
import { renderPlantImpact } from "../src/components/plant-impact-table.js";
import { computeKpis, renderKpiHeader } from "../src/components/ecn-kpi-header.js";
// MOCK_DATA_RETIREMENT_VALIDATION_DISABLED: import { renderProcurement } from "../src/components/procurement-risk.js";

// Mock ECharts to prevent canvas errors in jsdom
vi.mock("../src/components/ecn-charts.js", () => ({
  initChart: vi.fn(),
  disposeChart: vi.fn(),
  partsDonutOption: vi.fn(() => ({})),
  plantMiniBarOption: vi.fn(() => ({})),
  procurementBarOption: vi.fn(() => ({})),
  supplyChainGraphOption: vi.fn(() => ({})),
  C: {
    positive: "#256f3a", positiveLight: "#eaf6ed",
    negative: "#aa0808", negativeLight: "#ffeaf4",
    critical: "#8d5e00", criticalLight: "#fff3e1",
    information: "#004f9e", informationLight: "#e5f0fa",
    neutral: "#475e75", neutralLight: "#eef1f4",
    border: "#e5e5e5", label: "#556b82",
    newPart: "#004f9e", phaseOut: "#aa0808", revised: "#8d5e00", referenced: "#8a95a0",
  },
}));

beforeEach(() => { document.body.replaceChildren(); });

it("switches ECN views in place and re-initializes only for another ECN", async () => {
  document.body.innerHTML = '<main class="content"></main>';
  const views = [];
  const init = vi.fn(async (context) => { context.onViewChange?.((view) => views.push(view)); return vi.fn(); });
  const router = new PageRouter({
    routeEngine: Object.assign(vi.fn(), { start: vi.fn() }),
    loadModule: async () => ({ init }),
    loadHtml: async () => "<article></article>",
    loadCss: async () => undefined,
  });
  router.contentContainer = document.querySelector(".content");
  const onNavigate = vi.fn();
  router.onNavigate = onNavigate;
  const overview = { page: "ecn-detail", view: "overview", title: "ECN Overview", nav: "ecn-overview" };
  const impact = { page: "ecn-detail", view: "impact", title: "Impact Analysis", nav: "ecn-impact" };
  await router.loadPage(overview, { params: { ecn_row_id: "a" } });
  await router.loadPage(impact, { params: { ecn_row_id: "a" } });
  expect(init).toHaveBeenCalledTimes(1);
  expect(document.title).toBe("Impact Analysis | ECN Impact Analysis");
  expect(onNavigate).toHaveBeenLastCalledWith(impact, { ecn_row_id: "a" });
  await router.loadPage(impact, { params: { ecn_row_id: "b" } });
  expect(init).toHaveBeenCalledTimes(2);
});

it("shows ECN-specific navigation only while an ECN is open", () => {
  document.body.innerHTML = '<ui5-side-navigation><ui5-side-navigation-item data-nav="ecns" href="/ecns"></ui5-side-navigation-item><ui5-side-navigation-item data-nav="settings"></ui5-side-navigation-item></ui5-side-navigation>';
  const items = () => [...document.querySelectorAll("ui5-side-navigation > ui5-side-navigation-item")].slice(1, -1);
  syncEcnNavigation(undefined);
  expect(items()).toHaveLength(0);
  syncEcnNavigation("row-1");
  expect(items()).toHaveLength(1);
  expect(items().map(item => item.getAttribute("text"))).toEqual(["ECN Overview"]);
  expect(items().map(item => item.getAttribute("icon"))).toEqual(["detail-view"]);
  expect(items().map(item => item.getAttribute("href"))).toEqual(["/ecns/row-1"]);
  expect(items().map(item => item.dataset.nav)).toEqual(["ecn-overview"]);
  ["ecn-impact", "ecn-attention", "ecn-references"].forEach((key) => {
    expect(document.querySelector(`ui5-side-navigation-item[data-nav="${key}"]`)).toBeNull();
  });
  expect(items().some(item => item.hasAttribute("disabled") || item.hasAttribute("tooltip"))).toBe(false);
  syncEcnNavigation(undefined);
  expect(items()).toHaveLength(0);
});

it("lists parts and BOM matches per program and counts attention items from the snapshot", () => {
  const host = document.createElement("div");
  document.body.append(host);
  const inspect = vi.fn();
  const row = (number, extra = {}) => ({ number, parent: "Q923VV2526401_Q900", position: null, quantity: "2", valid_from: "2026-06-30", valid_to: "9999-12-31", plants: [{ plant_code: "PL10", plant_name: "Plant A" }], ...extra });
  renderChangeLists(host, {
    snapshot: {
      counts: { technical_objects: 2 },
      questions: [{ code: "A" }, { code: "B" }],
      programs: [
        { model_range: "Q900V", bom_uploaded: true, ebom: [row("QKX463881A", { parent: "Q923VV2526401", position: "0150" })], mbom: [row("QKX463881A")] },
        { model_range: "Q938V", bom_uploaded: false, ebom: [], mbom: [] },
      ],
    },
    changes: [{ number: "QKX4638810", action: "PHASE_OUT", counterparts: ["QKX463881A"] }, { number: "QKX463881A", action: "PHASE_IN", counterparts: ["QKX4638810"] }],
    warningCount: 1,
    attentionHref: "/ecns/x/attention",
    inspect,
  });
  const [parts, matches] = host.querySelectorAll("details");
  expect(parts.querySelector("summary").textContent).toBe("Parts in This ECN (2)");
  expect(parts.textContent).toContain("QKX4638810 · Phase-out · paired with QKX463881A");
  expect(matches.querySelector("summary").textContent).toBe("BOM Matches (2)");
  expect(matches.textContent).toContain("Engineering BOM (EBOM): 1");
  expect(matches.textContent).toContain("in Q923VV2526401 · position 0150");
  expect(matches.textContent).toContain("plants Plant A (PL10)");
  const select = matches.querySelector("select");
  select.value = "1";
  select.dispatchEvent(new Event("change"));
  expect(matches.textContent).toContain("No BOM uploaded for Q938V");
  const link = host.querySelector(".attention-count a");
  expect(link.textContent).toBe("Items needing attention: 2");
  expect(link.getAttribute("href")).toBe("/ecns/x/attention");
  parts.querySelector("button").click();
  expect(inspect).toHaveBeenCalledWith("QKX4638810");
});

it("shows material-oriented BOM evidence with ECN-specific status", () => {
  const host = document.createElement("div");
  renderBomChanges(host, {
    lines: [{
      plants: [{ plant_code: "PK01", plant_name: "Plant B" }, { plant_code: "PL10", plant_name: "Plant A" }],
      model_range: "Q900V", assembly: "Q923VV2526401", assembly_description: "M1V decal assembly",
      change: "REPLACED", change_label: "Replaced", position: "0150", engineering_date: "2026-01-16",
      outgoing: { number: "QKX4638810", description: "Old decal", quantity: "2", date: "2026-06-29" },
      incoming: { number: "QKX463881A", description: "Updated decal", quantity: "2", date: "2026-06-30" },
      levels: ["EBOM", "MBOM"], pairing_label: "Paired in the ECN", qualifiers: [],
    }],
    materials: [
      { number: "QKX4638810", status: "FULLY_PHASED_OUT", status_label: "Fully phased out", still_used_in: [] },
      { number: "QKX463881A", status: "NEW_PART", status_label: "New part", still_used_in: [] },
    ],
  }, vi.fn());
  expect(host.querySelector("h2").textContent).toBe("BOM Change Evidence");
  expect([...host.querySelectorAll("thead th")].map((header) => header.textContent)).toEqual([
    "Material", "Description", "ECN change status", "Parent assembly", "Affected plants", "Blocked status",
  ]);
  const groupHeader = host.querySelector(".bom-assembly-group-header th");
  expect(groupHeader.getAttribute("scope")).toBe("rowgroup");
  expect(groupHeader.textContent).toContain("M1V decal assembly");
  expect(groupHeader.textContent).toContain("Q900V · Q923VV2526401");
  expect(groupHeader.textContent).toContain("2 affected material rows");
  const rows = [...host.querySelectorAll(".bom-change-row")].map((row) => [...row.querySelectorAll("td")].map((cell) => cell.textContent));
  expect(rows).toHaveLength(2);
  expect(rows[0][0]).toBe("QKX4638810");
  expect(rows[0][1]).toBe("Old decal");
  expect(rows[0][2]).toBe("Removed");
  expect(rows[0][3]).toContain("M1V decal assembly");
  expect(rows[0][3]).toContain("Q900V · Q923VV2526401");
  expect(rows[0][4]).toBe("Plant B (PK01), Plant A (PL10)");
  expect(rows[1][0]).toBe("QKX463881A");
  expect(rows[1][1]).toBe("Updated decal");
  expect(rows[1][2]).toBe("New");
  expect(host.querySelector(".bom-part-link").textContent).toBe("QKX4638810");
  const tableViewport = host.querySelector(".table-scroll");
  expect(tableViewport.tabIndex).toBe(0);
  expect(tableViewport.getAttribute("role")).toBe("region");
  expect(tableViewport.getAttribute("aria-label")).toBe("BOM change evidence table. Scroll for additional rows.");
});

it("groups material rows by their exact parent assembly occurrence", () => {
  const host = document.createElement("div");
  renderBomChanges(host, {
    lines: [
      {
        plants: [], model_range: "Q900V", parent: "Q900VV3005101_Q900", assembly: "Q900VV3005101",
        assembly_description: "HYD_SUPPLY BASIC_PARTS_CVT", alternative: "01", change_label: "Added",
        incoming: { number: "NEW-BASIC", description: "Basic supply tube" }, qualifiers: [],
      },
      {
        plants: [], model_range: "Q900V", parent: "Q900VV3005211_Q900", assembly: "Q900VV3005211",
        assembly_description: "HYD_SUPPLY LS_PUMP_160_CVT", alternative: "01", change_label: "Added",
        incoming: { number: "NEW-160", description: "160 CVT tube" }, qualifiers: [],
      },
      {
        plants: [], model_range: "Q900V", parent: "Q900VV3005101_Q900", assembly: "Q900VV3005101",
        assembly_description: "HYD_SUPPLY BASIC_PARTS_CVT", alternative: "01", change_label: "Removed",
        outgoing: { number: "OLD-BASIC", description: "Old basic supply tube" }, qualifiers: [],
      },
    ],
    materials: [],
  }, vi.fn());

  const groups = [...host.querySelectorAll(".bom-assembly-group")];
  expect(groups).toHaveLength(2);
  expect(groups[0].querySelector(".bom-assembly-group-header").textContent).toContain("2 affected material rows");
  expect(groups[0].querySelectorAll(".bom-change-row")).toHaveLength(2);
  expect(groups[0].textContent).toContain("NEW-BASIC");
  expect(groups[0].textContent).toContain("OLD-BASIC");
  expect(groups[1].querySelector(".bom-assembly-group-header").textContent).toContain("HYD_SUPPLY LS_PUMP_160_CVT");
});

it("keeps long BOM evidence in one aligned table", () => {
  const host = document.createElement("div");
  const lines = Array.from({ length: 6 }, (_, index) => ({
    plants: [{ plant_code: "PL10", plant_name: "Plant A" }], model_range: "Q900V", assembly: "Q900VV0080402",
    assembly_description: "M1V decal assembly", change_label: "Added", outgoing: null,
    incoming: { number: `NEW${index}`, description: `Decal ${index}`, date: "2026-05-19" }, levels: ["EBOM", "MBOM"], qualifiers: [],
  }));
  renderBomChanges(host, { lines, materials: [] }, vi.fn());

  expect(host.querySelectorAll("table")).toHaveLength(1);
  expect(host.querySelectorAll(".bom-assembly-group")).toHaveLength(1);
  expect(host.querySelectorAll(".bom-change-row")).toHaveLength(6);
  expect(host.querySelector("details")).toBeNull();
});

it("keeps a same-material quantity change in one material row", () => {
  const host = document.createElement("div");
  renderBomChanges(host, {
    lines: [{
      plants: [{ plant_code: "PL10", plant_name: "Plant A" }],
      assembly: "Q900VV0080402", assembly_description: "Decal assembly",
      change_label: "Quantity changed",
      outgoing: { number: "8241814T1", description: "Warning decal", quantity: "1", date: "2026-05-18" },
      incoming: { number: "8241814T1", description: "Warning decal", quantity: "2", date: "2026-05-19" },
      levels: ["EBOM", "MBOM"], qualifiers: [],
    }],
    materials: [],
  }, vi.fn());

  const rows = host.querySelectorAll(".bom-change-row");
  expect(rows).toHaveLength(1);
  expect(rows[0].textContent).toContain("8241814T1");
  expect(rows[0].querySelectorAll("td")[2].textContent).toBe("Unchanged");
});

it("summarizes plant scope in one line without procurement counts", () => {
  const host = document.createElement("div");
  renderPlantImpact(host, {
    plants: [
      { plant_code: "PK01", plant_name: "Plant B", assemblies: [{
        assembly: "Q900VV0080301", assembly_description: "M1V decal assembly", model_range: "Q900V", change_count: 1,
        effective_from: "2026-05-19", effective_to: "2026-05-19", bom_alignment: "ALIGNED", bom_alignment_label: "EBOM and MBOM aligned",
      }] },
      { plant_code: "PL10", plant_name: "Plant A", assemblies: [{
        assembly: "Q900VV0080402", assembly_description: "M1V safety decal assembly", model_range: "Q900V", change_count: 3,
        effective_from: "2026-05-18", effective_to: "2026-05-19", bom_alignment: "MBOM_PENDING", bom_alignment_label: "Manufacturing BOM update pending",
      }] },
    ],
  });

  expect(host.querySelector("h2")).toBeNull();
  expect(host.querySelectorAll("p")).toHaveLength(1);
  expect(host.textContent).toContain("Plant B (PK01): 1 affected assembly, 1 component change; BOM evidence aligned.");
  expect(host.textContent).toContain("Plant A (PL10): 1 affected assembly, 3 component changes; 1 BOM row needs alignment review.");
  expect(host.textContent).not.toContain("M1V decal assembly");
  expect(host.textContent).not.toContain("priced");
  expect(host.textContent).not.toContain("inventory disposition");
});

it("derives a described plant summary from a stored snapshot without assembly summaries", () => {
  const host = document.createElement("div");
  renderPlantImpact(host, {
    plants: [{ plant_code: "PL10", plant_name: "Plant A" }],
  }, {
    lines: [{
      plants: [{ plant_code: "PL10", plant_name: "Plant A" }], assembly: "Q900VV0080402",
      assembly_description: "M1V decal assembly", model_range: "Q900V", change: "ADDED",
      manufacturing_date: "2026-05-19", end_date: null, levels: ["EBOM", "MBOM"], qualifiers: [],
    }],
  });

  expect(host.textContent).toBe("Plant A (PL10): 1 affected assembly, 1 component change; BOM evidence aligned.");
});

it("renders five clickable decision signals with named plants and sourcing action", () => {
  const host = document.createElement("div");
  const onSelect = vi.fn();
  const snapshot = {
    plant_impact: {
      plants: [
        { plant_code: "PL10", plant_name: "Plant A", status: "CHECK", new_parts_without_purchasing: ["NEW1"], new_parts_with_purchasing: [], materials: [] },
        { plant_code: "PK01", plant_name: "Plant B", status: "CONFIRMED", new_parts_without_purchasing: [], new_parts_with_purchasing: [], materials: [] },
      ],
    },
    bom_changes: { lines: [{ qualifiers: ["MBOM_NOT_UPDATED"] }], materials: [{ status: "STILL_USED" }] },
    phase_schedule: [{ planner_confirmed: false }],
  };

  renderKpiHeader(host, computeKpis(snapshot), { onSelect });

  const tiles = [...host.querySelectorAll(".kpi-tile")];
  expect(tiles).toHaveLength(5);
  expect(tiles[0].textContent).toContain("PL10 Plant A · PK01 Plant B");
  expect(tiles[1].textContent).toContain("1 part needs sourcing");
  expect(tiles[4].textContent).toContain("2 plants need timing");
  tiles[1].click();
  expect(onSelect).toHaveBeenCalledWith("sourcing");
});

/* MOCK_DATA_RETIREMENT_VALIDATION_DISABLED_START: Procurement / Make-or-Buy renderer tests.
it("renders make-or-buy evidence for new and unchanged BOM materials", () => {
  const host = document.createElement("div");
  renderProcurement(host, {
    plant_impact: {
      plants: [
        {
          plant_code: "PL10", plant_name: "Plant A", new_parts_without_purchasing: [],
          materials: [
            { number: "DECAL", procurement_types: ["External procurement"], purchase_info_records: [{ supplier_name: "Plant A Decals", vendor: "V100", info_record: "I100" }] },
            { number: "NEW1", procurement_types: ["External procurement"], purchase_info_records: [{ supplier_name: "Sticker Supply", vendor: "V200", info_record: "I200" }] },
          ],
        },
        {
          plant_code: "PK01", plant_name: "Plant B", new_parts_without_purchasing: ["UNREC"],
          materials: [
            { number: "DECAL", procurement_types: ["In-house and external procurement allowed"], purchase_info_records: [{ supplier_name: "Plant B Decals", vendor: "V300", info_record: "I300" }] },
            { number: "UNREC", procurement_types: ["External procurement"], purchase_info_records: [] },
          ],
        },
      ],
    },
    bom_changes: {
      lines: [
        { plants: [{ plant_code: "PK01", plant_name: "Plant B" }, { plant_code: "PL10", plant_name: "Plant A" }], incoming: { number: "DECAL", description: "Diesel warning decal" }, levels: ["EBOM", "MBOM"] },
        { plants: [{ plant_code: "PL10", plant_name: "Plant A" }], incoming: { number: "NEW1", description: "JB greasing sticker" }, qualifiers: ["MBOM_NOT_UPDATED"] },
        { plants: [{ plant_code: "PK01", plant_name: "Plant B" }], outgoing: { number: "UNREC", description: "Unrecommended material" }, incoming: { number: "UNREC", description: "Unrecommended material" } },
        { plants: [{ plant_code: "PK01", plant_name: "Plant B" }], outgoing: { number: "REMOVED", description: "Removed only material" } },
      ],
      materials: [
        { number: "DECAL", description: "Diesel warning decal", status: "ALREADY_USED", status_label: "Existing part, already used elsewhere", prior_manufacturing_bom_use: true },
        { number: "NEW1", description: "JB greasing sticker", status: "NEW_PART", status_label: "New part" },
      ],
    },
    purchase_prices: [
      { material_number: "DECAL", plant_code: "PK01", net_price: "0.55", currency: "EUR", price_per: "1", unit_of_measure: "PC", status: "ACTIVE" },
      { material_number: "DECAL", plant_code: "PL10", net_price: "0.85", currency: "EUR", price_per: "1", unit_of_measure: "PC", status: "ACTIVE" },
    ],
    make_buy_recommendations: [
      {
        material_number: "DECAL", material_description: "Diesel warning decal", plant_code: "PK01", recommendation: "MAKE",
        recommendation_summary: "Make at Plant B because the local process is lower cost.", owner: "Manufacturing engineering",
        make_cost: "0.42", currency: "EUR", price_per: "1", unit_of_measure: "PC",
        comparable_history: "Equivalent decals are made at Plant B.", tooling: "Qualified decal process.",
        capacity: "Capacity is available.", raw_materials: "Stock is available.", next_action: "Approve the internal routing",
      },
      {
        material_number: "DECAL", material_description: "Diesel warning decal", plant_code: "PL10", recommendation: "BUY",
        recommendation_summary: "Buy at Plant A because the current price is lower.", owner: "Procurement",
        make_cost: "1.10", currency: "EUR", price_per: "1", unit_of_measure: "PC",
        comparable_history: "No local production history.", tooling: "Application only.",
        capacity: "No print capacity.", raw_materials: "No printed stock.", next_action: "Release the supplier setup",
      },
      {
        material_number: "NEW1", material_description: "JB greasing sticker", plant_code: "PL10", recommendation: "BUY_PENDING_QUOTE",
        recommendation_summary: "Buy pending a supplier quote.", owner: "Procurement",
        make_cost: "", currency: "EUR", price_per: "1", unit_of_measure: "PC",
        comparable_history: "Comparable stickers are bought.", tooling: "No print tooling.",
        capacity: "Application capacity is available.", raw_materials: "No printed stock.", next_action: "Obtain a supplier quote",
      },
    ],
  }, vi.fn());

  expect(host.querySelectorAll(".surface")).toHaveLength(1);
  expect(host.textContent).toContain("Make or Buy");
  expect(host.querySelector(".eyebrow")).toBeNull();
  expect(host.textContent).not.toContain("Recommendations weigh comparable manufacturing history");
  expect(host.textContent).toContain("Plant A (PL10): EUR 0.85 / PC");
  expect(host.textContent).toContain("Plant B (PK01): EUR 0.55 / PC");
  expect(host.textContent).toContain("Plant B (PK01): EUR 0.42 / PC");
  expect(host.textContent).toContain("Plant A (PL10): EUR 1.10 / PC");
  expect(host.textContent).toContain("manufacturing BOM update pending");
  expect(host.textContent).not.toContain("Removed only material");
  const headers = host.querySelectorAll("th");
  expect([...headers].map((header) => header.textContent)).toEqual([
    "Material", "Affected plants", "Purchasing evidence", "Mock make cost", "Material history", "Decision status",
  ]);
  const rows = [...host.querySelectorAll("tbody tr")];
  expect(rows).toHaveLength(3);
  const unrecommended = rows.find((row) => row.textContent.includes("Unrecommended material"));
  expect(unrecommended.textContent).toContain("No History");
  expect(unrecommended.textContent).toContain("Buy eligible in material master");
  expect([...host.querySelectorAll(".procurement-history-status")].map((status) => status.textContent)).toEqual(["Available", "New", "Available"]);
  expect(host.querySelectorAll(".procurement-basis")).toHaveLength(0);
  expect(host.querySelector(".material-link").textContent).toBe("Diesel warning decal");
});

it("aggregates purchase evidence for each material across its affected plants", () => {
  const host = document.createElement("div");
  renderProcurement(host, {
    plant_impact: {
      plants: [
        {
          plant_code: "PL10", plant_name: "Plant A", new_parts_with_purchasing: ["NEW1"], new_parts_without_purchasing: [],
          materials: [{ number: "NEW1", procurement_types: ["External procurement"] }],
        },
        {
          plant_code: "PK01", plant_name: "Plant B", new_parts_with_purchasing: ["NEW1"], new_parts_without_purchasing: [],
          materials: [{ number: "NEW1", procurement_types: ["In-house and external procurement allowed"] }],
        },
      ],
    },
    bom_changes: {
      lines: [{
        plants: [{ plant_code: "PL10", plant_name: "Plant A" }, { plant_code: "PK01", plant_name: "Plant B" }],
        incoming: { number: "NEW1", description: "Flexible shim" },
        levels: ["MBOM"],
      }],
      materials: [{ number: "NEW1", description: "Flexible shim", status: "NEW_PART" }],
    },
    purchase_prices: [
      { material_number: "NEW1", plant_code: "PL10", net_price: "0.85", currency: "EUR", price_per: "1", unit_of_measure: "PC", valid_to: "2026-12-31", status: "ACTIVE", source: "CURATED_DEMONSTRATION" },
      { material_number: "NEW1", plant_code: "PK01", net_price: "1.10", currency: "USD", price_per: "1", unit_of_measure: "PC", valid_to: "2026-06-30", status: "ACTIVE", source: "CURATED_DEMONSTRATION" },
    ],
    make_buy_recommendations: [
      {
        material_number: "NEW1", material_description: "Flexible shim", plant_code: "PL10", recommendation: "BUY",
        recommendation_summary: "Buy at Plant A.", owner: "Procurement", make_cost: "1.10", currency: "EUR", price_per: "1", unit_of_measure: "PC",
      },
      {
        material_number: "NEW1", material_description: "Flexible shim", plant_code: "PK01", recommendation: "MAKE",
        recommendation_summary: "Make at Plant B.", owner: "Manufacturing engineering", make_cost: "0.95", currency: "USD", price_per: "1", unit_of_measure: "PC",
      },
    ],
  }, vi.fn());

  const rows = [...host.querySelectorAll("tbody tr")];
  expect(rows).toHaveLength(1);
  expect(rows[0].textContent).toContain("Plant A (PL10)");
  expect(rows[0].textContent).toContain("Plant B (PK01)");
  expect(rows[0].textContent).toContain("EUR 0.85 / PC");
  expect(rows[0].textContent).toContain("USD 1.10 / PC");
  expect(rows[0].textContent).toContain("Plant A (PL10): No History");
  expect(rows[0].textContent).toContain("Plant B (PK01): No History");
  expect(rows[0].textContent).toContain("Make or buy eligible in material master");
});

it("generates an illustrative make cost only with prior manufacturing BOM evidence", () => {
  const host = document.createElement("div");
  renderProcurement(host, {
    plant_impact: { plants: [{ plant_code: "PL10", plant_name: "Plant A", materials: [{ number: "NEW1" }] }] },
    bom_changes: {
      lines: [{
        plants: [{ plant_code: "PL10", plant_name: "Plant A" }],
        incoming: { number: "NEW1", description: "Flexible shim" },
        levels: ["MBOM"],
      }],
      materials: [{ number: "NEW1", description: "Flexible shim", status: "ALREADY_USED", prior_manufacturing_bom_use: true }],
    },
    purchase_prices: [{ material_number: "NEW1", plant_code: "PL10", net_price: "8.50", currency: "EUR", price_per: "1", unit_of_measure: "PC" }],
  }, vi.fn());

  expect(host.textContent).toContain("Plant A (PL10): EUR 9.10 / PC");
});

it("shows missing manufacturing BOM evidence without removing the make-cost column", () => {
  const host = document.createElement("div");
  renderProcurement(host, {
    plant_impact: {
      plants: [{
        plant_code: "PL10", plant_name: "Plant A",
        materials: [{ number: "NEW1" }],
      }],
    },
    bom_changes: {
      lines: [{
        plants: [{ plant_code: "PL10", plant_name: "Plant A" }],
        incoming: { number: "NEW1", description: "Flexible shim" },
      }],
      materials: [{ number: "NEW1", description: "Flexible shim", status: "NEW_PART" }],
    },
  }, vi.fn());

  expect([...host.querySelectorAll("th")].map((header) => header.textContent)).toContain("Mock make cost");
  expect(host.textContent).toContain("Plant A (PL10): no manufacturing BOM record found");
});
MOCK_DATA_RETIREMENT_VALIDATION_DISABLED_END: Procurement / Make-or-Buy renderer tests. */

it("shows every snapshot part and uses only the shared Readiness count", () => {
  const host = document.createElement("div");
  const parts = Array.from({ length: 125 }, (_, index) => ({ number: `PART${index}`, action: "PHASE_IN", types: ["Part"], revisions: ["A"] }));
  renderChangeLists(host, { snapshot: { counts: { technical_objects: 125, attention_items: 1 }, questions: [{}], parts, programs: [] }, changes: [], warningCount: 99, inspect: vi.fn(), attentionHref: "/attention" });
  expect(host.querySelector("details").querySelectorAll("li")).toHaveLength(125);
  expect(host.querySelector(".attention-count").textContent).toBe("Items needing attention: 1");
  expect(host.textContent).toContain("New part");
  expect(host.textContent).toContain("revision A");
});

it("states the empty case precisely and omits BOM alignment qualifiers", () => {
  const host = document.createElement("div");
  renderBomChanges(host, { lines: [], materials: [] }, vi.fn());
  expect(host.textContent).toContain("No part row in the uploaded BOMs carries this ECN's change number");
  expect(host.textContent).toContain("Document rows are not BOM changes");
  renderBomChanges(host, {
    lines: [{
      plants: [{ plant_code: "PL10", plant_name: "Plant A" }], model_range: "Q900V", assembly: "Q900VV0170801",
      change: "ADDED", change_label: "Added", position: "0010", engineering_date: "2017-03-08",
      outgoing: null, incoming: { number: "X596516039000", quantity: "2", date: "2017-03-08" },
      levels: ["EBOM"], pairing_label: null, qualifiers: ["EBOM_130_SOURCE", "MBOM_NOT_UPDATED"],
    }],
    materials: [],
  }, vi.fn());
  const cells = [...host.querySelectorAll("tbody td")].map(cell => cell.textContent);
  expect(cells).toHaveLength(6);
  expect(cells[0]).toBe("X596516039000");
  expect(cells[1]).toBe("Not supplied");
  expect(cells[2]).toBe("New");
  expect(cells[3]).toContain("Assembly Q900VV0170801");
  expect(cells[4]).toBe("Plant A (PL10)");
  expect(host.textContent).not.toContain("130% EBOM source");
  expect(host.textContent).not.toContain("Manufacturing BOM update pending");
});
