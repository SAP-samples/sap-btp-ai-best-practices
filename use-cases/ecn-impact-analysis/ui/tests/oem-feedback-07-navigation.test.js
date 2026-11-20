import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { handleNavigation, syncEcnNavigation } from "../src/modules/navigation.js";
import { PageRouter, pageRouter } from "../src/modules/router.js";
import { aliases, ecnViews } from "../src/routes.js";
import shellHtml from "../index.html?raw";
import detailHtml from "../src/pages/ecn-detail/ecn-detail.html?raw";

const rowId = "11111111-1111-4111-8111-111111111111";
const removedKeys = ["ecn-impact", "ecn-attention", "ecn-references"];
const item = (key) => document.querySelector(`ui5-side-navigation-item[data-nav="${key}"]`);
const keys = () => [...document.querySelectorAll("#sn1 ui5-side-navigation-item")].map((node) => node.dataset.nav);

function mountShell() {
  const source = new DOMParser().parseFromString(shellHtml, "text/html");
  document.body.append(document.importNode(source.querySelector("#app"), true));
}

function insertStaleShortcuts() {
  removedKeys.forEach((key) => {
    // Two copies prove reconciliation removes all obsolete items, not just the first.
    for (let count = 0; count < 2; count += 1) {
      const node = document.createElement("ui5-side-navigation-item");
      node.dataset.nav = key;
      node.setAttribute("href", `/ecns/${rowId}/stale`);
      node.selected = true;
      item("ecns").after(node);
    }
  });
}

beforeEach(() => { document.body.replaceChildren(); mountShell(); });
afterEach(() => { vi.restoreAllMocks(); document.body.replaceChildren(); });

describe("feedback 07: only the three duplicate ECN sidebar shortcuts are removed", () => {
  it("retains the shipped ECNs and Settings items and inserts one named Overview between them", () => {
    const ecns = item("ecns");
    const settings = item("settings");
    expect(keys()).toEqual(["ecns", "settings"]);
    syncEcnNavigation(rowId);
    expect(keys()).toEqual(["ecns", "ecn-overview", "settings"]);
    expect(item("ecns")).toBe(ecns);
    expect(item("settings")).toBe(settings);
    expect(ecns.getAttribute("href")).toBe("/ecns");
    expect(settings.getAttribute("href")).toBe("/settings");
    expect(settings.getAttribute("icon")).toBe("action-settings");
    expect(item("ecn-overview").getAttribute("text")).toBe("ECN Overview");
    expect(item("ecn-overview").getAttribute("icon")).toBe("detail-view");
    expect(item("ecn-overview").getAttribute("href")).toBe(`/ecns/${rowId}`);
    expect(document.querySelector("#assistant-shell-action").getAttribute("accessible-name")).toBe("Open ECN chat");
  });

  it("cleans stale shortcuts even for the same ECN and retains the existing overview identity/selection", () => {
    syncEcnNavigation(rowId);
    const overview = item("ecn-overview");
    overview.selected = true;
    overview.setAttribute("disabled", "");
    overview.setAttribute("tooltip", "Open an ECN to use this page");
    insertStaleShortcuts();
    syncEcnNavigation(rowId);
    syncEcnNavigation(rowId);
    expect(keys()).toEqual(["ecns", "ecn-overview", "settings"]);
    expect(item("ecn-overview")).toBe(overview);
    expect(overview.selected).toBe(true);
    expect(overview.hasAttribute("disabled")).toBe(false);
    expect(overview.hasAttribute("tooltip")).toBe(false);
  });

  it("restores a missing overview for the same ECN instead of skipping reconciliation", () => {
    item("ecns").dataset.ecnRowId = rowId;
    insertStaleShortcuts();
    syncEcnNavigation(rowId);
    expect(keys()).toEqual(["ecns", "ecn-overview", "settings"]);
    expect(item("ecn-overview").getAttribute("href")).toBe(`/ecns/${rowId}`);
  });

  it("updates the retained overview href when ECN identity changes and safely encodes the path segment", () => {
    syncEcnNavigation(rowId);
    const overview = item("ecn-overview");
    const nextId = 'synthetic /?#<>&"';
    syncEcnNavigation(nextId);
    expect(item("ecn-overview")).toBe(overview);
    expect(overview.getAttribute("href")).toBe(`/ecns/${encodeURIComponent(nextId)}`);
    expect(item("ecns").dataset.ecnRowId).toBe(nextId);
    expect(keys()).toEqual(["ecns", "ecn-overview", "settings"]);
    expect(document.querySelector("script")).toBeNull();
  });

  it.each([null, undefined, ""])("removes every ECN item on leaving (%s), including stale same-empty-row items", (emptyId) => {
    syncEcnNavigation(rowId);
    syncEcnNavigation(emptyId);
    insertStaleShortcuts();
    const staleOverview = document.createElement("ui5-side-navigation-item");
    staleOverview.dataset.nav = "ecn-overview";
    item("ecns").after(staleOverview);
    syncEcnNavigation(emptyId);
    expect(keys()).toEqual(["ecns", "settings"]);
    expect(item("ecns").dataset.ecnRowId).toBe("");
  });

  it("does nothing safely if the shell ECNs anchor has not mounted", () => {
    document.body.replaceChildren();
    expect(() => syncEcnNavigation(rowId)).not.toThrow();
    expect(document.body.children).toHaveLength(0);
  });

  it("leaves all four route definitions and Overview/Plants/BOM/Audit main-page navigation intact", () => {
    expect(ecnViews.map(({ path, view, nav }) => ({ path, view, nav }))).toEqual([
      { path: "/ecns/:ecn_row_id", view: "overview", nav: "ecn-overview" },
      { path: "/ecns/:ecn_row_id/impact", view: "impact", nav: "ecn-impact" },
      { path: "/ecns/:ecn_row_id/attention", view: "attention", nav: "ecn-attention" },
      { path: "/ecns/:ecn_row_id/references", view: "references", nav: "ecn-references" },
    ]);
    expect(aliases).toEqual({ "/": "/ecns", "/home": "/ecns" });
    const main = document.createElement("main");
    main.innerHTML = detailHtml;
    const navigation = main.querySelector('.ecn-section-navigation[aria-label="ECN sections"]');
    expect([...navigation.querySelectorAll("button")].map((node) => node.dataset.sectionTarget)).toEqual(["overview", "plants", "bom", "audit"]);
    ["source-record", "affected-sites", "bom-changes", "audit-log"].forEach((id) => expect(main.querySelector(`#${id}`)).not.toBeNull());
  });

  it.each(["ecns", "ecn-overview", "settings"])("selects only the existing %s shell item using the unchanged router", (navKey) => {
    syncEcnNavigation(rowId);
    const router = new PageRouter();
    router.updateNavigationSelection(navKey);
    expect([...document.querySelectorAll("[data-nav]")].filter((node) => node.selected).map((node) => node.dataset.nav)).toEqual([navKey]);
  });

  it.each(removedKeys)("does not falsely highlight overview for a retained %s deep link", (navKey) => {
    syncEcnNavigation(rowId);
    const router = new PageRouter();
    router.updateNavigationSelection("ecn-overview");
    router.updateNavigationSelection(navKey);
    expect([...document.querySelectorAll("[data-nav]")].filter((node) => node.selected)).toEqual([]);
    expect(keys()).toEqual(["ecns", "ecn-overview", "settings"]);
  });

  it("keeps menu expansion, same-tab overview navigation, and listener cleanup", () => {
    syncEcnNavigation(rowId);
    const layout = document.querySelector("#nl1");
    layout.isSideCollapsed = () => layout.mode !== "Expanded";
    const navigate = vi.spyOn(pageRouter, "navigate").mockImplementation(() => {});
    const cleanup = handleNavigation();
    const menu = document.querySelector("#startButton");
    menu.click();
    expect(layout.mode).toBe("Expanded");
    menu.click();
    expect(layout.mode).toBe("Collapsed");
    const event = new CustomEvent("ui5-click", { bubbles: true, cancelable: true });
    item("ecn-overview").dispatchEvent(event);
    expect(event.defaultPrevented).toBe(true);
    expect(navigate).toHaveBeenCalledExactlyOnceWith(`/ecns/${rowId}`);
    cleanup();
    menu.click();
    item("ecn-overview").dispatchEvent(new CustomEvent("ui5-click", { bubbles: true }));
    expect(layout.mode).toBe("Collapsed");
    expect(navigate).toHaveBeenCalledTimes(1);
  });

  it.each(["ctrlKey", "metaKey", "shiftKey", "altKey"])("preserves native modified navigation (%s)", (modifier) => {
    syncEcnNavigation(rowId);
    const navigate = vi.spyOn(pageRouter, "navigate").mockImplementation(() => {});
    const cleanup = handleNavigation();
    const event = new CustomEvent("ui5-click", { bubbles: true, cancelable: true, detail: { [modifier]: true } });
    item("ecn-overview").dispatchEvent(event);
    expect(event.defaultPrevented).toBe(false);
    expect(navigate).not.toHaveBeenCalled();
    cleanup();
  });

  it("keeps explicit new-tab targets native", () => {
    syncEcnNavigation(rowId);
    item("ecn-overview").setAttribute("target", "_blank");
    const navigate = vi.spyOn(pageRouter, "navigate").mockImplementation(() => {});
    const cleanup = handleNavigation();
    const event = new CustomEvent("ui5-click", { bubbles: true, cancelable: true });
    item("ecn-overview").dispatchEvent(event);
    expect(event.defaultPrevented).toBe(false);
    expect(navigate).not.toHaveBeenCalled();
    cleanup();
  });
});
