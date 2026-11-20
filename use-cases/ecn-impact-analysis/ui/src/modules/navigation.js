import NavigationLayoutMode from "@ui5/webcomponents-fiori/dist/types/NavigationLayoutMode.js";
import { pageRouter } from "./router.js";
import { ecnViews } from "../routes.js";

/** Attach shell expansion and same-tab navigation listeners and return their cleanup. */
export function handleNavigation() {
  const layout = document.querySelector("#nl1");
  const startButton = document.querySelector("#startButton");
  const sideNavigation = document.querySelector("#sn1");

  const toggleNavigation = () => {
    layout.mode = layout.isSideCollapsed()
      ? NavigationLayoutMode.Expanded
      : NavigationLayoutMode.Collapsed;
  };
  /** Cancel UI5's native anchor navigation so the router preserves the active ECN. */
  const navigateToSelection = (event) => {
    const item = event.target.closest?.("[data-nav]");
    if (!item || item.getAttribute("target")) return;
    const modifiers = event.detail || {};
    if (modifiers.ctrlKey || modifiers.metaKey || modifiers.shiftKey || modifiers.altKey) return;
    const href = item.getAttribute("href");
    if (!href) return;
    event.preventDefault();
    pageRouter.navigate(href);
  };

  startButton.addEventListener("click", toggleNavigation);
  sideNavigation.addEventListener("ui5-click", navigateToSelection);
  return () => {
    startButton.removeEventListener("click", toggleNavigation);
    sideNavigation.removeEventListener("ui5-click", navigateToSelection);
  };
}

/** Create the side-navigation item for one ECN view route (text, icon and `nav` key). */
function createViewItem(view) {
  const item = document.createElement("ui5-side-navigation-item");
  item.setAttribute("text", view.title);
  item.setAttribute("icon", view.icon);
  item.dataset.nav = view.nav;
  return item;
}

/** Keep only ECN Overview immediately after "ECNs" while an ECN is open.
 *
 * Input: the open ECN row ID, or null/undefined on any other page.
 * Impact, attention and references remain detail-page views/deep links, not
 * duplicate sidebar shortcuts. Reconcile on every call, including the same ECN,
 * to remove stale shortcuts. The overview retains its route's selection key.
 */
export function syncEcnNavigation(ecnRowId) {
  const ecnsItem = document.querySelector('ui5-side-navigation-item[data-nav="ecns"]');
  if (!ecnsItem) return;
  ecnsItem.dataset.ecnRowId = ecnRowId || "";
  const base = ecnRowId ? `/ecns/${encodeURIComponent(ecnRowId)}` : null;
  const sidebarViews = ecnViews.filter((view) => view.view === "overview");
  ecnViews.filter((view) => !base || view.view !== "overview").forEach((view) => {
    document.querySelectorAll(`ui5-side-navigation-item[data-nav="${view.nav}"]`).forEach((item) => item.remove());
  });
  if (!base) return;
  const items = sidebarViews.map(view => document.querySelector(`ui5-side-navigation-item[data-nav="${view.nav}"]`) || createViewItem(view));
  ecnsItem.after(...items);
  sidebarViews.forEach((view, index) => {
    const item = items[index];
    item.removeAttribute("disabled");
    item.removeAttribute("tooltip");
    item.setAttribute("href", base + view.path.replace("/ecns/:ecn_row_id", ""));
  });
}
