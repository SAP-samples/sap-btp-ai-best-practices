/** Product routes available after the historical ECN page is installed.
 *
 * The four ECN routes share one page module; `view` identifies the section to
 * scroll to, so switching between them keeps loaded data and polling.
 */
export const routes = [
  { path: "/ecns", page: "ecns", title: "ECNs", nav: "ecns" },
  { path: "/ecns/:ecn_row_id", page: "ecn-detail", view: "overview", title: "ECN Overview", nav: "ecn-overview", icon: "detail-view" },
  { path: "/ecns/:ecn_row_id/impact", page: "ecn-detail", view: "impact", title: "Impact Analysis", nav: "ecn-impact", icon: "chart-table-view" },
  { path: "/ecns/:ecn_row_id/attention", page: "ecn-detail", view: "attention", title: "What Needs Attention", nav: "ecn-attention", icon: "alert" },
  { path: "/ecns/:ecn_row_id/references", page: "ecn-detail", view: "references", title: "Reference Documents", nav: "ecn-references", icon: "documents" },
  { path: "/settings", page: "settings", title: "Settings", nav: "settings" },
];

/** Top-level side-navigation items for one ECN, shown only while an ECN is open. */
export const ecnViews = routes.filter(route => route.page === "ecn-detail");

/** Legacy entry points redirect to the ECN history. */
export const aliases = { "/": "/ecns", "/home": "/ecns" };
