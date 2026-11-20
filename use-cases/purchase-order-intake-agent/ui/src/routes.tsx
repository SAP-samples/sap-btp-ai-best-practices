/**
 * Route registry - the single source of truth for pages.
 *
 * Every entry here automatically registers a <Route> in App.tsx and renders a
 * SideNavigationItem in the side navigation.
 *
 * Pages are lazy-loaded (`React.lazy`), so each page becomes its own JS chunk.
 *
 * Add pages with `npm run create-page <name> "<Title>" [icon]`, which inserts an entry
 * between the `create-page` marker comments below. Manual edits work the same way.
 */
import { lazy, type ComponentType, type LazyExoticComponent } from "react";

/** One navigable page of the application. */
export interface AppRoute {
  /** URL path, e.g. "/orders" */
  path: string;
  /** Label used in the side navigation */
  title: string;
  /** UI5 icon name (any icon from the registered collections) */
  icon: string;
  /** Lazily loaded page component (default export of the page module) */
  Component: LazyExoticComponent<ComponentType>;
}

export const routes: AppRoute[] = [
  // create-page:routes-start
  { path: "/orders", title: "Purchase Orders", icon: "sales-order", Component: lazy(() => import("./pages/orders/OrdersPage")) },
  // create-page:routes-end
];

/** Pages reached by navigation only (not shown in the side navigation). */
export const detailRoutes: { path: string; Component: LazyExoticComponent<ComponentType> }[] = [
  { path: "/orders/:orderId", Component: lazy(() => import("./pages/order-review/OrderReviewPage")) }
];

/** Path the app opens on and where unknown aliases fall back to. */
export const DEFAULT_PATH = "/orders";

/** Redirects: key path is redirected to value path. */
export const aliases: Record<string, string> = {
  "/": DEFAULT_PATH
};
