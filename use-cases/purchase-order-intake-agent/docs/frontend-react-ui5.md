# Frontend: React + UI5 Web Components for React

## What it is

The `ui/` application is a React 19 + TypeScript single-page app built with Vite and
[UI5 Web Components for React](https://ui5.github.io/webcomponents-react/v2/?path=/docs/getting-started--docs)
(v2). It replaces the earlier plain-JavaScript UI5 Web Components frontend and keeps the same
capabilities: Fiori shell (ShellBar + collapsible SideNavigation), client-side routing with
lazy-loaded pages, a typed HTTP client with the shared API key, and a page generator script.

It hosts the purchase-order intake screens (orders list, upload dialog, review page); see
[ui-order-intake.md](ui-order-intake.md).

## What it is for

A starting point for SAP BTP PoCs that need a Fiori-looking UI on top of a FastAPI backend,
with React's component model and TypeScript types for every UI5 component prop and event.

## Inputs and outputs

| Input | Where | Purpose |
|---|---|---|
| `VITE_API_BASE_URL` | `ui/.env` locally; on CF `deploy.sh` reads the API's `API_BASE_URL` from `manifest.yaml` | Backend base URL (default `http://127.0.0.1:8000`) |
| `VITE_API_KEY` | `ui/.env` locally, injected by `deploy.sh` on CF | Sent as `X-API-Key`; must equal `API_KEY` of the API |

Output: a static build in `ui/dist/`, built locally by `deploy.sh` and served on Cloud Foundry by the
`staticfile_buildpack` (nginx, 64M; see `manifest.yaml`). `ui/public/Staticfile` (copied into `dist/`)
enables `pushstate` so deep links handled by `BrowserRouter` fall back to `index.html`.

## Related files

```
ui/
├── index.html                    # <div id="root"> + module script only
├── vite.config.ts                # React plugin
├── tsconfig.json                 # strict TS, checks src/
├── scripts/create-page.js        # page generator
└── src/
    ├── main.tsx                  # assets/icons, ThemeProvider, BrowserRouter, AppUiProvider
    ├── App.tsx                   # NavigationLayout: ShellBar, SideNavigation, Routes
    ├── routes.tsx                # route registry (single source of truth)
    ├── state/AppUiContext.tsx    # side nav collapsed, theme
    ├── services/api.ts           # request(), requestForm(), requestBlob(), ApiError
    ├── services/orders.ts        # typed purchase-order API calls
    └── pages/
        ├── NotFoundPage.tsx
        └── orders/               # purchase-order intake (see ui-order-intake.md)
```

## Key concepts and snippets

### Route registry drives everything

Every page is one entry in `src/routes.tsx`. `App.tsx` turns the list into `<Route>`s **and**
`SideNavigationItem`s. No other file needs editing when a page is added.

```tsx
export const routes: AppRoute[] = [
  // create-page:routes-start
  { path: "/orders", title: "Purchase Orders", icon: "sales-order", Component: lazy(() => import("./pages/orders/OrdersPage")) },
  // create-page:routes-end
];

export const aliases: Record<string, string> = { "/": DEFAULT_PATH };
```

Pages are `React.lazy` imports, so each page is its own JS chunk. Unknown paths render
`NotFoundPage`.

### Shared UI state

`AppUiContext` holds the shell state: the ShellBar menu button calls `toggleSideNav`; the theme
is applied to the UI5 runtime with `setTheme` from `@ui5/webcomponents-base`.

```tsx
const { sideNavCollapsed, toggleSideNav, theme, setTheme } = useAppUi();
```

Supported themes: `sap_horizon`, `sap_horizon_dark`, `sap_horizon_hcb`, `sap_horizon_hcw`
(`SUPPORTED_THEMES`).

### Calling the backend

```ts
import { request } from "../../services/api";

const health = await request("/api/health");
const orders = await request<OrderListResponse>("/api/orders");
```

`requestForm(endpoint, formData)` posts multipart uploads and `requestBlob(endpoint)` downloads
files with the API key (needed for the document viewer, because an `<iframe>` cannot send headers).
Non-2xx responses throw `ApiError` with `status` and the parsed FastAPI `detail` (e.g. the issue list
of a 422 confirm).

### Icons

`main.tsx` registers all icons of `@ui5/webcomponents-icons`, `-icons-tnt` and
`-icons-business-suite`. This is deliberate: icon names are chosen at runtime (route registry,
status icons), so they cannot be imported one by one. It is the main reason for
the large main chunk reported by `vite build`.

## Adding a page

```bash
cd ui
npm run create-page products "Product Catalog" product
```

This creates `src/pages/products/ProductsPage.tsx` and `products.css` and inserts a route entry
between the `create-page` markers in `routes.tsx`. The side navigation item appears automatically.
Manual creation works the same way: add a component with a default export and one entry in
`routes`.

Use the UI5 Web Components for React MCP server (`.mcp.json` at the repo root) to look up
component props, events and slots while building pages.

## Cloud Foundry note

The UI is not built on Cloud Foundry. `deploy.sh` runs `npm run build` (`tsc -b && vite build`)
locally after generating the deployment API key, with `VITE_API_KEY` and `VITE_API_BASE_URL` set, and
`cf push` uploads only `ui/dist/` to the staticfile buildpack. Building on CF at start-up needed
more memory than the container had (exit 137) and repeated the build on every restart.
Both `VITE_*` values are compiled into the public JS bundle, so the API key is visible to anyone
who can load the UI.

## How to test

```bash
cd ui
npm install
npm run typecheck     # TypeScript
npm run build         # production build
npm run dev           # http://localhost:5173 (run the API on :8000 in parallel)
```

Manual checks: `/` redirects to `/orders`; the menu button expands/collapses the side navigation;
an unknown URL shows "Page Not Found" with a working "Go to Purchase Orders" button; back/forward
browser buttons work.
