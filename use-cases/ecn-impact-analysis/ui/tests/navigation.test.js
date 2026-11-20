import { expect, it, vi } from "vitest";
import { PageRouter } from "../src/modules/router.js";
import { ecnViews } from "../src/routes.js";
import { syncEcnNavigation } from "../src/modules/navigation.js";

it("cleans a stale awaited init and passes params, query, container, and signal", async () => {
  document.body.innerHTML = '<main class="content"></main>';
  let release;
  let markInitStarted;
  const initStarted = new Promise((resolve) => { markInitStarted = resolve; });
  const staleCleanup = vi.fn();
  const firstInit = vi.fn(() => new Promise((resolve) => {
    release = () => resolve(staleCleanup);
    markInitStarted();
  }));
  const secondInit = vi.fn(async () => vi.fn());
  const modules = { first: { init: firstInit }, second: { init: secondInit } };
  const pageRouter = new PageRouter({
    routeEngine: Object.assign(vi.fn(), { start: vi.fn() }),
    loadModule: async (name) => modules[name],
    loadHtml: async (name) => `<h1>${name}</h1>`,
    loadCss: async () => undefined,
  });
  pageRouter.contentContainer = document.querySelector(".content");

  const first = pageRouter.loadPage({ page: "first", title: "First" }, {
    params: { ecn_row_id: "row-1" }, querystring: "page=2",
  });
  await initStarted;
  const second = pageRouter.loadPage({ page: "second", title: "Second" }, {});
  release();
  await Promise.all([first, second]);

  expect(staleCleanup).toHaveBeenCalledOnce();
  expect(firstInit.mock.calls[0][0]).toEqual(expect.objectContaining({
    params: { ecn_row_id: "row-1" },
    query: expect.any(URLSearchParams),
    container: pageRouter.contentContainer,
    signal: expect.any(AbortSignal),
  }));
  expect(document.querySelector("h1").textContent).toBe("second");
});

it("retains ECN deep-link view switching without reloading after duplicate sidebar removal", async () => {
  document.body.innerHTML = '<ui5-side-navigation id="sn1"><ui5-side-navigation-item data-nav="ecns" href="/ecns"></ui5-side-navigation-item><ui5-side-navigation-item data-nav="settings" href="/settings"></ui5-side-navigation-item></ui5-side-navigation><main class="content"></main>';
  const showView = vi.fn();
  const cleanup = vi.fn();
  const initialize = vi.fn(async (context) => {
    context.onViewChange(showView);
    return cleanup;
  });
  const loadModule = vi.fn(async () => ({ init: initialize }));
  const loadHtml = vi.fn(async () => '<h1>Retained ECN content</h1>');
  const router = new PageRouter({
    routeEngine: Object.assign(vi.fn(), { start: vi.fn() }),
    loadModule,
    loadHtml,
    loadCss: async () => undefined,
  });
  router.contentContainer = document.querySelector(".content");
  router.onNavigate = (_definition, params) => syncEcnNavigation(params.ecn_row_id);
  const params = { ecn_row_id: "11111111-1111-4111-8111-111111111111" };
  await router.loadPage(ecnViews[0], { params });
  const content = document.querySelector("h1");
  for (const definition of ecnViews.slice(1)) {
    await router.loadPage(definition, { params });
    expect(showView).toHaveBeenLastCalledWith(definition.view);
    expect(document.querySelector("h1")).toBe(content);
    expect(document.querySelector(`[data-nav="${definition.nav}"]`)).toBeNull();
    expect(document.querySelector('[data-nav="ecn-overview"]').selected).toBe(false);
  }
  await router.loadPage(ecnViews[0], { params });
  expect(showView).toHaveBeenLastCalledWith("overview");
  expect(document.querySelector('[data-nav="ecn-overview"]').selected).toBe(true);
  expect(loadModule).toHaveBeenCalledOnce();
  expect(loadHtml).toHaveBeenCalledOnce();
  expect(initialize).toHaveBeenCalledOnce();
  expect(cleanup).not.toHaveBeenCalled();
  expect(router.pageController.signal.aborted).toBe(false);
  expect([...document.querySelectorAll("[data-nav]")].map((node) => node.dataset.nav)).toEqual(["ecns", "ecn-overview", "settings"]);
});
