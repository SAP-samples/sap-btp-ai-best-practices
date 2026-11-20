import routeEngine from "page";
import { aliases, routes } from "../routes.js";

const defaultLoaders = {
  loadModule: (name) => import(`../pages/${name}/${name}.js`),
  loadHtml: async (name) => (await import(`../pages/${name}/${name}.html?raw`)).default,
  loadCss: (name) => import(`../pages/${name}/${name}.css`),
};

/** Own route registration and exactly one active page lifecycle. */
export class PageRouter {
  /** Create a router with injectable route and asset loaders for deterministic tests. */
  constructor({
    routeEngine: engine = routeEngine,
    loadModule = defaultLoaders.loadModule,
    loadHtml = defaultLoaders.loadHtml,
    loadCss = defaultLoaders.loadCss,
  } = {}) {
    this.routeEngine = engine;
    this.loadModule = loadModule;
    this.loadHtml = loadHtml;
    this.loadCss = loadCss;
    this.contentContainer = null;
    this.navigationGeneration = 0;
    this.pageController = null;
    this.pageCleanup = null;
    this.activePage = null;
    this.activeParams = null;
    this.viewHandler = null;
    /** Optional shell hook `(definition, params) => void` run after every navigation. */
    this.onNavigate = null;
  }

  /** Register configured routes and begin same-tab routing. */
  async init(selector = ".content") {
    this.contentContainer = document.querySelector(selector);
    if (!this.contentContainer) throw new Error(`Content container "${selector}" not found`);
    for (const definition of routes) {
      this.routeEngine(definition.path, (context) => this.loadPage(definition, context));
    }
    for (const [alias, target] of Object.entries(aliases)) {
      this.routeEngine(alias, () => this.navigate(target));
    }
    this.routeEngine("*", () => this.show404());
    this.routeEngine.start();
  }

  /** Abort and clean the previous page, then initialize the newest navigation only. */
  async loadPage(definition, rawContext = {}) {
    const params = rawContext.params || {};
    // Same page and parameters, different section: scroll in place and keep page state.
    if (this.viewHandler && definition.page === this.activePage
        && JSON.stringify(params) === JSON.stringify(this.activeParams)) {
      ++this.navigationGeneration;
      this.viewHandler(definition.view);
      this.finishNavigation(definition, params);
      return;
    }
    const generation = ++this.navigationGeneration;
    this.viewHandler = null;
    this.activePage = null;
    this.pageController?.abort();
    if (this.pageCleanup) await this.pageCleanup();
    this.pageCleanup = null;
    const controller = new AbortController();
    this.pageController = controller;

    try {
      const [module, html] = await Promise.all([
        this.loadModule(definition.page),
        this.loadHtml(definition.page),
        this.loadCss(definition.page),
      ]);
      if (generation !== this.navigationGeneration) return;
      this.contentContainer.innerHTML = html;
      let viewHandler = null;
      const context = {
        params,
        view: definition.view,
        query: new URLSearchParams(rawContext.querystring || ""),
        container: this.contentContainer,
        navigate: (path) => this.navigate(path),
        signal: controller.signal,
        /** Let a multi-section page move to another section without re-initializing. */
        onViewChange: (handler) => { viewHandler = handler; },
      };
      const initialize = module.init ?? module.default;
      const cleanup = typeof initialize === "function" ? await initialize(context) : null;
      if (generation !== this.navigationGeneration || controller.signal.aborted) {
        if (typeof cleanup === "function") await cleanup();
        return;
      }
      this.pageCleanup = typeof cleanup === "function" ? cleanup : null;
      this.activePage = definition.page;
      this.activeParams = params;
      this.viewHandler = viewHandler;
      this.finishNavigation(definition, params);
    } catch (error) {
      if (error?.name !== "AbortError" && generation === this.navigationGeneration) this.show404();
    }
  }

  /** Update title, shell hook and navigation selection for the active route. */
  finishNavigation(definition, params) {
    document.title = `${definition.title} | ECN Impact Analysis`;
    this.onNavigate?.(definition, params);
    this.updateNavigationSelection(definition.nav || definition.page);
  }

  /** Render a safe not-found view without inline event handlers. */
  show404() {
    this.viewHandler = null;
    this.activePage = null;
    this.contentContainer.replaceChildren();
    const heading = document.createElement("h1");
    heading.textContent = "Page Not Found";
    const button = document.createElement("button");
    button.type = "button";
    button.textContent = "Go to ECNs";
    button.addEventListener("click", () => this.navigate("/"));
    this.contentContainer.append(heading, button);
  }

  /** Select the product navigation item matching the active route. */
  updateNavigationSelection(navKey) {
    document.querySelectorAll("[data-nav]").forEach((item) => {
      item.selected = item.dataset.nav === navKey;
    });
  }

  /** Navigate in the same browser tab. */
  navigate(path) {
    this.routeEngine(path);
  }
}

export const pageRouter = new PageRouter();
