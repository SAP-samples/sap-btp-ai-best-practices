import router from "page";
import { routes, aliases } from "../routes.js";

const PAGE_LOADERS = {
  "purchase-requisition": {
    html: () => import("../pages/purchase-requisition/purchase-requisition.html?raw"),
    css: () => import("../pages/purchase-requisition/purchase-requisition.css"),
    js: () => import("../pages/purchase-requisition/purchase-requisition.js")
  },
  extraction: {
    html: () => import("../pages/extraction/extraction.html?raw"),
    css: () => import("../pages/extraction/extraction.css"),
    js: () => import("../pages/extraction/extraction.js")
  }
};

class PageRouter {
  constructor() {
    this.contentContainer = null;
    this.loadedCSS = new Set();
    this.currentPage = null;
  }

  async init(contentContainerSelector = ".content") {
    this.contentContainer = document.querySelector(contentContainerSelector);
    if (!this.contentContainer) throw new Error(`Content container "${contentContainerSelector}" not found`);
    await this.registerRoutes();
    router("*", async () => await this.show404());
    router.start();
  }

  async registerRoutes() {
    router("/", () => this.loadPage("purchase-requisition"));
    routes.forEach((route) => {
      if (typeof route === "string") router(`/${route}`, () => this.loadPage(route));
      else if (typeof route === "object" && route.path && route.page) router(route.path, (ctx) => this.loadPage(route.page, ctx));
    });
    Object.entries(aliases).forEach(([alias, target]) => {
      router(alias, () => this.loadPage(target.replace("/", "")));
    });
  }

  async loadPage(pageName, ctx = null) {
    if (this.currentPage === pageName && !ctx) return;
    try {
      this.contentContainer.style.opacity = "0";
      const cssPromise = this.loadCSS(pageName);
      const jsModule = await this.loadAndExecuteJS(pageName);
      await cssPromise;
      const htmlContent = await this.loadHTML(pageName);
      this.contentContainer.innerHTML = htmlContent;
      await this.executeJSInit(jsModule);
      await new Promise((resolve) => setTimeout(resolve, 50));
      this.contentContainer.style.opacity = "1";
      this.currentPage = pageName;
      this.updateNavigationSelection(pageName);
      window.dispatchEvent(new CustomEvent("pageChanged", { detail: { pageName, container: this.contentContainer } }));
    } catch (error) {
      console.error(`Error loading page ${pageName}:`, error);
      this.show404();
    }
  }

  loader(pageName) {
    const loader = PAGE_LOADERS[pageName];
    if (!loader) throw new Error(`Unknown page "${pageName}"`);
    return loader;
  }

  async loadHTML(pageName) {
    const module = await this.loader(pageName).html();
    return module.default;
  }

  async loadCSS(pageName) {
    if (this.loadedCSS.has(pageName)) return;
    await this.loader(pageName).css();
    this.loadedCSS.add(pageName);
  }

  async loadAndExecuteJS(pageName) {
    return this.loader(pageName).js();
  }

  async executeJSInit(jsModule) {
    if (jsModule?.default && typeof jsModule.default === "function") jsModule.default();
    else if (jsModule?.init && typeof jsModule.init === "function") jsModule.init();
  }

  updateNavigationSelection(pageName) {
    document.querySelectorAll("ui5-side-navigation-item, ui5-side-navigation-sub-item").forEach((item) => {
      const href = item.getAttribute("href");
      item.selected = href === `#${pageName}` || href === `/${pageName}`;
    });
  }

  async show404() {
    this.contentContainer.style.opacity = "0";
    this.contentContainer.innerHTML = `
      <div style="padding: 2rem; text-align: center;">
        <ui5-title level="H2">Page Not Found</ui5-title>
        <ui5-text>The requested page could not be found.</ui5-text>
        <ui5-button style="margin-top: 1rem;" onclick="pageRouter.navigate('/purchase-requisition')">Go to Upload</ui5-button>
      </div>
    `;
    await new Promise((resolve) => setTimeout(resolve, 50));
    this.contentContainer.style.opacity = "1";
  }

  navigate(path) {
    router(path);
  }

  getCurrentRoute() {
    return router.current;
  }
}

const pageRouter = new PageRouter();
export { pageRouter };
