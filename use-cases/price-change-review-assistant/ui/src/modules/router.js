import router from "page";
import { routes, aliases } from "../routes.js";

class PageRouter {
  constructor() {
    this.contentContainer = null;
    this.loadedCSS = new Set();
    this.currentPage = null;
  }

  async init(contentContainerSelector = ".content") {
    this.contentContainer = document.querySelector(contentContainerSelector);
    if (!this.contentContainer) {
      throw new Error(`Content container "${contentContainerSelector}" not found`);
    }

    await this.registerRoutes();

    router("*", async () => await this.show404());

    router.start();
  }

  async registerRoutes() {
    try {
      router("/", () => this.loadPage("home"));

      routes.forEach((route) => {
        if (typeof route === "string") {
          router(`/${route}`, () => this.loadPage(route));
        } else if (typeof route === "object" && route.path && route.page) {
          router(route.path, (ctx) => this.loadPage(route.page, ctx));
        }
      });

      Object.entries(aliases).forEach(([alias, target]) => {
        router(alias, () => {
          const targetPage = target.replace("/", "");
          this.loadPage(targetPage);
        });
      });
    } catch (error) {
      console.error("Error registering routes:", error);
      router("/", () => this.loadPage("home"));
    }
  }

  async loadPage(pageName, ctx = null) {
    if (this.currentPage === pageName && !ctx) {
      return;
    }

    try {
      this.contentContainer.style.opacity = "0";

      const cssPromise = this.loadCSS(pageName);

      const jsModule = await this.loadAndExecuteJS(pageName, "register");

      await cssPromise;

      const htmlContent = await this.loadHTML(pageName);

      this.contentContainer.innerHTML = htmlContent;

      await this.executeJSInit(jsModule, pageName);

      await new Promise((resolve) => setTimeout(resolve, 50));
      this.contentContainer.style.opacity = "1";

      this.currentPage = pageName;

      window.dispatchEvent(
        new CustomEvent("pageChanged", {
          detail: { pageName, container: this.contentContainer }
        })
      );
      document.querySelectorAll(".side-nav-link").forEach((link) => {
        link.classList.toggle("is-active", link.dataset.route === pageName);
      });
    } catch (error) {
      console.error(`Error loading page ${pageName}:`, error);
      this.show404();
    }
  }

  async loadHTML(pageName) {
    try {
      const htmlModule = await import(`../pages/${pageName}/${pageName}.html?raw`);
      return htmlModule.default;
    } catch (error) {
      throw new Error(`Failed to load HTML for ${pageName}: ${error.message}`);
    }
  }

  async loadCSS(pageName) {
    if (this.loadedCSS.has(pageName)) {
      return;
    }

    try {
      await import(`../pages/${pageName}/${pageName}.css`);
      this.loadedCSS.add(pageName);
    } catch (error) {
      console.warn(`CSS not found for ${pageName}:`, error.message);
    }
  }

  async loadAndExecuteJS(pageName, phase = "register") {
    try {
      const jsModule = await import(`../pages/${pageName}/${pageName}.js`);

      if (phase === "register") {
        return jsModule;
      } else if (phase === "init") {
        if (jsModule.default && typeof jsModule.default === "function") {
          jsModule.default();
        } else if (jsModule.init && typeof jsModule.init === "function") {
          jsModule.init();
        }
      }
    } catch (error) {
      console.warn(`JS not found for ${pageName}:`, error.message);
      return null;
    }
  }

  async executeJSInit(jsModule, pageName) {
    if (!jsModule) return;

    try {
      if (jsModule.default && typeof jsModule.default === "function") {
        jsModule.default();
      } else if (jsModule.init && typeof jsModule.init === "function") {
        jsModule.init();
      }
    } catch (error) {
      console.warn(`Error initializing JS for ${pageName}:`, error.message);
    }
  }

  async show404() {
    this.contentContainer.style.opacity = "0";

    this.contentContainer.innerHTML = `
      <div style="padding: 2rem; text-align: center;">
        <ui5-title level="H2">Page Not Found</ui5-title>
        <ui5-text>The requested page could not be found.</ui5-text>
        <ui5-button style="margin-top: 1rem;" onclick="pageRouter.navigate('/home')">
          Go to Home
        </ui5-button>
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
