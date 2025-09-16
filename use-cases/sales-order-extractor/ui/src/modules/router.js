import router from "page";
import { routes, aliases } from "../config/routes.js";

/**
 * PageRouter - Configuration-based routing system
 *
 * Features:
 * - Automatic route registration from src/config/routes.js
 * - Support for simple string routes and advanced route objects
 * - Route aliases for redirects and shortcuts
 * - Dynamic page loading with code splitting
 * - No manual router code editing required!
 *
 * To add a new route: Just add the page name to the routes array in src/config/routes.js
 */

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

    // Content is hidden by CSS initially - keep it hidden until first page loads

    // Auto-discover and register routes from pages directory
    await this.registerRoutes();

    // Handle 404s
    router("*", async () => await this.show404());

    // Start the router
    router.start();
  }

  async registerRoutes() {
    try {
      // Register root route to pdf-extraction
      router("/", () => this.loadPage("pdf-extraction"));

      // Register routes from configuration
      routes.forEach((route) => {
        if (typeof route === "string") {
          // Simple string route
          router(`/${route}`, () => this.loadPage(route));
        } else if (typeof route === "object" && route.path && route.page) {
          // Advanced route object - handle parameterized routes
          router(route.path, (ctx) => this.loadPage(route.page, ctx));
        }
      });

      // Register route aliases
      Object.entries(aliases).forEach(([alias, target]) => {
        router(alias, () => {
          const targetPage = target.replace("/", "");
          this.loadPage(targetPage);
        });
      });
    } catch (error) {
      console.error("Error registering routes:", error);
      // Fallback to pdf-extraction route
      router("/", () => this.loadPage("pdf-extraction"));
    }
  }

  async loadPage(pageName, ctx = null) {
    if (this.currentPage === pageName && !ctx) {
      return;
    }

    try {
      // Fade out current content for smooth transition
      this.contentContainer.style.opacity = "0";

      // Load CSS first (non-blocking)
      const cssPromise = this.loadCSS(pageName);

      // Phase 1: Import JavaScript module to register icons and components
      const jsModule = await this.loadAndExecuteJS(pageName, "register");

      // Wait for CSS to finish loading
      await cssPromise;

      // Now load HTML after JS has registered all icons/components
      const htmlContent = await this.loadHTML(pageName);

      // Update content container
      this.contentContainer.innerHTML = htmlContent;

      // Phase 2: Execute the initialization function now that DOM exists
      await this.executeJSInit(jsModule, pageName);

      // Brief pause to ensure styles are applied, then show content
      await new Promise((resolve) => setTimeout(resolve, 50));
      this.contentContainer.style.opacity = "1";

      this.currentPage = pageName;
      this.updateNavigationSelection(pageName);

      // Dispatch custom event
      window.dispatchEvent(
        new CustomEvent("pageChanged", {
          detail: { pageName, container: this.contentContainer }
        })
      );
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
      // Import CSS module (Vite will handle it)
      await import(`../pages/${pageName}/${pageName}.css`);
      this.loadedCSS.add(pageName);
    } catch (error) {
      console.warn(`⚠️ CSS not found for ${pageName}:`, error.message);
    }
  }

  async loadAndExecuteJS(pageName, phase = "register") {
    try {
      // Always import the module (this registers icons/components)
      const jsModule = await import(`../pages/${pageName}/${pageName}.js`);

      if (phase === "register") {
        // First phase: Module imported, icons/components registered
        return jsModule;
      } else if (phase === "init") {
        // Second phase: Execute the initialization function after HTML is loaded
        if (jsModule.default && typeof jsModule.default === "function") {
          jsModule.default();
        } else if (jsModule.init && typeof jsModule.init === "function") {
          jsModule.init();
        }
      }
    } catch (error) {
      console.warn(`⚠️ JS not found for ${pageName}:`, error.message);
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
      console.warn(`⚠️ Error initializing JS for ${pageName}:`, error.message);
    }
  }

  updateNavigationSelection(pageName) {
    const sideNavItems = document.querySelectorAll("ui5-side-navigation-item, ui5-side-navigation-sub-item");

    sideNavItems.forEach((item) => {
      const href = item.getAttribute("href");
      if (href === `#${pageName}` || href === `/${pageName}`) {
        item.selected = true;
      } else {
        item.selected = false;
      }
    });
  }

  async show404() {
    this.contentContainer.style.opacity = "0";

    this.contentContainer.innerHTML = `
      <div style="padding: 2rem; text-align: center;">
        <ui5-title level="H2">Page Not Found</ui5-title>
        <ui5-text>The requested page could not be found.</ui5-text>
        <ui5-button style="margin-top: 1rem;" onclick="pageRouter.navigate('/pdf-extraction')">
          Go to PDF Extraction
        </ui5-button>
      </div>
    `;

    // Show the 404 content
    await new Promise((resolve) => setTimeout(resolve, 50));
    this.contentContainer.style.opacity = "1";
  }

  // Public method to navigate programmatically
  navigate(path) {
    router(path);
  }

  // Get current route
  getCurrentRoute() {
    return router.current;
  }
}

// Create global instance
const pageRouter = new PageRouter();

export { pageRouter };
