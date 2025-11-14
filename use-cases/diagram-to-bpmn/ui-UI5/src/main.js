import "@ui5/webcomponents-icons/dist/AllIcons.js";
import "@ui5/webcomponents-icons-tnt/dist/AllIcons.js";
import "@ui5/webcomponents-icons-business-suite/dist/AllIcons.js";

/* ShellBar */
import "@ui5/webcomponents/dist/Button.js";

import "@ui5/webcomponents-fiori/dist/ShellBar.js";
import "@ui5/webcomponents-fiori/dist/ShellBarBranding.js";

/* Navigation Layout */
import "@ui5/webcomponents/dist/Text.js";
import "@ui5/webcomponents/dist/Title.js";

import "@ui5/webcomponents-fiori/dist/NavigationLayout.js";
import "@ui5/webcomponents-fiori/dist/SideNavigation.js";
import "@ui5/webcomponents-fiori/dist/SideNavigationGroup.js";
import "@ui5/webcomponents-fiori/dist/SideNavigationItem.js";
import "@ui5/webcomponents-fiori/dist/SideNavigationSubItem.js";

/* Core UI5 components used across multiple pages */
import "@ui5/webcomponents/dist/Icon.js";

/* App Modules */
import { handleNavigation } from "./modules/navigation.js";
import { pageRouter } from "./modules/router.js";

document.addEventListener("DOMContentLoaded", async () => {
  try {
    // Make pageRouter globally accessible for inline onclick handlers
    window.pageRouter = pageRouter;

    // Initialize router with auto-discovery
    await pageRouter.init(".content");

    // Initialize navigation
    handleNavigation();

    // Wait a moment for UI5 components to fully register and render
    await new Promise((resolve) => setTimeout(resolve, 100));

    // Show the application - this will hide the loader and fade in the app
    const app = document.getElementById("app");
    if (app) {
      app.classList.add("ready");
    }

    console.log("✅ Application ready");
  } catch (error) {
    console.error("❌ Error initializing application:", error);

    // Show app anyway to prevent permanent blank screen
    const app = document.getElementById("app");
    if (app) {
      app.classList.add("ready");
    }
  }
});
