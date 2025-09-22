/* ShellBar */
import "@ui5/webcomponents/dist/Avatar.js";
import "@ui5/webcomponents/dist/Button.js";
import "@ui5/webcomponents/dist/ToggleButton.js";

import "@ui5/webcomponents-fiori/dist/ShellBar.js";
import "@ui5/webcomponents-fiori/dist/ShellBarItem.js";
import "@ui5/webcomponents-fiori/dist/ShellBarSearch.js";
import "@ui5/webcomponents-fiori/dist/ShellBarBranding.js";

import "@ui5/webcomponents-icons/dist/menu2.js";
import "@ui5/webcomponents-icons/dist/sys-help.js";
import "@ui5/webcomponents-icons/dist/customer.js";
import "@ui5/webcomponents-icons/dist/da.js";
import "@ui5/webcomponents-icons/dist/ai.js";

/* Navigation Layout */
import "@ui5/webcomponents/dist/Text.js";
import "@ui5/webcomponents/dist/Title.js";

import "@ui5/webcomponents-fiori/dist/NavigationLayout.js";
import "@ui5/webcomponents-fiori/dist/SideNavigation.js";
import "@ui5/webcomponents-fiori/dist/SideNavigationGroup.js";
import "@ui5/webcomponents-fiori/dist/SideNavigationItem.js";
import "@ui5/webcomponents-fiori/dist/SideNavigationSubItem.js";

import "@ui5/webcomponents-icons/dist/home.js";
import "@ui5/webcomponents-icons/dist/group.js";
import "@ui5/webcomponents-icons/dist/menu.js";
import "@ui5/webcomponents-icons/dist/locate-me.js";
import "@ui5/webcomponents-icons/dist/calendar.js";
import "@ui5/webcomponents-icons/dist/history.js";
import "@ui5/webcomponents-icons/dist/source-code.js";
import "@ui5/webcomponents-icons/dist/background.js";

import "@ui5/webcomponents-icons/dist/action-settings.js";
import "@ui5/webcomponents-icons/dist/chain-link.js";
import "@ui5/webcomponents-icons/dist/document-text.js";
import "@ui5/webcomponents-icons/dist/compare.js";
import "@ui5/webcomponents-icons/dist/locked.js";
import "@ui5/webcomponents-icons/dist/product.js";

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
