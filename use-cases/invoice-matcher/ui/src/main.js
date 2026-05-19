import "@ui5/webcomponents-icons/dist/AllIcons.js";
import "@ui5/webcomponents-icons-tnt/dist/AllIcons.js";
import "@ui5/webcomponents-icons-business-suite/dist/AllIcons.js";

import "@ui5/webcomponents/dist/Avatar.js";
import "@ui5/webcomponents/dist/Button.js";
import "@ui5/webcomponents/dist/Text.js";
import "@ui5/webcomponents/dist/Title.js";
import "@ui5/webcomponents/dist/Icon.js";

import "@ui5/webcomponents-fiori/dist/ShellBar.js";
import "@ui5/webcomponents-fiori/dist/ShellBarBranding.js";
import "@ui5/webcomponents-fiori/dist/NavigationLayout.js";
import "@ui5/webcomponents-fiori/dist/SideNavigation.js";
import "@ui5/webcomponents-fiori/dist/SideNavigationItem.js";

import { handleNavigation } from "./modules/navigation.js";
import { pageRouter } from "./modules/router.js";

document.addEventListener("DOMContentLoaded", async () => {
  try {
    window.pageRouter = pageRouter;
    await pageRouter.init(".content");
    handleNavigation();
    await new Promise((resolve) => setTimeout(resolve, 100));
    const app = document.getElementById("app");
    if (app) app.classList.add("ready");
  } catch (error) {
    console.error("Error initializing application:", error);
    const app = document.getElementById("app");
    if (app) app.classList.add("ready");
  }
});
