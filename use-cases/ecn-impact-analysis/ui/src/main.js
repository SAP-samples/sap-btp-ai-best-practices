import "@ui5/webcomponents-icons/dist/AllIcons.js";
import "@ui5/webcomponents/dist/Button.js";
import "@ui5/webcomponents-fiori/dist/NavigationLayout.js";
import "@ui5/webcomponents-fiori/dist/ShellBar.js";
import "@ui5/webcomponents-fiori/dist/ShellBarBranding.js";
import "@ui5/webcomponents-fiori/dist/SideNavigation.js";
import "@ui5/webcomponents-fiori/dist/SideNavigationItem.js";

import { handleNavigation, syncEcnNavigation } from "./modules/navigation.js";
import { pageRouter } from "./modules/router.js";
import { mountShellChat } from "./modules/shell-chat.js";

/** Initialize the application shell after its DOM becomes available. */
async function initializeApplication() {
  try {
    mountShellChat(document.querySelector("#app"));
    pageRouter.onNavigate = (_definition, params) => syncEcnNavigation(params.ecn_row_id);
    await pageRouter.init(".content");
    handleNavigation();
  } catch (error) {
    console.error("Error initializing application:", error);
  } finally {
    const app = document.getElementById("app");
    app?.classList.add("ready");
    app?.setAttribute("aria-busy", "false");
  }
}

document.addEventListener("DOMContentLoaded", initializeApplication);
