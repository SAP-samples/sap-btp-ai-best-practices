import "@ui5/webcomponents-icons/dist/AllIcons.js";
import "@ui5/webcomponents-icons-tnt/dist/AllIcons.js";
import "@ui5/webcomponents-icons-business-suite/dist/AllIcons.js";

import "@ui5/webcomponents/dist/Icon.js";

import { pageRouter } from "./modules/router.js";

document.addEventListener("DOMContentLoaded", async () => {
  try {
    window.pageRouter = pageRouter;

    await pageRouter.init(".content");

    await new Promise((resolve) => setTimeout(resolve, 100));

    const app = document.getElementById("app");
    if (app) {
      app.classList.add("ready");
    }

    console.log("Application ready");
  } catch (error) {
    console.error("Error initializing application:", error);

    const app = document.getElementById("app");
    if (app) {
      app.classList.add("ready");
    }
  }
});
