import NavigationLayoutMode from "@ui5/webcomponents-fiori/dist/types/NavigationLayoutMode.js";
import { pageRouter } from "./router.js";

function handleNavigation() {
  const nl1 = document.querySelector("#nl1");
  const startButton = document.querySelector("#startButton");
  const sn1 = document.querySelector("#sn1");

  startButton.addEventListener("click", () => {
    nl1.mode = nl1.isSideCollapsed() ? NavigationLayoutMode.Expanded : NavigationLayoutMode.Collapsed;
  });

  sn1.addEventListener("selection-change", (event) => {
    if (event.detail.item.getAttribute("target")) {
      return;
    }

    const href = event.detail.item.getAttribute("href");
    if (href) {
      const pageName = href.replace("#", "").replace("/", "");
      pageRouter.navigate(`/${pageName}`);
    }
  });
}

export { handleNavigation };
