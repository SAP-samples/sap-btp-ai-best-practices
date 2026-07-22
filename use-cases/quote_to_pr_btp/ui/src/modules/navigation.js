import NavigationLayoutMode from "@ui5/webcomponents-fiori/dist/types/NavigationLayoutMode.js";
import { pageRouter } from "./router.js";

function handleNavigation() {
  const nl1 = document.querySelector("#nl1");
  const startButton = document.querySelector("#startButton");
  const sn1 = document.querySelector("#sn1");
  const compactViewport = window.matchMedia("(max-width: 700px)");

  const syncNavigationMode = (event) => {
    if (event.matches) {
      sn1.hidden = true;
      nl1.mode = NavigationLayoutMode.Auto;
    } else {
      sn1.hidden = false;
      nl1.mode = NavigationLayoutMode.Expanded;
    }
  };

  syncNavigationMode(compactViewport);
  compactViewport.addEventListener("change", syncNavigationMode);

  startButton.addEventListener("click", () => {
    if (compactViewport.matches) {
      sn1.hidden = !sn1.hidden;
      nl1.mode = sn1.hidden ? NavigationLayoutMode.Auto : NavigationLayoutMode.Expanded;
      return;
    }
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
      if (compactViewport.matches) {
        sn1.hidden = true;
        nl1.mode = NavigationLayoutMode.Auto;
      }
    }
  });
}

export { handleNavigation };
