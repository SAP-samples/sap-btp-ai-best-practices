import NavigationLayoutMode from "@ui5/webcomponents-fiori/dist/types/NavigationLayoutMode.js";
import { pageRouter } from "./router.js";

/** Bind shell navigation without a document reload, preserving in-memory advice chat. */
function handleNavigation() {
  const nl1 = document.querySelector("#nl1");
  const startButton = document.querySelector("#startButton");
  const sn1 = document.querySelector("#sn1");

  startButton.addEventListener("click", () => {
    nl1.mode = nl1.isSideCollapsed() ? NavigationLayoutMode.Expanded : NavigationLayoutMode.Collapsed;
  });

  // UI5 links live in shadow DOM: page.js cannot reliably intercept their anchors.
  sn1.addEventListener("click", (event) => {
    const item = event.composedPath().find(node => node.matches?.("ui5-side-navigation-item,ui5-side-navigation-sub-item"));
    const href = item?.getAttribute("href");
    if (!href?.startsWith("/") || item.getAttribute("target") || event.metaKey || event.ctrlKey || event.shiftKey) return;
    event.preventDefault();
    pageRouter.navigate(href);
  }, true);
}

export { handleNavigation };
