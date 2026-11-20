import { afterEach, expect, it, vi } from "vitest";
import {
  getActiveDynamicPageSection,
  observeDynamicPageScroll,
  scrollToDynamicPageSection,
  syncDynamicPageNavigationOffset,
} from "../src/pages/ecn-detail/dynamic-page-scroll-adapter.js";

function rect(top, bottom = top) {
  return { top, bottom, height: bottom - top };
}

function createDynamicPageInternals() {
  const dynamicPage = document.createElement("div");
  const shadowRoot = dynamicPage.attachShadow({ mode: "open" });
  shadowRoot.innerHTML = `
    <div class="ui5-dynamic-page-scroll-container"></div>
    <div class="ui5-dynamic-page-title-header-wrapper"></div>
  `;
  const scrollContainer = shadowRoot.querySelector(".ui5-dynamic-page-scroll-container");
  const titleWrapper = shadowRoot.querySelector(".ui5-dynamic-page-title-header-wrapper");
  scrollContainer.getBoundingClientRect = () => rect(10, 410);
  titleWrapper.getBoundingClientRect = () => rect(10, 60);
  Object.defineProperty(scrollContainer, "scrollTop", { configurable: true, value: 20, writable: true });
  scrollContainer.scrollTo = vi.fn();
  return { dynamicPage, scrollContainer };
}

afterEach(() => {
  vi.unstubAllGlobals();
  document.body.replaceChildren();
});

it("uses native section scrolling without touching unavailable DynamicPage internals", () => {
  vi.stubGlobal("requestAnimationFrame", (callback) => {
    callback();
    return 1;
  });
  const dynamicPage = document.createElement("div");
  const navigation = document.createElement("nav");
  const section = document.createElement("section");
  section.scrollIntoView = vi.fn();

  expect(syncDynamicPageNavigationOffset(dynamicPage, navigation)).toBe(false);
  expect(scrollToDynamicPageSection(dynamicPage, navigation, section)).toBe(true);
  expect(section.scrollIntoView).toHaveBeenCalledWith({ block: "start", behavior: "auto" });
  expect(getActiveDynamicPageSection(dynamicPage, navigation, [])).toBeNull();
});

it("treats a missing pinned-title wrapper as an unavailable internal layout", () => {
  vi.stubGlobal("requestAnimationFrame", (callback) => {
    callback();
    return 1;
  });
  const dynamicPage = document.createElement("div");
  const shadowRoot = dynamicPage.attachShadow({ mode: "open" });
  const scrollContainer = document.createElement("div");
  scrollContainer.className = "ui5-dynamic-page-scroll-container";
  scrollContainer.scrollTo = vi.fn();
  shadowRoot.append(scrollContainer);
  const navigation = document.createElement("nav");
  const section = document.createElement("section");
  section.scrollIntoView = vi.fn();

  expect(syncDynamicPageNavigationOffset(dynamicPage, navigation)).toBe(false);
  scrollToDynamicPageSection(dynamicPage, navigation, section);
  expect(scrollContainer.scrollTo).not.toHaveBeenCalled();
  expect(section.scrollIntoView).toHaveBeenCalledWith({ block: "start", behavior: "auto" });
});

it("keeps the pinned-title offset and scroll subscription inside the adapter", () => {
  const { dynamicPage, scrollContainer } = createDynamicPageInternals();
  const navigation = document.createElement("nav");
  navigation.getBoundingClientRect = () => rect(60, 84);
  const overview = document.createElement("section");
  const plants = document.createElement("section");
  const bom = document.createElement("section");
  overview.getBoundingClientRect = () => rect(30, 90);
  plants.getBoundingClientRect = () => rect(120, 180);
  bom.getBoundingClientRect = () => rect(240, 300);

  expect(syncDynamicPageNavigationOffset(dynamicPage, navigation)).toBe(true);
  expect(navigation.style.getPropertyValue("--ecn-navigation-top")).toBe("50px");
  expect(getActiveDynamicPageSection(dynamicPage, navigation, [
    { target: "overview", element: overview },
    { target: "plants", element: plants },
    { target: "bom", element: bom },
  ])).toBe("overview");

  const onScroll = vi.fn();
  const observer = observeDynamicPageScroll(dynamicPage, onScroll);
  scrollContainer.dispatchEvent(new Event("scroll"));
  expect(onScroll).toHaveBeenCalledOnce();
  observer.disconnect();
  scrollContainer.dispatchEvent(new Event("scroll"));
  expect(onScroll).toHaveBeenCalledOnce();
});

it("rebinds after an internal DynamicPage scroller replacement without a render event", async () => {
  const { dynamicPage, scrollContainer } = createDynamicPageInternals();
  const shadowRoot = dynamicPage.shadowRoot;
  const titleWrapper = shadowRoot.querySelector(".ui5-dynamic-page-title-header-wrapper");
  const onScroll = vi.fn();
  const observer = observeDynamicPageScroll(dynamicPage, onScroll);
  const replacement = document.createElement("div");
  replacement.className = "ui5-dynamic-page-scroll-container";

  shadowRoot.replaceChildren(replacement, titleWrapper);
  await vi.waitFor(() => expect(onScroll).toHaveBeenCalledOnce());

  scrollContainer.dispatchEvent(new Event("scroll"));
  expect(onScroll).toHaveBeenCalledOnce();
  replacement.dispatchEvent(new Event("scroll"));
  expect(onScroll).toHaveBeenCalledTimes(2);
  observer.disconnect();
});
