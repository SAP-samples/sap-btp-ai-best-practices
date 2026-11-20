// DynamicPage does not expose a public scrolling or pinned-title geometry API.
//
// Internal-selector boundary: the two selectors below are intentionally confined
// to this adapter. They match @ui5/webcomponents-fiori DynamicPage 2.26.0's
// rendered template and are not a public UI5 contract. Callers receive only
// behavioral operations; no internal node or selector escapes this module.
const SCROLL_CONTAINER_SELECTOR = ".ui5-dynamic-page-scroll-container";
const TITLE_HEADER_WRAPPER_SELECTOR = ".ui5-dynamic-page-title-header-wrapper";
const SECTION_GAP = 16;

function getDynamicPageInternals(dynamicPage) {
  const shadowRoot = dynamicPage?.shadowRoot;
  if (!shadowRoot) return null;
  const scrollContainer = shadowRoot.querySelector(SCROLL_CONTAINER_SELECTOR);
  if (!scrollContainer) return null;
  return {
    scrollContainer,
    titleWrapper: shadowRoot.querySelector(TITLE_HEADER_WRAPPER_SELECTOR),
  };
}

function getNavigationGeometry(dynamicPage, navigation) {
  const internals = getDynamicPageInternals(dynamicPage);
  if (!internals) return null;
  const { scrollContainer, titleWrapper } = internals;
  const scrollTop = scrollContainer.getBoundingClientRect().top;
  const titleBottom = titleWrapper?.getBoundingClientRect?.().bottom ?? scrollTop;
  const navigationHeight = navigation?.getBoundingClientRect?.().height || 0;
  return {
    scrollContainer,
    navigationHeight,
    visibleTop: Math.max(scrollTop, titleBottom) + navigationHeight + SECTION_GAP,
    navigationOffset: Math.max(0, titleBottom - scrollTop),
  };
}

function schedule(callback) {
  if (typeof requestAnimationFrame === "function") requestAnimationFrame(callback);
  else callback();
}

/**
 * Align the separate section navigation beneath the pinned DynamicPage title.
 * Returns false and makes no DOM changes when the internal implementation is
 * unavailable, so callers can retain ordinary document scrolling behavior.
 */
export function syncDynamicPageNavigationOffset(dynamicPage, navigation) {
  const geometry = getNavigationGeometry(dynamicPage, navigation);
  if (!geometry || !getDynamicPageInternals(dynamicPage)?.titleWrapper || !navigation?.style) return false;
  navigation.style.setProperty("--ecn-navigation-top", `${geometry.navigationOffset}px`);
  return true;
}

/**
 * Scroll a section into the DynamicPage reading position, or use native
 * document scrolling when DynamicPage internals are absent or changed.
 */
export function scrollToDynamicPageSection(dynamicPage, navigation, section) {
  if (!section) return false;
  if (dynamicPage?.headerPinned) dynamicPage.headerPinned = false;
  if (dynamicPage && !dynamicPage.headerSnapped) dynamicPage.headerSnapped = true;

  const scrollToSection = () => {
    const geometry = getNavigationGeometry(dynamicPage, navigation);
    const internals = getDynamicPageInternals(dynamicPage);
    if (!geometry || !internals?.titleWrapper || typeof geometry.scrollContainer.scrollTo !== "function") {
      section.scrollIntoView?.({ block: "start", behavior: "auto" });
      return;
    }

    syncDynamicPageNavigationOffset(dynamicPage, navigation);
    const { scrollContainer, visibleTop } = geometry;
    const scrollTop = scrollContainer.scrollTop + section.getBoundingClientRect().top - visibleTop;
    scrollContainer.scrollTo({ top: Math.max(0, scrollTop), behavior: "auto" });

    // Snapping can change the title height after the first alignment.
    schedule(() => schedule(() => {
      const nextGeometry = getNavigationGeometry(dynamicPage, navigation);
      const nextInternals = getDynamicPageInternals(dynamicPage);
      if (!nextGeometry || !nextInternals?.titleWrapper || typeof nextGeometry.scrollContainer.scrollTo !== "function") return;
      syncDynamicPageNavigationOffset(dynamicPage, navigation);
      const adjustment = section.getBoundingClientRect().top - nextGeometry.visibleTop;
      if (Math.abs(adjustment) > 1) {
        nextGeometry.scrollContainer.scrollTo({
          top: Math.max(0, nextGeometry.scrollContainer.scrollTop + adjustment),
          behavior: "auto",
        });
      }
    }));
  };

  schedule(scrollToSection);
  return true;
}

/**
 * Return the currently readable section for the DynamicPage scroll spy.
 * Null means unavailable geometry or a non-layout environment (for example
 * JSDOM), and callers should leave the current aria-current state unchanged.
 */
export function getActiveDynamicPageSection(dynamicPage, navigation, sections) {
  const geometry = getNavigationGeometry(dynamicPage, navigation);
  if (!geometry) return null;
  syncDynamicPageNavigationOffset(dynamicPage, navigation);
  const positioned = sections
    .filter(({ element }) => element)
    .map(({ target, element }) => ({ target, top: element.getBoundingClientRect().top }));
  if (new Set(positioned.map(({ top }) => top)).size < 2) return null;
  return positioned.reduce((selected, { target, top }) => (
    top <= geometry.visibleTop + 1 ? target : selected
  ), "overview");
}

/**
 * Attach to the current internal scroll container without exposing it to page
 * code. The adapter observes its own selector boundary so a deferred
 * DynamicPage render or an internal container replacement rebinds the listener
 * without requiring a component-specific render event.
 */
export function observeDynamicPageScroll(dynamicPage, onScroll) {
  let trackedScrollContainer = null;
  let observedShadowRoot = null;
  let replacementObserver = null;
  const listener = () => onScroll?.();

  const refresh = () => {
    const shadowRoot = dynamicPage?.shadowRoot || null;
    if (shadowRoot !== observedShadowRoot) {
      replacementObserver?.disconnect();
      replacementObserver = null;
      observedShadowRoot = shadowRoot;
      if (shadowRoot && typeof MutationObserver === "function") {
        replacementObserver = new MutationObserver(() => {
          if (refresh()) onScroll?.();
        });
        replacementObserver.observe(shadowRoot, { childList: true, subtree: true });
      }
    }

    const nextScrollContainer = getDynamicPageInternals(dynamicPage)?.scrollContainer || null;
    if (nextScrollContainer === trackedScrollContainer) return false;
    trackedScrollContainer?.removeEventListener("scroll", listener);
    trackedScrollContainer = nextScrollContainer;
    trackedScrollContainer?.addEventListener("scroll", listener);
    return !!trackedScrollContainer;
  };

  refresh();
  return {
    refresh,
    disconnect: () => {
      replacementObserver?.disconnect();
      replacementObserver = null;
      observedShadowRoot = null;
      trackedScrollContainer?.removeEventListener("scroll", listener);
      trackedScrollContainer = null;
    },
  };
}
