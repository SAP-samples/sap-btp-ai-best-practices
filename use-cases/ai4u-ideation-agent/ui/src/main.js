import "@ui5/webcomponents-icons/dist/AllIcons.js";
import "@ui5/webcomponents-icons-tnt/dist/AllIcons.js";
import "@ui5/webcomponents-icons-business-suite/dist/AllIcons.js";

/* ShellBar */
import "@ui5/webcomponents/dist/Avatar.js";
import "@ui5/webcomponents/dist/Button.js";
import "@ui5/webcomponents/dist/ToggleButton.js";

import "@ui5/webcomponents-fiori/dist/ShellBar.js";
import "@ui5/webcomponents-fiori/dist/ShellBarItem.js";
import "@ui5/webcomponents-fiori/dist/ShellBarBranding.js";

/* Navigation Layout */
import "@ui5/webcomponents/dist/Text.js";
import "@ui5/webcomponents/dist/Title.js";

import "@ui5/webcomponents-fiori/dist/NavigationLayout.js";
import "@ui5/webcomponents-fiori/dist/SideNavigation.js";
import "@ui5/webcomponents-fiori/dist/SideNavigationItem.js";

import NavigationLayoutMode from "@ui5/webcomponents-fiori/dist/types/NavigationLayoutMode.js";

/* Popover for user profile */
import "@ui5/webcomponents/dist/Popover.js";

/* Input & BusyIndicator for login page */
import "@ui5/webcomponents/dist/Input.js";
import "@ui5/webcomponents/dist/BusyIndicator.js";
import "@ui5/webcomponents/dist/MessageStrip.js";
import "@ui5/webcomponents/dist/Card.js";

/* Core */
import "@ui5/webcomponents/dist/Icon.js";

/* Services */
import { ensureAuthenticated, handleCallback, getUserInfo, logout, isAuthenticated } from "./services/auth.js";
import { trackLoginServer } from "./services/cdc.js";
import { showDisclaimerIfNeeded, showAboutDialog } from "./services/disclaimer.js";
import "./styles/disclaimer.css";

/* Page */
import initIdeationAgent from "./pages/ideation-agent/ideation-agent.js";
import pageHTML from "./pages/ideation-agent/ideation-agent.html?raw";
import "./pages/ideation-agent/ideation-agent.css";

/* Login page */
import initLoginPage from "./pages/login/login.js";
import "./pages/login/login.css";

/* Privacy page */
import initPrivacyPage from "./pages/privacy/privacy.js";
import "./pages/privacy/privacy.css";

/* Cookies page (reuses privacy-page styles) */
import initCookiesPage from "./pages/cookies/cookies.js";

/* Load SAP CDC Gigya SDK */
const cdcApiKey = import.meta.env.VITE_CDC_API_KEY;
if (cdcApiKey) {
  if (!document.getElementById("gigya-sdk")) {
    const script = document.createElement("script");
    script.id = "gigya-sdk";
    script.src = `https://cdns.gigya.com/js/gigya.js?apiKey=${cdcApiKey}`;
    script.async = true;
    script.defer = true;
    document.head.appendChild(script);
  }
}

function _initUserShell() {
  const avatar = document.querySelector('ui5-avatar[slot="profile"]');
  if (!avatar) return;

  const user = getUserInfo();
  if (user) {
    const initials = `${(user.givenName || "")[0] || ""}${(user.familyName || "")[0] || ""}`.toUpperCase();
    avatar.initials = initials || "?";

    const nameEl = document.getElementById("user-popover-name");
    const emailEl = document.getElementById("user-popover-email");
    const fullName = `${user.givenName} ${user.familyName}`.trim();
    if (nameEl) nameEl.textContent = fullName || user.email || user.userId;
    if (emailEl) {
      // Hide email line if it's already shown as the name
      emailEl.textContent = fullName ? (user.email || "") : "";
    }
  }

  const shellbar = document.querySelector("ui5-shellbar");
  if (shellbar) {
    shellbar.addEventListener("profile-click", () => {
      const popover = document.getElementById("user-popover");
      if (popover) {
        popover.opener = avatar;
        popover.open = !popover.open;
      }
    });
  }

  const logoutBtn = document.getElementById("logout-btn");
  if (logoutBtn) {
    logoutBtn.addEventListener("click", () => logout());
  }

  // Company Leaderboard button — works on any page
  const shellbarLeaderboard = document.getElementById("shellbar-leaderboard");
  if (shellbarLeaderboard) {
    shellbarLeaderboard.addEventListener("click", async () => {
      const { openLeaderboardDialog } = await import("./services/leaderboard.js");
      openLeaderboardDialog();
    });
  }

  // About/Info button in shellbar — opens disclaimer panel in read-only mode
  const shellbarAbout = document.getElementById("shellbar-about");
  if (shellbarAbout) {
    shellbarAbout.addEventListener("click", () => {
      showAboutDialog();
    });
  }

  // Update notification button — opens version popover
  const shellbarUpdate = document.getElementById("shellbar-update");
  if (shellbarUpdate) {
    shellbarUpdate.addEventListener("click", () => {
      const popover = document.getElementById("update-popover");
      if (popover) {
        popover.opener = shellbarUpdate;
        popover.open = !popover.open;
      }
    });
  }
}

async function _maybeShowAdminNav() {
  try {
    const { checkAdmin } = await import("./services/admin.js");
    const isAdmin = await checkAdmin();
    if (isAdmin) {
      const sideNav = document.getElementById("sn1");
      if (sideNav && !document.getElementById("admin-nav-item")) {
        const item = document.createElement("ui5-side-navigation-item");
        item.id = "admin-nav-item";
        item.setAttribute("text", "Admin");
        item.setAttribute("href", "/admin");
        item.setAttribute("icon", "settings");
        sideNav.appendChild(item);
      }
    }
  } catch {
    // Non-admin or endpoint unavailable — silently ignore
  }
}

function _checkAndShowUpdateIndicator() {
  import("./services/version.js").then(({ checkForUpdate }) => {
    checkForUpdate().then((result) => {
      if (!result || result.updateAvailable !== true) return;

      const shellbarItem = document.getElementById("shellbar-update");
      if (!shellbarItem) return;

      // Show the shellbar item and add notification dot
      shellbarItem.style.display = "";
      shellbarItem.classList.add("has-update");

      // Populate popover text
      const currentEl = document.getElementById("update-current-version");
      const latestEl = document.getElementById("update-latest-version");
      if (currentEl) currentEl.textContent = result.currentVersion;
      if (latestEl) latestEl.textContent = result.latestVersion;

      // Auto-open popover once per session (first page load only)
      if (!sessionStorage.getItem("update-popover-shown")) {
        sessionStorage.setItem("update-popover-shown", "1");
        const popover = document.getElementById("update-popover");
        if (popover) {
          popover.opener = shellbarItem;
          setTimeout(() => { popover.open = true; }, 500);
        }
      }
    });
  });
}

function _initNavigation() {
  const nl1 = document.querySelector("#nl1");
  const startButton = document.querySelector("#startButton");

  if (startButton && nl1) {
    startButton.addEventListener("click", () => {
      nl1.mode = nl1.isSideCollapsed()
        ? NavigationLayoutMode.Expanded
        : NavigationLayoutMode.Collapsed;
    });
  }
}

async function _handleAuthCallback() {
  if (window.location.pathname === "/auth/callback") {
    try {
      const returnUrl = await handleCallback();
      // Track XSUAA login in CDC via backend (non-blocking)
      trackLoginServer();
      window.history.replaceState({}, "", returnUrl || "/");
      return true;
    } catch (err) {
      console.error("Auth callback failed:", err);
      window.history.replaceState({}, "", "/");
      return true;
    }
  }
  return false;
}

function _showLoginPage() {
  // Hide the app shell and show login
  const app = document.getElementById("app");
  if (app) app.style.display = "none";

  // Create login root if not present
  let loginRoot = document.getElementById("login-root");
  if (!loginRoot) {
    loginRoot = document.createElement("div");
    loginRoot.id = "login-root";
    document.body.appendChild(loginRoot);
  }

  initLoginPage();

  // Make visible
  loginRoot.style.opacity = "1";
}

function _showPrivacyPage() {
  // Hide the app shell and show privacy page
  const app = document.getElementById("app");
  if (app) app.style.display = "none";

  // Create privacy root if not present
  let privacyRoot = document.getElementById("privacy-root");
  if (!privacyRoot) {
    privacyRoot = document.createElement("div");
    privacyRoot.id = "privacy-root";
    document.body.appendChild(privacyRoot);
  }

  initPrivacyPage();
}

function _showCookiesPage() {
  // Hide the app shell and show cookies page
  const app = document.getElementById("app");
  if (app) app.style.display = "none";

  // Create cookies root if not present
  let cookiesRoot = document.getElementById("cookies-root");
  if (!cookiesRoot) {
    cookiesRoot = document.createElement("div");
    cookiesRoot.id = "cookies-root";
    document.body.appendChild(cookiesRoot);
  }

  initCookiesPage();
}

function _showApp() {
  const app = document.getElementById("app");
  if (app) app.style.display = "";

  // Remove login root if present
  const loginRoot = document.getElementById("login-root");
  if (loginRoot) loginRoot.remove();
}

function _setActiveNavItem(href) {
  const sideNav = document.getElementById("sn1");
  if (!sideNav) return;
  const items = sideNav.querySelectorAll("ui5-side-navigation-item");
  items.forEach((item) => {
    item.selected = item.getAttribute("href") === href;
  });
}

document.addEventListener("DOMContentLoaded", async () => {
  try {
    await _handleAuthCallback();

    // Check if on /privacy path (public, no auth required)
    if (window.location.pathname === "/privacy") {
      _showPrivacyPage();
      return;
    }

    // Check if on /cookies path (public, no auth required)
    if (window.location.pathname === "/cookies") {
      _showCookiesPage();
      return;
    }

    // Check if on /login path explicitly
    if (window.location.pathname === "/login") {
      _showLoginPage();
      return;
    }

    const authenticated = await ensureAuthenticated();
    if (!authenticated) {
      // Preserve current URL for post-login redirect
      sessionStorage.setItem("auth_return_url", window.location.href);
      _showLoginPage();
      return;
    }

    // Authenticated — show app
    _showApp();

    // Route-based page loading
    const contentEl = document.querySelector(".content");
    const path = window.location.pathname;

    if (path === "/events") {
      // Events hub page (dynamic import for code-splitting)
      const [{ default: initEventsPage }, { default: eventsHTML }] = await Promise.all([
        import("./pages/events/events.js"),
        import("./pages/events/events.html?raw"),
      ]);
      await import("./pages/events/events.css");
      if (contentEl) contentEl.innerHTML = eventsHTML;
      initEventsPage();
      _setActiveNavItem("/events");
    } else if (path.startsWith("/event/")) {
      // Individual event workspace (dynamic import)
      const [{ default: initEventPage }, { default: eventHTML }] = await Promise.all([
        import("./pages/event/event.js"),
        import("./pages/event/event.html?raw"),
      ]);
      await import("./pages/event/event.css");
      if (contentEl) contentEl.innerHTML = eventHTML;
      const code = path.split("/event/")[1];
      initEventPage(code);
      _setActiveNavItem("/events");
    } else if (path === "/canvas") {
      // Whiteboard canvas page (dynamic import)
      const [{ default: initCanvasPage }, { default: canvasHTML }] = await Promise.all([
        import("./pages/canvas/canvas.js"),
        import("./pages/canvas/canvas.html?raw"),
      ]);
      await import("./pages/canvas/canvas.css");
      if (contentEl) contentEl.innerHTML = canvasHTML;
      initCanvasPage();
      _setActiveNavItem("/canvas");
    } else if (path.startsWith("/shared/")) {
      // Shared conversation join page — redirect to ideation workspace after joining
      const shareCode = path.split("/shared/")[1];
      if (shareCode) {
        try {
          const { joinByCode } = await import("./services/sharing.js");
          const result = await joinByCode(shareCode);
          if (result?.conversation_id) {
            // Redirect to the ideation agent page with the shared conversation loaded
            window.history.replaceState(null, "", `/?c=${result.conversation_id}`);
            if (contentEl) contentEl.innerHTML = pageHTML;
            initIdeationAgent();
            _setActiveNavItem("/");
          } else {
            // Fallback: just go to home
            window.history.replaceState(null, "", "/");
            if (contentEl) contentEl.innerHTML = pageHTML;
            initIdeationAgent();
            _setActiveNavItem("/");
          }
        } catch (err) {
          console.error("[Share] Failed to join shared conversation:", err);
          // Show error and redirect to home
          window.history.replaceState(null, "", "/");
          if (contentEl) contentEl.innerHTML = pageHTML;
          initIdeationAgent();
          _setActiveNavItem("/");
        }
      }
    } else if (path === "/handover") {
      const [{ default: initHandoverPage }, { default: handoverHTML }] = await Promise.all([
        import("./pages/handover/handover.js"),
        import("./pages/handover/handover.html?raw"),
      ]);
      await import("./pages/handover/handover.css");
      if (contentEl) contentEl.innerHTML = handoverHTML;
      initHandoverPage();
      _setActiveNavItem("/handover");
    } else if (path === "/admin") {
      // Admin dashboard (dynamic import for code-splitting)
      const [{ default: initAdminPage }, { default: adminHTML }] = await Promise.all([
        import("./pages/admin/admin.js"),
        import("./pages/admin/admin.html?raw"),
      ]);
      await import("./pages/admin/admin.css");
      if (contentEl) contentEl.innerHTML = adminHTML;
      initAdminPage();
      _setActiveNavItem("/admin");
    } else {
      // Default: personal ideation workspace
      if (contentEl) contentEl.innerHTML = pageHTML;
      initIdeationAgent();
      _setActiveNavItem("/");
    }

    // Init shell
    _initNavigation();
    _initUserShell();

    // Conditionally show admin nav item (non-blocking)
    _maybeShowAdminNav();

    // Check for newer version upstream (non-blocking)
    _checkAndShowUpdateIndicator();

    // Reveal
    await new Promise((resolve) => setTimeout(resolve, 100));
    const app = document.getElementById("app");
    if (app) app.classList.add("ready");

    // Show disclaimer/about panel on first use (overlays the visible app)
    await showDisclaimerIfNeeded();

    console.log("Application ready");
  } catch (error) {
    console.error("Error initializing application:", error);
    const app = document.getElementById("app");
    if (app) app.classList.add("ready");
  }
});
