/**
 * Application entry point.
 *
 * Load order matters:
 * 1. UI5 assets (themes, i18n) so `setTheme` can switch to any supported theme.
 * 2. Icon collections. All icons are registered because icon names can be chosen
 *    at runtime (route registry, status icons), so they cannot be
 *    imported one by one at build time.
 * 3. Global styles, then the React tree.
 */
import "@ui5/webcomponents-react/dist/Assets.js";
import "@ui5/webcomponents-icons/dist/AllIcons.js";
import "@ui5/webcomponents-icons-tnt/dist/AllIcons.js";
import "@ui5/webcomponents-icons-business-suite/dist/AllIcons.js";
import "./style.css";

import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { BrowserRouter } from "react-router";
import { ThemeProvider } from "@ui5/webcomponents-react";

import App from "./App";
import { AppUiProvider } from "./state/AppUiContext";

createRoot(document.getElementById("root") as HTMLElement).render(
  <StrictMode>
    {/* ThemeProvider is required by UI5 Web Components for React (theming + CSS vars) */}
    <ThemeProvider>
      <BrowserRouter>
        {/* Shared UI state (side nav, theme) used by the shell */}
        <AppUiProvider>
          <App />
        </AppUiProvider>
      </BrowserRouter>
    </ThemeProvider>
  </StrictMode>
);
