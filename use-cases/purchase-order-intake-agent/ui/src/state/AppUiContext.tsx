/**
 * Shared application UI state: side navigation collapse state and UI5 theme.
 * Plain React context - no state library.
 */
import { createContext, useCallback, useContext, useEffect, useMemo, useState, type ReactNode } from "react";
import { setTheme as applyUi5Theme } from "@ui5/webcomponents-base/dist/config/Theme.js";

/** UI5 themes the app may switch between. */
export const SUPPORTED_THEMES = ["sap_horizon", "sap_horizon_dark", "sap_horizon_hcb", "sap_horizon_hcw"] as const;
export type ThemeName = (typeof SUPPORTED_THEMES)[number];

/** Shape of the shared UI state and its setters. */
export interface AppUiState {
  sideNavCollapsed: boolean;
  toggleSideNav: () => void;
  theme: ThemeName;
  setTheme: (theme: ThemeName) => void;
}

const AppUiContext = createContext<AppUiState | null>(null);

/**
 * Provides the shared UI state to the component tree.
 *
 * @param props.children - Application tree that consumes the state.
 * @returns Context provider element.
 */
export function AppUiProvider({ children }: { children: ReactNode }) {
  const [sideNavCollapsed, setSideNavCollapsed] = useState(true);
  const [theme, setTheme] = useState<ThemeName>("sap_horizon");

  const toggleSideNav = useCallback(() => setSideNavCollapsed((collapsed) => !collapsed), []);

  // Keep the UI5 runtime theme in sync with React state.
  useEffect(() => {
    void applyUi5Theme(theme);
  }, [theme]);

  const value = useMemo(
    () => ({ sideNavCollapsed, toggleSideNav, theme, setTheme }),
    [sideNavCollapsed, toggleSideNav, theme]
  );

  return <AppUiContext.Provider value={value}>{children}</AppUiContext.Provider>;
}

/**
 * Reads the shared UI state.
 *
 * @returns Current UI state and setters.
 * @throws Error when used outside of `AppUiProvider`.
 */
export function useAppUi(): AppUiState {
  const context = useContext(AppUiContext);
  if (!context) {
    throw new Error("useAppUi must be used inside <AppUiProvider>");
  }
  return context;
}
