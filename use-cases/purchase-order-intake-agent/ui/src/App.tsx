/**
 * Application shell: ShellBar header, collapsible SideNavigation and the routed page area.
 *
 * Pages and navigation items are generated from `routes.tsx`; nothing here needs
 * editing when a page is added.
 */
import { Suspense } from "react";
import { Navigate, Route, Routes, useLocation, useNavigate } from "react-router";
import {
  BusyIndicator,
  Button,
  NavigationLayout,
  ShellBar,
  ShellBarBranding,
  SideNavigation,
  SideNavigationItem
} from "@ui5/webcomponents-react";

import { aliases, detailRoutes, routes } from "./routes";
import { useAppUi } from "./state/AppUiContext";
import NotFoundPage from "./pages/NotFoundPage";

/**
 * Root layout component.
 *
 * @returns The full application layout with header, side navigation and current page.
 */
export default function App() {
  const navigate = useNavigate();
  const location = useLocation();
  const { sideNavCollapsed, toggleSideNav } = useAppUi();

  const header = (
    <ShellBar
      startButton={<Button icon="menu2" onClick={toggleSideNav} tooltip="Toggle navigation" />}
      branding={
        <ShellBarBranding logo={<img src="/images/SAP_logo.svg" alt="SAP" />}>Purchase Order Intake</ShellBarBranding>
      }
    />
  );

  const sideNavigation = (
    <SideNavigation
      onSelectionChange={(event) => {
        // Each item carries its route path in a data attribute (see below).
        const path = event.detail.item.dataset.path;
        if (path) {
          navigate(path);
        }
      }}
    >
      {routes.map((route) => (
        <SideNavigationItem
          key={route.path}
          data-path={route.path}
          text={route.title}
          icon={route.icon}
          selected={location.pathname === route.path}
        />
      ))}
    </SideNavigation>
  );

  return (
    <NavigationLayout mode={sideNavCollapsed ? "Collapsed" : "Expanded"} header={header} sideContent={sideNavigation}>
      <div className="app-body">
        <div className="content">
          <Suspense fallback={<BusyIndicator active className="content-loading" />}>
            <Routes>
              {routes.map(({ path, Component }) => (
                <Route key={path} path={path} element={<Component />} />
              ))}
              {detailRoutes.map(({ path, Component }) => (
                <Route key={path} path={path} element={<Component />} />
              ))}
              {Object.entries(aliases).map(([from, to]) => (
                <Route key={from} path={from} element={<Navigate to={to} replace />} />
              ))}
              <Route path="*" element={<NotFoundPage />} />
            </Routes>
          </Suspense>
        </div>
      </div>
    </NavigationLayout>
  );
}
