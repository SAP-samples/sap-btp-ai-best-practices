import React, { useEffect } from "react";
import { AuthProvider } from "@site/src/authProviderBTP"; // Corrected path
import { trackEvent } from "@site/src/lib/trackingTool/trackingUtils";

// Default implementation, that you can customize
// https://docusaurus.io/docs/swizzling#wrapping-global-components
export default function Root({ children }) {
  useEffect(() => {
    const handleClick = (e: MouseEvent) => {
      const target = (e.target as HTMLElement).closest("a");
      if (!target) return;

      const href = target.getAttribute("href");
      if (!href) return;

      // Announcement bar link
      if (target.closest("[class*='announcementBar']")) {
        trackEvent({ featureName: `btn-announcement-bar:${href}` });
        return;
      }

      // Navbar AI4U link
      if (target.classList.contains("navbar__item--ai4u")) {
        trackEvent({ featureName: `btn-navbar:${href}` });
      }
    };

    document.addEventListener("click", handleClick);
    return () => document.removeEventListener("click", handleClick);
  }, []);

  return <AuthProvider>{children}</AuthProvider>;
}
