import React from "react";
import { TRACKING_EVENTS } from "@site/src/lib/trackingTool/config";
import { trackEvent } from "@site/src/lib/trackingTool/trackingUtils";

interface TrackableLinkProps {
  href: string;
  className?: string;
  target?: string;
  trackingFeature?: keyof typeof TRACKING_EVENTS.BUTTON_CLICK;
  children: React.ReactNode;
}

export default function TrackableLink({ href, className, target, trackingFeature, children }: TrackableLinkProps) {
  const getFeatureName = () => {
    return trackingFeature ? TRACKING_EVENTS.BUTTON_CLICK[trackingFeature] : `trackable-link:${href}`;
  };

  const trackLinkClick = (e: React.MouseEvent) => {
    // Track the click event with consent skipped (assuming consent was given at page load)
    trackEvent({
      featureName: getFeatureName(),
      // Add additional data for tracking new tab
      additionalData: {
        isNewTab: e.ctrlKey || e.metaKey || e.button === 1 || target === "_blank",
        interactionType: e.type
      }
    });
  };

  const handleContextMenu = (e: React.MouseEvent) => {
    // Track right-click events that might lead to "Open in New Tab"
    trackEvent({
      featureName: getFeatureName(),
      additionalData: {
        isNewTab: true, // Assume right-click might lead to opening in new tab
        interactionType: "contextmenu"
      }
    });
  };

  return (
    <a
      href={href}
      className={className}
      target={target}
      onClick={trackLinkClick}
      onAuxClick={trackLinkClick} // Capture middle clicks
      onContextMenu={handleContextMenu} // Capture right clicks
    >
      {children}
    </a>
  );
}
