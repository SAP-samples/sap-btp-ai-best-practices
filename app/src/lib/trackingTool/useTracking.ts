import SAPTrackingTool from "@sap_oss/automated-usage-tracking-tool";
import "@sap_oss/automated-usage-tracking-tool/theme/sap_horizon.css";
import { environment } from "@site/src/config/environment";
import { useEffect } from "react";
import { TRACKING_CONFIG } from "./config";

interface TrackingParams {
  toolName?: string;
  featureName: string;
}

export const useTracking = ({ toolName, featureName }: TrackingParams) => {
  useEffect(() => {
    const initializeTracking = async () => {
      // // Skip tracking in development mode
      // if (environment.isDevelopment) {
      //   console.log("Tracking disabled in development mode");
      //   return;
      // }

      try {
        const trackingTool = new SAPTrackingTool({
          apiKey: TRACKING_CONFIG.apiKey,
          dataCenter: TRACKING_CONFIG.dataCenter,
          storageName: environment.storageName
        });

        await trackingTool.requestConsentConfirmation({
          message: TRACKING_CONFIG.consentMessage
        });

        trackingTool.trackUsage({
          featureName,
          toolName: toolName ?? TRACKING_CONFIG.toolName
        });
      } catch (error) {
        console.error("Failed to initialize tracking:", error);
      }
    };

    initializeTracking();
  }, [toolName, featureName]);
};
