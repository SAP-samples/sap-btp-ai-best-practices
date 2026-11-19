/**
 * SAP Business AI Platform Best Practices Website Tracking Utility
 *
 * This utility handles user tracking for the SAP Business AI Platform Best Practices website with two main modes:
 *
 * 1. With User Consent (Cookie-based tracking):
 *    - Uses persistent storage to track user sessions
 *    - Associates tracking with user's email if logged in
 *    - Falls back to anonymous tracking if no email is available
 *    - Maintains usage history across sessions
 *
 * 2. Without User Consent (In-memory tracking):
 *    - Uses in-memory storage for the current session only
 *    - Creates a unique session ID for each visit
 *    - If user is logged in, links session data to user email via additionalData
 *    - Data is not persisted between sessions
 *
 * Consent Levels:
 * - Level 1: Basic tracking consent
 * - Level 2: Enhanced tracking consent
 * - Either level is sufficient for enabling cookie-based tracking
 *
 * Usage:
 * - Call trackEvent() with toolName and featureName to track user interactions
 * - The utility automatically handles consent checking and appropriate storage method
 * - Use clear() to reset tracking data when needed (e.g., on logout)
 *
 * Note: This utility integrates with SAP's TrustArc for consent management
 * and requires the global window.SAP.trustArc configuration to be present.
 */

import SAPTrackingTool from "@site/src/lib/trackingTool/automated-usage-tracking-tool/web";
import { InMemoryTrackingStorage } from "./InMemoryTrackingStorage";
import { environment } from "@site/src/config/environment";
import { TRACKING_CONFIG } from "./config";
import { authStorage } from "@site/src/utils/authStorage";

// Types
interface TrackingParams {
  toolName?: string;
  featureName: string;
  additionalData?: Record<string, any>;
}

// Constants
const ANONYMOUS_EMAIL_DOMAIN = "anonymous.btp-ai-bp.docs.sap";
const SESSION_EMAIL_DOMAIN = "session.btp-ai-bp.docs.sap";

// Global type declarations
declare global {
  interface Window {
    SAP?: {
      global: {
        trustArc: {
          domain: string;
        };
      };
    };
    isConsentEnabled: (domain: string, level: number) => boolean;
  }
}

// Create and export a singleton tracking tool instance
export const trackingTool = new SAPTrackingTool({
  apiKey: TRACKING_CONFIG.apiKey,
  dataCenter: TRACKING_CONFIG.dataCenter,
  storageName: environment.storageName
});

/**
 * Checks if user has given consent for tracking
 */
const hasConsentLevel = (): boolean => {
  if (!window.SAP?.global?.trustArc?.domain || !window.isConsentEnabled) {
    console.error("window.SAP.global.trustArc.domain or window.isConsentEnabled is not set");
    return false;
  }

  if (window.isConsentEnabled(`${window.SAP.global.trustArc.domain}`, 1) || window.isConsentEnabled(`${window.SAP.global.trustArc.domain}`, 2)) {
    return true;
  }

  return false;
};

/**
 * Generates an email based on the provided identifier and domain
 */
const generateEmail = (identifier: string, domain: string): string => `${identifier}@${domain}`;

/**
 * Handles tracking when user has not given consent
 */
const handleTrackingWithoutConsent = async (toolName: string, featureName: string, userEmail?: string): Promise<void> => {
  const memoryStorage = InMemoryTrackingStorage.getInstance();
  memoryStorage.addUsage(toolName, featureName);

  const sessionId = memoryStorage.getSessionId();
  const sessionEmail = generateEmail(sessionId, SESSION_EMAIL_DOMAIN);

  await trackingTool.customTrackUsage({
    toolName,
    featureName,
    email: sessionEmail,
    usages: memoryStorage.getUsages(),
    ...(userEmail && { additionalData: { userEmail } })
  });
};

/**
 * Handles tracking when user has given consent
 */
const handleTrackingWithConsent = async (toolName: string, featureName: string, userEmail?: string): Promise<void> => {
  const email = userEmail || trackingTool.storage.getEmail() || generateEmail(crypto.randomUUID(), ANONYMOUS_EMAIL_DOMAIN);

  if (email !== trackingTool.storage.getEmail()) {
    trackingTool.storage.clear();
    trackingTool.storage.setEmail(email);
  }

  trackingTool.storage.setLatestUsage(toolName, featureName);
  await trackingTool.customTrackUsage({
    toolName,
    featureName,
    email,
    usages: trackingTool.storage.getLatestUsages()
  });
};

/**
 * Shared tracking utility that can be used by both hooks and event handlers
 */
export const trackEvent = async ({ toolName = TRACKING_CONFIG.toolName, featureName = window.location.pathname, additionalData }: TrackingParams): Promise<void> => {
  const authData = authStorage.load();
  const hasUserConsent = hasConsentLevel();

  if (!hasUserConsent) {
    await handleTrackingWithoutConsent(toolName, featureName, authData?.email);
    return;
  }

  await handleTrackingWithConsent(toolName, featureName, authData?.email);
};

/**
 * Clears the in-memory tracking storage
 */
export const clear = (): void => {
  InMemoryTrackingStorage.getInstance().clear();
  trackingTool.storage.clear();
};
