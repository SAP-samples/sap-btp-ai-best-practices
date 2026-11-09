/**
 * SAP CDC (Customer Data Cloud) OTP login service.
 * Vanilla JS adaptation of the gamification project's CDCService.
 *
 * Requires the Gigya SDK to be loaded: https://cdns.gigya.com/js/gigya.js?apiKey=...
 */

const CDC_API_KEY = import.meta.env.VITE_CDC_API_KEY || "";
const CDC_BASE_URL = import.meta.env.VITE_CDC_BASE_URL || "https://accounts.us1.gigya.com";

/**
 * Wait for the Gigya SDK to be available on `window.gigya.accounts`.
 */
export function waitForGigyaSdk(timeoutMs = 10000) {
  if (typeof window === "undefined") {
    return Promise.reject(new Error("SAP CDC SDK is only available in browser environments."));
  }

  if (window.gigya?.accounts) {
    return Promise.resolve();
  }

  return new Promise((resolve, reject) => {
    const startedAt = Date.now();
    const intervalId = setInterval(() => {
      if (window.gigya?.accounts) {
        clearInterval(intervalId);
        resolve();
        return;
      }
      if (Date.now() - startedAt >= timeoutMs) {
        clearInterval(intervalId);
        reject(new Error("SAP CDC SDK did not load within the expected timeout."));
      }
    }, 100);
  });
}

/**
 * Run a Gigya SDK call and return the unwrapped response via Promise.
 */
async function _runGigyaCall(call) {
  await waitForGigyaSdk();

  return new Promise((resolve, reject) => {
    try {
      call((rawResponse) => {
        const response = _unwrapGigyaResponse(rawResponse);
        if ((response.errorCode || 0) !== 0) {
          reject(new Error(_getGigyaErrorMessage(response, "SAP CDC request failed.")));
          return;
        }
        resolve(response);
      });
    } catch (error) {
      reject(error);
    }
  });
}

function _unwrapGigyaResponse(response) {
  if (!response || typeof response !== "object") return response;

  const dataValue = response.data;
  if (!dataValue || typeof dataValue !== "object") return response;

  const topLevelHasPayload = _hasCorePayloadFields(response);
  const nestedHasPayload = _hasCorePayloadFields(dataValue);

  if (!topLevelHasPayload && nestedHasPayload) {
    return dataValue;
  }
  return response;
}

function _hasCorePayloadFields(obj) {
  return "UID" in obj || "vToken" in obj || "sessionInfo" in obj || "profile" in obj || "email" in obj || "regToken" in obj;
}

function _getGigyaErrorMessage(response, fallback) {
  const details = (response.errorDetails || "").toLowerCase();

  if (response.errorCode === 400006 && details.includes("code")) {
    return "The verification code is incorrect. Please try again.";
  }
  if (response.errorCode === 400006 && details.includes("token has been revoked")) {
    return "This verification code is no longer valid. Please request a new code.";
  }
  if (response.errorCode === 400006 && details.includes("vtoken")) {
    return "Your verification session expired. Please request a new code.";
  }

  return response.errorMessage || response.statusReason || fallback;
}

/**
 * Send an OTP verification code to the given email.
 * @returns {{ vToken: string }}
 */
export async function sendOtpCode(email, lang = "en") {
  if (!CDC_API_KEY) {
    throw new Error("VITE_CDC_API_KEY is not configured.");
  }

  const response = await _runGigyaCall((callback) => {
    window.gigya.accounts.otp.sendCode({ apiKey: CDC_API_KEY, lang, email, callback });
  });

  if (!response.vToken) {
    throw new Error("SAP CDC OTP response did not return a verification token.");
  }

  return { vToken: response.vToken };
}

/**
 * Verify an OTP code and log the user in.
 * @returns {{ email: string, uid: string, isNewUser: boolean }}
 */
export async function loginWithOtp(vToken, code) {
  if (!CDC_API_KEY) {
    throw new Error("VITE_CDC_API_KEY is not configured.");
  }

  const response = await _runGigyaCall((callback) => {
    window.gigya.accounts.otp.login({ apiKey: CDC_API_KEY, vToken, code, callback });
  });

  const identities = response.userInfo?.identities;
  const firstIdentity = Array.isArray(identities)
    ? identities[0]
    : identities
      ? Object.values(identities)[0]
      : undefined;

  const email = response.profile?.email || response.email || response.userInfo?.email || firstIdentity?.email;
  const uid = response.UID || response.userInfo?.UID || firstIdentity?.providerUID;

  if (!email || !uid) {
    throw new Error("SAP CDC OTP login response is missing email or UID.");
  }

  return { email, uid, isNewUser: Boolean(response.newUser) };
}

/**
 * Get a fresh CDC JWT (id_token) from the active Gigya session.
 * @returns {string} The JWT id_token
 */
export async function getCdcJwt(fields = "firstName,lastName,email") {
  if (!CDC_API_KEY) {
    throw new Error("VITE_CDC_API_KEY is not configured.");
  }

  const response = await _runGigyaCall((callback) => {
    window.gigya.accounts.getJWT({ fields, callback });
  });

  if (!response.id_token) {
    throw new Error("SAP CDC getJWT response did not return an id_token.");
  }

  return response.id_token;
}

/**
 * Check if there's an active CDC session and return account info.
 * @returns {{ email: string, uid: string } | null}
 */
export async function getCurrentAccount() {
  try {
    if (!CDC_API_KEY) return null;

    const response = await _runGigyaCall((callback) => {
      window.gigya.accounts.getAccountInfo({ callback });
    });

    const email = response.profile?.email || response.email;
    const uid = response.UID;

    if (!email || !uid) return null;
    return { email, uid };
  } catch (error) {
    console.warn("Could not restore SAP CDC session:", error);
    return null;
  }
}

/**
 * Log out of the CDC/Gigya session.
 */
export async function cdcLogout() {
  try {
    await _runGigyaCall((callback) => {
      window.gigya.accounts.logout({ callback });
    });
  } catch (error) {
    console.warn("CDC logout error (non-fatal):", error);
  }
}

/**
 * Track login via the backend endpoint.
 * Sends the user's auth token to the server, which writes consent + timestamps
 * to CDC using admin credentials. Works for both XSUAA and CDC users.
 * Non-blocking — always resolves, logs warning on failure.
 */
export async function trackLoginServer() {
  try {
    const { getAccessTokenAsync } = await import("./auth.js");
    const token = await getAccessTokenAsync();
    if (!token) return;

    const apiBase = (import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000").replace(/\/$/, "");
    await fetch(`${apiBase}/api/auth/track-login`, {
      method: "POST",
      headers: { Authorization: `Bearer ${token}` },
    });
  } catch (e) {
    console.warn("Login tracking failed (non-fatal):", e);
  }
}

/**
 * Check if CDC is configured (API key present).
 */
export function isCdcConfigured() {
  return Boolean(CDC_API_KEY);
}
