import { getCdcJwt, cdcLogout, getCurrentAccount } from "./cdc.js";

const XSUAA_URL = (import.meta.env.VITE_XSUAA_URL || "").replace(/\/$/, "");
const CLIENT_ID = import.meta.env.VITE_XSUAA_CLIENT_ID || "";
const IDP_ORIGIN = (import.meta.env.VITE_XSUAA_IDP_ORIGIN ?? "sap.default").trim();
const _API_BASE = (import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000").replace(/\/$/, "");
const TOKEN_PROXY_URL = `${_API_BASE}/api/auth/token`;

const TOKEN_KEY = "xsuaa_tokens";
const VERIFIER_KEY = "pkce_verifier";
const RETURN_URL_KEY = "auth_return_url";
const AUTH_PROVIDER_KEY = "auth_provider"; // "xsuaa" | "cdc"
const CDC_TOKEN_KEY = "cdc_token";

function _storage() {
  return sessionStorage;
}

// --- XSUAA token helpers ---

function _getTokens() {
  try {
    const raw = _storage().getItem(TOKEN_KEY);
    return raw ? JSON.parse(raw) : null;
  } catch {
    return null;
  }
}

function _setTokens(tokens) {
  _storage().setItem(TOKEN_KEY, JSON.stringify(tokens));
}

function _clearTokens() {
  _storage().removeItem(TOKEN_KEY);
  _storage().removeItem(VERIFIER_KEY);
}

// --- CDC token helpers ---

function _getCdcToken() {
  return _storage().getItem(CDC_TOKEN_KEY) || null;
}

function _setCdcToken(token) {
  _storage().setItem(CDC_TOKEN_KEY, token);
}

function _clearCdcToken() {
  _storage().removeItem(CDC_TOKEN_KEY);
}

// --- Provider helpers ---

export function getAuthProvider() {
  return _storage().getItem(AUTH_PROVIDER_KEY) || null;
}

function _setAuthProvider(provider) {
  _storage().setItem(AUTH_PROVIDER_KEY, provider);
}

function _clearAll() {
  _clearTokens();
  _clearCdcToken();
  _storage().removeItem(AUTH_PROVIDER_KEY);
  _storage().removeItem(RETURN_URL_KEY);
}

// --- JWT decoding ---

function _decodeJwtPayload(token) {
  const base64Url = token.split(".")[1];
  const base64 = base64Url.replace(/-/g, "+").replace(/_/g, "/");
  const json = decodeURIComponent(
    atob(base64)
      .split("")
      .map((c) => "%" + ("00" + c.charCodeAt(0).toString(16)).slice(-2))
      .join("")
  );
  return JSON.parse(json);
}

function _isTokenExpired(token) {
  try {
    const payload = _decodeJwtPayload(token);
    const now = Math.floor(Date.now() / 1000);
    return payload.exp <= now + 60; // 60s buffer
  } catch {
    return true;
  }
}

// --- PKCE ---

async function _generatePKCE() {
  const array = new Uint8Array(32);
  crypto.getRandomValues(array);
  const verifier = btoa(String.fromCharCode(...array))
    .replace(/\+/g, "-")
    .replace(/\//g, "_")
    .replace(/=+$/, "");

  const encoder = new TextEncoder();
  const data = encoder.encode(verifier);
  const digest = await crypto.subtle.digest("SHA-256", data);
  const challenge = btoa(String.fromCharCode(...new Uint8Array(digest)))
    .replace(/\+/g, "-")
    .replace(/\//g, "_")
    .replace(/=+$/, "");

  return { verifier, challenge };
}

// --- XSUAA login flow ---

export async function loginXsuaa(returnUrl) {
  if (!XSUAA_URL || !CLIENT_ID) {
    console.error("XSUAA not configured. Set VITE_XSUAA_URL and VITE_XSUAA_CLIENT_ID.");
    return;
  }

  const { verifier, challenge } = await _generatePKCE();
  _storage().setItem(VERIFIER_KEY, verifier);
  _storage().setItem(RETURN_URL_KEY, returnUrl || window.location.href);

  const redirectUri = `${window.location.origin}/auth/callback`;
  const params = new URLSearchParams({
    response_type: "code",
    client_id: CLIENT_ID,
    redirect_uri: redirectUri,
    code_challenge: challenge,
    code_challenge_method: "S256",
  });
  if (IDP_ORIGIN) {
    params.set("login_hint", JSON.stringify({ origin: IDP_ORIGIN }));
  }

  window.location.href = `${XSUAA_URL}/oauth/authorize?${params.toString()}`;
}

export async function handleCallback() {
  const params = new URLSearchParams(window.location.search);
  const code = params.get("code");
  if (!code) {
    throw new Error("No authorization code in callback URL");
  }

  const verifier = _storage().getItem(VERIFIER_KEY);
  if (!verifier) {
    throw new Error("No PKCE verifier found — login flow was not started properly");
  }

  const redirectUri = `${window.location.origin}/auth/callback`;
  const tokenResponse = await fetch(TOKEN_PROXY_URL, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      grant_type: "authorization_code",
      code,
      redirect_uri: redirectUri,
      code_verifier: verifier,
    }),
  });

  if (!tokenResponse.ok) {
    const err = await tokenResponse.text();
    throw new Error(`Token exchange failed: ${tokenResponse.status} ${err}`);
  }

  const tokens = await tokenResponse.json();
  _setTokens({
    access_token: tokens.access_token,
    refresh_token: tokens.refresh_token || null,
    expires_at: Math.floor(Date.now() / 1000) + (tokens.expires_in || 3600),
  });
  _setAuthProvider("xsuaa");

  _storage().removeItem(VERIFIER_KEY);
  const returnUrl = _storage().getItem(RETURN_URL_KEY) || "/";
  _storage().removeItem(RETURN_URL_KEY);
  return returnUrl;
}

export async function refreshAccessToken() {
  const tokens = _getTokens();
  if (!tokens?.refresh_token) {
    _clearTokens();
    return null;
  }

  const tokenResponse = await fetch(TOKEN_PROXY_URL, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      grant_type: "refresh_token",
      refresh_token: tokens.refresh_token,
    }),
  });

  if (!tokenResponse.ok) {
    _clearTokens();
    return null;
  }

  const newTokens = await tokenResponse.json();
  _setTokens({
    access_token: newTokens.access_token,
    refresh_token: newTokens.refresh_token || tokens.refresh_token,
    expires_at: Math.floor(Date.now() / 1000) + (newTokens.expires_in || 3600),
  });

  return newTokens.access_token;
}

// --- CDC login flow ---

/**
 * Store a CDC JWT token after successful OTP login.
 */
export function loginWithCdcToken(token) {
  _setCdcToken(token);
  _setAuthProvider("cdc");
}

/**
 * Refresh the CDC token silently via the Gigya SDK session.
 * Returns the new token or null if session expired.
 */
export async function refreshCdcToken() {
  try {
    const newToken = await getCdcJwt("firstName,lastName,email");
    _setCdcToken(newToken);
    return newToken;
  } catch {
    _clearCdcToken();
    _storage().removeItem(AUTH_PROVIDER_KEY);
    return null;
  }
}

// --- Unified interface ---

/**
 * Get the current access token regardless of provider (sync).
 * For XSUAA: returns the stored token if not expired.
 * For CDC: returns the stored token if not expired (may be stale).
 * Use getAccessTokenAsync() for guaranteed-fresh CDC tokens.
 */
export function getAccessToken() {
  const provider = getAuthProvider();

  if (provider === "cdc") {
    const token = _getCdcToken();
    if (!token) return null;
    if (_isTokenExpired(token)) return null;
    return token;
  }

  // XSUAA (default)
  const tokens = _getTokens();
  if (!tokens?.access_token) return null;
  if (_isTokenExpired(tokens.access_token)) return null;
  return tokens.access_token;
}

/**
 * Get a fresh access token (async). For CDC users, returns the stored token
 * if still valid, otherwise calls the Gigya SDK for a fresh one.
 * Returns null if session is expired (user must re-login).
 */
export async function getAccessTokenAsync() {
  const provider = getAuthProvider();

  if (provider === "cdc") {
    // Use stored token if still valid (avoid unnecessary SDK calls)
    const stored = _getCdcToken();
    if (stored && !_isTokenExpired(stored)) {
      return stored;
    }
    // Token expired or missing — get fresh one from Gigya SDK
    try {
      const freshToken = await getCdcJwt("firstName,lastName,email");
      _setCdcToken(freshToken);
      return freshToken;
    } catch {
      // Gigya session expired — user needs to re-login
      return null;
    }
  }

  // XSUAA: try stored token first, then refresh
  const tokens = _getTokens();
  if (tokens?.access_token && !_isTokenExpired(tokens.access_token)) {
    return tokens.access_token;
  }
  if (tokens?.refresh_token) {
    return await refreshAccessToken();
  }
  return null;
}

export function isAuthenticated() {
  return getAccessToken() !== null;
}

export function getUserInfo() {
  const token = getAccessToken();
  if (!token) return null;
  try {
    const payload = _decodeJwtPayload(token);
    const provider = getAuthProvider();

    if (provider === "cdc") {
      return {
        userId: payload.sub || "",
        email: payload.email || "",
        givenName: payload.firstName || payload.first_name || "",
        familyName: payload.lastName || payload.last_name || "",
        authProvider: "cdc",
      };
    }

    // XSUAA
    return {
      userId: payload.user_name || payload.sub || "",
      email: payload.email || "",
      givenName: payload.given_name || "",
      familyName: payload.family_name || "",
      authProvider: "xsuaa",
    };
  } catch {
    return null;
  }
}

/**
 * Whether the current user has a corporate (non-public) email domain.
 * Users with public providers (gmail, hotmail, etc.) don't have access
 * to the company leaderboard.
 */
const PUBLIC_EMAIL_DOMAINS = new Set([
  "gmail.com", "hotmail.com", "outlook.com", "yahoo.com", "icloud.com",
  "aol.com", "proton.me", "protonmail.com", "mail.com", "gmx.com",
  "live.com", "msn.com", "me.com", "yandex.com", "zoho.com", "pm.me",
]);

export function hasCompanyEmail() {
  const info = getUserInfo();
  if (!info?.email) return false;
  const domain = info.email.split("@")[1]?.toLowerCase();
  return domain && !PUBLIC_EMAIL_DOMAINS.has(domain);
}

/**
 * Redirect to login page. Called on 401 or when not authenticated.
 * Preserves the return URL so user lands back after login.
 */
export async function login(returnUrl) {
  _storage().setItem(RETURN_URL_KEY, returnUrl || window.location.href);
  // Navigate to login page — main.js will render it
  window.location.href = "/login";
}

export function getReturnUrl() {
  const url = _storage().getItem(RETURN_URL_KEY) || "/";
  _storage().removeItem(RETURN_URL_KEY);
  return url;
}

export async function logout() {
  const provider = getAuthProvider();

  _clearAll();

  if (provider === "cdc") {
    await cdcLogout();
    window.location.href = "/login";
  } else if (XSUAA_URL) {
    const redirectUri = `${window.location.origin}/`;
    window.location.href = `${XSUAA_URL}/logout.do?redirect=${encodeURIComponent(redirectUri)}&client_id=${CLIENT_ID}`;
  } else {
    window.location.href = "/login";
  }
}

/**
 * Attempt to ensure authentication. Returns true if authenticated.
 * For CDC: tries silent token refresh. For XSUAA: tries token refresh.
 * Returns false if not authenticated (caller should show login page).
 */
export async function ensureAuthenticated() {
  if (isAuthenticated()) return true;

  const provider = getAuthProvider();

  if (provider === "xsuaa") {
    const tokens = _getTokens();
    if (tokens?.refresh_token) {
      const newToken = await refreshAccessToken();
      if (newToken) return true;
    }
  }

  if (provider === "cdc") {
    const newToken = await refreshCdcToken();
    if (newToken) return true;
  }

  // Not authenticated
  return false;
}
