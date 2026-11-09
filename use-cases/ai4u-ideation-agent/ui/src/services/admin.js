/**
 * Admin API service — stats and access checks.
 */
import { request } from "./api.js";

/**
 * Check if the current user has admin access.
 * Non-blocking: returns false on error.
 */
export async function checkAdmin() {
  try {
    const res = await request("/api/admin/check");
    return res?.isAdmin === true;
  } catch {
    return false;
  }
}

/**
 * Fetch aggregated usage stats for the admin dashboard.
 */
export async function getStats() {
  return request("/api/admin/stats");
}

/**
 * Fetch telemetry data for the admin dashboard (SAP-hosted only).
 * Returns null on 404 (not SAP-hosted) or error.
 */
export async function getTelemetryData() {
  return request("/api/admin/telemetry");
}
