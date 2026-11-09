import { request } from "./api.js";

/**
 * Create a new event.
 * @param {string} name
 * @returns {Promise<{event_id: string, join_code: string, name: string, status: string, created_at: string}>}
 */
export async function createEvent(name) {
  return request("/api/events", "POST", { name });
}

/**
 * List all events the current user is a member of.
 * @returns {Promise<Array<{event_id, join_code, name, status, created_at, member_count, role, joined_at}>>}
 */
export async function getMyEvents() {
  return request("/api/events/my");
}

/**
 * Resolve a join code to event info (works for members and non-members).
 * @param {string} code
 * @returns {Promise<{event_id, join_code, name, status, member_count, is_member, role}>}
 */
export async function getEventByCode(code) {
  return request(`/api/events/by-code/${encodeURIComponent(code.toUpperCase())}`);
}

/**
 * Join an event.
 * @param {string} eventId
 * @returns {Promise<{status: string, message: string}>}
 */
export async function joinEvent(eventId) {
  return request(`/api/events/${eventId}/join`, "POST");
}

/**
 * Leave an event.
 * @param {string} eventId
 * @returns {Promise<{status: string, message: string}>}
 */
export async function leaveEvent(eventId) {
  return request(`/api/events/${eventId}/leave`, "POST");
}

/**
 * Update event status (admin only).
 * @param {string} eventId
 * @param {"active"|"ended"} status
 * @returns {Promise<{status: string, message: string}>}
 */
export async function updateEventStatus(eventId, status) {
  return request(`/api/events/${eventId}/status`, "PATCH", { status });
}

/**
 * Get event leaderboard.
 * @param {string} eventId
 * @param {number} [limit=50]
 * @returns {Promise<Array>}
 */
export async function getEventLeaderboard(eventId, limit = 50) {
  return request(`/api/events/${eventId}/leaderboard?limit=${limit}`);
}

/**
 * Get event members list.
 * @param {string} eventId
 * @returns {Promise<Array<{uid, email, display_name, role, joined_at}>>}
 */
export async function getEventMembers(eventId) {
  return request(`/api/events/${eventId}/members`);
}

/**
 * Update a member's role (admin only).
 * @param {string} eventId
 * @param {string} targetUid
 * @param {"admin"|"member"} role
 * @returns {Promise<{status: string, role: string}>}
 */
export async function updateMemberRole(eventId, targetUid, role) {
  return request(`/api/events/${eventId}/members/${encodeURIComponent(targetUid)}/role`, "PATCH", { role });
}
