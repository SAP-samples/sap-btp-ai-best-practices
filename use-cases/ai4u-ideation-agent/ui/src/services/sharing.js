import { request } from "./api.js";

/**
 * Create a share link for a conversation.
 * @param {string} convId - Conversation ID
 * @param {string} permission - 'read' or 'write'
 * @returns {Promise<{share_id: string, share_code: string, permission: string, created_at: string}>}
 */
export async function createShare(convId, permission = "read") {
    return request(`/api/ideation-agent/conversations/${convId}/share`, "POST", { permission });
}

/**
 * Get all active shares and members for a conversation.
 * @param {string} convId - Conversation ID
 * @returns {Promise<{shares: Array, members: Array}>}
 */
export async function getShares(convId) {
    return request(`/api/ideation-agent/conversations/${convId}/share`);
}

/**
 * Revoke a share link.
 * @param {string} convId - Conversation ID
 * @param {string} shareId - Share ID to revoke
 */
export async function revokeShare(convId, shareId) {
    return request(`/api/ideation-agent/conversations/${convId}/share/${shareId}`, "DELETE");
}

/**
 * Remove a member from a shared conversation.
 * @param {string} convId - Conversation ID
 * @param {string} uid - User ID to remove
 */
export async function removeMember(convId, uid) {
    return request(`/api/ideation-agent/conversations/${convId}/share/members/${uid}`, "DELETE");
}

/**
 * Join a shared conversation via code.
 * @param {string} code - Share code
 * @returns {Promise<{conversation_id: string, permission: string, title: string}>}
 */
export async function joinByCode(code) {
    return request(`/api/ideation-agent/shared/join/${code}`, "POST");
}

/**
 * Preview a shared conversation before joining.
 * @param {string} code - Share code
 * @returns {Promise<{conversation_id: string, title: string, permission: string, owner_name: string, share_code: string}>}
 */
export async function getSharePreview(code) {
    return request(`/api/ideation-agent/shared/preview/${code}`);
}

/**
 * Update the permission level for a conversation's share link (default for new joiners).
 * @param {string} convId - Conversation ID
 * @param {string} permission - 'read' or 'write'
 */
export async function updateSharePermission(convId, permission) {
    return request(`/api/ideation-agent/conversations/${convId}/share/permission`, "PUT", { permission });
}

/**
 * Update a single member's permission.
 * @param {string} convId - Conversation ID
 * @param {string} uid - Target user ID
 * @param {string} permission - 'read' or 'write'
 */
export async function updateMemberPermission(convId, uid, permission) {
    return request(`/api/ideation-agent/conversations/${convId}/share/members/${uid}/permission`, "PUT", { permission });
}
