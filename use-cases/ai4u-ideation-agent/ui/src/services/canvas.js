/**
 * Canvas persistence service — save/load tile positions per workspace.
 *
 * All canvases (personal + shared) use the same endpoints:
 * - GET /tiles → load (returns { tiles, server_time })
 * - PUT /tiles → full-board save
 * - DELETE /tiles/:id → remove single tile
 *
 * Shared canvases poll GET /tiles every 5s for near-instant sync.
 */
import { request } from "./api.js";

export const PERSONAL_WORKSPACE_ID = "__personal__";
export const COMPANY_WORKSPACE_ID = "__company__";

/**
 * Check if a workspace is shared (company or event).
 * @param {string} workspaceId
 * @returns {boolean}
 */
export function isSharedWorkspace(workspaceId) {
  return workspaceId !== PERSONAL_WORKSPACE_ID;
}

/**
 * Load saved tiles for a workspace.
 * @param {string} workspaceId
 * @returns {Promise<{tiles: Array, server_time: string}>}
 */
export async function loadCanvasTiles(workspaceId) {
  return request(`/api/canvas/${encodeURIComponent(workspaceId)}/tiles`);
}

/**
 * Bulk-save the full board state (full-replace).
 * @param {string} workspaceId
 * @param {Array} tiles
 * @returns {Promise<{saved: number}>}
 */
export async function saveCanvasTiles(workspaceId, tiles) {
  return request(`/api/canvas/${encodeURIComponent(workspaceId)}/tiles`, "PUT", { tiles });
}

/**
 * Lightweight position-only update for moved tiles (PATCH).
 * @param {string} workspaceId
 * @param {Array<{tile_id: string, pos_x: number, pos_y: number, z_index: number}>} moves
 * @returns {Promise<{status: string, updated: number}>}
 */
export async function patchTilePositions(workspaceId, moves) {
  return request(`/api/canvas/${encodeURIComponent(workspaceId)}/tiles`, "PATCH", { moves });
}

/**
 * Delete a single tile from a workspace.
 * @param {string} workspaceId
 * @param {string} tileId
 * @returns {Promise<{status: string}>}
 */
export async function deleteCanvasTile(workspaceId, tileId) {
  return request(
    `/api/canvas/${encodeURIComponent(workspaceId)}/tiles/${encodeURIComponent(tileId)}`,
    "DELETE",
  );
}

/**
 * Acquire an edit lock on a tile (shared canvases).
 * @param {string} workspaceId
 * @param {string} tileId
 * @returns {Promise<{status: string}>}
 */
export async function lockCanvasTile(workspaceId, tileId) {
  return request(
    `/api/canvas/${encodeURIComponent(workspaceId)}/tiles/${encodeURIComponent(tileId)}/lock`,
    "POST",
  );
}

/**
 * Release an edit lock on a tile.
 * @param {string} workspaceId
 * @param {string} tileId
 * @returns {Promise<{status: string}>}
 */
export async function unlockCanvasTile(workspaceId, tileId) {
  return request(
    `/api/canvas/${encodeURIComponent(workspaceId)}/tiles/${encodeURIComponent(tileId)}/unlock`,
    "POST",
  );
}
