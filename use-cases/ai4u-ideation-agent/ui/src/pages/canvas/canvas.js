import "@ui5/webcomponents/dist/Button.js";
import "@ui5/webcomponents/dist/Title.js";
import "@ui5/webcomponents/dist/Input.js";
import "@ui5/webcomponents/dist/TextArea.js";
import "@ui5/webcomponents/dist/Label.js";
import "@ui5/webcomponents/dist/Dialog.js";
import "@ui5/webcomponents/dist/BusyIndicator.js";
import "@ui5/webcomponents/dist/Icon.js";
import "@ui5/webcomponents/dist/Select.js";
import "@ui5/webcomponents/dist/Option.js";
import "@ui5/webcomponents/dist/Text.js";
import "@ui5/webcomponents/dist/Toast.js";
import "@ui5/webcomponents/dist/Popover.js";
import "@ui5/webcomponents-fiori/dist/IllustratedMessage.js";
import "@ui5/webcomponents-fiori/dist/illustrations/NoData.js";

import Panzoom from "@panzoom/panzoom";
import { getCompanyLeaderboard, openEntryDetailDialog } from "../../services/leaderboard.js";
import { getMyEvents, getEventLeaderboard } from "../../services/events.js";
import { hasCompanyEmail, getUserInfo } from "../../services/auth.js";
import { request, getAuthHeaders, API_BASE_URL } from "../../services/api.js";
import {
  PERSONAL_WORKSPACE_ID,
  COMPANY_WORKSPACE_ID,
  loadCanvasTiles,
  saveCanvasTiles,
  patchTilePositions,
  deleteCanvasTile,
  lockCanvasTile,
  unlockCanvasTile,
  isSharedWorkspace,
} from "../../services/canvas.js";


const SNAP_GRID = 20;
const TILE_WIDTH = 320;
const TILE_H_GAP = 360;
const TILE_V_GAP = 280;
// Axis geometry — cells align exactly with TILE_H_GAP / TILE_V_GAP
const AXIS_LEFT   = 260; // space for Y-axis labels + title
const AXIS_TOP    =  80; // space above chart
const AXIS_CELL_W = TILE_H_GAP;   // 360
const AXIS_CELL_H = TILE_V_GAP;   // 280
const AXIS_COLS   = 5;
const AXIS_ROWS   = 5;

const CHART_W = AXIS_LEFT + AXIS_COLS * AXIS_CELL_W + 60 + TILE_WIDTH + 60;
const CHART_H = AXIS_TOP  + AXIS_ROWS * AXIS_CELL_H + 110;

// Board canvas dimensions — must match .canvas-board CSS width/height
const BOARD_W = 4000;
const BOARD_H = 3000;

// Polling interval for shared canvases (ms)
const POLL_INTERVAL = 5000;

let panzoom = null;
let editingTileId = null;
let _activeWorkspaceId = PERSONAL_WORKSPACE_ID;

// Stores the raw entry object for each entry tile so drag-back-to-sidebar can recover it
const _tileEntryMap = new WeakMap();

// Lookup: contextId → full entry object (populated on workspace load, used by _diffAndApply)
let _entriesByContextId = new Map();

// Viewport and board refs kept module-level for clamp access
let _viewport = null;
let _board = null;
let _sidebar = null;
let _zoomLabel = null;
let _suppressClamp = false; // prevents clamp re-entrance during _fitToView
let _loadSeq = 0;           // incremented on each _loadWorkspace call; stale calls self-abort

// Tiles recently added locally but not yet confirmed by server — protect from SSE removal
const _optimisticTileIds = new Set();

// Tiles recently deleted locally — exclude from any pending _persistBoard saves
const _deletedTileIds = new Set();

// SSE / polling state
let _pollTimer = null;
let _sseAbort = null;  // AbortController for the SSE fetch connection
let _draggingTileId = null; // set during drag, cleared on drop — skip remote updates for this tile
// Tiles with a pending local save — ignore SSE position echoes until confirmed
const _pendingLocalMove = new Map(); // tile_id → { x, y, ts }
let _confirmingTileId = null; // set while delete-confirm dialog is open — skip remote removal
let _editLockPromise = null; // tracks in-flight lock request to prevent unlock-before-lock race

// Minimum pixels of content that must remain visible in each direction
const MIN_VISIBLE = 120;

export default async function initCanvasPage() {
  _viewport = document.getElementById("canvas-viewport");
  _board = document.getElementById("canvas-board");
  _sidebar = document.getElementById("canvas-sidebar");
  _zoomLabel = document.getElementById("canvas-zoom-label");

  // Cache user identity for lock/presence checks
  const userInfo = getUserInfo();
  window.__ai4u_user_id = userInfo?.userId || "";

  panzoom = Panzoom(_board, {
    maxScale: 3,
    minScale: 0.15,
    excludeClass: "tile",
    noBind: true, // we bind to viewport so pan works even off the scaled-down board
  });

  // Bind pan handlers to the viewport so the full viewport area is draggable,
  // not just the (potentially smaller) scaled board element.
  // Skip if the pointer went down on a tile — tile drag handles that itself.
  _viewport.addEventListener("pointerdown", (e) => {
    if (e.target.closest(".tile")) return;
    if (e.target.closest(".canvas-sidebar")) return;
    panzoom.handleDown(e);
  });
  _viewport.addEventListener("pointermove", panzoom.handleMove);
  _viewport.addEventListener("pointerup", panzoom.handleUp);

  _viewport.addEventListener("wheel", (e) => {
    panzoom.zoomWithWheel(e);
  });

  _board.addEventListener("panzoomzoom", (e) => {
    const scale = e.detail?.scale ?? panzoom.getScale();
    _zoomLabel.textContent = Math.round(scale * 100) + "%";
    if (!_suppressClamp) _clampPan();
  });

  _board.addEventListener("panzoompan", () => {
    if (!_suppressClamp) _clampPan();
  });

  // Toolbar buttons
  document.getElementById("canvas-add-tile-btn").addEventListener("click", () => {
    _addUserTile("card");
  });
  document.getElementById("canvas-add-note-btn").addEventListener("click", () => {
    _addUserTile("note");
  });
  document.getElementById("canvas-zoom-in-btn").addEventListener("click", () => panzoom.zoomIn());
  document.getElementById("canvas-zoom-out-btn").addEventListener("click", () => panzoom.zoomOut());
  document.getElementById("canvas-reset-btn").addEventListener("click", () => {
    _fitToView(_viewport, panzoom, _zoomLabel);
  });

  // Edit dialog (for user-added tiles)
  const editDialog = document.getElementById("canvas-tile-dialog");
  document.getElementById("canvas-tile-save-btn").addEventListener("click", () => {
    if (!editingTileId) return;
    const tile = document.getElementById(editingTileId);
    if (tile) {
      const titleEl = tile.querySelector(".tile-card-title");
      const bodyEl = tile.querySelector(".tile-card-body");
      if (titleEl) titleEl.textContent = document.getElementById("canvas-tile-title-input").value;
      if (bodyEl) bodyEl.textContent = document.getElementById("canvas-tile-body-input").value;
    }
    editDialog.open = false;
    _unlockEditingTile();
    editingTileId = null;
    _persistBoard();
  });
  document.getElementById("canvas-tile-cancel-btn").addEventListener("click", () => {
    editDialog.open = false;
    _unlockEditingTile();
    editingTileId = null;
  });
  document.getElementById("canvas-tile-delete-btn").addEventListener("click", () => {
    if (editingTileId) {
      const tileId = editingTileId;
      document.getElementById(tileId)?.remove();
      if (isSharedWorkspace(_activeWorkspaceId)) {
        _deletedTileIds.add(tileId);
        deleteCanvasTile(_activeWorkspaceId, tileId).catch(console.error);
      }
    }
    editDialog.open = false;
    _unlockEditingTile();
    editingTileId = null;
    _clampPan();
    _persistBoard();
  });

  // Safety net: unlock if dialog is dismissed via Escape or backdrop click
  editDialog.addEventListener("after-close", () => {
    if (editingTileId) {
      _unlockEditingTile();
      editingTileId = null;
    }
  });

  // Init sidebar (workspace selector + entry list)
  await _initSidebar();

  // Render axis backdrop on empty board
  _board.innerHTML = "";
  _renderAxisSvg(_board);
  document.getElementById("canvas-board-busy").active = true;

  // Determine initial workspace from URL or default to personal
  const urlParams = new URLSearchParams(window.location.search);
  const initialWorkspace = urlParams.get("workspace") || PERSONAL_WORKSPACE_ID;

  // Pre-select the workspace in the dropdown (if it exists)
  const select = document.getElementById("canvas-workspace-select");
  if (initialWorkspace !== PERSONAL_WORKSPACE_ID) {
    // Wait a tick for sidebar options to be populated, then select
    requestAnimationFrame(() => {
      const opt = select.querySelector(`ui5-option[value="${initialWorkspace}"]`);
      if (opt) opt.selected = true;
    });
  }

  // Load initial workspace
  await _loadWorkspace(initialWorkspace);
  if (isSharedWorkspace(initialWorkspace)) _startPolling();

  // Fit the chart into view on load, then set initial clamp bounds
  _fitToView(_viewport, panzoom, _zoomLabel);
  // Defer clamp so the fit transform is committed to layout first
  requestAnimationFrame(_clampPan);

  // Release locks on page unload (tab close, refresh, navigate away)
  window.addEventListener("beforeunload", _releaseOnUnload);
}

// ── Auto-save ────────────────────────────────────────────────────

let _savePromise = Promise.resolve();
let _saveDebounceTimer = null;

// Position-only batch pipeline: at most 1 in-flight + 1 pending
const _dirtyTiles = new Map(); // tile_id → { pos_x, pos_y, z_index }
let _patchInFlight = false;
let _patchTimer = null;

function _snapshotBoard() {
  if (!_board) return [];
  const tiles = [];
  _board.querySelectorAll(".tile").forEach((tile) => {
    // Skip tiles that were just deleted — prevents resurrection via queued PUT
    if (_deletedTileIds.has(tile.id)) return;

    const tileType = tile.classList.contains("tile-note")
      ? "note"
      : tile.classList.contains("tile-entry")
        ? "entry"
        : "card";

    tiles.push({
      tile_id: tile.id,
      tile_type: tileType,
      context_id: tile.dataset.contextId || null,
      title: tile.querySelector(".tile-card-title")?.textContent ?? "",
      body: tile.querySelector(".tile-card-body")?.textContent ?? "",
      pos_x: parseInt(tile.dataset.x, 10) || 0,
      pos_y: parseInt(tile.dataset.y, 10) || 0,
      color: tile.dataset.color || "",
      z_index: parseInt(tile.style.zIndex, 10) || 0,
    });
  });
  return tiles;
}

/**
 * Full-board save — used for structural changes (add/delete tile, content edits).
 * Personal canvases always use this (prune semantics).
 */
function _persistBoard() {
  const workspaceId = _activeWorkspaceId;
  if (!workspaceId) return;

  clearTimeout(_saveDebounceTimer);
  _saveDebounceTimer = setTimeout(() => {
    const boardSnapshot = _snapshotBoard();
    if (!boardSnapshot.length) return;
    _savePromise = _savePromise
      .then(() => saveCanvasTiles(workspaceId, boardSnapshot))
      .catch((err) => console.error("[canvas] Auto-save failed:", err));
  }, 300);
}

/**
 * Position-only save — used after drag-drop on shared canvases.
 * Batches multiple tile moves and sends a single lightweight PATCH.
 * At most 1 in-flight request; if moves arrive while one is flying,
 * they're batched into the next request (latest positions win).
 */
function _persistMove(tileId, posX, posY, zIndex) {
  const workspaceId = _activeWorkspaceId;
  if (!workspaceId || !isSharedWorkspace(workspaceId)) {
    // Personal canvas — use full-board save
    _persistBoard();
    return;
  }

  // Collect dirty tile position
  _dirtyTiles.set(tileId, { tile_id: tileId, pos_x: posX, pos_y: posY, z_index: zIndex });

  // Schedule flush (coalesce moves within 150ms window)
  clearTimeout(_patchTimer);
  _patchTimer = setTimeout(() => _flushMoves(workspaceId), 150);
}

async function _flushMoves(workspaceId) {
  if (_patchInFlight || !_dirtyTiles.size) return;

  // Grab all pending moves and clear the buffer
  const moves = [..._dirtyTiles.values()];
  _dirtyTiles.clear();
  _patchInFlight = true;

  try {
    await patchTilePositions(workspaceId, moves);
  } catch (err) {
    console.error("[canvas] Patch positions failed:", err);
  } finally {
    _patchInFlight = false;
    // If more moves arrived while we were in-flight, flush them now
    if (_dirtyTiles.size) _flushMoves(workspaceId);
  }
}

function _flushPersist() {
  // Fire any pending saves immediately (used on workspace switch)
  if (_saveDebounceTimer) {
    clearTimeout(_saveDebounceTimer);
    _saveDebounceTimer = null;
    const workspaceId = _activeWorkspaceId;
    const boardSnapshot = _snapshotBoard();
    if (workspaceId && boardSnapshot.length) {
      _savePromise = _savePromise
        .then(() => saveCanvasTiles(workspaceId, boardSnapshot))
        .catch((err) => console.error("[canvas] Auto-save failed:", err));
    }
  }
  // Also flush any pending position moves
  if (_patchTimer) {
    clearTimeout(_patchTimer);
    _patchTimer = null;
    if (_dirtyTiles.size) _flushMoves(_activeWorkspaceId);
  }
}

// ── SSE sync (shared canvases) ─────────────────────────────────

function _startPolling() {
  if (!isSharedWorkspace(_activeWorkspaceId)) return;
  if (_sseAbort || _pollTimer) return;
  _sseAbort = new AbortController();
  _connectSSE(_sseAbort.signal);
}

function _stopPolling() {
  if (_sseAbort) { _sseAbort.abort(); _sseAbort = null; }
  if (_pollTimer) { clearInterval(_pollTimer); _pollTimer = null; }
}

async function _connectSSE(signal) {
  const workspaceId = _activeWorkspaceId;
  const url = `${API_BASE_URL}/api/canvas/${encodeURIComponent(workspaceId)}/stream`;
  try {
    const headers = await getAuthHeaders();
    const response = await fetch(url, { headers, signal });
    if (!response.ok || !response.body) {
      throw new Error(`SSE response ${response.status}`);
    }
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      const parts = buffer.split("\n\n");
      buffer = parts.pop();
      for (const part of parts) {
        if (!part.trim()) continue;
        const match = part.match(/^data:\s*(.+)$/m);
        if (match) {
          try {
            const payload = JSON.parse(match[1]);
            if (payload.tiles) _diffAndApply(payload.tiles);
            if (payload.viewers) _renderPresence(payload.viewers);
          } catch { /* ignore parse errors */ }
        }
      }
    }
  } catch (err) {
    if (err.name === "AbortError") return;
    console.warn("[canvas] SSE disconnected, falling back to polling:", err.message);
    if (!signal.aborted && _activeWorkspaceId === workspaceId) {
      _sseAbort = null;
      _pollTimer = setInterval(_syncSharedCanvas, POLL_INTERVAL);
    }
  }
}

async function _syncSharedCanvas() {
  if (!isSharedWorkspace(_activeWorkspaceId)) return;
  const wsAtStart = _activeWorkspaceId;

  try {
    const data = await loadCanvasTiles(wsAtStart);
    // Guard: if workspace changed while fetch was in-flight, discard stale result
    if (_activeWorkspaceId !== wsAtStart) return;
    const remoteTiles = data?.tiles ?? data ?? [];
    _diffAndApply(remoteTiles);
  } catch (err) {
    console.warn("[canvas] Sync poll failed:", err);
  }
}

function _diffAndApply(remoteTiles) {
  if (!_board) return;

  // Keep _zTop in sync with remote z-index values
  const maxRemoteZ = remoteTiles.reduce((m, t) => Math.max(m, t.z_index || 0), 0);
  if (maxRemoteZ > _zTop) _zTop = maxRemoteZ;

  const remoteMap = new Map(remoteTiles.map((t) => [t.tile_id, t]));
  const localTiles = _board.querySelectorAll(".tile");
  const localIds = new Set();

  localTiles.forEach((el) => {
    localIds.add(el.id);
    const remote = remoteMap.get(el.id);

    // Tile confirmed by server — no longer optimistic
    if (remote && _optimisticTileIds.has(el.id)) _optimisticTileIds.delete(el.id);

    if (!remote) {
      if (el.id !== _draggingTileId && el.id !== editingTileId && el.id !== _confirmingTileId && !_optimisticTileIds.has(el.id)) {
        // Return entry tile to sidebar before removing
        if (el.classList.contains("tile-entry")) {
          const entry = _tileEntryMap.get(el) || _entriesByContextId.get(el.dataset.contextId);
          if (entry) _returnEntryToSidebar(entry);
        }
        el.style.transition = "opacity 0.3s ease";
        el.style.opacity = "0";
        setTimeout(() => { el.remove(); _clampPan(); }, 300);
      }
    } else if (el.id !== _draggingTileId && el.id !== editingTileId && el.id !== _confirmingTileId && !_optimisticTileIds.has(el.id)) {
      const newX = parseInt(remote.pos_x, 10) || 0;
      const newY = parseInt(remote.pos_y, 10) || 0;
      const curX = parseInt(el.dataset.x, 10) || 0;
      const curY = parseInt(el.dataset.y, 10) || 0;

      // Skip position echo if we have a pending local move for this tile
      const pending = _pendingLocalMove.get(el.id);
      if (pending) {
        if (newX === pending.x && newY === pending.y) {
          // Server confirmed our exact position — clear pending, no move needed
          _pendingLocalMove.delete(el.id);
        }
        // Otherwise: server is behind our local state — skip position update entirely
        // (content/z-index/lock still update below)
      } else if (newX !== curX || newY !== curY) {
        el.dataset.x = newX;
        el.dataset.y = newY;
        el.style.transition = "transform 0.3s ease";
        el.style.transform = `translate(${newX}px, ${newY}px)`;
        setTimeout(() => { el.style.transition = ""; }, 320);
      }

      const titleEl = el.querySelector(".tile-card-title");
      const bodyEl = el.querySelector(".tile-card-body");
      if (titleEl && remote.title && titleEl.textContent !== remote.title) {
        titleEl.textContent = remote.title;
      }
      if (bodyEl && remote.body !== undefined && bodyEl.textContent !== remote.body) {
        bodyEl.textContent = remote.body;
      }

      if (remote.z_index !== undefined) {
        el.style.zIndex = remote.z_index;
      }

      _updateLockVisual(el, remote.locked_by);
    }
  });

  // Add new tiles from remote
  for (const remote of remoteTiles) {
    if (localIds.has(remote.tile_id)) continue;

    const x = parseInt(remote.pos_x, 10) || 0;
    const y = parseInt(remote.pos_y, 10) || 0;
    let tileEl;

    if (remote.tile_type === "entry" && remote.context_id) {
      const known = _entriesByContextId.get(remote.context_id);
      const entryData = known
        ? { ...known, contextId: remote.context_id }
        : {
            rank: 0, score: 0, contextId: remote.context_id,
            displayName: "", title: remote.title || "",
            problem: remote.body || "", industry: "", app_type: "", solution: "",
          };
      tileEl = _createEntryTile(entryData, x, y);
    } else {
      tileEl = _createUserTile({
        id: remote.tile_id,
        title: remote.title || (remote.tile_type === "note" ? "Note" : "Tile"),
        body: remote.body || "", x, y,
        color: remote.color || "accent2",
        isNote: remote.tile_type === "note",
      });
    }
    tileEl.style.zIndex = remote.z_index || 0;
    _updateLockVisual(tileEl, remote.locked_by);

    tileEl.style.opacity = "0";
    tileEl.style.transition = "opacity 0.3s ease";
    _board.appendChild(tileEl);
    _initTileDrag(tileEl);
    requestAnimationFrame(() => { tileEl.style.opacity = "1"; });
    setTimeout(() => { tileEl.style.transition = ""; }, 320);

    // Hide sidebar card when entry tile arrives from remote
    if (remote.tile_type === "entry" && remote.context_id) {
      const list = document.getElementById("canvas-entry-list");
      const card = list?.querySelector(`.canvas-entry-card[data-context-id="${remote.context_id}"]`);
      if (card) card.remove();
      if (list && !list.querySelector(".canvas-entry-card") && !list.querySelector("ui5-illustrated-message")) {
        list.innerHTML = `<ui5-illustrated-message name="NoData" design="Spot" title-text="All on the board" subtitle-text="All entries are placed on the canvas."></ui5-illustrated-message>`;
      }
    }
  }

  _clampPan();
}

// ── Sidebar ───────────────────────────────────────────────────

async function _initSidebar() {
  const select = document.getElementById("canvas-workspace-select");

  // Add Company option (only for corporate email users)
  if (hasCompanyEmail()) {
    const companyOpt = document.createElement("ui5-option");
    companyOpt.setAttribute("icon", "building");
    companyOpt.setAttribute("value", COMPANY_WORKSPACE_ID);
    companyOpt.setAttribute("additional-text", "shared");
    companyOpt.textContent = "Company";
    select.appendChild(companyOpt);

    // Update label with actual domain after first load
    getCompanyLeaderboard(1).then((data) => {
      if (data?.companyDomain) companyOpt.textContent = data.companyDomain;
    }).catch(() => {});
  }

  // Add Event options
  let events = [];
  try {
    events = await getMyEvents();
  } catch { /* API unavailable — no events shown */ }

  events?.forEach((ev) => {
    const opt = document.createElement("ui5-option");
    opt.setAttribute("icon", "calendar");
    opt.setAttribute("value", ev.event_id);
    const extra = ev.status === "ended" ? "ended" : "shared";
    opt.setAttribute("additional-text", extra);
    opt.textContent = ev.name;
    select.appendChild(opt);
  });

  select.addEventListener("change", async () => {
    const workspaceId = select.selectedOption?.getAttribute("value") ?? PERSONAL_WORKSPACE_ID;

    // Update URL to reflect current workspace (enables sharing links + refresh)
    const url = new URL(window.location);
    if (workspaceId === PERSONAL_WORKSPACE_ID) {
      url.searchParams.delete("workspace");
    } else {
      url.searchParams.set("workspace", workspaceId);
    }
    history.replaceState(null, "", url);

    // Release any open edit lock before switching
    _unlockEditingTile();
    editingTileId = null;
    document.getElementById("canvas-tile-dialog").open = false;

    // Clear per-workspace transient state
    _optimisticTileIds.clear();
    _deletedTileIds.clear();
    _pendingLocalMove.clear();
    _confirmingTileId = null;

    // Immediately clear UI (don't wait for async save flush)
    if (_board) _board.querySelectorAll(".tile").forEach((t) => t.remove());
    const list = document.getElementById("canvas-entry-list");
    if (list) list.innerHTML = "";
    document.getElementById("canvas-sidebar-busy").active = true;
    document.getElementById("canvas-board-busy").active = true;
    document.getElementById("canvas-presence").innerHTML = "";

    // Stop SSE on old workspace
    _stopPolling();

    // Flush any pending debounced save immediately
    _flushPersist();

    await _loadWorkspace(workspaceId);
    if (panzoom && _viewport && _zoomLabel) _fitToView(_viewport, panzoom, _zoomLabel);

    // Start polling if shared
    if (isSharedWorkspace(workspaceId)) _startPolling();
  });
}

async function _loadWorkspace(workspaceId) {
  const seq = ++_loadSeq;
  _activeWorkspaceId = workspaceId;
  const busy = document.getElementById("canvas-sidebar-busy");
  const list = document.getElementById("canvas-entry-list");
  busy.active = true;
  list.innerHTML = "";

  // 1. Fetch leaderboard entries (data source depends on scope)
  let entries = [];
  try {
    if (workspaceId === PERSONAL_WORKSPACE_ID) {
      const data = await request("/api/ideation-agent/submissions/history?limit=50");
      entries = (data?.entries ?? []).map((e, i) => ({
        rank: i + 1,
        score: e.score ?? 0,
        contextId: e.context_id ?? "",
        displayName: "You",
        industry: e.industry ?? "",
        title: e.title ?? "",
        problem: e.problem ?? "",
        solution: e.solution ?? "",
        app_type: e.app_type ?? "",
        evaluatorFeedback: e.evaluator_feedback ?? "",
        evaluatorDetailedAnalysis: e.evaluator_detailed_analysis ?? "",
        evaluatorSuggestions: e.evaluator_suggestions ?? [],
        appropriatenessLevel: e.appropriateness_level ?? "",
        rating: e.rating ?? 0,
        timestamp: e.submitted_at ?? "",
        isCurrentUser: true,
        handover_status: e.handover_status ?? null,
      }));
    } else if (workspaceId === COMPANY_WORKSPACE_ID) {
      const data = await getCompanyLeaderboard(50);
      entries = data?.entries ?? [];
    } else {
      // Event workspace
      const data = await getEventLeaderboard(workspaceId, 50);
      entries = Array.isArray(data) ? data : (data?.entries ?? []);
    }
  } catch { /* API unavailable — canvas loads with empty sidebar */ }

  if (seq !== _loadSeq) return; // a newer switch arrived while we were fetching

  // Build context_id → entry lookup for SSE diff (remote tiles need rank/score)
  _entriesByContextId = new Map();
  for (const e of entries) {
    const cid = e.contextId ?? e.context_id ?? "";
    if (cid) _entriesByContextId.set(cid, e);
  }

  // 2. Load saved tile positions from backend
  let savedTiles = [];
  try {
    const data = await loadCanvasTiles(workspaceId);
    savedTiles = data?.tiles ?? data ?? [];
  } catch { /* no saved state — fresh canvas */ }

  if (seq !== _loadSeq) return;

  busy.active = false;
  document.getElementById("canvas-board-busy").active = false;

  // 3. Build set of context_ids already on board (from saved tiles)
  const onBoardContextIds = new Set(
    savedTiles
      .filter((t) => t.tile_type === "entry" && t.context_id)
      .map((t) => t.context_id),
  );

  // 4. Restore saved tiles to board (with dedup guard)
  const placedIds = new Set();
  for (const saved of savedTiles) {
    if (!saved.still_valid && saved.tile_type === "entry") continue; // stale entry

    // Compute the DOM id this tile will get — skip if already placed
    const expectedId = saved.tile_type === "entry"
      ? `entry-${(saved.context_id || "").replace(/[^a-zA-Z0-9]/g, "-")}`
      : saved.tile_id;
    if (placedIds.has(expectedId)) continue;
    placedIds.add(expectedId);

    if (saved.tile_type === "entry") {
      // Find the matching entry data for this tile
      const entryData = entries.find((e) => {
        const cid = e.contextId ?? e.context_id ?? "";
        return cid === saved.context_id;
      });
      if (entryData) {
        const tile = _createEntryTile(entryData, saved.pos_x, saved.pos_y);
        tile.style.zIndex = saved.z_index || 0;
        _board.appendChild(tile);
        _initTileDrag(tile);
      }
    } else {
      // User tile (card or note)
      const tile = _createUserTile({
        id: saved.tile_id,
        title: saved.title || (saved.tile_type === "note" ? "Note" : "Tile"),
        body: saved.body || "",
        x: saved.pos_x,
        y: saved.pos_y,
        color: saved.color || "accent2",
        isNote: saved.tile_type === "note",
      });
      tile.style.zIndex = saved.z_index || 0;
      _board.appendChild(tile);
      _initTileDrag(tile);
    }
  }

  // 5. Sidebar shows only entries NOT already on the board
  const sidebarEntries = entries.filter((e) => {
    const cid = e.contextId ?? e.context_id ?? "";
    return cid && !onBoardContextIds.has(cid);
  });

  if (!sidebarEntries.length && !savedTiles.length) {
    list.innerHTML = `<ui5-illustrated-message name="NoData" design="Spot" title-text="No entries yet" subtitle-text="Start an ideation session to add entries to this workspace."></ui5-illustrated-message>`;
    return;
  }

  if (!sidebarEntries.length) {
    list.innerHTML = `<ui5-illustrated-message name="NoData" design="Spot" title-text="All on the board" subtitle-text="All entries are placed on the canvas."></ui5-illustrated-message>`;
    return;
  }

  sidebarEntries.forEach((entry) => {
    const card = _createSidebarCard(entry);
    list.appendChild(card);
    _initSidebarCardDrag(card, entry);
  });
}

function _createSidebarCard(entry) {
  entry = _normalizeEntry(entry);
  const card = document.createElement("div");
  card.className = `canvas-entry-card tile-color-${_rankAccent(entry.rank)}`;
  card.dataset.contextId = entry.contextId;

  const scoreColor = _scoreColor(entry.score);
  const industry = entry.industry
    ? `<ui5-text class="tile-badge">${_esc(entry.industry)}</ui5-text>` : "";
  const appType = entry.app_type
    ? `<ui5-text class="tile-badge tile-badge--type">${_esc(entry.app_type)}</ui5-text>` : "";

  card.innerHTML = `
    <div class="canvas-entry-card-header">
      <span class="tile-rank ${_rankClass(entry.rank)}">#${entry.rank}</span>
      <span class="tile-score" style="color:${scoreColor}">${entry.score}</span>
    </div>
    <div class="canvas-entry-card-body">
      <div class="tile-card-title">${_esc(entry.title)}</div>
      <div class="tile-badges">${industry}${appType}</div>
      <div class="tile-card-body">${_esc(_truncate(entry.problem || "", 100))}</div>
    </div>
    <div class="canvas-entry-card-footer">
      <span class="tile-author">${_esc(entry.displayName)}</span>
      <div class="canvas-card-actions">
        <ui5-button class="canvas-card-handover-btn" icon="journey-arrive" design="Default" tooltip="${entry.handover_status ? "Open Package" : "Create Package"}" aria-label="Handover Package">${entry.handover_status ? "Open" : "Package"}</ui5-button>
        <ui5-button class="tile-detail-btn canvas-card-detail-btn" icon="resize" design="Default" tooltip="View full entry" aria-label="View full entry"></ui5-button>
      </div>
    </div>
  `;

  card.querySelector(".canvas-card-detail-btn").addEventListener("click", (e) => {
    e.stopPropagation();
    openEntryDetailDialog(_normalizeEntry(entry));
  });

  const cardHandoverBtn = card.querySelector(".canvas-card-handover-btn");
  cardHandoverBtn.addEventListener("click", async (e) => {
    e.stopPropagation();
    cardHandoverBtn.setAttribute("loading", "");
    cardHandoverBtn.disabled = true;
    try {
      const { createHandover, listHandovers } = await import("../../services/handover.js");
      if (entry.handover_status) {
        const { items } = await listHandovers();
        const existing = items.find((h) => h.linked_conversation_id === entry.contextId);
        if (existing) {
          window.location.href = `/handover?id=${existing.id}`;
        }
      } else {
        const payload = {};
        if (entry.title) payload.usecase_title = entry.title;
        if (entry.industry) payload.industry = entry.industry;
        const result = await createHandover({
          linkedConversationId: entry.contextId,
          payload: Object.keys(payload).length ? payload : undefined,
        });
        if (result?.id) {
          window.location.href = `/handover?id=${result.id}`;
        }
      }
    } catch (err) {
      console.error("Handover action failed:", err);
      cardHandoverBtn.removeAttribute("loading");
      cardHandoverBtn.disabled = false;
    }
  });

  return card;
}

function _initSidebarCardDrag(card, entry) {
  let dragging = false;
  let ghost = null;

  card.addEventListener("pointerdown", (e) => {
    if (e.target.closest(".canvas-card-detail-btn, .canvas-card-handover-btn")) return;
    dragging = true;
    card.setPointerCapture(e.pointerId);
    card.classList.add("is-dragging");
    // Clone the actual card as the drag ghost
    ghost = card.cloneNode(true);
    ghost.classList.remove("is-dragging");
    ghost.classList.add("canvas-drag-ghost-card");
    ghost.style.left = (e.clientX - card.offsetWidth / 2) + "px";
    ghost.style.top  = (e.clientY - 20) + "px";
    document.body.appendChild(ghost);
    _viewport?.classList.add("drag-over");
  });

  card.addEventListener("pointermove", (e) => {
    if (!dragging) return;
    ghost.style.left = (e.clientX - card.offsetWidth / 2) + "px";
    ghost.style.top  = (e.clientY - 20) + "px";
  });

  card.addEventListener("pointerup", (e) => {
    if (!dragging) return;
    card.releasePointerCapture(e.pointerId);
    dragging = false;
    card.classList.remove("is-dragging");
    ghost?.remove();
    ghost = null;
    _viewport?.classList.remove("drag-over");

    const vpRect = _viewport?.getBoundingClientRect();
    if (
      vpRect &&
      e.clientX >= vpRect.left && e.clientX <= vpRect.right &&
      e.clientY >= vpRect.top  && e.clientY <= vpRect.bottom
    ) {
      const boardRect = _board.getBoundingClientRect();
      const scale = panzoom.getScale();
      const tx = _snap((e.clientX - boardRect.left) / scale - TILE_WIDTH / 2);
      const ty = _snap((e.clientY - boardRect.top)  / scale - 60);
      const tile = _createEntryTile(entry, tx, ty);
      _board.appendChild(tile);
      _initTileDrag(tile);
      _clampPan();
      card.remove();
      // Protect from SSE removal until server confirms
      _optimisticTileIds.add(tile.id);
      _persistBoard();
    }
  });

  card.addEventListener("pointercancel", () => {
    dragging = false;
    card.classList.remove("is-dragging");
    ghost?.remove();
    ghost = null;
    _viewport?.classList.remove("drag-over");
  });
}

function _confirmDeleteTile(tile, origX, origY) {
  const title = tile.querySelector(".tile-card-title")?.textContent?.trim() || "Untitled";
  const body  = tile.querySelector(".tile-card-body")?.textContent?.trim()  || "";
  const isNote = tile.classList.contains("tile-note");
  const label = isNote ? "note" : "tile";

  // Protect tile from remote removal while confirm dialog is open
  _confirmingTileId = tile.id;

  const existing = document.getElementById("canvas-delete-confirm-dialog");
  if (existing) existing.remove();

  const dialog = document.createElement("ui5-dialog");
  dialog.id = "canvas-delete-confirm-dialog";
  dialog.setAttribute("header-text", `Delete ${label}?`);
  dialog.style.cssText = "max-width: 480px;";

  dialog.innerHTML = `
    <div class="canvas-delete-confirm-body">
      ${title ? `<div class="canvas-delete-confirm-field"><span class="canvas-delete-confirm-label">Title</span><span class="canvas-delete-confirm-value">${_esc(title)}</span></div>` : ""}
      ${body  ? `<div class="canvas-delete-confirm-field"><span class="canvas-delete-confirm-label">Content</span><span class="canvas-delete-confirm-value canvas-delete-confirm-body-text">${_esc(body)}</span></div>` : ""}
    </div>
    <div slot="footer" class="canvas-delete-confirm-footer">
      <ui5-button id="canvas-delete-cancel-btn" design="Default">Cancel</ui5-button>
      <ui5-button id="canvas-delete-confirm-btn" design="Negative" icon="delete">Delete permanently</ui5-button>
    </div>
  `;

  document.body.appendChild(dialog);
  dialog.open = true;

  dialog.querySelector("#canvas-delete-cancel-btn").addEventListener("click", () => {
    dialog.open = false;
    _confirmingTileId = null;
    // Restore tile to its position before the drag started
    const nx = _snap(origX);
    const ny = _snap(origY);
    tile.style.transform = `translate(${nx}px, ${ny}px)`;
    tile.dataset.x = nx;
    tile.dataset.y = ny;
    tile.style.zIndex = ++_zTop;
    _clampPan();
  });
  dialog.querySelector("#canvas-delete-confirm-btn").addEventListener("click", () => {
    dialog.open = false;
    _confirmingTileId = null;
    const tileId = tile.id;
    tile.remove();
    _clampPan();
    _persistBoard();
    if (isSharedWorkspace(_activeWorkspaceId)) {
      _deletedTileIds.add(tileId);
      deleteCanvasTile(_activeWorkspaceId, tileId).catch(console.error);
    }
  });
  dialog.addEventListener("close", () => { _confirmingTileId = null; dialog.remove(); });
}

function _returnEntryToSidebar(entry) {
  const list = document.getElementById("canvas-entry-list");
  if (!list) return;

  // Remove empty-state message if present
  const empty = list.querySelector("ui5-illustrated-message");
  if (empty) empty.remove();

  const card = _createSidebarCard(entry);
  _initSidebarCardDrag(card, entry);

  // Insert in rank order so the list stays sorted
  const rank = _normalizeEntry(entry).rank;
  const existing = [...list.querySelectorAll(".canvas-entry-card")];
  const after = existing.find((c) => {
    const r = Number(c.querySelector(".tile-rank")?.textContent?.replace("#", "") ?? 9999);
    return r > rank;
  });
  if (after) {
    list.insertBefore(card, after);
  } else {
    list.appendChild(card);
  }
}

// ── Leaderboard entry tile ──────────────────────────────────

function _createEntryTile(entry, x, y) {
  entry = _normalizeEntry(entry);
  const id = `entry-${entry.contextId.replace(/[^a-zA-Z0-9]/g, "-")}`;
  const tile = document.createElement("div");
  tile.className = `tile tile-entry tile-color-${_rankAccent(entry.rank)}`;
  tile.id = id;
  tile.dataset.x = x;
  tile.dataset.y = y;
  tile.dataset.contextId = entry.contextId;
  tile.style.transform = `translate(${x}px, ${y}px)`;

  const scoreColor = _scoreColor(entry.score);
  const industry = entry.industry
    ? `<ui5-text class="tile-badge">${_esc(entry.industry)}</ui5-text>`
    : "";
  const appType = entry.app_type
    ? `<ui5-text class="tile-badge tile-badge--type">${_esc(entry.app_type)}</ui5-text>`
    : "";
  const handoverBadge = entry.handover_status
    ? `<ui5-text class="tile-badge tile-badge--handover tile-badge--handover-${entry.handover_status}">${entry.handover_status === "validated" ? "✓ Validated" : "Package"}</ui5-text>`
    : "";
  const problem = _truncate(entry.problem || "", 130);

  tile.innerHTML = `
    <div class="tile-drag-handle">
      <span class="tile-drag-dots"></span>
    </div>
    <div class="tile-entry-header">
      <span class="tile-rank ${_rankClass(entry.rank)}">#${entry.rank}</span>
      <span class="tile-score" style="color:${scoreColor}">${entry.score}</span>
    </div>
    <div class="tile-content">
      <div class="tile-card-title">${_esc(entry.title)}</div>
      <div class="tile-badges">${industry}${appType}${handoverBadge}</div>
      <div class="tile-card-body">${_esc(problem)}</div>
    </div>
    <div class="tile-entry-footer">
      <span class="tile-author">${_esc(entry.displayName)}</span>
      <div class="tile-footer-actions">
        <ui5-button class="tile-handover-btn" icon="journey-arrive" design="Default" tooltip="${entry.handover_status ? "Open Package" : "Create Package"}" aria-label="Handover Package">${entry.handover_status ? "Open" : "Package"}</ui5-button>
        <ui5-button class="tile-detail-btn" icon="resize" design="Default" tooltip="View full entry" aria-label="View full entry"></ui5-button>
      </div>
    </div>
    <ui5-button class="tile-remove-btn" icon="delete" design="Default" tooltip="Remove from canvas" aria-label="Remove from canvas"></ui5-button>
  `;

  tile.querySelector(".tile-detail-btn").addEventListener("click", (e) => {
    e.stopPropagation();
    openEntryDetailDialog(entry, () => {
      const tileId = tile.id;
      tile.remove();
      _clampPan();
      _returnEntryToSidebar(entry);
      _persistBoard();
      if (isSharedWorkspace(_activeWorkspaceId)) {
        _deletedTileIds.add(tileId);
        deleteCanvasTile(_activeWorkspaceId, tileId).catch(console.error);
      }
    });
  });

  tile.querySelector(".tile-remove-btn").addEventListener("click", (e) => {
    e.stopPropagation();
    const tileId = tile.id;
    tile.remove();
    _clampPan();
    _returnEntryToSidebar(entry);
    _persistBoard();
    if (isSharedWorkspace(_activeWorkspaceId)) {
      deleteCanvasTile(_activeWorkspaceId, tileId).catch(console.error);
    }
  });

  // Handover button: Start or Open handover (only when feature is enabled)
  const handoverBtn = tile.querySelector(".tile-handover-btn");
  if (handoverBtn) {
  handoverBtn.addEventListener("click", async (e) => {
    e.stopPropagation();
    handoverBtn.setAttribute("loading", "");
    handoverBtn.disabled = true;
    try {
      const { createHandover, listHandovers } = await import("../../services/handover.js");
      if (entry.handover_status) {
        // Navigate to existing handover (find by conversation ID)
        const { items } = await listHandovers();
        const existing = items.find((h) => h.linked_conversation_id === entry.contextId);
        if (existing) {
          window.location.href = `/handover?id=${existing.id}`;
        }
      } else {
        // Create new handover linked to this conversation
        const payload = {};
        if (entry.title) payload.usecase_title = entry.title;
        if (entry.industry) payload.industry = entry.industry;
        const result = await createHandover({
          linkedConversationId: entry.contextId,
          payload: Object.keys(payload).length ? payload : undefined,
        });
        if (result?.id) {
          window.location.href = `/handover?id=${result.id}`;
        }
      }
    } catch (err) {
      console.error("Handover action failed:", err);
      handoverBtn.removeAttribute("loading");
      handoverBtn.disabled = false;
    }
  });
  }

  // Store entry reference so drag-back-to-sidebar can recover it
  _tileEntryMap.set(tile, entry);

  return tile;
}

// ── User-added blank tile ────────────────────────────────────

function _addUserTile(type) {
  const id = `user-${crypto.randomUUID()}`;

  const scale = panzoom.getScale();
  const viewRect = _viewport.getBoundingClientRect();
  const boardRect = _board.getBoundingClientRect();
  const cx = (viewRect.left + viewRect.width / 2 - boardRect.left) / scale;
  const cy = (viewRect.top + viewRect.height / 2 - boardRect.top) / scale;
  const tileX = _snap(cx - TILE_WIDTH / 2);
  const tileY = _snap(cy - 80);

  const colors = ["accent1", "accent2", "accent3", "accent4", "accent5", "accent6"];
  const color = colors[Math.floor(Math.random() * colors.length)];

  const tile = _createUserTile({
    id,
    title: type === "note" ? "New Note" : "New Tile",
    body: "",
    x: tileX,
    y: tileY,
    color,
    isNote: type === "note",
  });

  document.getElementById("canvas-board").appendChild(tile);
  _initTileDrag(tile);
  _optimisticTileIds.add(id);
  _clampPan(); // expand bounds to include new tile
  _openEditDialog(id);
}

function _createUserTile(data) {
  const tile = document.createElement("div");
  tile.className = `tile tile-color-${data.color}${data.isNote ? " tile-note" : ""}`;
  tile.id = data.id;
  tile.dataset.x = data.x;
  tile.dataset.y = data.y;
  tile.dataset.color = data.color;
  tile.style.transform = `translate(${data.x}px, ${data.y}px)`;

  tile.innerHTML = `
    <div class="tile-drag-handle">
      <span class="tile-drag-dots"></span>
    </div>
    <div class="tile-content">
      <div class="tile-card-title">${_esc(data.title)}</div>
      <div class="tile-card-body">${_esc(data.body)}</div>
    </div>
    <ui5-button class="tile-edit-btn" icon="edit" design="Default" tooltip="Edit tile" aria-label="Edit tile"></ui5-button>
    <ui5-button class="tile-remove-btn" icon="delete" design="Default" tooltip="Delete tile" aria-label="Delete tile"></ui5-button>
  `;

  tile.querySelector(".tile-edit-btn").addEventListener("click", (e) => {
    e.stopPropagation();
    _openEditDialog(tile.id);
  });

  tile.querySelector(".tile-remove-btn").addEventListener("click", (e) => {
    e.stopPropagation();
    _confirmDeleteTile(tile, Number(tile.dataset.x) || 0, Number(tile.dataset.y) || 0);
  });

  return tile;
}

function _openEditDialog(tileId) {
  // Guard: prevent double-open (rapid clicks or re-entry while lock in-flight)
  if (editingTileId) return;
  const tile = document.getElementById(tileId);
  if (!tile) return;

  // Open dialog immediately (optimistic)
  editingTileId = tileId;
  _editLockPromise = null;
  document.getElementById("canvas-tile-title-input").value =
    tile.querySelector(".tile-card-title")?.textContent ?? "";
  document.getElementById("canvas-tile-body-input").value =
    tile.querySelector(".tile-card-body")?.textContent ?? "";
  document.getElementById("canvas-tile-dialog").open = true;

  // On shared canvases, acquire lock in background — close dialog if rejected.
  // Skip lock for optimistic tiles that haven't been persisted to DB yet.
  if (isSharedWorkspace(_activeWorkspaceId) && !_optimisticTileIds.has(tileId)) {
    _editLockPromise = lockCanvasTile(_activeWorkspaceId, tileId).catch(() => {
      document.getElementById("canvas-tile-dialog").open = false;
      editingTileId = null;
      _editLockPromise = null;
      _showToast("This tile is being edited by another user");
    });
  }
}

// ── Tile drag ─────────────────────────────────────────────────

function _createDragGhost(tile, scale, rect) {
  const ghost = tile.cloneNode(true);
  ghost.style.cssText = `position:fixed;left:${rect.left}px;top:${rect.top}px;` +
    `transform-origin:top left;transform:scale(${scale});z-index:9999;` +
    `pointer-events:none;opacity:0.85;box-shadow:var(--sapContent_Shadow3);margin:0;`;
  ghost.dataset.startLeft = rect.left;
  ghost.dataset.startTop  = rect.top;
  return ghost;
}

function _initTileDrag(tile) {
  let dragging = false;
  let startClientX, startClientY, startTileX, startTileY;
  let dragGhost = null;

  tile.addEventListener("pointerdown", (e) => {
    if (e.target.closest(".tile-edit-btn, .tile-detail-btn, .tile-remove-btn, .tile-handover-btn")) return;
    e.stopPropagation();
    // Safety: clear any stuck drag state from a previous interaction
    if (_draggingTileId && _draggingTileId !== tile.id) _draggingTileId = null;
    dragging = true;
    _draggingTileId = tile.id;
    tile.setPointerCapture(e.pointerId);
    startClientX = e.clientX;
    startClientY = e.clientY;
    startTileX = Number(tile.dataset.x) || 0;
    startTileY = Number(tile.dataset.y) || 0;

    // For entry tiles: use a fixed-position ghost that floats above everything
    // including the sidebar stacking context, while hiding the real tile.
    const tileRect = tile.getBoundingClientRect();
    dragGhost = _createDragGhost(tile, panzoom.getScale(), tileRect);
    document.body.appendChild(dragGhost);
    tile.style.opacity = "0";
    _sidebar?.classList.add(_tileEntryMap.has(tile) ? "drop-target" : "drop-delete");
  });

  tile.addEventListener("pointermove", (e) => {
    if (!dragging || !tile.hasPointerCapture(e.pointerId)) return;

    const scale = panzoom.getScale();

    // Update tile board position
    const nx = startTileX + (e.clientX - startClientX) / scale;
    const ny = startTileY + (e.clientY - startClientY) / scale;
    tile.style.transform = `translate(${nx}px, ${ny}px)`;
    tile.dataset.x = nx;
    tile.dataset.y = ny;

    // Ghost: clamp its screen position to sidebar + viewport combined bounds
    if (dragGhost) {
      const startLeft = parseFloat(dragGhost.dataset.startLeft);
      const startTop  = parseFloat(dragGhost.dataset.startTop);
      let gx = startLeft + (e.clientX - startClientX);
      let gy = startTop  + (e.clientY - startClientY);

      if (_sidebar && _viewport) {
        const sRect = _sidebar.getBoundingClientRect();
        const vRect = _viewport.getBoundingClientRect();
        const toolbar = document.querySelector(".canvas-toolbar");
        const toolbarBottom = toolbar ? toolbar.getBoundingClientRect().bottom : vRect.top;
        const minX = sRect.left;
        const maxX = vRect.right  - dragGhost.offsetWidth  * scale;
        const minY = toolbarBottom;
        const maxY = Math.max(sRect.bottom, vRect.bottom) - dragGhost.offsetHeight * scale;
        gx = Math.max(minX, Math.min(maxX, gx));
        gy = Math.max(minY, Math.min(maxY, gy));
      }

      dragGhost.style.left = gx + "px";
      dragGhost.style.top  = gy + "px";
      dragGhost.style.transform = `scale(${scale})`;
    }
  });

  tile.addEventListener("pointerup", (e) => {
    if (!dragging || !tile.hasPointerCapture(e.pointerId)) return;
    tile.releasePointerCapture(e.pointerId);
    dragging = false;
    _draggingTileId = null;
    dragGhost?.remove();
    dragGhost = null;
    tile.style.opacity = "";
    _sidebar?.classList.remove("drop-target");
    _sidebar?.classList.remove("drop-delete");

    // Check if dropped over the sidebar
    const sr = _sidebar?.getBoundingClientRect();
    const overSidebar = sr &&
      e.clientX >= sr.left && e.clientX <= sr.right &&
      e.clientY >= sr.top  && e.clientY <= sr.bottom;

    const entry = _tileEntryMap.get(tile);
    if (overSidebar) {
      if (entry) {
        // Entry tile → return to sidebar (no confirmation needed)
        const tileId = tile.id;
        tile.remove();
        _clampPan();
        _returnEntryToSidebar(entry);
        _persistBoard();
        if (isSharedWorkspace(_activeWorkspaceId)) {
          _deletedTileIds.add(tileId);
          deleteCanvasTile(_activeWorkspaceId, tileId).catch(console.error);
        }
      } else {
        // User tile → confirm before deleting; restore to pre-drag position on cancel
        _confirmDeleteTile(tile, startTileX, startTileY);
      }
      return;
    }

    const nx = _snap(Number(tile.dataset.x));
    const ny = _snap(Number(tile.dataset.y));
    tile.style.transform = `translate(${nx}px, ${ny}px)`;
    tile.dataset.x = nx;
    tile.dataset.y = ny;
    tile.style.zIndex = ++_zTop;
    // Mark tile as locally moved (AFTER snap) — ignore SSE echoes until server confirms
    _pendingLocalMove.set(tile.id, { x: nx, y: ny, ts: Date.now() });
    _clampPan();
    _persistMove(tile.id, nx, ny, _zTop);
  });

  tile.addEventListener("pointercancel", () => {
    dragging = false;
    _draggingTileId = null;
    dragGhost?.remove();
    dragGhost = null;
    tile.style.opacity = "";
    _sidebar?.classList.remove("drop-target");
    _sidebar?.classList.remove("drop-delete");
  });
}

// ── Axis SVG backdrop ────────────────────────────────────────

function _renderAxisSvg(board) {
  // W covers only the axis plot area (not the tile overflow to the right)
  const W = AXIS_LEFT + AXIS_COLS * AXIS_CELL_W + 40;
  const H = CHART_H;

  const yLabels = ["High", "Medium-High", "Medium", "Medium-Low", "Low"];
  const xLabels = ["Low", "Medium-Low", "Medium", "Medium-High", "High"];

  // Quadrant background fills (top-right = high value+feasibility = green, etc.)
  const quadrants = [
    { col: 3, row: 0, fill: "rgba(29,125,57,0.06)"  },
    { col: 4, row: 0, fill: "rgba(29,125,57,0.06)"  },
    { col: 3, row: 1, fill: "rgba(29,125,57,0.04)"  },
    { col: 4, row: 1, fill: "rgba(29,125,57,0.04)"  },
    { col: 0, row: 3, fill: "rgba(187,0,0,0.04)"    },
    { col: 0, row: 4, fill: "rgba(187,0,0,0.05)"    },
    { col: 1, row: 4, fill: "rgba(187,0,0,0.04)"    },
  ];

  let svgContent = "";

  // Quadrant tints
  for (const q of quadrants) {
    const rx = AXIS_LEFT + q.col * AXIS_CELL_W;
    const ry = AXIS_TOP  + q.row * AXIS_CELL_H;
    svgContent += `<rect x="${rx}" y="${ry}" width="${AXIS_CELL_W}" height="${AXIS_CELL_H}" fill="${q.fill}" />`;
  }

  // Grid lines
  for (let c = 0; c <= AXIS_COLS; c++) {
    const x = AXIS_LEFT + c * AXIS_CELL_W;
    const isEdge = c === 0 || c === AXIS_COLS;
    svgContent += `<line x1="${x}" y1="${AXIS_TOP}" x2="${x}" y2="${AXIS_TOP + AXIS_ROWS * AXIS_CELL_H}"
      stroke="${isEdge ? "var(--sapTextColor, #1d2d3e)" : "var(--sapContent_LabelColor, #8c9cb0)"}"
      stroke-width="${isEdge ? 2 : 1}"
      stroke-dasharray="${isEdge ? "none" : "8 6"}" />`;
  }
  for (let r = 0; r <= AXIS_ROWS; r++) {
    const y = AXIS_TOP + r * AXIS_CELL_H;
    const isEdge = r === 0 || r === AXIS_ROWS;
    svgContent += `<line x1="${AXIS_LEFT}" y1="${y}" x2="${AXIS_LEFT + AXIS_COLS * AXIS_CELL_W}" y2="${y}"
      stroke="${isEdge ? "var(--sapTextColor, #1d2d3e)" : "var(--sapContent_LabelColor, #8c9cb0)"}"
      stroke-width="${isEdge ? 2 : 1}"
      stroke-dasharray="${isEdge ? "none" : "8 6"}" />`;
  }

  // Y-axis tick labels
  for (let r = 0; r < AXIS_ROWS; r++) {
    const y = AXIS_TOP + r * AXIS_CELL_H + AXIS_CELL_H / 2;
    svgContent += `<text x="${AXIS_LEFT - 16}" y="${y}" text-anchor="end" dominant-baseline="middle"
      font-size="30" fill="var(--sapContent_LabelColor, #556b7e)" font-family="var(--sapFontFamily, Arial)">${yLabels[r]}</text>`;
  }

  // X-axis tick labels
  for (let c = 0; c < AXIS_COLS; c++) {
    const x = AXIS_LEFT + c * AXIS_CELL_W + AXIS_CELL_W / 2;
    const y = AXIS_TOP + AXIS_ROWS * AXIS_CELL_H + 36;
    svgContent += `<text x="${x}" y="${y}" text-anchor="middle" dominant-baseline="auto"
      font-size="30" fill="var(--sapContent_LabelColor, #556b7e)" font-family="var(--sapFontFamily, Arial)">${xLabels[c]}</text>`;
  }

  // Y-axis title (rotated)
  const axisTitleY = AXIS_TOP + (AXIS_ROWS * AXIS_CELL_H) / 2;
  svgContent += `<text transform="translate(60,${axisTitleY}) rotate(-90)" text-anchor="middle"
    font-size="30" font-weight="600" fill="var(--sapTextColor, #1d2d3e)" font-family="var(--sapFontFamily, Arial)">Business Value ($)</text>`;

  // X-axis title
  const axisTitleX = AXIS_LEFT + (AXIS_COLS * AXIS_CELL_W) / 2;
  svgContent += `<text x="${axisTitleX}" y="${H - 8}" text-anchor="middle"
    font-size="30" font-weight="600" fill="var(--sapTextColor, #1d2d3e)" font-family="var(--sapFontFamily, Arial)">Technical Feasibility</text>`;

  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  svg.setAttribute("class", "canvas-axis-svg");
  svg.setAttribute("width", W);
  svg.setAttribute("height", H);
  svg.style.cssText = `position:absolute;top:0;left:0;pointer-events:none;z-index:0;overflow:visible;`;
  svg.innerHTML = svgContent;

  board.insertBefore(svg, board.firstChild);
}

// ── Pan clamping ─────────────────────────────────────────────

function _getContentBbox() {
  let minX = 0, minY = 0, maxX = CHART_W, maxY = CHART_H;

  if (_board) {
    _board.querySelectorAll(".tile").forEach((tile) => {
      const tx = Number(tile.dataset.x) || 0;
      const ty = Number(tile.dataset.y) || 0;
      const tw = tile.offsetWidth  || TILE_WIDTH;
      const th = tile.offsetHeight || 200;
      minX = Math.min(minX, tx);
      minY = Math.min(minY, ty);
      maxX = Math.max(maxX, tx + tw);
      maxY = Math.max(maxY, ty + th);
    });
  }

  return { minX, minY, maxX, maxY };
}

function _clampPan() {
  if (!panzoom || !_viewport || !_board) return;

  const s   = panzoom.getScale();
  const vw  = _viewport.clientWidth;
  const vh  = _viewport.clientHeight;
  const bb  = _getContentBbox();
  const boardRect  = _board.getBoundingClientRect();
  const vpRect     = _viewport.getBoundingClientRect();

  const contentLeft   = boardRect.left - vpRect.left + bb.minX * s;
  const contentRight  = boardRect.left - vpRect.left + bb.maxX * s;
  const contentTop    = boardRect.top  - vpRect.top  + bb.minY * s;
  const contentBottom = boardRect.top  - vpRect.top  + bb.maxY * s;

  let dx = 0, dy = 0;

  if (contentRight < MIN_VISIBLE)       dx = MIN_VISIBLE - contentRight;
  if (contentLeft  > vw - MIN_VISIBLE)  dx = (vw - MIN_VISIBLE) - contentLeft;
  if (contentBottom < MIN_VISIBLE)      dy = MIN_VISIBLE - contentBottom;
  if (contentTop    > vh - MIN_VISIBLE) dy = (vh - MIN_VISIBLE) - contentTop;

  if (dx !== 0 || dy !== 0) {
    panzoom.pan(dx / s, dy / s, { relative: true, animate: false });
  }
}

// ── Helpers ───────────────────────────────────────────────────

let _zTop = 10;

function _fitToView(viewport, panzoom, zoomLabel) {
  _suppressClamp = true;

  const vw = viewport.clientWidth;
  const vh = viewport.clientHeight;

  const padding = 20;
  const s = Math.min(
    (vw - padding * 2) / CHART_W,
    (vh - padding * 2) / CHART_H,
    1
  );

  const ox = (vw - CHART_W * s) / 2;
  const oy = (vh - CHART_H * s) / 2;

  panzoom.zoom(1, { animate: false });
  panzoom.pan(0, 0, { animate: false });
  panzoom.zoom(s, { animate: false });

  const curOx = (BOARD_W / 2) * (1 - s);
  const curOy = (BOARD_H / 2) * (1 - s);
  panzoom.pan((ox - curOx) / s, (oy - curOy) / s, { relative: true, animate: false });

  zoomLabel.textContent = Math.round(s * 100) + "%";
  requestAnimationFrame(() => { _suppressClamp = false; });
}

function _normalizeEntry(e) {
  return {
    rank:          e.rank,
    score:         e.score ?? 0,
    contextId:     e.contextId   ?? e.context_id   ?? "",
    displayName:   e.displayName ?? e.display_name ?? "",
    industry:      e.industry    ?? "",
    title:         e.title       ?? "",
    problem:       e.problem     ?? "",
    solution:      e.solution    ?? "",
    app_type:      e.app_type    ?? e.appType       ?? "",
    evaluatorFeedback:         e.evaluatorFeedback         ?? e.evaluator_feedback         ?? "",
    evaluatorDetailedAnalysis: e.evaluatorDetailedAnalysis ?? e.evaluator_detailed_analysis ?? "",
    evaluatorSuggestions:      e.evaluatorSuggestions      ?? e.evaluator_suggestions      ?? [],
    appropriatenessLevel:      e.appropriatenessLevel      ?? e.appropriateness_level      ?? "",
    rating:        e.rating      ?? 0,
    timestamp:     e.timestamp   ?? "",
    isCurrentUser: e.isCurrentUser ?? e.is_current_user ?? false,
    handover_status: e.handover_status ?? e.handoverStatus ?? null,
  };
}

function _snap(v) {
  return Math.round(v / SNAP_GRID) * SNAP_GRID;
}

function _esc(str) {
  return String(str ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function _truncate(str, max) {
  if (!str || str.length <= max) return str;
  const sub = str.slice(0, max);
  const lastSpace = sub.lastIndexOf(" ");
  return (lastSpace > max * 0.6 ? sub.slice(0, lastSpace) : sub).trim() + "…";
}

function _scoreColor(score) {
  if (score >= 800) return "#1d7d39";
  if (score >= 600) return "#0070f2";
  if (score >= 400) return "#e76500";
  return "#bb0000";
}

function _rankAccent(rank) {
  if (rank === 1) return "rank-gold";
  if (rank === 2) return "rank-silver";
  if (rank === 3) return "rank-bronze";
  const cycle = ["accent2", "accent4", "accent5", "accent6"];
  return cycle[(rank - 4) % cycle.length];
}

function _rankClass(rank) {
  if (rank === 1) return "tile-rank--gold";
  if (rank === 2) return "tile-rank--silver";
  if (rank === 3) return "tile-rank--bronze";
  return "";
}

// ── Lock visual ────────────────────────────────────────────────

function _updateLockVisual(el, lockedBy) {
  if (!lockedBy || lockedBy === _getCurrentUserId()) {
    el.classList.remove("tile-locked");
  } else {
    el.classList.add("tile-locked");
  }
}

function _getCurrentUserId() {
  return window.__ai4u_user_id || "";
}

function _unlockEditingTile() {
  if (editingTileId && isSharedWorkspace(_activeWorkspaceId)) {
    const tileId = editingTileId;
    const wsId = _activeWorkspaceId;
    // Wait for lock to land before sending unlock (prevents unlock-before-lock race)
    const p = _editLockPromise || Promise.resolve();
    p.finally(() => unlockCanvasTile(wsId, tileId).catch(() => {}));
    _editLockPromise = null;
  }
}

function _releaseOnUnload() {
  if (editingTileId && isSharedWorkspace(_activeWorkspaceId)) {
    const url = `${API_BASE_URL}/api/canvas/${encodeURIComponent(_activeWorkspaceId)}/tiles/${encodeURIComponent(editingTileId)}/unlock`;
    const headers = { type: "application/json" };
    navigator.sendBeacon(url, new Blob([JSON.stringify({})], headers));
  }
}

function _showToast(message) {
  let toast = document.querySelector("ui5-toast");
  if (!toast) {
    toast = document.createElement("ui5-toast");
    toast.setAttribute("placement", "BottomCenter");
    document.body.appendChild(toast);
  }
  toast.textContent = message;
  // Ensure the custom element is upgraded before calling .show()
  if (typeof toast.show === "function") {
    toast.show();
  } else {
    customElements.whenDefined("ui5-toast").then(() => toast.show());
  }
}

// ── Presence ───────────────────────────────────────────────────

function _renderPresence(viewers) {
  const container = document.getElementById("canvas-presence");
  if (!container) return;
  if (!viewers || !viewers.length) { container.innerHTML = ""; return; }

  const shown = viewers.slice(0, 5);
  const overflow = viewers.length - 5;
  container.innerHTML = shown.map((v, i) =>
    `<span class="presence-avatar" id="presence-av-${i}"><ui5-text>${_esc(v.initials)}</ui5-text></span>`
  ).join("") + (overflow > 0 ? `<span class="presence-overflow"><ui5-text>+${overflow}</ui5-text></span>` : "");

  // Attach hover popover for each avatar
  shown.forEach((v, i) => {
    const el = container.querySelector(`#presence-av-${i}`);
    if (!el) return;
    el.addEventListener("mouseenter", () => _showPresencePopover(el, v.display_name));
    el.addEventListener("mouseleave", _hidePresencePopover);
  });
}

let _presencePopover = null;

function _showPresencePopover(opener, name) {
  if (!_presencePopover) {
    _presencePopover = document.createElement("ui5-popover");
    _presencePopover.setAttribute("placement", "Bottom");
    _presencePopover.setAttribute("hide-arrow", "");
    _presencePopover.classList.add("presence-popover");
    _presencePopover.innerHTML = `<ui5-text class="presence-popover-name"></ui5-text>`;
    document.body.appendChild(_presencePopover);
  }
  _presencePopover.querySelector(".presence-popover-name").textContent = name;
  _presencePopover.opener = opener;
  _presencePopover.open = true;
}

function _hidePresencePopover() {
  if (_presencePopover) _presencePopover.open = false;
}
