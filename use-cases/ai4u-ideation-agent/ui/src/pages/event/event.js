/**
 * Event workspace page — individual event at /event/<CODE>.
 * Single-column leaderboard dashboard with hero header.
 * Handles: join flow, leaderboard, admin controls, members.
 */
import {
    getEventByCode,
    joinEvent,
    getEventLeaderboard,
    getEventMembers,
    updateEventStatus,
    updateMemberRole,
} from "../../services/events.js";
import { getUserInfo } from "../../services/auth.js";
import { normalizeEntry, renderLeaderboardEntry, attachLeaderboardInteractions } from "../../components/leaderboard-card.js";
import "@ui5/webcomponents/dist/Toast.js";
import "@ui5/webcomponents/dist/Dialog.js";
import "@ui5/webcomponents-fiori/dist/IllustratedMessage.js";
import "@ui5/webcomponents-fiori/dist/illustrations/NoData.js";
import "@ui5/webcomponents-icons/dist/share-2.js";

export default function initEventPage(code) {
    if (!code) {
        window.location.href = "/events";
        return;
    }

    const user = getUserInfo();
    let _event = null;
    let _role = null;

    // DOM refs
    const loadingEl = document.getElementById("event-loading");
    const joinOverlay = document.getElementById("event-join-overlay");
    const mainEl = document.getElementById("event-main");

    // Hero
    const titleEl = document.getElementById("event-title");
    const memberCountEl = document.getElementById("event-member-count");
    const statusBadge = document.getElementById("event-status-badge");
    const newSolutionBtn = document.getElementById("event-new-solution");

    // Admin
    const adminControls = document.getElementById("event-admin-controls");
    const toggleStatusBtn = document.getElementById("event-toggle-status");

    // Leaderboard
    const lbList = document.getElementById("event-leaderboard-list");
    const lbEmpty = document.getElementById("event-lb-empty");
    const lbBusy = document.getElementById("event-lb-busy");

    // ── Initialize ──────────────────────────────────────────────────────

    async function init() {
        try {
            _event = await getEventByCode(code);
        } catch (err) {
            console.error("Event not found:", err);
            loadingEl.innerHTML = `<ui5-message-strip design="Negative">Event not found. <a href="/events">Back to Events</a></ui5-message-strip>`;
            return;
        }

        loadingEl.hidden = true;

        if (!_event.is_member) {
            _showJoinOverlay();
            return;
        }

        _role = _event.role;
        _showWorkspace();
    }

    // ── Join overlay ────────────────────────────────────────────────────

    function _showJoinOverlay() {
        document.getElementById("join-overlay-name").textContent = _event.name;

        document.getElementById("join-overlay-members").textContent =
            `${_event.member_count} participant${_event.member_count !== 1 ? "s" : ""}`;

        const statusEl = document.getElementById("join-overlay-status");
        const statusText = _event.status.charAt(0).toUpperCase() + _event.status.slice(1);
        statusEl.textContent = statusText;
        if (_event.status === "ended") statusEl.closest(".join-status-badge").classList.add("ended");

        const joinBtn = document.getElementById("join-overlay-btn");
        if (_event.status === "ended") {
            joinBtn.textContent = "Event has ended";
            joinBtn.disabled = true;
        }

        joinOverlay.hidden = false;

        joinBtn.addEventListener("click", async () => {
            joinBtn.disabled = true;
            try {
                await joinEvent(_event.event_id);
                joinOverlay.hidden = true;
                _role = "member";
                _event.is_member = true;
                _showWorkspace();
            } catch (err) {
                joinBtn.disabled = false;
                console.error("Failed to join:", err);
            }
        });
    }

    // ── Workspace ───────────────────────────────────────────────────────

    function _showWorkspace() {
        mainEl.hidden = false;

        // Hero header
        titleEl.textContent = _event.name;
        memberCountEl.textContent = `${_event.member_count} participant${_event.member_count !== 1 ? "s" : ""}`;
        statusBadge.textContent = _event.status === "active" ? "Active" : "Ended";
        if (_event.status === "ended") {
            statusBadge.classList.add("ended");
        }

        // Ended state
        if (_event.status === "ended") {
            newSolutionBtn.disabled = true;
            newSolutionBtn.textContent = "Event Ended";
        }

        // Admin controls
        if (_role === "admin") {
            adminControls.hidden = false;
            toggleStatusBtn.textContent = _event.status === "active" ? "End Event" : "Reopen Event";
            toggleStatusBtn.design = _event.status === "active" ? "Attention" : "Positive";
        }

        // Load data
        _loadLeaderboard();
        _setupEventListeners();
    }

    // ── Leaderboard ─────────────────────────────────────────────────────

    async function _loadLeaderboard() {
        lbBusy.active = true;
        lbEmpty.hidden = true;

        try {
            const entries = await getEventLeaderboard(_event.event_id);
            lbBusy.active = false;

            if (!entries || entries.length === 0) {
                lbEmpty.hidden = false;
                lbList.innerHTML = "";
                return;
            }

            lbList.innerHTML = entries.map((entry) => {
                const normalized = normalizeEntry(entry, "event");
                return renderLeaderboardEntry(normalized);
            }).join("");

            attachLeaderboardInteractions(lbList);
        } catch (err) {
            lbBusy.active = false;
            console.error("Failed to load leaderboard:", err);
        }
    }

    // ── New solution (navigate to main page) ─────────────────────────────

    function _createNewSolution() {
        if (_event.status === "ended") return;
        window.location.href = `/?event_id=${_event.event_id}&event_code=${_event.join_code}`;
    }

    // ── Event listeners ─────────────────────────────────────────────────

    function _setupEventListeners() {
        // Share event — open dialog
        document.getElementById("event-share")?.addEventListener("click", () => {
            const dialog = document.getElementById("event-share-dialog");
            const codeEl = document.getElementById("share-event-code");
            const linkEl = document.getElementById("share-event-link");
            const link = `${window.location.origin}/event/${_event.join_code}`;
            codeEl.textContent = _event.join_code;
            linkEl.value = link;
            dialog.open = true;
        });

        document.getElementById("share-copy-code-btn")?.addEventListener("click", () => {
            navigator.clipboard.writeText(_event.join_code);
            const toast = document.getElementById("event-toast");
            if (toast) { toast.textContent = "Code copied!"; toast.open = true; }
        });

        document.getElementById("share-copy-link-btn")?.addEventListener("click", () => {
            const linkEl = document.getElementById("share-event-link");
            navigator.clipboard.writeText(linkEl.value);
            const toast = document.getElementById("event-toast");
            if (toast) { toast.textContent = "Link copied!"; toast.open = true; }
        });

        document.getElementById("event-share-close")?.addEventListener("click", () => {
            document.getElementById("event-share-dialog").open = false;
        });

        // New solution
        newSolutionBtn.addEventListener("click", _createNewSolution);

        // Back to events hub
        document.getElementById("event-back-hub").addEventListener("click", () => {
            window.location.href = "/events";
        });

        // Refresh leaderboard
        document.getElementById("event-refresh-lb").addEventListener("click", _loadLeaderboard);

        // Admin: toggle status
        if (_role === "admin") {
            toggleStatusBtn.addEventListener("click", async () => {
                const newStatus = _event.status === "active" ? "ended" : "active";
                const confirmed = confirm(
                    newStatus === "ended"
                        ? "End this event? Participants won't be able to submit new solutions."
                        : "Reopen this event? Participants will be able to submit again."
                );
                if (!confirmed) return;

                try {
                    await updateEventStatus(_event.event_id, newStatus);
                    _event.status = newStatus;
                    toggleStatusBtn.textContent = newStatus === "active" ? "End Event" : "Reopen Event";
                    toggleStatusBtn.design = newStatus === "active" ? "Attention" : "Positive";
                    newSolutionBtn.disabled = newStatus === "ended";
                    newSolutionBtn.textContent = newStatus === "ended" ? "Event Ended" : "New Solution";
                    statusBadge.textContent = newStatus === "active" ? "Active" : "Ended";
                    statusBadge.classList.toggle("ended", newStatus === "ended");
                } catch (err) {
                    console.error("Failed to update status:", err);
                }
            });

            // Members dialog
            document.getElementById("event-show-members").addEventListener("click", _showMembersDialog);
            document.getElementById("event-members-close").addEventListener("click", () => {
                document.getElementById("event-members-dialog").open = false;
            });
        }
    }

    // ── Members dialog ──────────────────────────────────────────────────

    async function _showMembersDialog() {
        const dialog = document.getElementById("event-members-dialog");
        const listEl = document.getElementById("event-members-list");
        listEl.innerHTML = `<ui5-busy-indicator active size="S"></ui5-busy-indicator>`;
        dialog.open = true;

        try {
            const members = await getEventMembers(_event.event_id);
            listEl.innerHTML = members.map((m) => `
                <div class="member-row">
                    <ui5-text class="member-name">${_escapeHtml(m.display_name)}</ui5-text>
                    <ui5-text class="member-role">${m.role}</ui5-text>
                    ${_role === "admin" && m.uid !== user?.userId ? `
                        <ui5-button design="Transparent" class="member-role-btn"
                            data-uid="${m.uid}" data-current-role="${m.role}"
                            tooltip="${m.role === "admin" ? "Demote to member" : "Promote to admin"}">
                            ${m.role === "admin" ? "Demote" : "Promote"}
                        </ui5-button>
                    ` : ""}
                </div>
            `).join("");

            // Attach promote/demote handlers
            listEl.querySelectorAll(".member-role-btn").forEach((btn) => {
                btn.addEventListener("click", async () => {
                    const uid = btn.dataset.uid;
                    const newRole = btn.dataset.currentRole === "admin" ? "member" : "admin";
                    try {
                        await updateMemberRole(_event.event_id, uid, newRole);
                        _showMembersDialog(); // Refresh
                    } catch (err) {
                        console.error("Failed to update role:", err);
                    }
                });
            });
        } catch (err) {
            listEl.innerHTML = `<ui5-message-strip design="Negative">Failed to load members</ui5-message-strip>`;
        }
    }

    // ── Helpers ──────────────────────────────────────────────────────────

    function _escapeHtml(str) {
        const div = document.createElement("div");
        div.textContent = str || "";
        return div.innerHTML;
    }

    // Start
    init();
}
