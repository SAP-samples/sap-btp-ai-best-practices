/**
 * Events hub page — list, create, and join events.
 */
import { getMyEvents, createEvent, getEventByCode, joinEvent } from "../../services/events.js";

export default function initEventsPage() {
    const loadingEl = document.getElementById("events-loading");
    const emptyEl = document.getElementById("events-empty");
    const activeSection = document.getElementById("events-active-section");
    const endedSection = document.getElementById("events-ended-section");
    const activeList = document.getElementById("events-active-list");
    const endedList = document.getElementById("events-ended-list");

    // ── Load events ──────────────────────────────────────────────────────
    async function loadEvents() {
        loadingEl.hidden = false;
        emptyEl.hidden = true;
        activeSection.hidden = true;
        endedSection.hidden = true;

        try {
            const events = await getMyEvents();
            loadingEl.hidden = true;

            const active = events.filter((e) => e.status === "active");
            const ended = events.filter((e) => e.status === "ended");

            if (active.length === 0 && ended.length === 0) {
                emptyEl.hidden = false;
                return;
            }

            if (active.length > 0) {
                activeSection.hidden = false;
                activeList.innerHTML = active.map(_renderEventCard).join("");
            }

            if (ended.length > 0) {
                endedSection.hidden = false;
                endedList.innerHTML = ended.map(_renderEventCard).join("");
            }

            // Attach click handlers to cards
            document.querySelectorAll(".event-card").forEach((card) => {
                card.addEventListener("click", () => {
                    const code = card.dataset.code;
                    if (code) window.location.href = `/event/${code}`;
                });
            });
        } catch (err) {
            loadingEl.hidden = true;
            emptyEl.hidden = false;
            console.error("Failed to load events:", err);
        }
    }

    function _renderEventCard(event) {
        const isCreator = event.role === "admin";
        const metaParts = [
            `Code: ${event.join_code}`,
            `${event.member_count} participant${event.member_count !== 1 ? "s" : ""}`,
        ];
        if (isCreator) metaParts.push("Created by you");

        return `
            <div class="event-card" data-code="${event.join_code}" data-id="${event.event_id}">
                <div class="event-card-status ${event.status}"></div>
                <div class="event-card-info">
                    <ui5-text class="event-card-name">${_escapeHtml(event.name)}</ui5-text>
                    <ui5-text class="event-card-meta">${metaParts.join(" • ")}</ui5-text>
                </div>
                <ui5-icon name="navigation-right-arrow" class="event-card-arrow"></ui5-icon>
            </div>
        `;
    }

    // ── Create event dialog ──────────────────────────────────────────────
    const createDialog = document.getElementById("create-event-dialog");
    const createNameInput = document.getElementById("event-name-input");
    const createConfirmBtn = document.getElementById("create-event-confirm");
    const createCancelBtn = document.getElementById("create-event-cancel");

    document.getElementById("create-event-btn").addEventListener("click", () => {
        createNameInput.value = "";
        createDialog.open = true;
    });

    createCancelBtn.addEventListener("click", () => {
        createDialog.open = false;
    });

    createConfirmBtn.addEventListener("click", async () => {
        const name = createNameInput.value.trim();
        if (!name) {
            createNameInput.valueState = "Negative";
            createNameInput.valueStateMessage = "Event name is required";
            return;
        }
        createNameInput.valueState = "None";
        createConfirmBtn.disabled = true;

        try {
            const result = await createEvent(name);
            createDialog.open = false;
            createConfirmBtn.disabled = false;
            _showCreatedDialog(result.join_code);
            loadEvents(); // Refresh list in background
        } catch (err) {
            createConfirmBtn.disabled = false;
            console.error("Failed to create event:", err);
        }
    });

    // ── Event created dialog ─────────────────────────────────────────────
    const createdDialog = document.getElementById("event-created-dialog");
    const createdCodeEl = document.getElementById("created-event-code");
    const createdLinkEl = document.getElementById("created-event-link");

    function _showCreatedDialog(code) {
        const link = `${window.location.origin}/event/${code}`;
        createdCodeEl.textContent = code;
        createdLinkEl.value = link;
        createdDialog.open = true;
    }

    document.getElementById("copy-code-btn").addEventListener("click", () => {
        navigator.clipboard.writeText(createdCodeEl.textContent);
    });

    document.getElementById("copy-link-btn").addEventListener("click", () => {
        navigator.clipboard.writeText(createdLinkEl.value);
    });

    document.getElementById("event-created-open").addEventListener("click", () => {
        createdDialog.open = false;
        window.location.href = `/event/${createdCodeEl.textContent}`;
    });

    // ── Join event dialog ────────────────────────────────────────────────
    const joinDialog = document.getElementById("join-event-dialog");
    const joinCodeInput = document.getElementById("join-code-input");
    const joinPreview = document.getElementById("join-event-preview");
    const joinPreviewName = document.getElementById("join-preview-name");
    const joinPreviewInfo = document.getElementById("join-preview-info");
    const joinConfirmBtn = document.getElementById("join-event-confirm");
    const joinCancelBtn = document.getElementById("join-event-cancel");
    let _joinEventId = null;

    document.getElementById("join-event-btn").addEventListener("click", () => {
        joinCodeInput.value = "";
        joinPreview.hidden = true;
        joinConfirmBtn.disabled = true;
        _joinEventId = null;
        joinDialog.open = true;
    });

    joinCancelBtn.addEventListener("click", () => {
        joinDialog.open = false;
    });

    // Lookup event as user types code
    let _lookupTimeout = null;
    joinCodeInput.addEventListener("input", () => {
        const code = joinCodeInput.value.trim().toUpperCase();
        joinPreview.hidden = true;
        joinConfirmBtn.disabled = true;
        _joinEventId = null;

        if (code.length >= 6) {
            clearTimeout(_lookupTimeout);
            _lookupTimeout = setTimeout(async () => {
                try {
                    const event = await getEventByCode(code);
                    joinPreviewName.textContent = event.name;
                    joinPreviewInfo.textContent = `${event.member_count} participants • ${event.status}`;
                    joinPreview.hidden = false;
                    _joinEventId = event.event_id;

                    if (event.is_member) {
                        joinConfirmBtn.textContent = "Open Event";
                    } else if (event.status === "ended") {
                        joinConfirmBtn.textContent = "Event Ended";
                        joinConfirmBtn.disabled = true;
                        return;
                    } else {
                        joinConfirmBtn.textContent = "Join Event";
                    }
                    joinConfirmBtn.disabled = false;
                } catch {
                    joinPreview.hidden = true;
                    joinConfirmBtn.disabled = true;
                }
            }, 300);
        }
    });

    joinConfirmBtn.addEventListener("click", async () => {
        if (!_joinEventId) return;

        const code = joinCodeInput.value.trim().toUpperCase();
        // Check if already a member — just navigate
        try {
            const event = await getEventByCode(code);
            if (event.is_member) {
                joinDialog.open = false;
                window.location.href = `/event/${code}`;
                return;
            }
        } catch { /* proceed with join */ }

        joinConfirmBtn.disabled = true;
        try {
            await joinEvent(_joinEventId);
            joinDialog.open = false;
            window.location.href = `/event/${code}`;
        } catch (err) {
            joinConfirmBtn.disabled = false;
            console.error("Failed to join event:", err);
        }
    });

    // ── Helpers ──────────────────────────────────────────────────────────
    function _escapeHtml(str) {
        const div = document.createElement("div");
        div.textContent = str;
        return div.innerHTML;
    }

    // Initial load
    loadEvents();
}
