import "@ui5/webcomponents/dist/Title.js";
import { createStreamingRenderer, renderMarkdown } from "../../modules/streaming-renderer.js";
import { Chart, LineController, LineElement, PointElement, LinearScale, CategoryScale, Tooltip, Legend, Filler } from "chart.js";
import { streamNDJSON, API_BASE_URL, API_KEY } from "../../services/api.js";

Chart.register(LineController, LineElement, PointElement, LinearScale, CategoryScale, Tooltip, Legend, Filler);

/* ── State ─────────────────────────────────────────────── */
let sessions = [];       // [{ id, title, messages: [{role, content}] }]
let activeId = null;
let isWaiting = false;

/* ── DOM helpers ───────────────────────────────────────── */
const $ = id => document.getElementById(id);

/* ── Session management ────────────────────────────────── */
function newSession() {
  const id = crypto.randomUUID();
  sessions.unshift({ id, title: "New conversation", messages: [] });
  activeId = id;
  renderSessions();
  renderConversation();
  $("jlWelcomeState").style.display = "";
}

function loadSession(id) {
  activeId = id;
  renderSessions();
  renderConversation();
  const sess = getActive();
  $("jlWelcomeState").style.display = sess.messages.length ? "none" : "";
}

function getActive() {
  return sessions.find(s => s.id === activeId);
}

function renderSessions() {
  const list = $("jlSessionList");
  list.innerHTML = "";
  sessions.forEach(sess => {
    const li = document.createElement("li");
    li.className = "jl-session-item" + (sess.id === activeId ? " active" : "");
    li.textContent = sess.title;
    li.title = sess.title;
    li.onclick = () => loadSession(sess.id);
    list.appendChild(li);
  });
}

/* ── Conversation render ───────────────────────────────── */
function renderConversation() {
  $("jlConversation").innerHTML = "";
  getActive().messages.forEach(msg => appendRenderedMessage(msg));
  scrollToBottom();
}

/* ── Message rendering ─────────────────────────────────── */
const AVATAR_SVG = `
  <svg width="16" height="18" viewBox="0 0 130 130" fill="none">
    <path fill-rule="evenodd" clip-rule="evenodd"
      d="M46.6026,37C45.3357,37,44.1364,37.5715,43.3379,38.5556L24.9405,61.23C23.7324,62.7189,23.6834,64.8368,24.8211,66.3802L61.616,116.29C62.4087,117.365,63.6647,118,65,118C66.3353,118,67.5913,117.365,68.384,116.29L105.179,66.3802C106.317,64.8368,106.268,62.7189,105.06,61.23L86.6621,38.5556C85.8636,37.5715,84.6643,37,83.3974,37H46.6026ZM88.3249,63.5392C79.643,62.0748,76.8647,55.2489,75.9469,50.9797C75.8477,50.5577,75.302,50.5825,75.2276,51.0045C73.764,59.6919,66.9425,62.4719,62.6759,63.3903C62.2543,63.4896,62.2791,64.0357,62.7008,64.1101C71.3827,65.5746,74.1609,72.4004,75.0787,76.6697C75.178,77.0917,75.7237,77.0668,75.7981,76.6449C77.2616,67.9574,84.0832,65.1774,88.3497,64.259C88.7714,64.1598,88.7466,63.6137,88.3249,63.5392Z"
      fill="white"/>
  </svg>`;

function appendRenderedMessage(msg) {
  const container = $("jlConversation");

  if (msg.role === "user") {
    const row = document.createElement("div");
    row.className = "jl-msg-row user";
    row.innerHTML = `<div class="jl-bubble">${escHtml(msg.content)}</div>`;
    container.appendChild(row);
  } else {
    const row = document.createElement("div");
    row.className = "jl-msg-row agent";
    row.innerHTML = `
      <div class="jl-agent-avatar">${AVATAR_SVG}</div>
      <div>
        <div class="jl-bubble">${renderMarkdown(msg.content)}</div>
        <div class="jl-agent-meta">
          <span class="jl-routing-badge">GR/IR</span>
        </div>
        <div class="jl-msg-actions">
          <button class="jl-msg-action-btn" title="Copy" data-copy="${escAttr(msg.content)}">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none"><rect x="9" y="9" width="13" height="13" rx="2" stroke="currentColor" stroke-width="1.8"/><path d="M5 15H4a2 2 0 01-2-2V4a2 2 0 012-2h9a2 2 0 012 2v1" stroke="currentColor" stroke-width="1.8"/></svg>
          </button>
          <button class="jl-msg-action-btn" title="Helpful">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none"><path d="M14 9V5a3 3 0 00-3-3l-4 9v11h11.28a2 2 0 002-1.7l1.38-9a2 2 0 00-2-2.3H14z" stroke="currentColor" stroke-width="1.8"/><path d="M7 22H4a2 2 0 01-2-2v-7a2 2 0 012-2h3" stroke="currentColor" stroke-width="1.8"/></svg>
          </button>
          <button class="jl-msg-action-btn" title="Not helpful">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none"><path d="M10 15v4a3 3 0 003 3l4-9V2H5.72a2 2 0 00-2 1.7l-1.38 9a2 2 0 002 2.3H10z" stroke="currentColor" stroke-width="1.8"/><path d="M17 2h2.67A2.31 2.31 0 0122 4v7a2.31 2.31 0 01-2.33 2H17" stroke="currentColor" stroke-width="1.8"/></svg>
          </button>
        </div>
      </div>`;
    container.appendChild(row);
  }

  scrollToBottom();
  return container.lastElementChild;
}

/* ── Thinking indicator ────────────────────────────────── */
function showThinking() {
  const container = $("jlConversation");
  const row = document.createElement("div");
  row.className = "jl-thinking-row";
  row.id = "jlThinkingRow";
  row.innerHTML = `
    <div class="jl-agent-avatar">${AVATAR_SVG}</div>
    <div class="jl-thinking-bubble">
      <div class="jl-thinking-dots"><span></span><span></span><span></span></div>
      <span class="jl-thinking-label">Joule is thinking…</span>
    </div>`;
  container.appendChild(row);
  scrollToBottom();
}

function hideThinking() {
  const el = $("jlThinkingRow");
  if (el) el.remove();
}

/* ── Chart card ────────────────────────────────────────── */
function appendChartCard(data) {
  const container = $("jlConversation");
  const card = document.createElement("div");
  card.className = "jl-chart-card";

  const metaLabel = data.metric === "rate" ? "Discrepancy Rate %" : "Issue Volume";
  const issueLabel = data.issue_type || "All Issues";

  card.innerHTML = `
    <div class="jl-chart-header">
      <svg class="jl-chart-icon" width="18" height="18" viewBox="0 0 24 24" fill="none">
        <polyline points="22 12 18 12 15 21 9 3 6 12 2 12" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/>
      </svg>
      <span class="jl-chart-title">${escHtml(data.supplier)} — Monthly Trend</span>
      <span class="jl-chart-meta">${escHtml(metaLabel)} · ${escHtml(issueLabel)}</span>
    </div>
    <div class="jl-chart-body">
      <canvas id="jlChart_${Date.now()}"></canvas>
    </div>`;

  container.appendChild(card);
  scrollToBottom();

  // Defer chart creation until the card is in the DOM and has layout
  requestAnimationFrame(() => {
    const canvas = card.querySelector("canvas");
    const isDark = document.getElementById("jlShell")?.getAttribute("data-theme") === "dark";
    const gridColor = isDark ? "rgba(255,255,255,0.08)" : "rgba(0,0,0,0.06)";
    const textColor = isDark ? "#9aaec6" : "#556b82";

    new Chart(canvas, {
      type: "line",
      data: {
        labels: data.months,
        datasets: [{
          label: `${issueLabel} (${metaLabel})`,
          data: data.values,
          borderColor: "#5d36ff",
          backgroundColor: "rgba(93,54,255,0.1)",
          borderWidth: 2,
          pointBackgroundColor: "#5d36ff",
          pointRadius: 4,
          pointHoverRadius: 6,
          fill: true,
          tension: 0.3,
        }],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { display: false },
          tooltip: {
            callbacks: {
              label: ctx => data.metric === "rate"
                ? `${ctx.parsed.y}%`
                : `${ctx.parsed.y} issue${ctx.parsed.y !== 1 ? "s" : ""}`,
            },
          },
        },
        scales: {
          x: {
            ticks: { color: textColor, maxRotation: 45 },
            grid: { color: gridColor },
          },
          y: {
            beginAtZero: true,
            ticks: {
              color: textColor,
              callback: v => data.metric === "rate" ? `${v}%` : v,
            },
            grid: { color: gridColor },
          },
        },
      },
    });
  });
}

/* ── Notification card ─────────────────────────────────── */
function appendNotificationCard(data) {
  const container = $("jlConversation");
  const card = document.createElement("div");
  card.className = "jl-notification-card";
  card.innerHTML = `
    <div class="jl-notif-header">
      <svg class="jl-notif-icon" width="18" height="18" viewBox="0 0 24 24" fill="none">
        <path d="M4 4h16c1.1 0 2 .9 2 2v12c0 1.1-.9 2-2 2H4c-1.1 0-2-.9-2-2V6c0-1.1.9-2 2-2z" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/>
        <polyline points="22,6 12,13 2,6" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/>
      </svg>
      <span class="jl-notif-title">Notification Sent</span>
      <span class="jl-notif-status">Delivered</span>
    </div>
    <div class="jl-notif-body">
      <div class="jl-notif-row"><span class="jl-notif-label">To</span><span class="jl-notif-value">${escHtml(data.recipient_type || "")}</span></div>
      ${data.supplier_name ? `<div class="jl-notif-row"><span class="jl-notif-label">Supplier</span><span class="jl-notif-value">${escHtml(data.supplier_name)}</span></div>` : ""}
      ${data.po_number ? `<div class="jl-notif-row"><span class="jl-notif-label">PO</span><span class="jl-notif-value">${escHtml(data.po_number)}</span></div>` : ""}
      ${data.notification_message ? `<div class="jl-notif-row jl-notif-msg"><span class="jl-notif-label">Message</span><span class="jl-notif-value">${escHtml(data.notification_message)}</span></div>` : ""}
    </div>`;
  container.appendChild(card);
}

/* ── Streaming send ────────────────────────────────────── */
async function sendMessage(text) {
  const message = text.trim();
  if (!message || isWaiting) return;

  isWaiting = true;
  setInputEnabled(false);

  const sess = getActive();
  $("jlWelcomeState").style.display = "none";

  const userMsg = { role: "user", content: message };
  sess.messages.push(userMsg);

  if (sess.messages.length === 1) {
    sess.title = message.length > 42 ? message.slice(0, 42) + "…" : message;
    renderSessions();
  }

  appendRenderedMessage(userMsg);
  showThinking();

  let renderer = null;
  let agentBubble = null;

  try {
    await streamNDJSON("/api/grir-chat/chat", {
      method: "POST",
      body: { message, session_id: getActive().id },
      onChunk(obj) {
        if (obj.type === "notification") {
          hideThinking();
          appendNotificationCard(obj);
          scrollToBottom();
        } else if (obj.type === "chart") {
          hideThinking();
          appendChartCard(obj);
          scrollToBottom();
        } else if (obj.type === "token") {
          if (!agentBubble) {
            hideThinking();
            const container = $("jlConversation");
            const row = document.createElement("div");
            row.className = "jl-msg-row agent";
            row.innerHTML = `
              <div class="jl-agent-avatar">${AVATAR_SVG}</div>
              <div>
                <div class="jl-bubble jl-streaming-bubble"></div>
                <div class="jl-agent-meta">
                  <span class="jl-routing-badge">GR/IR</span>
                </div>
              </div>`;
            container.appendChild(row);
            agentBubble = row.querySelector(".jl-streaming-bubble");
            renderer = createStreamingRenderer(agentBubble);
          }
          renderer.append(obj.content);
          scrollToBottom();
        } else if (obj.type === "error") {
          hideThinking();
          const errMsg = { role: "agent", content: `Error: ${obj.content}` };
          sess.messages.push(errMsg);
          if (agentBubble) {
            agentBubble.innerHTML = renderMarkdown(errMsg.content);
          } else {
            appendRenderedMessage(errMsg);
          }
        }
      },
    });

    // Commit final message to session history and add action buttons
    const accumulatedText = renderer ? renderer.text : "";
    if (accumulatedText) {
      renderer.finish();
      const agentMsg = { role: "agent", content: accumulatedText };
      sess.messages.push(agentMsg);
      // Upgrade streaming bubble row with action buttons
      if (agentBubble) {
        const agentDiv = agentBubble.parentElement;
        const actions = document.createElement("div");
        actions.className = "jl-msg-actions";
        actions.innerHTML = `
          <button class="jl-msg-action-btn" title="Copy" data-copy="${escAttr(accumulatedText)}">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none"><rect x="9" y="9" width="13" height="13" rx="2" stroke="currentColor" stroke-width="1.8"/><path d="M5 15H4a2 2 0 01-2-2V4a2 2 0 012-2h9a2 2 0 012 2v1" stroke="currentColor" stroke-width="1.8"/></svg>
          </button>
          <button class="jl-msg-action-btn" title="Helpful">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none"><path d="M14 9V5a3 3 0 00-3-3l-4 9v11h11.28a2 2 0 002-1.7l1.38-9a2 2 0 00-2-2.3H14z" stroke="currentColor" stroke-width="1.8"/><path d="M7 22H4a2 2 0 01-2-2v-7a2 2 0 012-2h3" stroke="currentColor" stroke-width="1.8"/></svg>
          </button>
          <button class="jl-msg-action-btn" title="Not helpful">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none"><path d="M10 15v4a3 3 0 003 3l4-9V2H5.72a2 2 0 00-2 1.7l-1.38 9a2 2 0 002 2.3H10z" stroke="currentColor" stroke-width="1.8"/><path d="M17 2h2.67A2.31 2.31 0 0122 4v7a2.31 2.31 0 01-2.33 2H17" stroke="currentColor" stroke-width="1.8"/></svg>
          </button>`;
        agentDiv.appendChild(actions);
        agentBubble.classList.remove("jl-streaming-bubble");
      }
    }
  } catch (err) {
    hideThinking();
    const errMsg = { role: "agent", content: `Could not connect to server: ${err.message}` };
    sess.messages.push(errMsg);
    appendRenderedMessage(errMsg);
  } finally {
    isWaiting = false;
    setInputEnabled(true);
    $("jlChatInput").focus();
  }
}

/* ── Input helpers ─────────────────────────────────────── */
function setInputEnabled(enabled) {
  const inp = $("jlChatInput");
  const btn = $("jlSendBtn");
  inp.disabled = !enabled;
  btn.disabled = !enabled || inp.value.trim() === "";
}

function clearInput() {
  const inp = $("jlChatInput");
  inp.value = "";
  inp.style.height = "auto";
  $("jlSendBtn").disabled = true;
}

function scrollToBottom() {
  const area = $("jlMessagesArea");
  if (area) requestAnimationFrame(() => { area.scrollTop = area.scrollHeight; });
}

/* ── Utility ───────────────────────────────────────────── */
function escHtml(str) {
  return String(str)
    .replace(/&/g, "&amp;").replace(/</g, "&lt;")
    .replace(/>/g, "&gt;").replace(/"/g, "&quot;");
}

function escAttr(str) {
  return String(str).replace(/"/g, "&quot;").replace(/'/g, "&#39;");
}

/* ── Page init ─────────────────────────────────────────── */
export default function initGrirChatPage() {
  // Boot first session
  newSession();

  // Send button
  $("jlSendBtn").addEventListener("click", () => {
    const txt = $("jlChatInput").value.trim();
    if (txt) { clearInput(); sendMessage(txt); }
  });

  // Textarea: auto-resize + Enter to send
  $("jlChatInput").addEventListener("input", e => {
    const ta = e.target;
    ta.style.height = "auto";
    ta.style.height = Math.min(ta.scrollHeight, 160) + "px";
    $("jlSendBtn").disabled = ta.value.trim() === "";
  });

  $("jlChatInput").addEventListener("keydown", e => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      const txt = $("jlChatInput").value.trim();
      if (txt && !isWaiting) { clearInput(); sendMessage(txt); }
    }
  });

  // Suggestion chips
  document.querySelectorAll(".jl-suggestion-chip").forEach(btn => {
    btn.addEventListener("click", () => sendMessage(btn.dataset.query));
  });

  $("jlNewChatBtn").addEventListener("click", newSession);
  $("jlRestartBtn").addEventListener("click", newSession);

  // Sidebar toggle
  const toggleSidebar = () => {
    const sb = $("jlSidebar");
    sb.classList.toggle("collapsed");
    sb.classList.toggle("open");
  };
  $("jlSidebarToggle").addEventListener("click", toggleSidebar);
  $("jlSidebarToggleMobile").addEventListener("click", toggleSidebar);

  // Theme toggle
  $("jlThemeToggle").addEventListener("click", () => {
    const shell = $("jlShell");
    const isDark = shell.getAttribute("data-theme") === "dark";
    shell.setAttribute("data-theme", isDark ? "light" : "dark");
    const btn = $("jlThemeToggle");
    btn.querySelector(".jl-icon-sun").style.display = isDark ? "" : "none";
    btn.querySelector(".jl-icon-moon").style.display = isDark ? "none" : "";
  });

  // Copy / thumbs via event delegation
  document.getElementById("jlConversation").addEventListener("click", e => {
    const copyBtn = e.target.closest(".jl-msg-action-btn[data-copy]");
    if (copyBtn) {
      const text = copyBtn.getAttribute("data-copy");
      navigator.clipboard.writeText(text).then(() => {
        copyBtn.classList.add("active");
        setTimeout(() => copyBtn.classList.remove("active"), 1500);
      });
      return;
    }
    const thumbBtn = e.target.closest(".jl-msg-action-btn:not([data-copy])");
    if (thumbBtn) thumbBtn.classList.toggle("active");
  });
}
