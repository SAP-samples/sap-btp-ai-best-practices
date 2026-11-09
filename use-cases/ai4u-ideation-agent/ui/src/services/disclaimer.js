/**
 * Disclaimer / About panel service.
 *
 * Shows a premium right-side panel on first use that informs users about
 * data practices, AI limitations, and legal terms. Inspired by the
 * SAP Customer Strategy Explorer pattern.
 *
 * Acceptance is versioned — bumping DISCLAIMER_VERSION re-displays
 * the panel for all users.
 */

const DISCLAIMER_KEY = "ai4u_disclaimer_accepted";
const DISCLAIMER_VERSION = "1.0";

/* ─── Public API ──────────────────────────────────────────────── */

/**
 * Returns true if the disclaimer has already been accepted (current version).
 */
export function isDisclaimerAccepted() {
  try {
    const stored = JSON.parse(localStorage.getItem(DISCLAIMER_KEY));
    return stored?.version === DISCLAIMER_VERSION;
  } catch {
    return false;
  }
}

/**
 * If disclaimer not yet accepted, shows the blocking panel and returns
 * a Promise that resolves once the user clicks "I Accept".
 * If already accepted, resolves immediately.
 */
export function showDisclaimerIfNeeded() {
  if (isDisclaimerAccepted()) return Promise.resolve();
  return new Promise((resolve) => {
    _renderPanel(resolve, /* gateMode */ true);
  });
}

/**
 * Opens the panel in read-only (non-blocking) mode.
 * Used by the "About" button in the user popover.
 */
export function showAboutDialog() {
  _renderPanel(null, /* gateMode */ false);
}

/* ─── Internal ────────────────────────────────────────────────── */

const SECTIONS = [
  {
    title: "How it works",
    defaultOpen: false,
    content: `
      <ul>
        <li>You describe a business challenge and the AI guides you through approximately
        10 questions to shape it into a concrete AI use case.</li>
        <li>The final output is a structured solution blueprint you can export and act on.</li>
        <li>External research (company profile, market context) may be conducted automatically
        to enrich the recommendations.</li>
      </ul>
    `,
  },
  {
    title: "Data & Privacy",
    defaultOpen: true,
    content: `
      <ul>
        <li><strong>Your conversations are stored</strong> on SAP infrastructure and may be
        reviewed to improve the service.</li>
        <li><strong>Your input is processed by AI models</strong> (via SAP AI Core) to generate
        recommendations.</li>
        <li><strong>Company names you provide</strong> may be used for external research
        (e.g. LinkedIn, Perplexity) to enrich context.</li>
        <li><strong>Do NOT share confidential information</strong>, trade secrets, or data you
        would not want stored or processed by AI.</li>
      </ul>
    `,
  },
  {
    title: "AI Limitations",
    defaultOpen: false,
    content: `
      <ul>
        <li>AI-generated content may be inaccurate, incomplete, or misleading.</li>
        <li>Always verify suggestions with domain experts before acting on them.</li>
        <li>This tool does not replace professional business analysis or strategic consulting.</li>
      </ul>
    `,
  },
  {
    title: "Legal",
    defaultOpen: false,
    content: `
      <ul>
        <li>By using this tool you agree to the
          <a href="/privacy" target="_blank" rel="noopener">Privacy Statement</a> and
          <a href="https://www.sap.com/about/legal/terms-of-use.html" target="_blank" rel="noopener noreferrer">SAP Terms of Use</a>.
        </li>
        <li>Usage is logged in pseudonymized form for quality assurance and service improvement.</li>
        <li>SAP may update these terms; continued use implies acceptance of updates.</li>
      </ul>
      <div class="disclaimer-legal-links">
        <a href="/privacy" target="_blank" rel="noopener">Privacy Statement</a>
        <a href="/cookies" target="_blank" rel="noopener">Cookie Statement</a>
        <a href="https://www.sap.com/about/legal/terms-of-use.html" target="_blank" rel="noopener noreferrer">Terms of Use</a>
      </div>
    `,
  },
];

function _renderPanel(onAccept, gateMode) {
  // Prevent duplicate panels
  if (document.getElementById("disclaimer-panel")) {
    return;
  }

  // Backdrop (blocks interaction with app behind)
  const backdrop = document.createElement("div");
  backdrop.id = "disclaimer-backdrop";
  backdrop.className = "disclaimer-backdrop";
  if (!gateMode) {
    backdrop.addEventListener("click", () => _closePanel());
  }
  document.body.appendChild(backdrop);

  // Panel
  const panel = document.createElement("aside");
  panel.id = "disclaimer-panel";
  panel.className = "disclaimer-panel";
  panel.setAttribute("role", "dialog");
  panel.setAttribute("aria-label", "About AI4U Ideation Agent");
  panel.setAttribute("aria-modal", gateMode ? "true" : "false");

  // Build sections HTML
  const sectionsHTML = SECTIONS.map(
    (s, i) => `
    <div class="disclaimer-section">
      <button class="disclaimer-section-btn" aria-expanded="${s.defaultOpen ? "true" : "false"}" data-index="${i}">
        <span>${s.title}</span>
        <svg class="disclaimer-chevron${s.defaultOpen ? " open" : ""}" width="14" height="14" viewBox="0 0 24 24" fill="none"
          stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
          <path d="M6 9l6 6 6-6"></path>
        </svg>
      </button>
      <div class="disclaimer-section-body" aria-hidden="${s.defaultOpen ? "false" : "true"}">
        ${s.content}
      </div>
    </div>
  `
  ).join("");

  panel.innerHTML = `
    <div class="disclaimer-panel-header">
      <span class="disclaimer-panel-title">About this tool</span>
    </div>
    <div class="disclaimer-panel-body">
      <div class="disclaimer-panel-intro">
        <h2 class="disclaimer-panel-heading">AI4U Ideation Agent</h2>
        <p class="disclaimer-panel-subtext">
          An AI-powered brainstorming partner that guides you through a structured interview
          to define, challenge, and refine AI use cases. The result is an actionable solution
          blueprint grounded in market research and best practices.
        </p>
      </div>
      ${sectionsHTML}
    </div>
    <div class="disclaimer-panel-footer">
      <button class="disclaimer-panel-footer-btn" type="button">
        ${gateMode ? "I Understand & Accept" : "Close"}
      </button>
    </div>
  `;

  document.body.appendChild(panel);

  // Animate in
  requestAnimationFrame(() => {
    backdrop.classList.add("open");
    panel.classList.add("open");
  });

  // Section toggle handlers
  panel.querySelectorAll(".disclaimer-section-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      const expanded = btn.getAttribute("aria-expanded") === "true";
      btn.setAttribute("aria-expanded", String(!expanded));
      const body = btn.nextElementSibling;
      body.setAttribute("aria-hidden", String(expanded));
      btn.querySelector(".disclaimer-chevron").classList.toggle("open", !expanded);
    });
  });

  // Accept/Close handler (footer button)
  function _handleAccept() {
    if (gateMode) {
      localStorage.setItem(
        DISCLAIMER_KEY,
        JSON.stringify({
          version: DISCLAIMER_VERSION,
          acceptedAt: new Date().toISOString(),
        })
      );
    }
    _closePanel();
    if (onAccept) onAccept();
  }

  panel.querySelector(".disclaimer-panel-footer-btn").addEventListener("click", _handleAccept);

  // Block escape in gate mode
  function _handleKeydown(e) {
    if (e.key === "Escape") {
      if (gateMode) {
        e.preventDefault();
        e.stopPropagation();
      } else {
        _closePanel();
      }
    }
  }
  document.addEventListener("keydown", _handleKeydown, true);

  // Store cleanup reference
  panel._cleanup = () => {
    document.removeEventListener("keydown", _handleKeydown, true);
  };
}

function _closePanel() {
  const panel = document.getElementById("disclaimer-panel");
  const backdrop = document.getElementById("disclaimer-backdrop");

  if (panel) {
    panel.classList.remove("open");
    if (panel._cleanup) panel._cleanup();
  }
  if (backdrop) {
    backdrop.classList.remove("open");
  }

  // Remove after animation
  setTimeout(() => {
    panel?.remove();
    backdrop?.remove();
  }, 320);
}
