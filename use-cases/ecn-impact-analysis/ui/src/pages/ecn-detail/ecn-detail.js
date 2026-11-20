import { generateSummaries, getSnapshot, getAnalysisStages } from "../../services/workspace-api.js";
import { element } from "../../components/workspace-dom.js";
import { snapshotWorkspace } from "../../components/snapshot-workspace.js";
import { renderBomChanges } from "../../components/bom-changes-table.js";
// MOCK_DATA_RETIREMENT_VALIDATION_DISABLED: import { renderProcurement } from "../../components/procurement-risk.js";
import {
  getEcn,
  getRdfProjection,
  retryAnalysis,
  retryRdfProjection,
} from "../../services/ecn-api.js";
import { startAnalysisPolling } from "../../modules/analysis-polling.js";
import { buildIdentifierIndex, describePartIdentifiers, linkifyIdentifiers } from "../../components/identifier-glossary.js";
import { parseListState, serializeListState } from "../../modules/list-state.js";
import { createLatestRequestGate } from "../../modules/request-generation.js";
import { renderMarkdown } from "../../modules/markdown.js";
import { buildPlantNameIndex, describeRenderedPlants } from "../../modules/plant-display.js";
import { renderLinkedEcns } from "../../components/linked-ecns.js";
import {
  getActiveDynamicPageSection,
  observeDynamicPageScroll,
  scrollToDynamicPageSection,
  syncDynamicPageNavigationOffset,
} from "./dynamic-page-scroll-adapter.js";

// UI5 web components used on this page — imported here so the router's dynamic
// import() executes them (innerHTML injection does not run <script> tags).
import "@ui5/webcomponents-fiori/dist/DynamicPage.js";
import "@ui5/webcomponents-fiori/dist/DynamicPageTitle.js";
import "@ui5/webcomponents-fiori/dist/DynamicPageHeader.js";
import "@ui5/webcomponents/dist/Bar.js";
import "@ui5/webcomponents/dist/Dialog.js";
import "@ui5/webcomponents/dist/Card.js";
import "@ui5/webcomponents/dist/CardHeader.js";
import "@ui5/webcomponents/dist/Panel.js";
import "@ui5/webcomponents/dist/Tag.js";
import "@ui5/webcomponents/dist/Title.js";
import "@ui5/webcomponents/dist/MessageStrip.js";
import "@ui5/webcomponents/dist/Icon.js";
import "@ui5/webcomponents/dist/Input.js";
import "@ui5/webcomponents/dist/TextArea.js";
import "@ui5/webcomponents-icons/dist/ai.js";

const defaultApi = {
  generateSummaries,
  getEcn,
  getRdfProjection,
  retryAnalysis,
  retryRdfProjection,
  getSnapshot,
  getAnalysisStages,
};

const unfinished = new Set(["QUEUED", "RUNNING"]);
const projectionPresentation = Object.freeze({
  CURRENT: ["Current", "success"],
  PENDING: ["Pending", "warning"],
  FAILED: ["Failed", "error"],
  STALE: ["Stale", "warning"],
});

/** Apply a readable semantic label and tone to an existing pill. */
function setPill(node, value, presentation, prefix = "") {
  if (!node) return;
  const [label, tone] = presentation[value] || ["Not assessed", "neutral"];
  node.className = `semantic-pill tone-${tone}`;
  node.textContent = `${prefix}${label}`;
}

/** Collapse old generated headlines that enumerate raw CAD or material names. */
function conciseLegacyBrief(text) {
  if (!/(?:\b[A-Z0-9]+(?:_[A-Z0-9]+)+\b)/.test(text)) return "";
  const context = text.match(/\b([A-Z0-9]+)_PLATFORM\b/i)?.[1];
  const location = context ? ` to the ${context.toUpperCase()} platform` : "";
  const addition = text.match(/\b(\d+)\s+new\s+(assemblies|parts)\b/i);
  if (addition) return `Adds ${addition[1]} new ${addition[2]}${location}.`;
  if (/\bnew\s+(part|assembly)\b/i.test(text)) return `Adds a new component${location}.`;
  if (/\b(replace|replaced|replacing)\b/i.test(text)) return `Replaces an existing component with an updated design${location}.`;
  if (/\b(remove|removed|phase-out)\b/i.test(text)) return `Removes an obsolete component${location}.`;
  if (/\bquantity\b/i.test(text)) return `Adjusts the quantity of an assembly component${location}.`;
  return "";
}

/** Fill legacy BOM assembly labels from existing snapshot part descriptions. */
function describedBomChanges(snapshot) {
  const descriptions = new Map();
  const add = (entry) => {
    if (entry?.number && entry.description && !descriptions.has(entry.number)) descriptions.set(entry.number, entry.description);
  };
  (snapshot.part_descriptions || []).forEach(add);
  (snapshot.parts || []).forEach(add);
  (snapshot.objects || []).forEach(add);
  for (const program of snapshot.programs || []) {
    (program.ebom || []).forEach(add);
    (program.mbom || []).forEach(add);
  }
  const bomChanges = snapshot.bom_changes || {};
  return {
    ...bomChanges,
    lines: (bomChanges.lines || []).map((line) => ({
      ...line,
      assembly_description: line.assembly_description || descriptions.get(line.assembly),
    })),
  };
}

/** Use the stored AI change brief without triggering a model call while reading the ECN. */
function decisionStory(detail, identifierIndex = null) {
  const executive = detail.analysis?.sections?.executive_summary?.assessment_markdown
    || detail.analysis_stages?.SUMMARY?.sections?.executive_summary?.assessment_markdown;
  const legacyBrief = executive?.match(/^- \*\*What changes:\*\*\s*(.+)$/m)?.[1];
  const source = detail.analysis?.change_brief
    || detail.analysis_stages?.SUMMARY?.change_brief
    || legacyBrief;
  if (!source) {
    return detail.analysis
      ? "No AI change brief is available. Regenerate the summary to create one."
      : "Generate the summary to create a one-line AI change brief.";
  }
  const plain = describePartIdentifiers(
    String(source).replace(/^DescriptionSummary:\s*/i, "").replace(/\s+/g, " ").trim(),
    identifierIndex,
  );
  const concise = conciseLegacyBrief(plain);
  if (concise) return concise;
  const words = plain.split(" ");
  return words.length > 16 ? `${words.slice(0, 16).join(" ")}...` : plain;
}

/** Repaint the headline once the snapshot makes part descriptions available. */
function renderDecisionStory(root, detail, identifierIndex = null) {
  const story = root.querySelector("#decision-story");
  if (story) story.textContent = decisionStory(detail, identifierIndex);
}

/** The earliest cutover is the timing signal analysts need before the CIB discussion. */
function cutoverDate(snapshot) {
  return [
    // MOCK_DATA_RETIREMENT_VALIDATION_DISABLED: ...(snapshot.phase_schedule || []).map((item) => item.effective_date),
    ...((snapshot.plant_impact?.plants || []).map((plant) => plant.cutover_from)),
  ].filter(Boolean).sort()[0] || null;
}

function setActiveNavigation(root, target) {
  root.querySelectorAll("[data-section-target]").forEach((item) => {
    if (item.dataset.sectionTarget === target) item.setAttribute("aria-current", "true");
    else item.removeAttribute("aria-current");
  });
}

/** Highlight a section link and optionally scroll to the corresponding evidence. */
function showDecisionTarget(root, target, { scroll = false } = {}) {
  const targets = {
    overview: ["overview", "#source-record"],
    plants: ["plants", "#affected-sites"],
    // MOCK_DATA_RETIREMENT_VALIDATION_DISABLED_START: Procurement-section targets.
    // sourcing: ["make-buy", "#procurement-content"],
    // procurement: ["make-buy", "#procurement-content"],
    // "make-buy": ["make-buy", "#procurement-content"],
    // "phase-out": ["make-buy", "#procurement-content"],
    // manufacturing: ["make-buy", "#procurement-content"],
    // MOCK_DATA_RETIREMENT_VALIDATION_DISABLED_END: Procurement-section targets.
    bom: ["bom", "#bom-changes"],
    audit: ["audit", "#audit-log-card"],
    references: ["audit", "#audit-log-card"],
    attention: ["overview", "#source-record"],
    impact: ["overview", "#source-record"],
  };
  const [navigationTarget, sectionSelector] = targets[target] || targets.overview;
  setActiveNavigation(root, navigationTarget);
  const section = sectionSelector ? root.querySelector(sectionSelector) : null;
  if (scroll) {
    scrollToDynamicPageSection(
      root.querySelector("#ecn-page"),
      root.querySelector(".ecn-section-navigation"),
      section,
    );
  }
}

/** Format a source date without treating a missing or invalid value as a real date. */
function shortDate(value) {
  if (!value) return "";
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? String(value) : new Intl.DateTimeFormat(undefined, { dateStyle: "medium" }).format(date);
}

function bomEvidenceStatus(snapshot) {
  if (!snapshot) return "Loading BOM evidence...";
  const lines = snapshot.bom_changes?.lines || [];
  if (!lines.length) return "No BOM change evidence found in the supplied BOM.";
  const alignments = [...new Set((snapshot.plant_impact?.plants || [])
    .map((plant) => plant.bom_alignment_label)
    .filter(Boolean))];
  const count = `${lines.length} BOM change${lines.length === 1 ? "" : "s"} found`;
  return alignments.length ? `${count}; ${alignments.join(", ")}.` : `${count}.`;
}

function executiveSummary(detail) {
  return detail.analysis?.sections?.executive_summary?.assessment_markdown
    || detail.analysis_stages?.SUMMARY?.sections?.executive_summary?.assessment_markdown
    || "";
}

/** Reuse the stored AI executive summary in the Overview without regenerating it. */
function changeSummaryField(detail, snapshot, inspect) {
  const summary = executiveSummary(detail);
  const field = sourceRecordField(
    "Change summary",
    summary ? "" : detail.analysis ? "No AI change summary is available. Regenerate the summary to create one." : "Generate the summary to create an AI change summary.",
  );
  const label = field.querySelector("dt");
  label.classList.add("ai-change-summary-label");
  const icon = document.createElement("ui5-icon");
  icon.setAttribute("name", "ai");
  icon.setAttribute("aria-hidden", "true");
  label.prepend(icon);
  if (!summary) return field;
  const value = field.querySelector("dd");
  value.className = "markdown ai-change-summary";
  value.replaceChildren(renderMarkdown(summary));
  const identifiers = buildIdentifierIndex(snapshot);
  const materialDescriptions = [...identifiers.values()]
    .filter(entry => entry.kind === "part" && entry.description).map(entry => entry.description);
  describeRenderedPlants(value, buildPlantNameIndex(snapshot?.plant_impact?.plants), [...materialDescriptions, ...identifiers.keys()]);
  linkifyIdentifiers(value, identifiers, inspect);
  return field;
}

function sourceRecordField(label, value) {
  const field = document.createElement("div");
  const term = element("dt", label);
  const detail = element("dd", value || "Not supplied");
  field.append(term, detail);
  return field;
}

/** Render source-backed record facts without inferring unavailable business data. */
function renderSourceRecord(root, detail, snapshot = null, linkedDemo = null, inspect = null, listQuery = "") {
  const fields = root.querySelector("#source-record-fields");
  if (!fields) return;
  const release = sourceRecordField("Release date", shortDate(detail.resolution_date));
  release.className = "source-record-compact-field";
  const linked = sourceRecordField("Linked ECN", "Not supplied");
  linked.id = "linked-ecn-field";
  linked.className = "source-record-compact-field";
  renderLinkedEcns(linked, detail.linked_ecns, listQuery);
  if (!detail.linked_ecns?.length) linkedDemo?.render(linked);
  fields.replaceChildren(
    sourceRecordField("Source status", detail.source_status),
    release,
    linked,
    sourceRecordField("Requestor", detail.requestor_name),
    sourceRecordField("Affected products", detail.affected_products?.join(", ")),
    sourceRecordField("BOM evidence", bomEvidenceStatus(snapshot)),
    changeSummaryField(detail, snapshot, inspect),
  );
}

function siteEvidenceStatus(plant, snapshot) {
  if (!plant.plant_code) {
    return { label: "Manufacturing site not determined", tone: "neutral" };
  }
  const hasMbomGap = (snapshot.bom_changes?.lines || []).some((line) =>
    line.plants?.some((item) => item.plant_code === plant.plant_code)
      && line.qualifiers?.includes("MBOM_NOT_UPDATED"),
  );
  if (hasMbomGap) return { label: "Manufacturing BOM update pending", tone: "warning" };
  if (plant.status === "CONFIRMED") return { label: "Dated BOM change found", tone: "success" };
  return { label: "BOM evidence needs review", tone: "warning" };
}

/** List source-derived manufacturing sites and the status of their BOM evidence. */
function renderAffectedManufacturingSites(root, snapshot) {
  const list = root.querySelector("#affected-sites-list");
  if (!list) return;
  const sites = new Map();
  const names = buildPlantNameIndex(snapshot.plant_impact?.plants);
  for (const plant of snapshot.plant_impact?.plants || []) {
    const code = plant.plant_code?.trim?.();
    const key = code || "unresolved";
    if (sites.has(key)) continue;
    sites.set(key, {
      code,
      name: names.get(code) || code || plant.plant_name?.trim?.() || "Manufacturing site not determined",
      status: siteEvidenceStatus(plant, snapshot),
    });
  }
  list.replaceChildren();
  if (!sites.size) {
    list.append(element("li", "No manufacturing site could be determined from the supplied BOM evidence.", "helper-text"));
    return;
  }
  for (const { code, name, status } of sites.values()) {
    const site = document.createElement("li");
    site.className = "affected-site";
    site.append(element("strong", name, "affected-site-name"));
    if (code && name !== code) site.append(element("span", code, "affected-site-code"));
    site.append(element("span", status.label, `semantic-pill tone-${status.tone} affected-site-status`));
    list.append(site);
  }
}

/** Render source-backed header facts above the Overview card. */
function renderOverviewFacts(root, snapshot) {
  const context = snapshot.ecn_context || {};
  const facts = root.querySelector("#overview-change-facts");
  const timing = shortDate(cutoverDate(snapshot) || context.cib_target_date);
  if (facts) {
    facts.textContent = [["Owner", context.owner_name], ["Cutover", timing]]
      .filter(([, value]) => value)
      .map(([label, value]) => `${label}: ${value}`).join(" | ");
    facts.hidden = !facts.textContent;
  }

}

/** Build one exact-row detail initializer with injectable IO. */
export function createEcnDetailPage({
  api = defaultApi,
  startPolling = startAnalysisPolling,
  uuid = () => crypto.randomUUID(),
} = {}) {
  /** Initialize summary, lazy business evidence, polling, and retry behavior. */
  return async function init(context) {
    const root = context.container;
    const rowId = context.params.ecn_row_id;
    const linkedListQuery = serializeListState(parseListState(context.query)).toString();
    const detailReads = createLatestRequestGate(context.signal);
    const projectionReads = createLatestRequestGate(context.signal);
    const retryMutations = createLatestRequestGate(context.signal);
    const projectionMutations = createLatestRequestGate(context.signal);
    const generationMutations = createLatestRequestGate(context.signal);
    const listeners = [];
    let stopPolling = () => {};
    let generationRequestId = null;
    let projection = null;
    let workspace = null;
    let latestDescriptions = [];
    let latestDetail = null;
    let latestSnapshot = null;
    let identifierIndex = null;
    let linkedDemo = null;
    let latestReviewState = { acceptanceStatus: "loading" };
    let renderedBomSnapshot = null, renderedBomState = null;
    let pageDisposed = false;
    // Vite removes this entire import branch and its fictional asset graph from production.
    if (import.meta.env.DEV && context.query.get("demo-linked-ecn") === "1") {
      // Keep the helper outside pages/**: router's variable imports include every page JS file.
      void import("../../components/demo/linked-ecn-demo.js").then(({ mountLinkedEcnDemo }) => {
        if (context.signal.aborted || pageDisposed) return;
        linkedDemo = mountLinkedEcnDemo(root, context.signal);
        if (latestDetail) renderSourceRecord(root, latestDetail, latestSnapshot, linkedDemo, inspectPart, linkedListQuery);
      }).catch(() => {
        if (context.signal.aborted || pageDisposed) return;
        linkedDemo = { render: (field) => {
          const notice = element("p", "Demo unavailable. Reload to try again; no source relationship is supplied.", "helper-text");
          notice.setAttribute("role", "status");
          field.querySelector("dd").append(notice);
        }, dispose: () => {} };
        if (latestDetail) renderSourceRecord(root, latestDetail, latestSnapshot, linkedDemo, inspectPart, linkedListQuery);
      });
    }
    const snapshotRequest = api.getSnapshot ? api.getSnapshot(rowId, context.signal) : null;
    snapshotRequest?.catch(() => {});

    /** Open a part by number via the mounted workspace, if ready. */
    const inspectPart = (number) => void (workspace?.inspect ?? (() => {}))(number);
    const inspectMaterial = (...args) => {
      if (!workspace?.inspect) throw new Error("Material details are not available. Reload this ECN and try again.");
      return workspace.inspect(...args);
    };
    const renderBomEvidence = () => {
      if (!latestSnapshot || context.signal.aborted || pageDisposed) return;
      const stateKey = JSON.stringify(latestReviewState);
      if (latestSnapshot === renderedBomSnapshot && stateKey === renderedBomState) return;
      renderedBomSnapshot = latestSnapshot;
      renderedBomState = stateKey;
      renderBomChanges(root.querySelector("#bom-changes"), describedBomChanges(latestSnapshot), inspectMaterial,
        { reviewState: latestReviewState, snapshot: latestSnapshot });
    };

    // Once the snapshot is loaded, build the identifier glossary and re-render
    // the AI prose so its IDs become clickable/annotated.
    snapshotRequest?.then((snapshot) => {
      if (context.signal.aborted) return;
      identifierIndex = buildIdentifierIndex(snapshot);
      if (latestDetail) {
        renderDecisionStory(root, latestDetail, identifierIndex);
      }
      latestSnapshot = snapshot;
      renderBomEvidence();
      renderOverviewFacts(root, snapshot);
      renderAffectedManufacturingSites(root, snapshot);
      if (latestDetail) {
        renderSourceRecord(root, latestDetail, snapshot, linkedDemo, inspectPart, linkedListQuery);
      }
    }).catch(() => {});

    /** Register an event listener and retain its cleanup. */
    const on = (node, type, handler) => {
      if (!node) return;
      node.addEventListener(type, handler);
      listeners.push(() => node.removeEventListener(type, handler));
    };

    const rules = root.querySelector("#acceptance-rules");
    const rulesTrigger = root.querySelector("#acceptance-rules-open");
    const syncRulesTrigger = () => {
      if (rulesTrigger) rulesTrigger.accessibilityAttributes = {
        controls: "acceptance-rules", expanded: String(!!rules?.open), hasPopup: "dialog",
      };
    };
    syncRulesTrigger();
    on(rulesTrigger, "click", () => {
      if (!context.signal.aborted && !pageDisposed && rules) {
        rules.open = true;
        syncRulesTrigger();
      }
    });
    on(root.querySelector("#acceptance-rules-close"), "click", () => {
      if (rules) rules.open = false;
      syncRulesTrigger();
    });
    on(rules, "close", syncRulesTrigger);
    on(context.signal, "abort", () => {
      if (rules) rules.open = false;
      if (rulesTrigger) rulesTrigger.accessibilityAttributes = {};
    });

    root.querySelector("#back-to-list").href = `/ecns?${serializeListState(parseListState(context.query))}`;

    const showRouteView = (view) => {
      if (view === "references") showDecisionTarget(root, "references", { scroll: true });
      else if (view === "attention") showDecisionTarget(root, "attention", { scroll: true });
      else if (view === "impact") showDecisionTarget(root, "impact", { scroll: true });
      else showDecisionTarget(root, "overview");
    };
    context.onViewChange?.(showRouteView);
    showRouteView(context.view || "overview");

    root.querySelectorAll(".ecn-section-navigation").forEach((navigation) => on(navigation, "click", (event) => {
      const button = event.target.closest?.("[data-section-target]");
      if (button) showDecisionTarget(root, button.dataset.sectionTarget, { scroll: true });
    }));
    on(root, "click", (event) => {
      const link = event.target.closest?.("[data-decision-target]");
      if (!link) return;
      event.preventDefault();
      showDecisionTarget(root, link.dataset.decisionTarget, { scroll: true });
    });

    const updateActiveSection = () => {
      const active = getActiveDynamicPageSection(
        dynamicPage,
        root.querySelector(".ecn-section-navigation"),
        [
          { target: "overview", element: root.querySelector("#source-record") },
          { target: "plants", element: root.querySelector("#affected-sites") },
          { target: "bom", element: root.querySelector("#bom-changes") },
          // MOCK_DATA_RETIREMENT_VALIDATION_DISABLED: { target: "make-buy", element: root.querySelector("#procurement-content") },
          { target: "audit", element: root.querySelector("#audit-log-card") },
        ],
      );
      if (active) setActiveNavigation(root, active);
    };
    const dynamicPage = root.querySelector("#ecn-page");
    const scrollObserver = observeDynamicPageScroll(dynamicPage, updateActiveSection);
    const refreshActiveSection = () => {
      const update = () => {
        scrollObserver.refresh();
        updateActiveSection();
      };
      if (typeof requestAnimationFrame === "function") requestAnimationFrame(update); else update();
    };
    // title-toggle can change pinned-title geometry; container replacement is
    // observed inside the adapter because UI5 2.26 exposes no render event.
    on(dynamicPage, "title-toggle", refreshActiveSection);
    refreshActiveSection();

    /** Render independent graph provenance without widening the ECN detail payload. */
    const renderProjection = (nextProjection) => {
      projection = nextProjection;
      setPill(root.querySelector("#rdf-projection-status"), projection.status, projectionPresentation);
      const current = projection.status === "CURRENT";
      root.querySelector("#source-preparation").hidden = current;
      root.querySelector("#rdf-projection-retry").hidden = !["FAILED", "STALE"].includes(projection.status);
      root.querySelector("#generate-summary").disabled = !current;
      root.querySelector("#rdf-projection-notice").textContent = current
        ? "Source preparation is complete."
        : projection.status === "PENDING" ? "Preparing source data. Reload this page shortly to use summary generation and chat. Material evidence remains available." : "Source preparation needs attention before summary generation or chat can run. Retry preparation, then reload this page. Material evidence remains available.";
    };

    /** Mount snapshot-backed sections and chat once the graph gate is known. */
    const mountWorkspace = (graphReady) => {
      if (!snapshotRequest || workspace) return;
      workspace = snapshotWorkspace(root, rowId, context.signal, {
        graphReady,
        snapshotRequest,
        auditDetail: latestDetail,
        showSourceData: () => {
          showDecisionTarget(root, "references", { scroll: true });
        },
      });
      workspace.updateAnalysisDescriptions(latestDescriptions);

      // Wire footer decision buttons once review API is ready
      workspace.reviewApi?.then(reviewApi => {
        if (!reviewApi) return;
        const footerAccept = root.querySelector("#accept-ecn-footer");
        const footerSendBack = root.querySelector("#send-back-footer");
        const footerName = root.querySelector("#reviewer-name-footer");
        const sendBackDialog = root.querySelector("#send-back-dialog");

        on(footerAccept, "click", () => {
          if (!footerName?.value?.trim?.()) {
            footerName.focus();
            return;
          }
          if (footerName) reviewApi.setReviewerName(footerName.value?.trim?.() ?? "");
          reviewApi.accept();
        });

        // Keep the footer buttons in sync with the review card's acceptance plan:
        // a blocked or already-decided ECN cannot be accepted; a decided ECN cannot be sent back.
        reviewApi.onStateChange?.((nextState) => {
          latestReviewState = nextState;
          renderBomEvidence();
          const { blocked, decided, blockReason } = nextState;
          if (footerAccept) footerAccept.disabled = blocked || decided;
          if (footerSendBack) footerSendBack.disabled = decided;

          const reason = root.querySelector("#accept-ecn-reason");
          const disabledReason = blocked
            ? blockReason || "Acceptance is blocked by the current review preconditions."
            : decided
              ? "An engineering decision is already recorded."
              : "";
          const statusReason = nextState.acceptanceStatus === "loading" ? "Checking acceptance status…"
            : nextState.acceptanceStatus === "unavailable" || nextState.acceptanceAvailable === false
              ? "Acceptance status is unavailable." : "No material acceptance blocker is reported.";
          if (reason) reason.textContent = disabledReason || statusReason;
          // Public UI5 property forwards this description to the actual button.
          // Keep the visible, keyboard-expandable rules available even when disabled.
          if (footerAccept) footerAccept.accessibleDescription = [
            disabledReason,
            "See Blocked status in BOM Change Evidence and Acceptance rules beside this control. Reviewer name and confirmation are required; the SAP update is simulated.",
          ].filter(Boolean).join(" ");
        });
        on(footerSendBack, "click", () => {
          const errorEl = root.querySelector("#send-back-error");
          if (errorEl) errorEl.textContent = "";
          if (sendBackDialog) {
            if (typeof sendBackDialog.showModal === "function") sendBackDialog.showModal();
            else sendBackDialog.open = true;
          }
        });
        on(root.querySelector("#send-back-cancel"), "click", () => {
          if (sendBackDialog) {
            if (typeof sendBackDialog.close === "function") sendBackDialog.close();
            else sendBackDialog.open = false;
          }
        });
        on(root.querySelector("#send-back-confirm"), "click", () => {
          const commentsEl = root.querySelector("#send-back-comments");
          const text = commentsEl?.value?.trim?.() || (commentsEl?.shadowRoot?.querySelector("textarea")?.value?.trim?.());
          const errorEl = root.querySelector("#send-back-error");
          if (!text) {
            if (errorEl) errorEl.textContent = "Say what must change.";
            return;
          }
          const reviewer = footerName?.value?.trim?.() ?? "";
          if (!reviewer) {
            if (footerName) footerName.focus();
            if (errorEl) errorEl.textContent = "Enter your name in the decision footer.";
            return;
          }
          reviewApi.setReviewerName(reviewer);
          if (!reviewApi.reject(text)) return;
          if (sendBackDialog) {
            if (typeof sendBackDialog.close === "function") sendBackDialog.close();
            else sendBackDialog.open = false;
          }
        });
      });
    };

    /** Render the Overview facts and its Plants, BOM, and Procurement drill-downs. */
    const renderSnapshotSections = async (inspect) => {
      try {
        const snapshot = await snapshotRequest;
        if (context.signal.aborted) return;

        latestSnapshot = snapshot;
        renderOverviewFacts(root, snapshot);
        renderAffectedManufacturingSites(root, snapshot);
        if (latestDetail) {
          renderSourceRecord(root, latestDetail, snapshot, linkedDemo, inspectPart, linkedListQuery);
        }
        renderBomEvidence();
        // MOCK_DATA_RETIREMENT_VALIDATION_DISABLED: renderProcurement(root.querySelector("#procurement-content"), snapshot, inspect);
      } catch (error) {
        if (error.name !== "AbortError") {
          const sites = root.querySelector("#affected-sites-list");
          if (sites) sites.replaceChildren(element("li", "Plant and BOM evidence are unavailable. Reload to try again.", "helper-text"));
        }
      }
    };

    /** Render current metadata while preserving narrative accordion choices. */
    const renderDetail = (detail) => {
      latestDetail = detail;
      const title = detail.display_ecn_number || "ECN Details";
      root.querySelectorAll("#detail-title, #detail-title-snapped").forEach((heading) => {
        heading.textContent = title;
      });
      if (typeof requestAnimationFrame === "function") requestAnimationFrame(() => syncDynamicPageNavigationOffset(
        root.querySelector("#ecn-page"),
        root.querySelector(".ecn-section-navigation"),
      ));
      renderDecisionStory(root, detail, identifierIndex);
      renderSourceRecord(root, detail, latestSnapshot, linkedDemo, inspectPart, linkedListQuery);
      workspace?.updateAuditDetail(detail);

      root.querySelector("#generate-summary").textContent = detail.analysis ? "Regenerate Summary" : "Generate Summary";
      if (latestSnapshot) renderOverviewFacts(root, latestSnapshot);

      const descriptions = (detail.job.state !== "COMPLETED" && detail.analysis_stages?.DATA_READINESS?.descriptions)
        || detail.analysis?.data_readiness_descriptions || [];
      latestDescriptions = descriptions;
      workspace?.updateAnalysisDescriptions(descriptions);

      const retryBtn = root.querySelector("#retry-analysis");
      if (retryBtn) retryBtn.hidden = detail.job.state !== "FAILED";
    };

    /** Load canonical detail through the shared metadata request lane. */
    const readDetail = async () => {
      const request = detailReads.next();
      const [detail, stages] = await Promise.all([
        api.getEcn(rowId, { signal: request.signal }),
        api.getAnalysisStages ? api.getAnalysisStages(rowId, request.signal).catch(() => ({})) : Promise.resolve({}),
      ]);
      if (!request.isCurrent()) throw new DOMException("Superseded", "AbortError");
      detail.analysis_stages = stages;
      return detail;
    };

    /** Load independent graph status without blocking relational detail rendering. */
    const readProjection = async () => {
      if (!api.getRdfProjection) return null;
      const request = projectionReads.next();
      const result = await api.getRdfProjection(rowId, { signal: request.signal });
      if (!request.isCurrent()) throw new DOMException("Superseded", "AbortError");
      return result;
    };

    /** Start serial metadata polling until generation reaches a terminal state. */
    const beginPolling = () => {
      stopPolling();
      stopPolling = startPolling({
        load: readDetail,
        onUpdate: renderDetail,
        onError: (error) => {
          const errEl = root.querySelector("#detail-error");
          if (errEl) errEl.textContent = error.message;
        },
        isFinished: (detail) => !unfinished.has(detail.job.state),
        signal: context.signal,
      });
    };

    if (api.generateSummaries) {
      on(root.querySelector("#generate-summary"), "click", async () => {
        if (projection && projection.status !== "CURRENT") return;
        const request = generationMutations.next();
        const button = root.querySelector("#generate-summary");
        button.disabled = true;
        generationRequestId ||= uuid();
        try {
          const batch = await api.generateSummaries(
            { mode: "explicit", ecn_ids: [rowId] },
            generationRequestId,
            request.signal,
          );
          if (!request.isCurrent()) return;
          const scheduled = batch.selection.scheduled_job_ids.length;
          const completed = batch.jobs.filter((job) => job.state === "COMPLETED").length;
          const alreadyRunning = Math.max(0, batch.jobs.length - scheduled - completed);
          const genStatus = root.querySelector("#detail-generation-status");
          if (genStatus) genStatus.textContent = `${scheduled} scheduled · ${alreadyRunning} already running · ${completed} completed`;
          generationRequestId = null;
          renderDetail(await readDetail());
          beginPolling();
        } catch (error) {
          if (request.isCurrent() && error.name !== "AbortError") {
            const errEl = root.querySelector("#detail-error");
            if (errEl) errEl.textContent = error.message;
          }
        } finally {
          if (request.isCurrent()) button.disabled = false;
        }
      });
    }

    on(root.querySelector("#retry-analysis"), "click", async () => {
      const request = retryMutations.next();
      const retryBtn = root.querySelector("#retry-analysis");
      if (retryBtn) retryBtn.disabled = true;
      try {
        await api.retryAnalysis(rowId, { signal: request.signal });
        if (!request.isCurrent()) return;
        const detail = await readDetail();
        renderDetail(detail);
        beginPolling();
      } catch (error) {
        if (!request.isCurrent() || error.name === "AbortError") return;
        if (error.status === 409) {
          const detail = await readDetail();
          renderDetail(detail);
          if (unfinished.has(detail.job.state)) beginPolling();
        } else {
          const errEl = root.querySelector("#detail-error");
          if (errEl) errEl.textContent = error.message;
        }
      } finally {
        if (request.isCurrent() && retryBtn) retryBtn.disabled = false;
      }
    });

    on(root.querySelector("#rdf-projection-retry"), "click", async () => {
      if (!api.retryRdfProjection) return;
      const request = projectionMutations.next();
      const button = root.querySelector("#rdf-projection-retry");
      if (button) button.disabled = true;
      try {
        const pending = await api.retryRdfProjection(rowId, { signal: request.signal });
        if (request.isCurrent()) renderProjection(pending);
      } catch (error) {
        if (request.isCurrent() && error.name !== "AbortError") {
          const errEl = root.querySelector("#detail-error");
          if (errEl) errEl.textContent = error.message;
        }
      } finally {
        if (request.isCurrent() && button) button.disabled = false;
      }
    });

    try {
      const detail = await readDetail();
      renderDetail(detail);
      let graphReady = !api.getRdfProjection;
      try {
        const graphStatus = await readProjection();
        if (graphStatus) {
          renderProjection(graphStatus);
          graphReady = graphStatus.status === "CURRENT";
        }
      } catch (error) {
        if (error.name !== "AbortError") {
          const genBtn = root.querySelector("#generate-summary");
          if (genBtn) genBtn.disabled = true;
          root.querySelector("#source-preparation").hidden = false;
          root.querySelector("#rdf-projection-notice").textContent = "Source preparation status is unavailable. Reload to try again; material evidence remains available.";
        }
      }
      mountWorkspace(graphReady);
      void renderSnapshotSections(workspace?.inspect ?? (() => {}));
      if (unfinished.has(detail.job.state)) beginPolling();
    } catch (error) {
      if (error.name !== "AbortError") {
        const errEl = root.querySelector("#detail-error");
        if (errEl) errEl.textContent = error.message;
      }
    }

    /** Release page-owned asynchronous work, listeners, and dialogs. */
    return () => {
      pageDisposed = true;
      stopPolling();
      detailReads.dispose();
      projectionReads.dispose();
      retryMutations.dispose();
      projectionMutations.dispose();
      generationMutations.dispose();
      scrollObserver.disconnect();
      if (rules) rules.open = false;
      if (rulesTrigger) rulesTrigger.accessibilityAttributes = {};
      linkedDemo?.dispose();
      listeners.forEach((remove) => remove());
      for (const dialog of root.querySelectorAll("dialog[open]")) {
        if (typeof dialog.close === "function") dialog.close(); else dialog.open = false;
      }
    };
  };
}

export const init = createEcnDetailPage();
