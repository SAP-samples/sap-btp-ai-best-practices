import * as api from "../services/workspace-api.js";
import { element, table } from "./workspace-dom.js";
import { reviewHistoryLabels, reviewPresentation } from "../modules/review-outcomes.js";

function timestamp(value) {
  if (!value) return "Not recorded";
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? String(value) : date.toLocaleString();
}

function label(value) {
  return String(value || "Not recorded").replaceAll("_", " ").toLowerCase().replace(/\b\w/g, (character) => character.toUpperCase());
}

function eventPill(labelText, tone = "neutral") {
  return element("span", labelText, `semantic-pill tone-${tone}`);
}

function reviewEvent(review) {
  const presentation = reviewPresentation[review.outcome] || [review.outcome, "neutral"];
  return {
    timestamp: review.created_at,
    event: eventPill("Human decision", presentation[1]),
    actor: review.reviewer || "Not recorded",
    action: reviewHistoryLabels[review.outcome] || review.outcome,
    details: review.comments || "No comments recorded",
  };
}

/** Render source-backed assessment, analysis, and review events in timestamp order. */
export function mountAuditLog(host, assessmentId, snapshotRequest, signal, initialDetail = null, badgeHost = null) {
  let detail = initialDetail;
  let snapshot = null;
  let reviews = [];
  let versions = [];
  let badge = null;
  let acceptance = null;
  let acceptanceStatus = "loading";
  let reviewerName = "";
  let saving = false;
  let pending = null;
  let onState = null;

  const currentState = () => {
    const latest = reviews.at(-1);
    return {
      blocked: !!acceptance?.blocked,
      decided: ["ACCEPTED", "CHANGES_REQUESTED"].includes(latest?.outcome),
      outcome: latest?.outcome || null,
      blockReason: acceptance?.block_reason || null,
      acceptanceStatus,
      acceptanceAvailable: acceptance?.available === true,
      acceptancePlan: acceptance,
    };
  };

  const render = () => {
    const events = [];
    if (detail?.uploaded_at) {
      events.push({
        timestamp: detail.uploaded_at,
        event: eventPill("Source import"),
        actor: "Not recorded",
        action: "ECN source imported",
        details: detail.ecn_artifact?.file_name || detail.source_file_name || "Source file not recorded",
      });
    }
    for (const version of versions) {
      events.push({
        timestamp: version.created_at,
        event: eventPill("Assessment"),
        actor: "Not recorded",
        action: "Assessment version created",
        details: version.ecn_row_id === assessmentId
          ? `Assessed as of ${snapshot?.assessment_date || "not recorded"}`
          : "Earlier assessment version",
      });
    }
    if (detail?.analysis?.created_at) {
      events.push({
        timestamp: detail.analysis.created_at,
        event: eventPill("Generated analysis"),
        actor: "Not recorded",
        action: "Generated assessment recorded",
        details: detail.analysis.completeness ? `Assessment status: ${label(detail.analysis.completeness)}` : "Assessment status not recorded",
      });
    }
    events.push(...reviews.map(reviewEvent));
    events.sort((left, right) => new Date(right.timestamp || 0) - new Date(left.timestamp || 0));
    host.replaceChildren();
    if (!events.length) {
      host.append(element("p", "No audit events are recorded for this assessment.", "helper-text"));
    } else {
      host.append(table(["Timestamp", "Event type", "Actor", "Action", "Details"], events.map((entry) => [
        timestamp(entry.timestamp), entry.event, entry.actor, entry.action, entry.details,
      ])));
    }
    const latest = reviews.at(-1);
    if (badgeHost) {
      badge?.remove();
      badge = latest ? eventPill(...(reviewPresentation[latest.outcome] || [latest.outcome, "neutral"]))
        : acceptance?.blocked ? eventPill("Blocked", "error") : null;
      if (badge) {
        badge.title = "Engineering Review";
        badgeHost.append(badge);
      }
    }
    onState?.(currentState());
  };

  const reload = async () => {
    try {
      reviews = await api.listReviews(assessmentId, signal);
      if (!signal.aborted) render();
    } catch (_) {
      // The card still renders the source-backed events that were available.
    }
  };

  const save = async (outcome, comments) => {
    if (saving || !reviewerName.trim()) return false;
    const key = JSON.stringify([reviewerName, outcome, comments]);
    if (!pending || pending.key !== key) pending = { key, id: crypto.randomUUID() };
    saving = true;
    try {
      const currentSnapshot = await snapshotRequest;
      reviews = await api.saveReview(assessmentId, {
        request_id: pending.id,
        evidence_hash: currentSnapshot.evidence_hash,
        snapshot_version: currentSnapshot.version,
        reviewer: reviewerName.trim(),
        outcome,
        comments,
      }, signal);
      pending = null;
      if (!signal.aborted) render();
      return true;
    } catch (_) {
      return false;
    } finally {
      saving = false;
    }
  };

  snapshotRequest.then((value) => {
    snapshot = value;
    if (!signal.aborted) render();
  }).catch(() => {});
  api.versions(assessmentId, signal).then((value) => {
    versions = value;
    if (!signal.aborted) render();
  }).catch(() => {});
  void reload();
  api.getAcceptancePlan(assessmentId, signal).then((value) => {
    acceptance = value;
    acceptanceStatus = "ready";
    if (!signal.aborted) render();
  }).catch(() => {
    acceptanceStatus = "unavailable";
    if (!signal.aborted) render();
  });
  const onReviewSaved = (event) => {
    if (event.detail?.id === assessmentId) void reload();
  };
  window.addEventListener("ecn-review-saved", onReviewSaved);
  signal.addEventListener("abort", () => window.removeEventListener("ecn-review-saved", onReviewSaved), { once: true });
  render();

  return {
    reviewApi: {
      accept: () => !!reviewerName.trim() && window.confirm("Accept this ECN? The SAP update is simulated.") && save("ACCEPTED", ""),
      reject: (comments) => comments.trim() ? save("CHANGES_REQUESTED", comments.trim()) : false,
      setReviewerName: (value) => { reviewerName = String(value || "").trim(); },
      onStateChange: (callback) => { onState = callback; callback(currentState()); },
    },
    updateDetail: (value) => {
      detail = value;
      render();
    },
  };
}
