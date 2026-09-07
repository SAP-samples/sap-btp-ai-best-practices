import "@ui5/webcomponents/dist/Button.js";
import "@ui5/webcomponents/dist/Icon.js";
import "@ui5/webcomponents/dist/Title.js";

import "@ui5/webcomponents-icons/dist/form.js";
import "@ui5/webcomponents-icons/dist/accept.js";
import "@ui5/webcomponents-icons/dist/business-objects-experience.js";
import "@ui5/webcomponents-icons/dist/reset.js";
import "@ui5/webcomponents-icons/dist/save.js";
import "@ui5/webcomponents-icons/dist/sys-find.js";

import {
  customerClassOptions,
  filterAllowedQuestions,
  filterSelectedAnswerIds,
  isQuestionAvailable,
  maxAllowedLevel,
  normalizeCustomerClass
} from "./customerClassScope.js";
import {
  benchmarkCustomerClasses,
  buildAssessmentProfilePayload,
  naceOptionsForClass,
  resolveAssessmentProfile
} from "./assessmentProfileState.js";
import { getLanguage, t } from "../../modules/i18n.js";
import { request } from "../../services/api.js";
import { RequestGeneration } from "../../services/requestGeneration.js";

const POLL_INTERVAL_MS = 2500;
const JOB_STORAGE_PREFIX = "document-assessment-review-job";
const BATCH_JOB_STORAGE_PREFIX = "document-assessment-review-batch-job";
const CUSTOMER_CLASS_STORAGE_KEY = "document-assessment-assistant-customer-class";
const ASSESSMENT_ID = "demo-assessment";
const RESPONSE_SAVE_DEBOUNCE_MS = 350;
let languageChangeListenerAttached = false;
const assessmentLoadOwner = new RequestGeneration();

const state = {
  dimensions: [],
  language: getLanguage(),
  selectedDimension: "Strategy",
  customerClassScope: null,
  selectedCustomerClass: localStorage.getItem(CUSTOMER_CLASS_STORAGE_KEY),
  selectedNace1: null,
  benchmarkOptions: null,
  assessmentProfile: null,
  profileSaveInFlight: false,
  questionsByDimension: new Map(),
  selectedAnswers: new Map(),
  latestResults: new Map(),
  latestTaskStatuses: new Map(),
  activePolls: new Map(),
  activeBatchPolls: new Set(),
  pendingResponseSaveTimer: null,
  pendingResponseQuestionIds: new Set(),
  persistedResponseQuestionIds: new Set(),
  responseSaveInFlight: false,
  responseSaveQueue: Promise.resolve(),
  aiMarksApplyInFlight: new Set()
};

/**
 * Return whether a load still owns the mounted assessment page.
 *
 * @param {number} requestGeneration - Generation captured before asynchronous reads.
 * @returns {boolean} True only for the newest load on the currently mounted route.
 */
function ownsMountedAssessmentLoad(requestGeneration) {
  return Boolean(
    assessmentLoadOwner.owns(requestGeneration) &&
      document.getElementById("assessment-page-title")
  );
}

/**
 * Escape API-provided text before injecting it into page templates.
 *
 * @param {unknown} value - Value to render as text in HTML.
 * @returns {string} HTML-escaped string.
 */
function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

/**
 * Format backend enum-style labels for end users.
 *
 * @param {unknown} value - Raw backend status or decision value.
 * @returns {string} Human-readable label without enum separators.
 */
function formatDisplayLabel(value) {
  const rawValue = String(value ?? "").trim();
  if (!rawValue) {
    return "";
  }

  const translatedLabel = t(`label.${rawValue}`);
  if (translatedLabel !== `label.${rawValue}`) {
    return translatedLabel;
  }

  const normalized = rawValue.replaceAll("_", " ").replaceAll("-", " ");
  return normalized.charAt(0).toUpperCase() + normalized.slice(1);
}

/**
 * Return the cache key for dimension-scoped language-specific state.
 *
 * @param {string} dimension - Canonical dimension name.
 * @param {string} language - Language code associated with the cached value.
 * @returns {string} Cache key combining language and canonical dimension.
 */
function dimensionLanguageKey(dimension, language = state.language) {
  return `${language}:${dimension}`;
}

/**
 * Return the cache key for question-scoped language-specific AI state.
 *
 * @param {string} questionId - Canonical question ID.
 * @param {string} language - Language code associated with the cached value.
 * @returns {string} Cache key combining language and question ID.
 */
function questionLanguageKey(questionId, language = state.language) {
  return `${language}:${questionId}`;
}

/**
 * Clear cached AI result/status state before starting a fresh review.
 *
 * @param {Array<string>} questionIds - Question IDs whose result state is stale.
 * @param {string} language - Language associated with the cached review state.
 * @returns {void}
 */
function clearQuestionReviewState(questionIds, language = state.language) {
  questionIds.forEach((questionId) => {
    const cacheKey = questionLanguageKey(questionId, language);
    state.latestResults.delete(cacheKey);
    state.latestTaskStatuses.delete(cacheKey);
  });
}

/**
 * Return the localized display name for a canonical dimension key.
 *
 * @param {string} dimension - Canonical dimension name.
 * @returns {string} Localized dimension display name when loaded.
 */
function dimensionDisplayName(dimension) {
  const match = state.dimensions.find((item) => item.dimension === dimension);
  return match?.display_name || dimension;
}

/**
 * Group answer items by maturity level while preserving item order.
 *
 * @param {Array<object>} answerItems - Answer items from the assessment API.
 * @returns {Map<number, Array<object>>} Answer items keyed by level number.
 */
function groupedByLevel(answerItems) {
  const groups = new Map();
  answerItems.forEach((item) => {
    const level = Number(item.level);
    if (!groups.has(level)) {
      groups.set(level, []);
    }
    groups.get(level).push(item);
  });
  return new Map([...groups.entries()].sort(([left], [right]) => left - right));
}

/**
 * Return the loaded questions for the active dimension.
 *
 * @returns {Array<object>} Current dimension questions.
 */
function currentQuestions() {
  return (
    state.questionsByDimension.get(
      dimensionLanguageKey(state.selectedDimension)
    ) || []
  );
}

/**
 * Find one loaded question by ID across cached dimensions.
 *
 * @param {string} questionId - Assessment question identifier.
 * @returns {object | null} Loaded question payload, when available.
 */
function loadedQuestionById(questionId) {
  for (const questions of state.questionsByDimension.values()) {
    const match = questions.find((question) => question.question_id === questionId);
    if (match) {
      return match;
    }
  }
  return null;
}

/**
 * Show a page-level status or error message.
 *
 * @param {string} message - User-visible message to show.
 * @param {"info" | "error" | "success"} tone - Visual tone for the message.
 * @returns {void}
 */
function showJobSummary(message, tone = "info") {
  const summary = document.getElementById("ai-job-summary");
  if (!summary) {
    return;
  }
  summary.hidden = false;
  summary.className = `ai-job-summary ${tone}`;
  summary.textContent = message;
}

/**
 * Show benchmark-profile persistence or availability status independently.
 *
 * @param {string | null} message - Localized message, or null to hide it.
 * @param {"info" | "error" | "success"} [tone] - Visual status tone.
 * @returns {void}
 */
function showProfileStatus(message, tone = "info") {
  const summary = document.getElementById("benchmark-profile-status");
  if (!summary) {
    return;
  }
  summary.hidden = !message;
  summary.className = `benchmark-profile-status ${tone}`;
  summary.textContent = message || "";
}

/**
 * Convert selected-answer state to a plain object for API persistence.
 *
 * @param {Array<string> | null} questionIds - Optional question IDs to include.
 * @returns {object} Answer IDs keyed by question ID.
 */
function selectedAnswersPayload(questionIds = null) {
  const ids = questionIds || [...state.selectedAnswers.keys()];
  const payload = {};
  ids.forEach((questionId) => {
    const question = loadedQuestionById(questionId);
    if (!question) {
      payload[questionId] = state.selectedAnswers.get(questionId) || [];
      return;
    }
    payload[questionId] = filterSelectedAnswerIds(
      state.customerClassScope,
      question,
      state.selectedAnswers.get(questionId) || [],
      state.selectedCustomerClass
    );
  });
  return payload;
}

/**
 * Persist selected answers for submitted questions in HANA.
 *
 * @param {Array<string>} questionIds - Question IDs whose answers changed.
 * @param {string} source - Persistence source label.
 * @returns {Promise<object>} Persisted answers keyed by question ID.
 */
async function saveResponses(questionIds, source = "manual") {
  const submittedQuestionIds = [...new Set(questionIds)];
  return queueResponseWrite(async () => {
    const response = await request("/api/assessment/responses", "PUT", {
      assessment_id: ASSESSMENT_ID,
      customer_class: state.selectedCustomerClass,
      source,
      answers: selectedAnswersPayload(submittedQuestionIds)
    });
    submittedQuestionIds.forEach((questionId) => {
      state.persistedResponseQuestionIds.add(questionId);
    });
    return response.answers || {};
  });
}

/**
 * Serialize one response-writing API operation after earlier writes finish.
 *
 * @param {() => Promise<object>} writeOperation - Async operation that writes response state.
 * @returns {Promise<object>} Promise resolved with the operation result.
 */
function queueResponseWrite(writeOperation) {
  const savePromise = state.responseSaveQueue
    .catch(() => {})
    .then(async () => {
      state.responseSaveInFlight = true;
      try {
        return await writeOperation();
      } finally {
        state.responseSaveInFlight = false;
      }
    });
  state.responseSaveQueue = savePromise.catch(() => {});
  return savePromise;
}

/**
 * Schedule a debounced response save after local checkbox changes.
 *
 * @param {string} questionId - Question whose answer state changed.
 * @returns {void}
 */
function scheduleResponseSave(questionId) {
  state.pendingResponseQuestionIds.add(questionId);
  window.clearTimeout(state.pendingResponseSaveTimer);
  state.pendingResponseSaveTimer = window.setTimeout(() => {
    const questionIds = [...state.pendingResponseQuestionIds];
    state.pendingResponseQuestionIds.clear();
    if (questionIds.length === 0) {
      return;
    }
    saveResponses(questionIds)
      .catch((error) => {
        showJobSummary(t("assessment.responsesSaveFailed", { message: error.message }), "error");
      });
  }, RESPONSE_SAVE_DEBOUNCE_MS);
}

/**
 * Persist empty selections for every question ID known by the UI.
 *
 * @returns {Promise<object>} Persisted empty answers keyed by cleared question ID.
 */
function clearPersistedResponses() {
  window.clearTimeout(state.pendingResponseSaveTimer);
  const questionIds = [
    ...new Set([
      ...state.persistedResponseQuestionIds,
      ...state.selectedAnswers.keys(),
      ...state.pendingResponseQuestionIds
    ])
  ];
  state.pendingResponseQuestionIds.clear();
  if (questionIds.length === 0) {
    return state.responseSaveQueue;
  }
  const answers = Object.fromEntries(questionIds.map((questionId) => [questionId, []]));
  return queueResponseWrite(async () => {
    const response = await request("/api/assessment/responses", "PUT", {
      assessment_id: ASSESSMENT_ID,
      customer_class: state.selectedCustomerClass,
      source: "reset",
      answers
    });
    questionIds.forEach((questionId) => {
      state.persistedResponseQuestionIds.add(questionId);
    });
    return response.answers || {};
  });
}

/**
 * Store or remove the currently selected answer for one checkbox change.
 *
 * @param {string} questionId - Question whose answer selection changed.
 * @param {string} answerItemId - Answer item toggled by the user.
 * @param {boolean} checked - Whether the answer is now selected.
 * @returns {void}
 */
function setSelectedAnswer(questionId, answerItemId, checked) {
  const selected = new Set(state.selectedAnswers.get(questionId) || []);
  if (checked) {
    selected.add(answerItemId);
  } else {
    selected.delete(answerItemId);
  }

  if (selected.size === 0) {
    state.selectedAnswers.delete(questionId);
  } else {
    state.selectedAnswers.set(questionId, [...selected]);
  }
  renderDimensions();
  scheduleResponseSave(questionId);
}

/**
 * Remove selected answers that are no longer allowed for the active class.
 *
 * @returns {void}
 */
function pruneSelectedAnswersForCustomerClass() {
  state.questionsByDimension.forEach((questions) => {
    questions.forEach((question) => {
      const selected = state.selectedAnswers.get(question.question_id) || [];
      const allowed = filterSelectedAnswerIds(
        state.customerClassScope,
        question,
        selected,
        state.selectedCustomerClass
      );
      if (allowed.length === 0) {
        state.selectedAnswers.delete(question.question_id);
      } else {
        state.selectedAnswers.set(question.question_id, allowed);
      }
    });
  });
}

/**
 * Render the dimension tab strip.
 *
 * @returns {void}
 */
function renderDimensions() {
  const container = document.getElementById("dimension-tabs");
  if (!container) {
    return;
  }

  container.innerHTML = state.dimensions
    .map((dimension) => {
      const answeredCount = currentAnsweredCount(dimension.dimension);
      const activeClass = dimension.dimension === state.selectedDimension ? " active" : "";
      return `
        <button class="dimension-tab${activeClass}" data-dimension="${escapeHtml(dimension.dimension)}">
          <span>${escapeHtml(t("assessment.answered", {
            answered: answeredCount,
            total: dimension.question_count
          }))}</span>
          <strong>${escapeHtml((dimension.display_name || dimension.dimension).toUpperCase())}</strong>
        </button>
      `;
    })
    .join("");

  container.querySelectorAll("[data-dimension]").forEach((button) => {
    button.addEventListener("click", async () => {
      state.selectedDimension = button.dataset.dimension;
      await loadQuestions(state.selectedDimension);
      renderDimensions();
      renderQuestions();
      resumeStoredJobPolling(state.selectedDimension);
    });
  });
}

/**
 * Count questions with at least one selected answer in a dimension.
 *
 * @param {string} dimension - Dimension name to count.
 * @returns {number} Count of answered questions.
 */
function currentAnsweredCount(dimension) {
  const questions = state.questionsByDimension.get(dimensionLanguageKey(dimension)) || [];
  return questions.filter((question) => {
    const selected = state.selectedAnswers.get(question.question_id) || [];
    return selected.length > 0;
  }).length;
}

/**
 * Update static Assessment page labels for the active language.
 *
 * @returns {void}
 */
function renderStaticLabels() {
  const pageTitle = document.getElementById("assessment-page-title");
  if (pageTitle) {
    pageTitle.textContent = t("assessment.title");
  }

  const versionWarning = document.getElementById("assessment-version-warning");
  if (versionWarning) {
    versionWarning.textContent = t("assessment.versionWarning");
  }

  const resetButton = document.getElementById("reset-assessment-button");
  if (resetButton) {
    resetButton.textContent = t("assessment.reset");
  }

  const scoreButton = document.getElementById("score-page-button");
  if (scoreButton) {
    scoreButton.textContent = t("assessment.score");
  }

  const batchButton = document.getElementById("batch-upload-button");
  if (batchButton) {
    batchButton.textContent = t("assessment.batchUpload");
  }

  const customerClassLabel = document.getElementById("customer-class-label");
  if (customerClassLabel) {
    customerClassLabel.textContent = t("assessment.customerClassLabel");
  }

  const nace1Label = document.getElementById("nace1-label");
  if (nace1Label) {
    nace1Label.textContent = t("assessment.nace1Label");
  }

  const customerClassSelect = document.getElementById("customer-class-select");
  if (customerClassSelect) {
    const cohortClasses = benchmarkCustomerClasses(state.benchmarkOptions);
    const allowedClasses = new Set(cohortClasses);
    if (state.selectedCustomerClass) {
      allowedClasses.add(state.selectedCustomerClass);
    }
    const options = customerClassOptions(state.customerClassScope, state.language).filter(
      (option) => allowedClasses.size === 0 || allowedClasses.has(option.value)
    );
    customerClassSelect.innerHTML = options
      .map((option) => `
        <option value="${escapeHtml(option.value)}">${escapeHtml(option.label)}</option>
      `)
      .join("");
    customerClassSelect.value = state.selectedCustomerClass || "";
    customerClassSelect.disabled = state.profileSaveInFlight;
  }

  const nace1Select = document.getElementById("nace1-select");
  if (nace1Select) {
    const naceOptions = naceOptionsForClass(
      state.benchmarkOptions,
      state.selectedCustomerClass
    );
    if (state.selectedNace1 && !naceOptions.includes(state.selectedNace1)) {
      naceOptions.unshift(state.selectedNace1);
    }
    nace1Select.innerHTML = naceOptions.length
      ? naceOptions
          .map(
            (nace1) =>
              `<option value="${escapeHtml(nace1)}">${escapeHtml(nace1)}</option>`
          )
          .join("")
      : `<option value="">${escapeHtml(t("assessment.noNaceOptions"))}</option>`;
    nace1Select.value = state.selectedNace1 || "";
    nace1Select.disabled = state.profileSaveInFlight || naceOptions.length === 0;
  }

  const saveButton = document.getElementById("save-draft-button");
  if (saveButton) {
    saveButton.textContent = t("assessment.saveDraft");
  }
}

/**
 * Render the current questionnaire answer column for one question.
 *
 * @param {object} question - Assessment question payload.
 * @returns {string} HTML for the current answer column.
 */
function renderCurrentAnswers(question) {
  const grouped = groupedByLevel(question.answer_items || []);
  const maxLevel = maxAllowedLevel(
    state.customerClassScope,
    question.question_id,
    state.selectedCustomerClass
  );
  return [...grouped.entries()]
    .map(([level, items]) => `
      <div class="level-group">
        <div class="level-label">${escapeHtml(t("assessment.level", { level }))}</div>
        ${items
          .map((item) => {
            const selected = state.selectedAnswers.get(question.question_id) || [];
            const checked = selected.includes(item.answer_item_id) ? "checked" : "";
            const unavailable = Number(item.level) > maxLevel;
            return `
              <label class="answer-row${unavailable ? " answer-row-unavailable" : ""}">
                <input
                  type="checkbox"
                  data-question="${escapeHtml(question.question_id)}"
                  data-answer="${escapeHtml(item.answer_item_id)}"
                  ${checked}
                  ${unavailable ? "disabled" : ""}
                >
                <span>${escapeHtml(item.text)}</span>
              </label>
            `;
          })
          .join("")}
      </div>
    `)
    .join("");
}

/**
 * Render all AI decisions for one reviewed level.
 *
 * @param {object} question - Assessment question payload.
 * @param {object} level - AI level review result.
 * @returns {string} HTML for the level-level AI decisions.
 */
function renderAiLevel(question, level) {
  const decisions = level.answer_item_decisions || [];
  const decisionRows = decisions.length
    ? decisions
        .map((decision) => {
          const item = question.answer_items.find(
            (candidate) => candidate.answer_item_id === decision.answer_item_id
          );
          const checked = ["keep_selected", "select", "low_confidence"].includes(decision.decision)
            ? "checked"
            : "";
          return `
            <div class="ai-answer-row">
              <input type="checkbox" disabled ${checked}>
              <span>${escapeHtml(item?.text || decision.answer_item_id)}</span>
              <span class="decision ${escapeHtml(decision.decision)}">${escapeHtml(formatDisplayLabel(decision.decision))}</span>
            </div>
          `;
        })
        .join("")
    : `<div class="ai-empty small">${escapeHtml(t("assessment.noDecisions"))}</div>`;

  return `
    <div class="ai-level-block">
      <div class="ai-level-head">
        <strong>${escapeHtml(t("assessment.level", { level: level.level }))}</strong>
        <span>${escapeHtml(formatDisplayLabel(level.level_status))}</span>
      </div>
      ${decisionRows}
      <div class="ai-level-reason">
        <strong>${escapeHtml(t("assessment.reasoningEvidence"))}</strong>
        <p>${escapeHtml(level.level_reasoning)}</p>
      </div>
    </div>
  `;
}

/**
 * Render the AI verified answer column for one question.
 *
 * @param {object} question - Assessment question payload.
 * @returns {string} HTML for the AI verified answer column.
 */
function renderAiVerified(question) {
  const taskStatus = state.latestTaskStatuses.get(
    questionLanguageKey(question.question_id)
  );
  const activeStatuses = new Set([
    "in_progress",
    "extracting_documents",
    "embedding_documents",
    "retrieving_evidence",
    "finalizing_answer",
    "retrying"
  ]);
  if (activeStatuses.has(taskStatus?.status)) {
    return `
      <div class="ai-progress">
        <strong>${escapeHtml(t("assessment.questionInProgress"))}</strong>
        <p>${escapeHtml(taskStatus.progress_message || t("assessment.questionProgressFallback"))}</p>
      </div>
    `;
  }
  if (taskStatus?.status === "failed") {
    return `
      <div class="ai-error">
        <strong>${escapeHtml(t("assessment.aiFailed"))}</strong>
        <p>${escapeHtml(taskStatus.error_message || t("assessment.aiFailedFallback"))}</p>
      </div>
    `;
  }

  const result = state.latestResults.get(questionLanguageKey(question.question_id));
  if (!result) {
    return `
      <div class="ai-empty">
        ${escapeHtml(t("assessment.noAi"))}
      </div>
    `;
  }

  const levels = result.level_results || [];
  const rerunWarning = taskStatus?.result_requires_rerun
    ? `
      <div class="ai-warning">
        <strong>${escapeHtml(t("assessment.legacyResultTitle"))}</strong>
        <p>${escapeHtml(t("assessment.legacyResultMessage"))}</p>
      </div>
    `
    : "";
  const levelHtml = levels.length
    ? levels.map((level) => renderAiLevel(question, level)).join("")
    : `
      <div class="ai-empty">
        ${escapeHtml(t("assessment.noLevelResults"))}
      </div>
    `;

  return `
    ${rerunWarning}
    <div class="ai-result-summary">
      <span>${escapeHtml(formatDisplayLabel(result.overall_status))}</span>
      <strong>${escapeHtml(result.highest_supported_level ? t("assessment.highestSupportedLevel", {
        level: result.highest_supported_level
      }) : t("assessment.noSupportedLevel"))}</strong>
    </div>
    ${levelHtml}
  `;
}

/**
 * Return whether a question has AI verified answer IDs available to apply.
 *
 * @param {string} questionId - Question ID to inspect.
 * @returns {boolean} Whether Apply AI marks should be enabled.
 */
function hasApplicableAiMarks(questionId) {
  const taskStatus = state.latestTaskStatuses.get(questionLanguageKey(questionId));
  if (taskStatus?.result_requires_rerun) {
    return false;
  }
  const result = state.latestResults.get(questionLanguageKey(questionId));
  return Array.isArray(result?.verified_selected_answer_item_ids)
    && result.verified_selected_answer_item_ids.length > 0;
}

/**
 * Render all question rows for the active dimension.
 *
 * @returns {void}
 */
function renderQuestions() {
  const container = document.getElementById("questions-container");
  if (!container) {
    return;
  }

  const questions = currentQuestions();
  if (questions.length === 0) {
    container.innerHTML = `
      <section class="empty-state">
        ${escapeHtml(t("assessment.noFramework"))}
      </section>
    `;
    return;
  }

  container.innerHTML = questions
    .map((question) => `
      <article class="question-block">
        <header class="question-header">
          <span class="expand-icon">⌄</span>
          <span class="status-dot"></span>
          <span class="question-id">${escapeHtml(question.question_id)}</span>
          <span class="section-pill">${escapeHtml(question.section)}</span>
          <strong>${escapeHtml(question.question).toUpperCase()}</strong>
          <div class="question-header-actions">
            <ui5-button
              class="apply-ai-marks-button"
              data-apply-ai-marks="${escapeHtml(question.question_id)}"
              design="Transparent"
              icon="accept"
              ${hasApplicableAiMarks(question.question_id)
                && !state.aiMarksApplyInFlight.has(question.question_id) ? "" : "disabled"}
            >
              ${escapeHtml(t("assessment.applyAiMarks"))}
            </ui5-button>
            <span class="handled-pill">${escapeHtml(t("assessment.handled"))}</span>
          </div>
        </header>
        <div class="question-split">
          <section class="current-answers">
            <div class="panel-title">${escapeHtml(t("assessment.currentAnswers"))}</div>
            ${renderCurrentAnswers(question)}
            <ui5-button
              class="question-analysis-button"
              data-question-analysis="${escapeHtml(question.question_id)}"
              icon="sys-find"
            >
              ${escapeHtml(t("assessment.attachFile"))}
            </ui5-button>
          </section>
          <section class="ai-verified">
            <div class="panel-title">${escapeHtml(t("assessment.aiVerified"))}</div>
            ${renderAiVerified(question)}
          </section>
        </div>
      </article>
    `)
    .join("");

  attachQuestionHandlers(container);
}

/**
 * Attach event listeners after the question HTML has been rendered.
 *
 * @param {Element} container - Questions container element.
 * @returns {void}
 */
function attachQuestionHandlers(container) {
  container.querySelectorAll("input[type='checkbox'][data-question]").forEach((input) => {
    input.addEventListener("change", () => {
      setSelectedAnswer(input.dataset.question, input.dataset.answer, input.checked);
    });
  });

  container.querySelectorAll("[data-question-analysis]").forEach((button) => {
    button.addEventListener("click", () => {
      submitQuestionAnalysis(button.dataset.questionAnalysis).catch((error) => {
        showJobSummary(t("assessment.startFailed", { message: error.message }), "error");
      });
    });
  });

  container.querySelectorAll("[data-apply-ai-marks]").forEach((button) => {
    button.addEventListener("click", () => {
      applyAiMarks(button.dataset.applyAiMarks).catch((error) => {
        showJobSummary(t("assessment.startFailed", { message: error.message }), "error");
      });
    });
  });
}

/**
 * Load dimensions from the backend.
 *
 * @returns {Promise<void>} Promise resolved after dimensions are loaded.
 */
async function loadDimensions() {
  state.language = getLanguage();
  const currentDimension = state.selectedDimension;
  state.dimensions = await request(
    `/api/assessment/dimensions?language=${encodeURIComponent(state.language)}`
  );
  state.selectedDimension = state.dimensions.some(
    (dimension) => dimension.dimension === currentDimension
  )
    ? currentDimension
    : state.dimensions[0]?.dimension || "Strategy";
}

/**
 * Load questions for one dimension, using the cache on repeated visits.
 *
 * @param {string} dimension - Dimension name selected by the user.
 * @returns {Promise<void>} Promise resolved after questions are loaded.
 */
async function loadQuestions(dimension) {
  const cacheKey = dimensionLanguageKey(dimension);
  if (state.questionsByDimension.has(cacheKey)) {
    return;
  }
  const questions = await request(
    `/api/assessment/dimensions/${encodeURIComponent(dimension)}/questions?language=${encodeURIComponent(state.language)}`
  );
  state.questionsByDimension.set(cacheKey, questions);
}

/**
 * Load persisted user answers from HANA into local UI state.
 *
 * @returns {Promise<void>} Promise resolved after selected answer state is loaded.
 */
async function loadPersistedResponses() {
  const response = await request(
    `/api/assessment/responses?assessment_id=${encodeURIComponent(ASSESSMENT_ID)}`
  );
  state.selectedAnswers = new Map(
    Object.entries(response.answers || {}).map(([questionId, answerIds]) => [
      questionId,
      Array.isArray(answerIds) ? answerIds : []
    ])
  );
  state.persistedResponseQuestionIds = new Set(Object.keys(response.answers || {}));
  pruneSelectedAnswersForCustomerClass();
}

/**
 * Load backend-owned customer class scope and normalize stored selection.
 *
 * @returns {Promise<void>} Promise resolved after scope and selected class are ready.
 */
async function loadCustomerClassScope() {
  if (!state.customerClassScope) {
    state.customerClassScope = await request("/api/assessment/customer-class-scope");
  }
  state.selectedCustomerClass = normalizeCustomerClass(
    state.customerClassScope,
    state.selectedCustomerClass
  );
  localStorage.setItem(CUSTOMER_CLASS_STORAGE_KEY, state.selectedCustomerClass);
}

/**
 * Persist the selected exact class/NACE profile and preserve company identity.
 *
 * @param {boolean} [showSuccess] - Whether to announce successful persistence.
 * @returns {Promise<{profile: object, requestGeneration: number} | null>} Persisted profile and its authoritative generation, or null without a cohort.
 */
async function persistAssessmentProfile(showSuccess = true) {
  if (!state.selectedCustomerClass || !state.selectedNace1) {
    return null;
  }
  // Invalidate reads that started before this mutation. A second invalidation
  // after the PUT prevents reads started while the write was in flight from
  // replacing the newly persisted profile.
  assessmentLoadOwner.invalidate();
  state.profileSaveInFlight = true;
  renderStaticLabels();
  showProfileStatus(t("assessment.profileSaving"), "info");
  try {
    const profile = await request(
      "/api/assessment/profile",
      "PUT",
      buildAssessmentProfilePayload(
        ASSESSMENT_ID,
        state.selectedCustomerClass,
        state.selectedNace1,
        state.assessmentProfile
      )
    );
    state.assessmentProfile = profile;
    state.selectedCustomerClass = profile.customer_class;
    state.selectedNace1 = profile.nace1;
    localStorage.setItem(CUSTOMER_CLASS_STORAGE_KEY, profile.customer_class);
    const requestGeneration = assessmentLoadOwner.invalidate();
    if (showSuccess) {
      showProfileStatus(t("assessment.profileSaved"), "success");
    } else {
      showProfileStatus(null);
    }
    return { profile, requestGeneration };
  } catch (error) {
    showProfileStatus(
      t("assessment.profileSaveFailed", { message: error.message }),
      "error"
    );
    throw error;
  } finally {
    state.profileSaveInFlight = false;
    renderStaticLabels();
  }
}

/**
 * Load active cohort options and the authoritative HANA profile for assessment.
 *
 * A missing profile is initialized once from a valid imported cohort; browser
 * storage is consulted only as a class hint for that first persisted selection.
 *
 * @param {number} requestGeneration - Generation owning this profile/options read.
 * @returns {Promise<number | null>} Authoritative continuation generation, or null when stale.
 */
async function loadBenchmarkProfileContext(requestGeneration) {
  let options;
  try {
    options = await request("/api/assessment/benchmark-options");
  } catch (error) {
    if (!ownsMountedAssessmentLoad(requestGeneration)) {
      return null;
    }
    options = { available: false, reason: "request_failed", cohorts: [] };
    showProfileStatus(
      t("assessment.profileOptionsFailed", { message: error.message }),
      "error"
    );
  }

  let persistedProfile = null;
  try {
    persistedProfile = await request(
      `/api/assessment/profile?assessment_id=${encodeURIComponent(ASSESSMENT_ID)}`
    );
  } catch (error) {
    if (error.status !== 404) {
      if (!ownsMountedAssessmentLoad(requestGeneration)) {
        return null;
      }
      throw error;
    }
  }

  if (!ownsMountedAssessmentLoad(requestGeneration)) {
    return null;
  }

  state.benchmarkOptions = options;
  const resolved = resolveAssessmentProfile(
    options,
    persistedProfile,
    state.selectedCustomerClass
  );
  if (resolved) {
    state.assessmentProfile = persistedProfile;
    state.selectedCustomerClass = resolved.customer_class;
    state.selectedNace1 = resolved.nace1;
    localStorage.setItem(CUSTOMER_CLASS_STORAGE_KEY, resolved.customer_class);
    if (resolved.needsPersistence) {
      const persisted = await persistAssessmentProfile(false);
      return persisted?.requestGeneration || null;
    } else {
      showProfileStatus(null);
    }
    return requestGeneration;
  }

  state.assessmentProfile = null;
  state.selectedNace1 = null;
  showProfileStatus(t("assessment.profileUnavailable"), "info");
  return requestGeneration;
}

/**
 * Flush debounced answer changes while the old profile class is authoritative.
 *
 * @returns {Promise<void>} Promise resolved after pending response writes finish.
 */
async function flushPendingResponsesBeforeProfileChange() {
  window.clearTimeout(state.pendingResponseSaveTimer);
  const pendingQuestionIds = [...state.pendingResponseQuestionIds];
  state.pendingResponseQuestionIds.clear();
  if (pendingQuestionIds.length > 0) {
    await saveResponses(pendingQuestionIds);
  }
  await state.responseSaveQueue;
}

/**
 * Load the initial framework view.
 *
 * @returns {Promise<void>} Promise resolved after the first render completes.
 */
async function loadAssessment() {
  let requestGeneration = assessmentLoadOwner.begin();
  try {
    await loadCustomerClassScope();
    if (!ownsMountedAssessmentLoad(requestGeneration)) {
      return;
    }
    requestGeneration = await loadBenchmarkProfileContext(requestGeneration);
    if (requestGeneration === null || !ownsMountedAssessmentLoad(requestGeneration)) {
      return;
    }
    await loadDimensions();
    if (!ownsMountedAssessmentLoad(requestGeneration)) {
      return;
    }
    await loadQuestions(state.selectedDimension);
    if (!ownsMountedAssessmentLoad(requestGeneration)) {
      return;
    }
    await loadPersistedResponses();
    if (!ownsMountedAssessmentLoad(requestGeneration)) {
      return;
    }
    renderStaticLabels();
    renderDimensions();
    renderQuestions();
    resumeStoredJobPolling(state.selectedDimension);
    resumeStoredBatchJobPolling();
  } catch (error) {
    if (!ownsMountedAssessmentLoad(requestGeneration)) {
      return;
    }
    showJobSummary(
      t("assessment.loadingError", { message: error.message }),
      "error"
    );
    renderQuestions();
  }
}

/**
 * Reload localized framework text after the shell language changes.
 *
 * @returns {Promise<void>} Promise resolved after the page is re-rendered.
 */
async function handleLanguageChange() {
  let requestGeneration = assessmentLoadOwner.begin();
  state.language = getLanguage();
  await loadCustomerClassScope();
  if (!ownsMountedAssessmentLoad(requestGeneration)) {
    return;
  }
  requestGeneration = await loadBenchmarkProfileContext(requestGeneration);
  if (requestGeneration === null || !ownsMountedAssessmentLoad(requestGeneration)) {
    return;
  }
  await loadDimensions();
  if (!ownsMountedAssessmentLoad(requestGeneration)) {
    return;
  }
  await loadQuestions(state.selectedDimension);
  if (!ownsMountedAssessmentLoad(requestGeneration)) {
    return;
  }
  await loadPersistedResponses();
  if (!ownsMountedAssessmentLoad(requestGeneration)) {
    return;
  }
  renderStaticLabels();
  renderDimensions();
  renderQuestions();
  resumeStoredJobPolling(state.selectedDimension);
  resumeStoredBatchJobPolling();
}

/**
 * Persist a customer class selection and refresh scoped UI state.
 *
 * @param {string} value - Selected native option value.
 * @returns {Promise<void>} Promise resolved after profile and responses refresh.
 */
async function handleCustomerClassChange(value) {
  await flushPendingResponsesBeforeProfileChange();
  const previousClass = state.selectedCustomerClass;
  const previousNace1 = state.selectedNace1;
  state.selectedCustomerClass = normalizeCustomerClass(state.customerClassScope, value);
  localStorage.setItem(CUSTOMER_CLASS_STORAGE_KEY, state.selectedCustomerClass);
  const naceOptions = naceOptionsForClass(
    state.benchmarkOptions,
    state.selectedCustomerClass
  );
  state.selectedNace1 = naceOptions.includes(state.selectedNace1)
    ? state.selectedNace1
    : naceOptions[0] || null;
  try {
    if (state.selectedNace1) {
      await persistAssessmentProfile();
      // The profile route revalidates stored answers transactionally when class
      // applicability narrows, so the UI reloads that authoritative result.
      await loadPersistedResponses();
    } else {
      showProfileStatus(t("assessment.profileUnavailable"), "info");
      pruneSelectedAnswersForCustomerClass();
    }
  } catch (error) {
    // Keep browser controls aligned with the still-authoritative HANA profile
    // when persistence fails instead of leaving a misleading local selection.
    state.selectedCustomerClass = previousClass;
    state.selectedNace1 = previousNace1;
    localStorage.setItem(CUSTOMER_CLASS_STORAGE_KEY, previousClass);
    renderStaticLabels();
    throw error;
  }
  renderStaticLabels();
  renderDimensions();
  renderQuestions();
}

/**
 * Persist an exact NACE-1 selection and refresh visible profile context.
 *
 * @param {string} value - NACE-1 selector value from the active exact cohort.
 * @returns {Promise<void>} Promise resolved after the HANA profile is saved.
 */
async function handleNace1Change(value) {
  if (!naceOptionsForClass(state.benchmarkOptions, state.selectedCustomerClass).includes(value)) {
    return;
  }
  await flushPendingResponsesBeforeProfileChange();
  const previousNace1 = state.selectedNace1;
  state.selectedNace1 = value;
  try {
    await persistAssessmentProfile();
  } catch (error) {
    state.selectedNace1 = previousNace1;
    renderStaticLabels();
    throw error;
  }
  renderStaticLabels();
}

/**
 * Build a JSON review request for selected questions in the active dimension.
 *
 * @param {Array<string>} questionIds - Question IDs to analyze.
 * @returns {object} Request body accepted by the corpus review endpoint.
 */
function buildReviewRequest(questionIds) {
  const requestedQuestions = questionIds
    .map((questionId) => loadedQuestionById(questionId))
    .filter(Boolean);
  const allowedQuestions = filterAllowedQuestions(
    state.customerClassScope,
    requestedQuestions,
    state.selectedCustomerClass
  );
  if (allowedQuestions.length === 0) {
    throw new Error(t("assessment.customerClassDimensionUnavailable"));
  }
  const currentAnswers = {};
  allowedQuestions.forEach((question) => {
    currentAnswers[question.question_id] = filterSelectedAnswerIds(
      state.customerClassScope,
      question,
      state.selectedAnswers.get(question.question_id) || [],
      state.selectedCustomerClass
    );
  });

  return {
    assessment_id: ASSESSMENT_ID,
    dimension: state.selectedDimension,
    language: state.language,
    customer_class: state.selectedCustomerClass,
    current_answers: currentAnswers,
    question_ids: allowedQuestions.map((question) => question.question_id)
  };
}

/**
 * Ensure questions from every dimension are loaded before a batch submission.
 *
 * @returns {Promise<void>} Promise resolved when all dimension question caches exist.
 */
async function loadAllQuestionsForBatch() {
  await Promise.all(
    state.dimensions.map((dimension) => loadQuestions(dimension.dimension))
  );
}

/**
 * Build the all-question analysis request from loaded framework answers.
 *
 * @returns {Promise<object>} JSON body accepted by the all-question endpoint.
 */
async function buildBatchReviewRequest() {
  await loadAllQuestionsForBatch();
  const currentAnswers = {};
  state.dimensions.forEach((dimension) => {
    const questions = state.questionsByDimension.get(
      dimensionLanguageKey(dimension.dimension)
    ) || [];
    filterAllowedQuestions(
      state.customerClassScope,
      questions,
      state.selectedCustomerClass
    ).forEach((question) => {
      currentAnswers[question.question_id] = filterSelectedAnswerIds(
        state.customerClassScope,
        question,
        state.selectedAnswers.get(question.question_id) || [],
        state.selectedCustomerClass
      );
    });
  });

  if (Object.keys(currentAnswers).length === 0) {
    throw new Error(t("assessment.batchAttachRequired"));
  }

  return {
    assessment_id: ASSESSMENT_ID,
    language: state.language,
    customer_class: state.selectedCustomerClass,
    current_answers: currentAnswers
  };
}

/**
 * Submit a dimension review job and start background polling.
 *
 * @returns {Promise<void>} Promise resolved after the job is accepted.
 */
async function submitDimensionAnalysis() {
  const saveButton = document.getElementById("save-draft-button");
  if (saveButton) {
    saveButton.loading = true;
  }

  try {
    const allowedQuestions = filterAllowedQuestions(
      state.customerClassScope,
      currentQuestions(),
      state.selectedCustomerClass
    );
    if (allowedQuestions.length === 0) {
      showJobSummary(t("assessment.customerClassDimensionUnavailable"), "info");
      return;
    }
    const questionIds = allowedQuestions.map((question) => question.question_id);
    const body = buildReviewRequest(questionIds);
    clearQuestionReviewState(body.question_ids);
    renderQuestions();
    showJobSummary(t("assessment.submitting"), "info");
    const response = await request("/api/ai-review/jobs", "POST", body);
    localStorage.setItem(storageKey(state.selectedDimension), response.job_id);
    showJobSummary(
      t("assessment.createdJob", {
        jobId: response.job_id,
        taskCount: response.task_count
      }),
      "success"
    );
    startJobPolling(response.job_id, state.selectedDimension, state.language);
  } finally {
    if (saveButton) {
      saveButton.loading = false;
    }
  }
}

/**
 * Submit one question review job and start background polling.
 *
 * @param {string} questionId - Question ID selected for analysis.
 * @returns {Promise<void>} Promise resolved after the job is accepted.
 */
async function submitQuestionAnalysis(questionId) {
  if (!isQuestionAvailable(state.customerClassScope, questionId, state.selectedCustomerClass)) {
    showJobSummary(t("assessment.customerClassQuestionUnavailable"), "info");
    return;
  }

  const button = document.querySelector(`[data-question-analysis="${CSS.escape(questionId)}"]`);
  if (button) {
    button.loading = true;
  }

  try {
    const body = buildReviewRequest([questionId]);
    clearQuestionReviewState(body.question_ids);
    renderQuestions();
    showJobSummary(t("assessment.submittingQuestion"), "info");
    const response = await request("/api/ai-review/jobs", "POST", body);
    localStorage.setItem(storageKey(state.selectedDimension), response.job_id);
    showJobSummary(
      t("assessment.createdJob", {
        jobId: response.job_id,
        taskCount: response.task_count
      }),
      "success"
    );
    startJobPolling(response.job_id, state.selectedDimension, state.language);
  } finally {
    if (button) {
      button.loading = false;
    }
  }
}

/**
 * Submit an all-question review job and start background polling.
 *
 * @returns {Promise<void>} Promise resolved after the job is accepted.
 */
async function submitAllQuestionsAnalysis() {
  const batchButton = document.getElementById("batch-upload-button");
  if (batchButton) {
    batchButton.loading = true;
  }

  try {
    const body = await buildBatchReviewRequest();
    clearQuestionReviewState(Object.keys(body.current_answers || {}));
    renderQuestions();
    showJobSummary(t("assessment.submittingBatch"), "info");
    const response = await request("/api/ai-review/batch-jobs", "POST", body);
    localStorage.setItem(batchStorageKey(state.language), response.job_id);
    showJobSummary(
      t("assessment.createdBatchJob", {
        jobId: response.job_id,
        taskCount: response.task_count
      }),
      "success"
    );
    startBatchJobPolling(response.job_id, state.language);
  } finally {
    if (batchButton) {
      batchButton.loading = false;
    }
  }
}

/**
 * Apply AI verified marks to one question and persist them in HANA.
 *
 * @param {string} questionId - Question receiving AI selected answers.
 * @returns {Promise<void>} Promise resolved after the UI and HANA state update.
 */
async function applyAiMarks(questionId) {
  if (state.aiMarksApplyInFlight.has(questionId)) {
    return;
  }
  const result = state.latestResults.get(questionLanguageKey(questionId));
  const answerIds = result?.verified_selected_answer_item_ids || [];
  if (answerIds.length === 0) {
    showJobSummary(t("assessment.applyAiMarksUnavailable"), "info");
    return;
  }
  state.aiMarksApplyInFlight.add(questionId);
  renderQuestions();
  try {
    const taskStatus = state.latestTaskStatuses.get(questionLanguageKey(questionId));
    const response = await queueResponseWrite(async () => {
      const applyResponse = await request("/api/assessment/responses/apply-ai", "POST", {
        assessment_id: ASSESSMENT_ID,
        customer_class: state.selectedCustomerClass,
        question_id: questionId,
        task_id: taskStatus?.task_id || null,
        answer_item_ids: answerIds
      });
      state.persistedResponseQuestionIds.add(questionId);
      return applyResponse;
    });
    const persisted = response.answers?.[questionId] || [];
    if (persisted.length === 0) {
      state.selectedAnswers.delete(questionId);
    } else {
      state.selectedAnswers.set(questionId, persisted);
    }
    renderDimensions();
    showJobSummary(t("assessment.applyAiMarksDone"), "success");
  } finally {
    state.aiMarksApplyInFlight.delete(questionId);
    renderQuestions();
  }
}

/**
 * Return the localStorage key used for one dimension's latest job.
 *
 * @param {string} dimension - Dimension whose job ID should be stored.
 * @returns {string} localStorage key.
 */
function storageKey(dimension, language = state.language) {
  return `${JOB_STORAGE_PREFIX}:${language}:${dimension}`;
}

/**
 * Return the localStorage key for the latest global batch job.
 *
 * @param {string} language - Language associated with the stored batch job.
 * @returns {string} localStorage key.
 */
function batchStorageKey(language = state.language) {
  return `${BATCH_JOB_STORAGE_PREFIX}:${language}`;
}

/**
 * Resume polling for a stored job when the user returns to a dimension.
 *
 * @param {string} dimension - Dimension whose stored job should be polled.
 * @returns {void}
 */
function resumeStoredJobPolling(dimension) {
  const jobId = localStorage.getItem(storageKey(dimension));
  if (jobId) {
    startJobPolling(jobId, dimension, state.language);
  }
}

/**
 * Resume polling for a stored global batch job in the current language.
 *
 * @returns {void}
 */
function resumeStoredBatchJobPolling() {
  const jobId = localStorage.getItem(batchStorageKey(state.language));
  if (jobId) {
    startBatchJobPolling(jobId, state.language);
  }
}

/**
 * Stop polling one review job if a timer is active.
 *
 * @param {string | null} jobId - Job identifier whose timer should stop.
 * @returns {void}
 */
function stopJobPolling(jobId) {
  if (!jobId) {
    return;
  }

  if (state.activePolls.has(jobId)) {
    clearInterval(state.activePolls.get(jobId));
    state.activePolls.delete(jobId);
  }
  state.activeBatchPolls.delete(jobId);
}

/**
 * Stop every active review poller tracked by the page.
 *
 * @returns {void}
 */
function stopAllJobPolling() {
  [...state.activePolls.keys()].forEach((jobId) => {
    stopJobPolling(jobId);
  });
}

/**
 * Return whether a job status is terminal for polling.
 *
 * @param {string} status - Job status returned by the backend.
 * @returns {boolean} Whether polling can stop for this job.
 */
function isTerminalJobStatus(status) {
  return ["completed", "failed", "partial_failed"].includes(status);
}

/**
 * Stop all active batch pollers except the supplied current job ID.
 *
 * @param {string} currentJobId - Batch job ID that should remain active.
 * @returns {void}
 */
function stopOtherBatchPolls(currentJobId) {
  [...state.activeBatchPolls].forEach((jobId) => {
    if (jobId !== currentJobId) {
      stopJobPolling(jobId);
    }
  });
}

/**
 * Return whether a polling error means the browser stored a deleted job ID.
 *
 * @param {Error} error - Polling error raised by the API helper.
 * @returns {boolean} Whether the backend reported an unknown job.
 */
function isUnknownStoredJobError(error) {
  const message = String(error?.message || "");
  return message.includes("status: 404") && message.includes("Unknown AI review job ID");
}

/**
 * Remove one stale dimension-scoped job ID when another browser reset it.
 *
 * @param {string} jobId - Job identifier that failed polling.
 * @param {string} dimension - Dimension whose local storage key should be checked.
 * @param {string} language - Language associated with the stored job ID.
 * @returns {void}
 */
function clearStoredDimensionJob(jobId, dimension, language) {
  const key = storageKey(dimension, language);
  if (localStorage.getItem(key) === jobId) {
    localStorage.removeItem(key);
  }
}

/**
 * Remove one stale all-question job ID when another browser reset it.
 *
 * @param {string} jobId - Job identifier that failed polling.
 * @param {string} language - Language associated with the stored batch job ID.
 * @returns {void}
 */
function clearStoredBatchJob(jobId, language) {
  const key = batchStorageKey(language);
  if (localStorage.getItem(key) === jobId) {
    localStorage.removeItem(key);
  }
}

/**
 * Start polling one review job without blocking navigation.
 *
 * @param {string} jobId - AI review job identifier.
 * @param {string} dimension - Dimension that owns the job.
 * @param {string} language - Language associated with the stored job ID.
 * @returns {void}
 */
function startJobPolling(jobId, dimension, language = state.language) {
  if (state.activePolls.has(jobId)) {
    return;
  }

  const poll = async () => {
    try {
      const payload = await request(`/api/ai-review/jobs/${encodeURIComponent(jobId)}`);
      applyJobStatus(payload);
      if (isTerminalJobStatus(payload.status)) {
        stopJobPolling(jobId);
      }
      const payloadLanguage = payload.language || language;
      if (dimension === state.selectedDimension && payloadLanguage === state.language) {
        showJobSummary(formatJobStatus(payload), jobSummaryTone(payload.status));
        renderQuestions();
      }
    } catch (error) {
      stopJobPolling(jobId);
      if (isUnknownStoredJobError(error)) {
        clearStoredDimensionJob(jobId, dimension, language);
        if (dimension === state.selectedDimension && language === state.language) {
          clearQuestionReviewState(
            currentQuestions().map((question) => question.question_id),
            language
          );
          renderQuestions();
          showJobSummary(t("assessment.staleJobCleared"), "info");
        }
        return;
      }
      if (dimension === state.selectedDimension && language === state.language) {
        showJobSummary(t("assessment.pollFailed", { message: error.message }), "error");
      }
    }
  };

  state.activePolls.set(jobId, window.setInterval(poll, POLL_INTERVAL_MS));
  poll();
}

/**
 * Start polling one global batch review job.
 *
 * @param {string} jobId - Batch AI review job identifier.
 * @param {string} language - Language associated with the stored batch job ID.
 * @returns {void}
 */
function startBatchJobPolling(jobId, language = state.language) {
  stopOtherBatchPolls(jobId);
  if (state.activePolls.has(jobId)) {
    return;
  }

  const poll = async () => {
    try {
      const payload = await request(
        `/api/ai-review/batch-jobs/${encodeURIComponent(jobId)}`
      );
      if (!state.activeBatchPolls.has(jobId)) {
        return;
      }
      applyJobStatus(payload);
      if (isTerminalJobStatus(payload.status)) {
        stopJobPolling(jobId);
        const payloadLanguage = payload.language || language;
        if (localStorage.getItem(batchStorageKey(payloadLanguage)) === jobId) {
          localStorage.removeItem(batchStorageKey(payloadLanguage));
        }
      }
      const payloadLanguage = payload.language || language;
      if (payloadLanguage === state.language) {
        showJobSummary(formatJobStatus(payload), jobSummaryTone(payload.status));
        renderDimensions();
        renderQuestions();
      }
    } catch (error) {
      stopJobPolling(jobId);
      if (isUnknownStoredJobError(error)) {
        clearStoredBatchJob(jobId, language);
        if (language === state.language) {
          clearQuestionReviewState(
            currentQuestions().map((question) => question.question_id),
            language
          );
          renderQuestions();
          showJobSummary(t("assessment.staleJobCleared"), "info");
        }
        return;
      }
      if (language === state.language) {
        showJobSummary(t("assessment.pollFailed", { message: error.message }), "error");
      }
    }
  };

  state.activePolls.set(jobId, window.setInterval(poll, POLL_INTERVAL_MS));
  state.activeBatchPolls.add(jobId);
  poll();
}

/**
 * Copy completed task results into the UI state map.
 *
 * @param {object} payload - Job status response from the backend.
 * @returns {void}
 */
function applyJobStatus(payload) {
  const resultLanguage = payload.language || state.language;
  (payload.tasks || []).forEach((task) => {
    state.latestTaskStatuses.set(
      questionLanguageKey(task.question_id, resultLanguage),
      task
    );
    if (task.result) {
      state.latestResults.set(
        questionLanguageKey(task.question_id, resultLanguage),
        task.result
      );
    }
  });
}

/**
 * Map backend job status values to summary visual tones.
 *
 * @param {string} status - Job status returned by the backend.
 * @returns {"info" | "error" | "success"} Summary tone.
 */
function jobSummaryTone(status) {
  if (status === "completed") {
    return "success";
  }
  if (status === "failed" || status === "partial_failed") {
    return "error";
  }
  return "info";
}

/**
 * Format a compact job status summary.
 *
 * @param {object} payload - Job status response from the backend.
 * @returns {string} User-visible status text.
 */
function activeTaskProgress(payload) {
  const activeStatuses = new Set([
    "in_progress",
    "extracting_documents",
    "embedding_documents",
    "retrieving_evidence",
    "finalizing_answer",
    "retrying"
  ]);
  const activeTasks = (payload.tasks || []).filter((task) =>
    activeStatuses.has(task.status)
  );
  if (activeTasks.length === 0) {
    return null;
  }
  const activeQuestionTask = activeTasks.find(
    (task) => task.question_id === payload.active_question_id
  );
  if (activeQuestionTask) {
    return activeQuestionTask;
  }
  return [...activeTasks].sort((left, right) => {
    const leftTime = Date.parse(left.updated_at || "") || 0;
    const rightTime = Date.parse(right.updated_at || "") || 0;
    return rightTime - leftTime;
  })[0];
}

/**
 * Return progress counts appropriate for the active batch phase.
 *
 * @param {object} payload - Job status response from the backend.
 * @returns {{completed: number, total: number}} Counts for the summary text.
 */
function jobProgressCounts(payload) {
  if (["pending_indexing", "extracting_documents", "embedding_documents"].includes(payload.batch_phase)) {
    const documentTotal = Number(payload.document_count || 0);
    if (documentTotal > 0) {
      return {
        completed: Number(payload.processed_document_count || 0),
        total: documentTotal
      };
    }
  }
  return {
    completed: Number(payload.completed_count || 0),
    total: Number(payload.task_count || 0)
  };
}

function formatJobStatus(payload) {
  const failedText = payload.failed_count
    ? t("assessment.failedText", { failed: payload.failed_count })
    : "";
  const activeTask = activeTaskProgress(payload);
  const progressCounts = jobProgressCounts(payload);
  const statusForMessage = isTerminalJobStatus(payload.status)
    ? payload.status
    : payload.batch_phase || payload.status;
  const message = activeTask?.progress_message
    || formatDisplayLabel(statusForMessage);
  return t("assessment.activeProgress", {
    message,
    completed: progressCounts.completed,
    total: progressCounts.total,
    failedText
  });
}

/**
 * Reset all local and persisted draft/review state for the assessment.
 *
 * @returns {Promise<void>} Promise resolved after backend reset succeeds.
 */
async function resetAssessment() {
  const dimension = state.selectedDimension;
  stopAllJobPolling();
  showJobSummary(t("assessment.resetting"), "info");

  const query = new URLSearchParams({
    assessment_id: ASSESSMENT_ID,
    dimension
  });
  const response = await request(`/api/ai-review/jobs?${query.toString()}`, "DELETE");
  await clearPersistedResponses();
  const dimensions = state.dimensions.map((item) => item.dimension);
  ["en", "it"].forEach((language) => {
    dimensions.forEach((dimensionName) => {
      localStorage.removeItem(storageKey(dimensionName, language));
    });
    localStorage.removeItem(batchStorageKey(language));
  });

  state.selectedAnswers.clear();
  state.persistedResponseQuestionIds.clear();
  state.latestResults.clear();
  state.latestTaskStatuses.clear();
  renderDimensions();
  renderQuestions();
  showJobSummary(
    t("assessment.resetDone", {
      dimension: dimensionDisplayName(dimension),
      jobs: response.deleted_job_count,
      tasks: response.deleted_task_count
    }),
    "info"
  );
}

/**
 * Initialize the assessment page after its HTML has been mounted.
 *
 * @returns {void}
 */
export default function initAssessmentPage() {
  // Route remounts immediately revoke ownership from requests issued by the
  // previous DOM before any new asynchronous work is scheduled.
  assessmentLoadOwner.invalidate();
  renderStaticLabels();
  if (!languageChangeListenerAttached) {
    document.addEventListener("language-change", () => {
      handleLanguageChange().catch((error) => {
        showJobSummary(t("assessment.loadingError", { message: error.message }), "error");
      });
    });
    languageChangeListenerAttached = true;
  }

  const saveButton = document.getElementById("save-draft-button");
  if (saveButton) {
    saveButton.addEventListener("click", () => {
      submitDimensionAnalysis().catch((error) => {
        showJobSummary(t("assessment.startFailed", { message: error.message }), "error");
      });
    });
  }

  const scoreButton = document.getElementById("score-page-button");
  if (scoreButton) {
    scoreButton.addEventListener("click", () => {
      window.pageRouter?.navigate("/score");
    });
  }

  const resetButton = document.getElementById("reset-assessment-button");
  if (resetButton) {
    resetButton.addEventListener("click", () => {
      resetAssessment().catch((error) => {
        showJobSummary(t("assessment.resetFailed", { message: error.message }), "error");
      });
    });
  }

  const batchButton = document.getElementById("batch-upload-button");
  if (batchButton) {
    batchButton.addEventListener("click", () => {
      submitAllQuestionsAnalysis().catch((error) => {
        showJobSummary(t("assessment.startFailed", { message: error.message }), "error");
      });
    });
  }

  const customerClassSelect = document.getElementById("customer-class-select");
  if (customerClassSelect) {
    customerClassSelect.addEventListener("change", () => {
      handleCustomerClassChange(customerClassSelect.value).catch((error) => {
        showProfileStatus(
          t("assessment.profileSaveFailed", { message: error.message }),
          "error"
        );
      });
    });
  }

  const nace1Select = document.getElementById("nace1-select");
  if (nace1Select) {
    nace1Select.addEventListener("change", () => {
      handleNace1Change(nace1Select.value).catch((error) => {
        showProfileStatus(
          t("assessment.profileSaveFailed", { message: error.message }),
          "error"
        );
      });
    });
  }

  loadAssessment();
}
