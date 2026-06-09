import { evaluateAccount, getSavedAnswers } from "../../services/offerApi.js";

function escapeHtml(text) {
  return String(text)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function optionValue(value) {
  if (value === null || value === undefined) {
    return "__NULL__";
  }
  return String(value);
}

function humanizeIdentifier(value) {
  return String(value || "").replaceAll("_", " ");
}

function isAnswerableQuestion(question) {
  return question.source === "customer" && Boolean(question.answer_options?.length);
}

function systemStepText(question) {
  if (question.expected_fact === "has_current_snapshot") {
    return "A current billing snapshot is needed before this recommendation can continue.";
  }
  return question.prompt;
}

function questionMarkup(question, selectedAnswers = {}) {
  const options = question.answer_options || [];
  const optionMarkup = options.length
    ? `
      <select data-fact="${question.expected_fact}" class="inline-select">
        <option value="">Select an answer</option>
        ${options
          .map(
            (option) => {
              const value = optionValue(option.value);
              const isSelected = selectedAnswers[question.expected_fact] === option.value;
              return `<option value="${escapeHtml(value)}"${isSelected ? " selected" : ""}>${escapeHtml(option.label)}</option>`;
            }
          )
          .join("")}
      </select>
    `
    : `<p class="muted">This step requires a system or data update before the recommendation can continue.</p>`;

  return `
    <div class="question-card">
      <strong>${escapeHtml(options.length ? question.prompt : systemStepText(question))}</strong>
      ${optionMarkup}
    </div>
  `;
}

function explanationMarkup(result) {
  const explanation = result.explanation;
  if (!explanation) {
    return "";
  }

  const sections = [];
  if (explanation.summary) {
    sections.push(`
      <div class="explanation-section">
        <div class="section-label">Summary</div>
        <p>${escapeHtml(explanation.summary)}</p>
      </div>
    `);
  }

  if (explanation.facts_used?.length) {
    sections.push(`
      <div class="explanation-section">
        <div class="section-label">Facts Used</div>
        <ul>
          ${explanation.facts_used.map((fact) => `<li>${escapeHtml(fact)}</li>`).join("")}
        </ul>
      </div>
    `);
  }

  if (explanation.missing_core_facts?.length) {
    sections.push(`
      <div class="explanation-section">
        <div class="section-label">Missing Core Facts</div>
        <ul>
          ${explanation.missing_core_facts.map((fact) => `<li>${escapeHtml(fact)}</li>`).join("")}
        </ul>
      </div>
    `);
  }

  if (explanation.next_step) {
    sections.push(`
      <div class="explanation-section">
        <div class="section-label">Next Step</div>
        <p>${escapeHtml(explanation.next_step)}</p>
      </div>
    `);
  }

  return sections.join("");
}

/**
 * Render a compact expandable explanation for a recommendation or question.
 *
 * @param {object | null | undefined} explanation - API explanation payload.
 * @param {string} heading - Summary label shown before the explanation text.
 * @returns {string} HTML markup for the explanation or an empty string.
 */
function decisionExplanationMarkup(explanation, heading = "Why") {
  if (!explanation?.summary) {
    return "";
  }

  const details = explanation.details || [];
  const facts = explanation.facts_used || [];
  const blockers = explanation.blockers || [];
  const detailItems = [
    ...details,
    ...facts,
    ...blockers.map((blocker) => `Needed information: ${humanizeIdentifier(blocker)}`)
  ];

  return `
    <details class="decision-explanation">
      <summary>${escapeHtml(heading)}: ${escapeHtml(explanation.summary)}</summary>
      ${
        detailItems.length
          ? `<ul>${detailItems.map((item) => `<li>${escapeHtml(item)}</li>`).join("")}</ul>`
          : ""
      }
    </details>
  `;
}

/**
 * Render explanation rows for offer lists while excluding the primary offer when needed.
 *
 * @param {string} title - Section label shown above the offer list.
 * @param {Array<object>} offers - Offer response objects from the API.
 * @param {string | null} primaryProgramId - Program id to exclude from additional offers.
 * @returns {string} HTML markup for the offer explanations or an empty string.
 */
function offerExplanationMarkup(title, offers = [], primaryProgramId = null) {
  const visibleOffers = offers.filter((offer) => offer.program_id !== primaryProgramId);
  if (!visibleOffers.length) {
    return "";
  }

  return `
    <div class="explanation-section">
      <div class="section-label">${escapeHtml(title)}</div>
      ${visibleOffers
        .map(
          (offer) => `
            <div class="offer-explanation-row">
              <strong>${escapeHtml(offer.display_name)}</strong>
              ${decisionExplanationMarkup(offer.explanation, "Why")}
            </div>
          `
        )
        .join("")}
    </div>
  `;
}

function renderLookupResult(result, selectedAnswers = {}) {
  const summary = document.getElementById("lookup-summary");
  const questions = document.getElementById("lookup-questions");
  const allQuestions = result.questions || [];
  const answerableQuestions = allQuestions.filter(isAnswerableQuestion);
  const visibleQuestions = answerableQuestions.length ? answerableQuestions : allQuestions;

  const offerText = result.final_offer
    ? `${result.final_offer.display_name} (${result.final_offer.confidence})`
    : "No final offer yet";

  summary.innerHTML = `
    <div class="result-card">
      <div class="section-label">Account</div>
      <h2>${result.billing_account}</h2>
      <p>${result.customer_type}</p>
      <div class="section-label">Recommendation</div>
      <p>${offerText}</p>
      ${decisionExplanationMarkup(result.final_offer?.explanation, "Why this recommendation")}
      ${offerExplanationMarkup(
        "Additional Eligible Offers",
        result.eligible_offers || [],
        result.final_offer?.program_id
      )}
      ${offerExplanationMarkup("Blocked Or Pending Offers", result.blocked_offers || [])}
      ${explanationMarkup(result)}
      <div class="muted">${(result.flags || []).join(" | ")}</div>
    </div>
  `;

  questions.innerHTML = visibleQuestions.length
    ? visibleQuestions
        .map((question) => questionMarkup(question, selectedAnswers))
        .join("")
    : `<div class="result-card"><p>No follow-up questions remain.</p></div>`;
}

export async function initLookupPage(api = { evaluateAccount, getSavedAnswers }) {
  const input = document.getElementById("lookup-account-input");
  const submit = document.getElementById("lookup-submit");
  const recalc = document.getElementById("lookup-recalculate");
  const state = {
    answers: {},
    billingAccount: null,
  };

  /**
   * Collect currently selected lookup answers into the provided base mapping.
   *
   * @param {object} baseAnswers - Existing saved answers to preserve.
   * @returns {object} Merged answer mapping for evaluation.
   */
  const collectAnswers = (baseAnswers = state.answers) => {
    const answers = { ...baseAnswers };
    document.querySelectorAll("[data-fact]").forEach((element) => {
      if (!element.value) {
        return;
      }
      if (element.value === "__NULL__") {
        answers[element.dataset.fact] = null;
      } else if (element.value === "true") {
        answers[element.dataset.fact] = true;
      } else if (element.value === "false") {
        answers[element.dataset.fact] = false;
      } else if (!Number.isNaN(Number(element.value)) && element.value.trim() !== "") {
        answers[element.dataset.fact] = Number(element.value);
      } else {
        answers[element.dataset.fact] = element.value;
      }
    });
    return answers;
  };

  /**
   * Load saved answers for an account when the API method is available.
   *
   * @param {string} billingAccount - Billing account to load saved answers for.
   * @returns {Promise<object>} Saved answer values keyed by fact id.
   */
  const loadSavedAnswers = async (billingAccount) => {
    if (typeof api.getSavedAnswers !== "function") {
      return {};
    }
    const response = await api.getSavedAnswers(billingAccount);
    return response.answers || {};
  };

  const runLookup = async () => {
    const billingAccount = input.value.trim();
    if (!billingAccount) {
      return;
    }
    const accountChanged = state.billingAccount !== billingAccount;
    const baseAnswers = accountChanged && typeof api.getSavedAnswers === "function"
      ? await loadSavedAnswers(billingAccount)
      : accountChanged
        ? {}
        : state.answers;
    state.answers = accountChanged ? baseAnswers : collectAnswers(baseAnswers);
    state.billingAccount = billingAccount;
    const result = await api.evaluateAccount(billingAccount, state.answers, []);
    renderLookupResult(result, state.answers);
  };

  submit?.addEventListener("click", runLookup);
  recalc?.addEventListener("click", runLookup);
}
