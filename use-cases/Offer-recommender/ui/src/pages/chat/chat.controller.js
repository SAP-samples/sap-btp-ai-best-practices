import {
  createChatThread,
  declineChatProgram,
  getChatThread,
  postChatMessage
} from "../../services/offerApi.js";

const STORAGE_KEY = "customer_offer_advisor_thread_id";

function getPrimaryOffer(state) {
  return state.decision_result?.final_offer || null;
}

function getAdditionalOffers(state) {
  const primaryOffer = getPrimaryOffer(state);
  return (state.decision_result?.eligible_offers || []).filter(
    (offer) => offer.program_id !== primaryOffer?.program_id
  );
}

function formatCurrency(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) {
    return null;
  }

  return Number(value).toLocaleString("en-US", {
    style: "currency",
    currency: "USD",
    minimumFractionDigits: 2,
    maximumFractionDigits: 2
  });
}

function formatConfidence(confidence) {
  if (!confidence) {
    return null;
  }

  const normalized = confidence.toLowerCase().replaceAll("_", " ");
  return normalized.charAt(0).toUpperCase() + normalized.slice(1);
}

function humanizeIdentifier(value) {
  return String(value || "").replaceAll("_", " ");
}

/**
 * Append a compact expandable explanation to an existing DOM container.
 *
 * @param {HTMLElement | null} container - Parent element that receives the explanation.
 * @param {object | null | undefined} explanation - API explanation payload.
 * @param {string} label - Summary label shown before the explanation text.
 * @returns {void}
 */
function appendDecisionExplanation(container, explanation, label = "Why") {
  if (!container || !explanation?.summary) {
    return;
  }

  const wrapper = document.createElement("details");
  wrapper.className = "decision-explanation";

  const summary = document.createElement("summary");
  summary.textContent = `${label}: ${explanation.summary}`;
  wrapper.appendChild(summary);

  const items = [
    ...(explanation.details || []),
    ...(explanation.facts_used || []),
    ...(explanation.blockers || []).map((blocker) => `Needed information: ${humanizeIdentifier(blocker)}`)
  ];
  if (items.length) {
    const list = document.createElement("ul");
    items.forEach((itemText) => {
      const item = document.createElement("li");
      item.textContent = itemText;
      list.appendChild(item);
    });
    wrapper.appendChild(list);
  }

  container.appendChild(wrapper);
}

function renderMessages(container, messages) {
  if (!container) {
    return;
  }

  container.innerHTML = "";
  messages.forEach((message) => {
    const bubble = document.createElement("div");
    bubble.className = `chat-bubble ${message.role}`;
    bubble.textContent = message.content;
    container.appendChild(bubble);
  });
}

function renderOptions(container, state, api, storage, controls) {
  if (!container) {
    return;
  }

  container.innerHTML = "";
  const options = state.current_question?.answer_options || [];
  if (!options.length) {
    return;
  }

  options.forEach((option) => {
    const button = document.createElement("button");
    button.className = "action-button secondary";
    button.textContent = option.label;
    button.disabled = controls.isBusy();
    button.addEventListener("click", async () => {
      const nextState = await controls.runExclusive(() =>
        api.postChatMessage(state.thread_id, option.label)
      );
      if (!nextState) {
        return;
      }
      renderChatState(nextState, api, storage, controls);
    });
    container.appendChild(button);
  });
}

function renderPrimarySummary(container, state) {
  if (!container) {
    return;
  }

  container.innerHTML = "";
  const offer = getPrimaryOffer(state);
  if (!offer) {
    container.textContent = "No primary recommendation yet.";
    return;
  }

  const title = document.createElement("div");
  title.className = "section-label";
  title.textContent = offer.display_name;
  container.appendChild(title);

  const details = [];
  const savings = formatCurrency(offer.metadata?.estimated_monthly_savings);
  if (savings) {
    details.push(`Estimated monthly savings: ${savings}`);
  }

  const confidence = formatConfidence(offer.confidence);
  if (confidence) {
    details.push(`${confidence} confidence`);
  }

  const summaryText = state.decision_result?.explanation?.summary || details.join(" • ");
  if (summaryText) {
    const summary = document.createElement("p");
    summary.textContent = summaryText;
    container.appendChild(summary);
  }
  appendDecisionExplanation(container, offer.explanation, "Why this recommendation");
}

function renderExplanation(container, state) {
  if (!container) {
    return;
  }

  container.innerHTML = "";
  const explanation = state.decision_result?.explanation;
  if (!explanation) {
    return;
  }

  if (explanation.facts_used?.length) {
    const factsSection = document.createElement("div");
    factsSection.className = "explanation-block";

    const factsLabel = document.createElement("div");
    factsLabel.className = "section-label";
    factsLabel.textContent = "Facts Used";
    factsSection.appendChild(factsLabel);

    const factsList = document.createElement("ul");
    explanation.facts_used.forEach((fact) => {
      const item = document.createElement("li");
      item.textContent = fact;
      factsList.appendChild(item);
    });
    factsSection.appendChild(factsList);
    container.appendChild(factsSection);
  }

  if (explanation.missing_core_facts?.length) {
    const missingSection = document.createElement("div");
    missingSection.className = "explanation-block";

    const missingLabel = document.createElement("div");
    missingLabel.className = "section-label";
    missingLabel.textContent = "Missing Core Facts";
    missingSection.appendChild(missingLabel);

    const missingList = document.createElement("ul");
    explanation.missing_core_facts.forEach((fact) => {
      const item = document.createElement("li");
      item.textContent = fact;
      missingList.appendChild(item);
    });
    missingSection.appendChild(missingList);
    container.appendChild(missingSection);
  }

}

function renderGuidance(container, state, additionalOffers) {
  if (!container) {
    return;
  }

  container.innerHTML = "";
}

function renderAdditionalOffers(container, state, additionalOffers) {
  if (!container) {
    return;
  }

  container.innerHTML = "";
  if (!getPrimaryOffer(state)) {
    return;
  }

  const heading = document.createElement("div");
  heading.className = "section-label";
  heading.textContent = "Additional programs now eligible";
  container.appendChild(heading);

  if (!additionalOffers.length) {
    const empty = document.createElement("p");
    empty.textContent = "No additional offers unlocked yet.";
    container.appendChild(empty);
    return;
  }

  const list = document.createElement("ul");
  additionalOffers.forEach((offer) => {
    const item = document.createElement("li");
    const name = document.createElement("strong");
    name.textContent = offer.display_name;
    item.appendChild(name);
    appendDecisionExplanation(item, offer.explanation, "Why");
    list.appendChild(item);
  });
  container.appendChild(list);
}

function renderDeclines(container, state, api, storage, controls) {
  if (!container) {
    return;
  }

  container.innerHTML = "";
  const offers = state.decision_result?.eligible_offers || [];
  if (!offers.length) {
    return;
  }

  const heading = document.createElement("div");
  heading.className = "section-label";
  heading.textContent = "Remove an offer from this session:";
  container.appendChild(heading);

  const actions = document.createElement("div");
  actions.className = "decline-actions-list";

  offers.forEach((offer) => {
    const button = document.createElement("button");
    button.className = "action-button tertiary";
    button.textContent = `Remove ${offer.display_name} for this session`;
    button.disabled = controls.isBusy();
    button.addEventListener("click", async () => {
      const nextState = await controls.runExclusive(() =>
        api.declineChatProgram(state.thread_id, offer.program_id)
      );
      if (!nextState) {
        return;
      }
      renderChatState(nextState, api, storage, controls);
    });
    actions.appendChild(button);
  });

  container.appendChild(actions);
}

function renderChatState(state, api, storage, controls) {
  if (state.thread_id) {
    storage.setItem(STORAGE_KEY, state.thread_id);
  }

  const status = document.getElementById("chat-status");
  const primarySummary = document.getElementById("chat-primary-summary");
  const explanation = document.getElementById("chat-explanation");
  const guidance = document.getElementById("chat-guidance");
  const additionalOffers = document.getElementById("chat-additional-offers");
  const messages = document.getElementById("chat-messages");
  const options = document.getElementById("chat-options");
  const declines = document.getElementById("chat-summary-declines");
  const extraOffers = getAdditionalOffers(state);

  if (status) {
    status.textContent = `Status: ${state.status_phase}`;
  }
  renderPrimarySummary(primarySummary, state);
  renderExplanation(explanation, state);
  renderGuidance(guidance, state, extraOffers);
  renderAdditionalOffers(additionalOffers, state, extraOffers);
  renderMessages(messages, state.messages || []);
  renderOptions(options, state, api, storage, controls);
  renderDeclines(declines, state, api, storage, controls);
  controls.syncBusyState();
}

function buildResumeFailureState(threadId) {
  return {
    thread_id: threadId,
    billing_account: null,
    customer_type: null,
    messages: [
      {
        role: "assistant",
        content: "We couldn't reconnect to your saved chat thread. Please try again later or start a new thread.",
        timestamp: new Date().toISOString()
      }
    ],
    pending_questions: [],
    current_question: null,
    decision_result: null,
    declined_programs: [],
    status_phase: "resume_failed",
    user_answers: {}
  };
}

export async function initChatPage(
  api = {
    createChatThread,
    getChatThread,
    postChatMessage,
    declineChatProgram
  },
  storage = window.localStorage
) {
  const sendButton = document.getElementById("chat-send");
  const input = document.getElementById("chat-input");
  const newThreadButton = document.getElementById("chat-new-thread");
  let requestInFlight = false;

  function syncBusyState() {
    if (sendButton) {
      sendButton.disabled = requestInFlight;
    }
    if (newThreadButton) {
      newThreadButton.disabled = requestInFlight;
    }
    if (input) {
      input.disabled = requestInFlight;
    }
    document
      .querySelectorAll("#chat-options button, #chat-summary-declines button")
      .forEach((button) => {
        button.disabled = requestInFlight;
      });
  }

  async function runExclusive(operation) {
    if (requestInFlight) {
      return null;
    }

    requestInFlight = true;
    syncBusyState();
    try {
      return await operation();
    } finally {
      requestInFlight = false;
      syncBusyState();
    }
  }

  const controls = {
    isBusy() {
      return requestInFlight;
    },
    runExclusive,
    syncBusyState
  };

  let threadId = storage.getItem(STORAGE_KEY);
  let state;

  if (threadId) {
    try {
      state = await api.getChatThread(threadId);
    } catch (error) {
      if (error?.status === 404) {
        state = null;
        storage.removeItem(STORAGE_KEY);
      } else {
        state = buildResumeFailureState(threadId);
      }
    }
  }

  if (!state) {
    const created = await api.createChatThread();
    threadId = created.thread_id;
    state = created.state;
  }

  renderChatState(state, api, storage, controls);

  sendButton?.addEventListener("click", async () => {
    if (!input.value.trim()) {
      return;
    }
    const nextState = await runExclusive(() =>
      api.postChatMessage(threadId, input.value.trim())
    );
    if (!nextState) {
      return;
    }
    input.value = "";
    renderChatState(nextState, api, storage, controls);
  });

  input?.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      sendButton?.click();
    }
  });

  newThreadButton?.addEventListener("click", async () => {
    const created = await runExclusive(() => api.createChatThread());
    if (!created) {
      return;
    }
    storage.removeItem(STORAGE_KEY);
    threadId = created.thread_id;
    renderChatState(created.state, api, storage, controls);
  });
}
