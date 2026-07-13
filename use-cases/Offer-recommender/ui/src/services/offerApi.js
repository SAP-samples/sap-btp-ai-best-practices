import { API_BASE_URL, API_KEY, request } from "./api.js";

export async function createChatThread() {
  return request("/api/chat/threads", "POST");
}

export async function getChatThread(threadId) {
  return request(`/api/chat/threads/${threadId}`);
}

export async function postChatMessage(threadId, message) {
  return request(`/api/chat/threads/${threadId}/messages`, "POST", { message });
}

export async function declineChatProgram(threadId, programId) {
  return request(`/api/chat/threads/${threadId}/decline`, "POST", { program_id: programId });
}

/**
 * Fetch saved customer answer overlays for one billing account.
 *
 * @param {string} billingAccount - Billing account identifier.
 * @returns {Promise<object>} Saved answer response from the API.
 */
export async function getSavedAnswers(billingAccount) {
  return request(`/api/answers/${encodeURIComponent(billingAccount)}`);
}

/**
 * Reset saved customer answer overlays for one billing account.
 *
 * @param {string} billingAccount - Billing account identifier.
 * @returns {Promise<object>} Reset response from the API.
 */
export async function resetSavedAnswersForAccount(billingAccount) {
  return request(`/api/answers/${encodeURIComponent(billingAccount)}`, "DELETE");
}

/**
 * Reset saved customer answer overlays for every account.
 *
 * @returns {Promise<object>} Reset response from the API.
 */
export async function resetAllSavedAnswers() {
  return request("/api/answers", "DELETE");
}

export async function evaluateAccount(billingAccount, userAnswers = {}, declinedPrograms = []) {
  return request("/api/accounts/evaluate", "POST", {
    billing_account: billingAccount,
    user_answers: userAnswers,
    declined_programs: declinedPrograms
  });
}

export async function runBatch() {
  return request("/api/batch/runs", "POST");
}

export async function getBatchRun(runId) {
  return request(`/api/batch/runs/${runId}`);
}

export function getBatchArtifactUrl(runId, fileName) {
  return `${API_BASE_URL}/api/batch/runs/${runId}/artifacts/${fileName}`;
}

export async function downloadBatchArtifact(runId, fileName) {
  const response = await fetch(getBatchArtifactUrl(runId, fileName), {
    headers: {
      "X-API-Key": API_KEY
    }
  });

  if (!response.ok) {
    const error = new Error(`HTTP error! status: ${response.status}`);
    error.status = response.status;
    throw error;
  }

  return response.blob();
}
