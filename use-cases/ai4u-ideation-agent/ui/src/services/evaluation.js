import { request } from "./api.js";

/**
 * Evaluate an ideation solution via AI scoring.
 * @param {Object} params - The solution fields to evaluate
 * @param {string} params.industry
 * @param {string} params.title
 * @param {string} params.problem
 * @param {string} params.solution
 * @param {string} [params.app_type]
 * @param {string} [params.company_context]
 * @param {string} [params.context_id]
 * @returns {Promise<{score: number, feedback: string, detailedAnalysis: string, suggestions: string[], appropriatenessLevel: string, success: boolean, error?: string}>}
 */
export async function evaluateSolution({ industry, title, problem, solution, app_type, company_context, context_id }) {
  return request("/api/ideation-agent/evaluate", "POST", {
    industry,
    title,
    problem,
    solution,
    app_type: app_type || "Agent",
    company_context: company_context || null,
    context_id: context_id || null,
  });
}

/**
 * Submit an evaluated solution to the leaderboard.
 * @param {Object} submission - Full submission data including evaluation results
 * @returns {Promise<{success: boolean, submission_id: string, message: string}>}
 */
export async function submitSolution(submission) {
  return request("/api/ideation-agent/submissions", "POST", submission);
}

/**
 * Fetch a stored evaluation by context_id.
 * Returns {found: true, score, feedback, ...} if evaluation is stored,
 * or {found: false} if this context has not been evaluated yet.
 * @param {string} contextId
 * @returns {Promise<{found: boolean, score?: number, feedback?: string, detailedAnalysis?: string, suggestions?: string[], appropriatenessLevel?: string, success?: boolean}>}
 */
export async function getStoredEvaluation(contextId) {
  if (!contextId) return { found: false };
  return request(`/api/ideation-agent/submissions/by-context/${encodeURIComponent(contextId)}`);
}

/**
 * Fetch ALL stored evaluations for a conversation (multiple final messages).
 * @param {string} contextId
 * @returns {Promise<{entries: Array<{messageIndex: number|null, score: number, feedback: string, detailedAnalysis: string, suggestions: string[], appropriatenessLevel: string}>}>}
 */
export async function getStoredEvaluations(contextId) {
  if (!contextId) return { entries: [] };
  return request(`/api/ideation-agent/submissions/by-context/${encodeURIComponent(contextId)}/all`);
}

/**
 * Get the current user's submission history.
 * @param {number} [limit=50]
 * @param {number} [cursor=0]
 * @returns {Promise<{entries: Array, total: number, limit: number, cursor: number}>}
 */
export async function getSubmissionHistory(limit = 50, cursor = 0) {
  return request(`/api/ideation-agent/submissions/history?limit=${limit}&cursor=${cursor}`);
}
