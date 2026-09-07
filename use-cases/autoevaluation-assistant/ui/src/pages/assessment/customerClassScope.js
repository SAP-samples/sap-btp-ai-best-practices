/**
 * Return a configured customer class, falling back to the scope default.
 *
 * @param {object | null} scope - Customer class scope payload from the API.
 * @param {string | null | undefined} value - Requested customer class value.
 * @returns {string} Known customer class identifier, or the configured default.
 */
export function normalizeCustomerClass(scope, value) {
  const defaultClass = scope?.default_customer_class || "";
  return value && Object.hasOwn(scope?.classes || {}, value) ? value : defaultClass;
}

/**
 * Build localized customer class options ordered by backend rank.
 *
 * @param {object | null} scope - Customer class scope payload from the API.
 * @param {string} language - Preferred UI language.
 * @returns {Array<{value: string, label: string}>} Native select options.
 */
export function customerClassOptions(scope, language) {
  return Object.entries(scope?.classes || {})
    .sort(([, left], [, right]) => Number(left.rank || 0) - Number(right.rank || 0))
    .map(([value, details]) => ({
      value,
      label: details.labels?.[language] || details.labels?.en || value
    }));
}

/**
 * Return the maximum answer level available for one question and class.
 *
 * @param {object | null} scope - Customer class scope payload from the API.
 * @param {string} questionId - Assessment question identifier.
 * @param {string | null | undefined} customerClass - Requested customer class.
 * @returns {number} Maximum allowed level, or zero when the question is unknown.
 */
export function maxAllowedLevel(scope, questionId, customerClass) {
  const levels = scope?.question_max_levels?.[questionId];
  if (!levels) {
    return 0;
  }
  const normalizedClass = normalizeCustomerClass(scope, customerClass);
  return Number(levels[normalizedClass] || 0);
}

/**
 * Return whether a question has any available answer level for a class.
 *
 * @param {object | null} scope - Customer class scope payload from the API.
 * @param {string} questionId - Assessment question identifier.
 * @param {string | null | undefined} customerClass - Requested customer class.
 * @returns {boolean} Whether the question is available.
 */
export function isQuestionAvailable(scope, questionId, customerClass) {
  return maxAllowedLevel(scope, questionId, customerClass) > 0;
}

/**
 * Keep only questions available to the selected customer class.
 *
 * @param {object | null} scope - Customer class scope payload from the API.
 * @param {Array<object>} questions - Assessment question payloads.
 * @param {string | null | undefined} customerClass - Requested customer class.
 * @returns {Array<object>} Questions whose max level is greater than zero.
 */
export function filterAllowedQuestions(scope, questions, customerClass) {
  return questions.filter((question) =>
    isQuestionAvailable(scope, question.question_id, customerClass)
  );
}

/**
 * Keep selected answer IDs that are still within the selected class max level.
 *
 * @param {object | null} scope - Customer class scope payload from the API.
 * @param {object} question - Assessment question payload with answer items.
 * @param {Array<string>} selectedAnswerIds - Selected answer item IDs.
 * @param {string | null | undefined} customerClass - Requested customer class.
 * @returns {Array<string>} Allowed selected answer IDs, preserving caller order.
 */
export function filterSelectedAnswerIds(scope, question, selectedAnswerIds, customerClass) {
  const maxLevel = maxAllowedLevel(scope, question.question_id, customerClass);
  const allowedIds = new Set(
    (question.answer_items || [])
      .filter((item) => Number(item.level) <= maxLevel)
      .map((item) => item.answer_item_id)
  );
  return selectedAnswerIds.filter((answerId) => allowedIds.has(answerId));
}
