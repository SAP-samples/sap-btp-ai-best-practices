/**
 * Return sorted NACE-1 values valid for an exact customer-class cohort.
 *
 * @param {object | null} options - Benchmark-options API response.
 * @param {string | null | undefined} customerClass - Selected company class.
 * @returns {string[]} Unique valid NACE-1 selector values.
 */
export function naceOptionsForClass(options, customerClass) {
  return [
    ...new Set(
      (options?.cohorts || [])
        .filter((cohort) => cohort.customer_class === customerClass)
        .map((cohort) => cohort.nace1)
        .filter(Boolean)
    )
  ].sort((left, right) => left.localeCompare(right));
}

/**
 * Return sorted customer classes that have at least one active exact cohort.
 *
 * @param {object | null} options - Benchmark-options API response.
 * @returns {string[]} Unique customer-class selector values.
 */
export function benchmarkCustomerClasses(options) {
  return [
    ...new Set((options?.cohorts || []).map((cohort) => cohort.customer_class).filter(Boolean))
  ].sort((left, right) => left.localeCompare(right));
}

/**
 * Resolve authoritative persisted context or a one-time valid default.
 *
 * @param {object | null} options - Benchmark-options API response.
 * @param {object | null} persistedProfile - Profile loaded from HANA.
 * @param {string | null | undefined} customerClassHint - Browser-only default hint.
 * @returns {object | null} Persisted profile, transient default, or no context.
 */
export function resolveAssessmentProfile(options, persistedProfile, customerClassHint) {
  if (persistedProfile) {
    return persistedProfile;
  }
  if (!options?.available || !(options.cohorts || []).length) {
    return null;
  }
  const classValues = benchmarkCustomerClasses(options);
  const customerClass = classValues.includes(customerClassHint)
    ? customerClassHint
    : classValues[0];
  const nace1 = naceOptionsForClass(options, customerClass)[0];
  if (!customerClass || !nace1) {
    return null;
  }
  return { customer_class: customerClass, nace1, needsPersistence: true };
}

/**
 * Build the persisted profile request for the fixed demo assessment.
 *
 * @param {string} assessmentId - Assessment identifier being configured.
 * @param {string} customerClass - Exact selected customer class.
 * @param {string} nace1 - Exact selected NACE level-one cohort.
 * @param {object | null} [existingProfile] - Existing names/identity to preserve.
 * @returns {object} PUT `/api/assessment/profile` request body.
 */
export function buildAssessmentProfilePayload(
  assessmentId,
  customerClass,
  nace1,
  existingProfile = null
) {
  return {
    assessment_id: assessmentId,
    display_name: existingProfile?.display_name || "Demo assessment",
    source_company_id: existingProfile?.source_company_id || null,
    customer_class: customerClass,
    nace1
  };
}
