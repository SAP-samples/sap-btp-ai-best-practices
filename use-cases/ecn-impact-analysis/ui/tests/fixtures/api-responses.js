export const rowOne = Object.freeze({
  ecn_row_id: "11111111-1111-4111-8111-111111111111",
  ecn_number: "ECN100",
  display_ecn_number: "ECN100",
  title: "Steering bracket update",
  description_summary: "Change bracket geometry",
  source_ordinal: 0,
  resolution_date: "2016-06-21T02:00:00+02:00",
  uploaded_at: "2026-09-14T08:00:00Z",
  source_file_name: "ecn.json",
  counts: {
    affected_item_count: 3,
    raw_ecn_match_count: 4,
    ebom_match_count: 2,
    ebom_context_count: 1,
    mbom_match_count: 1,
    warning_count: 2,
  },
  analysis_state: "RUNNING",
  completeness: null,
  overall_impact: null,
  affected_plants: ["PL10 Plant A", "DE01 Plant C"],
});

export const missingIdentifierRow = Object.freeze({
  ...rowOne,
  affected_plants: null,
  ecn_row_id: "22222222-2222-4222-8222-222222222222",
  ecn_number: null,
  display_ecn_number: "Missing ECN ID",
  description_summary: null,
  title: "Imported occurrence 2",
  source_ordinal: 1,
  counts: { ...rowOne.counts, warning_count: 0 },
  analysis_state: "COMPLETED",
  completeness: "COMPLETED",
  overall_impact: "MEDIUM",
});

/** Return an exact historical-list response for the supplied rows. */
export function listEnvelope(items = [rowOne, missingIdentifierRow]) {
  return {
    items,
    page: 1,
    page_size: 25,
    total_items: items.length,
    total_pages: items.length ? 1 : 0,
  };
}

/** Return an exact progress response with optional state overrides. */
export function progressEnvelope(overrides = {}) {
  return {
    upload_id: null,
    queued_count: 0,
    blocked_count: 0,
    running_count: 1,
    completed_count: 1,
    completed_with_gaps_count: 0,
    failed_count: 0,
    unfinished_count: 1,
    total_count: 2,
    ...overrides,
  };
}

export const uploadEnvelope = Object.freeze({
  upload_id: "33333333-3333-4333-8333-333333333333",
  request_id: "44444444-4444-4444-8444-444444444444",
  created: true,
  imported_ecn_count: 2,
  bom_artifact_id: "55555555-5555-4555-8555-555555555555",
  bom_reused: false,
  queued_count: 2,
  blocked_count: 0,
  running_count: 0,
  completed_count: 0,
  completed_with_gaps_count: 0,
  failed_count: 0,
  list_url: "/api/ecns",
  progress_url: "/api/analysis-progress?upload_id=33333333-3333-4333-8333-333333333333",
});

export const sectionKeys = [
  "executive_summary",
  "change_intent",
  "affected_materials",
  "ebom_mbom_impact",
  "configuration_scope",
  "uncertainties_and_missing_evidence",
];

/** Return one sufficient generated section tied to deterministic evidence. */
function sufficient(text, evidenceId) {
  return {
    status: "SUFFICIENT",
    assessment_markdown: text,
    evidence_ids: [evidenceId],
    missing_evidence: [],
  };
}

/** Return an exact completed-with-gaps detail response with safe and adversarial text. */
export function detailEnvelope(overrides = {}) {
  return {
    ecn_row_id: rowOne.ecn_row_id,
    ecn_number: rowOne.ecn_number,
    display_ecn_number: rowOne.display_ecn_number,
    title: rowOne.title,
    description_summary: rowOne.description_summary,
    description_text: "Long source description",
    source_ordinal: 0,
    source_status: "Released",
    requestor_name: "Dana Whitfield",
    affected_products: ["Q900V", "M1V"],
    resolution_date: rowOne.resolution_date,
    uploaded_at: rowOne.uploaded_at,
    source_file_name: "ecn.json",
    ecn_artifact: {
      artifact_id: "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa",
      kind: "ECN_JSON",
      file_name: "ecn.json",
      sha256: "a".repeat(64),
    },
    bom_artifact: {
      artifact_id: "bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb",
      kind: "BOM_WORKBOOK",
      file_name: "bom.xlsx",
      sha256: "b".repeat(64),
    },
    rule_artifact: null,
    counts: rowOne.counts,
    job: {
      job_id: "cccccccc-cccc-4ccc-8ccc-cccccccccccc",
      state: "COMPLETED",
      attempt_count: 1,
      error_code: null,
      error_message: null,
    },
    analysis: {
      analysis_id: "dddddddd-dddd-4ddd-8ddd-dddddddddddd",
      completeness: "COMPLETED_WITH_GAPS",
      change_brief: "Replace the steering bracket with the revised production design.",
      sections: {
        executive_summary: sufficient("## Summary\n\n|Area|Impact|\n|---|---|\n|EBOM|Review|", "E-1"),
        change_intent: {
          status: "INSUFFICIENT_EVIDENCE",
          assessment_markdown: "Intent is only partly established.",
          evidence_ids: ["E-2"],
          missing_evidence: ["Authoritative affected-to-resulting relation"],
        },
        affected_materials: sufficient("Three affected materials are listed.", "E-3"),
        ebom_mbom_impact: sufficient("An exact EBOM match exists.", "E-4"),
        configuration_scope: {
          status: "NOT_APPLICABLE",
          assessment_markdown: "No configuration rule applies to this occurrence.",
          evidence_ids: ["E-5"],
          missing_evidence: [],
        },
        uncertainties_and_missing_evidence: {
          status: "INSUFFICIENT_EVIDENCE",
          assessment_markdown: "One relationship remains unresolved.",
          evidence_ids: ["E-6"],
          missing_evidence: ["Authoritative relationship evidence"],
        },
      },
      created_at: "2026-09-14T08:01:00Z",
      evidence_hash: "c".repeat(64),
      prompt_version: "ecn-impact-v1",
      output_schema_version: "ecn-impact-sections-v1",
      model_name: "gpt-5.4",
      model_deployment: null,
      impact_factors: {
        manufacturing: {
          severity: "MEDIUM",
          assessment: "One MBOM occurrence needs production review.",
          evidence_ids: ["E-4"],
          missing_evidence: [],
        },
        procurement: {
          severity: "LOW",
          assessment: "No supported purchasing change was found.",
          evidence_ids: ["E-3"],
          missing_evidence: [],
        },
        configuration: {
          severity: "UNKNOWN",
          assessment: "Plant applicability is not established.",
          evidence_ids: [],
          missing_evidence: ["Applicable plant rule"],
        },
        data_readiness: {
          severity: "MEDIUM",
          assessment: "Two source warnings require review.",
          evidence_ids: [],
          missing_evidence: [],
        },
      },
      overall_impact: "MEDIUM",
    },
    ...overrides,
  };
}

/** Return one paginated business evidence section response. */
export function businessEvidenceEnvelope(section = "modified_materials", overrides = {}) {
  return {
    section,
    items: [{
      label: "Affected material",
      summary: "Material 495452W1 is explicitly affected by this change.",
      material_number: "495452W1",
      description: "Steering bracket",
      revision: "B",
      direction: "AFFECTED",
      parent: "Q900V",
      bom_path: ["Q900V", "CAB"],
      quantity: "1",
      effectivity: null,
      plant: "PL10",
      matched_row_count: 2,
      sources: [{
        artifact_kind: "BOM_WORKBOOK",
        json_path: null,
        sheet_name: "EBOM Q900V",
        source_row_number: 1392,
      }],
      technical_details: null,
    }],
    page: 1,
    page_size: 25,
    total_items: 1,
    total_pages: 1,
    ...overrides,
  };
}

/** Add optional internal audit detail to a business evidence page. */
export function technicalBusinessEvidenceEnvelope(section = "modified_materials") {
  const response = businessEvidenceEnvelope(section);
  response.items[0].technical_details = {
    internal_evidence_ids: ["E-0123456789abcdef"],
    raw_kinds: ["DETERMINISTIC_FINDING"],
    raw_tiers: ["A_DIRECT_ECN_RAW"],
    raw_values: [{ component_number: "495452W1" }],
    quality_qualifiers: ["SOURCE_RELATION_ERROR_VALUE"],
    related_ids: ["eeeeeeee-eeee-4eee-8eee-eeeeeeeeeeee"],
  };
  return response;
}

/** Return safe provider readiness for Settings. */
export function providerStatusEnvelope(overrides = {}) {
  return {
    checked_at: "2026-09-15T12:00:00Z",
    oauth: { status: "AVAILABLE", http_status: 200, provider_code: null, request_id: null },
    models: [
      { capability: "text_generation", model: "gpt-5.4", status: "AVAILABLE", http_status: null, provider_code: null, request_id: null },
    ],
    ...overrides,
  };
}

/** Return one publication list containing active snapshots for the five maintained types. */
export function referenceDatasetsEnvelope(overrides = {}) {
  return {
    items: [
      {
        dataset_id: "99999999-9999-4999-8999-999999999999",
        dataset_type: "BOM",
        file_name: "EBOM MBOM Q900V.xlsx",
        sha256: "d".repeat(64),
        scope_code: null,
        scope_name: null,
        row_count: 2100,
        warning_count: 4,
        warning_counts: { INVALID_RELATION: 4 },
        parser_version: "ecn-reference-v1",
        uploaded_at: "2026-09-15T11:00:00Z",
        active: true,
      },
      {
        dataset_id: "88888888-8888-4888-8888-888888888888",
        dataset_type: "MATERIAL_MASTER",
        file_name: "materials.xlsx",
        sha256: "e".repeat(64),
        scope_code: null,
        scope_name: null,
        row_count: 183,
        warning_count: 0,
        warning_counts: {},
        parser_version: "ecn-reference-v1",
        uploaded_at: "2026-09-15T11:30:00Z",
        active: true,
      },
    ],
    status_legend_version: "sap-procurement-status-v1",
    procurement_status_legend: {
      E: "In-house production",
      F: "External procurement",
      X: "In-house and external procurement allowed",
    },
    ...overrides,
  };
}

/** Return saved Plant Code Mapping settings as served by GET /api/plant-settings. */
export function plantSettingsEnvelope(overrides = {}) {
  return {
    mappings: [
      { suffix: "JB", plant_code: "PL10", plant_name: "Plant A" },
      { suffix: "JD", plant_code: "PK01", plant_name: "Plant B" },
    ],
    excluded_rule_states: ["To be Obsolete"],
    available_rule_states: ["App For Series", "In Work", "To be Obsolete"],
    ...overrides,
  };
}
