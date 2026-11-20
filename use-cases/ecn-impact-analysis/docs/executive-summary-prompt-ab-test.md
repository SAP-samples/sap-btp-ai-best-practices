# Executive-summary prompt A/B test

## Purpose

This opt-in acceptance test compares the current ECN executive-summary prompt with a
candidate description-first policy before changing production generation behavior.

The production path uses the description-first policy for the detailed Change
Summary only:

- `SummaryNarrativeStyle.DESCRIPTION_FIRST` is the default for new staged summaries;
- `PROMPT_VERSION` is `ecn-snapshot-v7`;
- the one-line `change_brief` keeps its existing independent prompt rules;
- existing stored summaries are immutable and change only when explicitly regenerated.

## Variants

| Variant | Policy |
| --- | --- |
| A — current | Identifiers can appear in the executive summary where the shared style requests them. |
| B — description-first | Uses supplied descriptions, change type, and counts in the detailed four-bullet Change Summary; does not list material, assembly, or VC-group identifiers there. The one-line `change_brief` is not part of this policy. |

Both variants receive the same bounded evidence projection. The comparison is
therefore about wording policy, not different source data.

## Data boundary and cost

`tests/integration/test_summary_prompt_ab_live.py` constructs a synthetic hydraulic
line replacement scenario in process. It does **not** read uploaded ECNs, workbooks,
HANA data, or customer records.

Running it makes exactly **two billable SAP Generative AI Hub SUMMARY-stage calls**.
It uses the existing Gen AI Hub adapter and reads credentials only from the standard
`api/.env` / environment variables. It prints only the generated summaries and their
token usage; it never writes HANA data or changes an analysis job.

## Run

Configure the usual AI Core environment variables without committing them:

```bash
AICORE_AUTH_URL=""
AICORE_CLIENT_ID=""
AICORE_CLIENT_SECRET=""
AICORE_BASE_URL=""
AICORE_RESOURCE_GROUP=""
LLM_MODEL="gpt-5.4"
```

Then explicitly accept the two calls:

```bash
RUN_SUMMARY_AB_TEST=1 PYTHONPATH=api python -m pytest -q -s -m llm_live \
  tests/integration/test_summary_prompt_ab_live.py
```

Without `RUN_SUMMARY_AB_TEST=1`, pytest skips the test and makes no provider call.

## Read-only HANA acceptance

`tests/integration/test_summary_prompt_ab_real_live.py` is the corresponding
real-data acceptance test for persisted ECN346863. It reads the latest HANA
occurrence and its immutable evidence bundle plus pinned BOM, rule, stock,
purchase-info, and material-master datasets. A read-only `SnapshotService`
subclass suppresses the normal plant-label cache update.

It makes four shared context-stage calls and two SUMMARY calls:

```bash
RUN_REAL_SUMMARY_AB_TEST=1 PYTHONPATH=api python -m pytest -q -s -m llm_live \
  tests/integration/test_summary_prompt_ab_real_live.py
```

This test must be explicitly approved because it sends the bounded prompt
projection derived from real source data to SAP Generative AI Hub. It creates
no HANA rows, analyses, attempts, provider-call records, or prompt updates.

## Acceptance criteria

Variant B must:

1. pass the existing structured `SummaryStageResult` schema and fixed executive-summary shape;
2. omit the fixture's material, assembly, and VC-group identifiers from `sections.executive_summary.assessment_markdown`;
3. use a supplied meaningful description such as the hydraulic/load-sensing context;
4. retain grounded scope and timing (the Plant A/PL10 location and the manufacturing cut-over date);
5. make no unsupported replacement or functional claim.

The A/B contract remains as a regression check: the explicit identifier-led
variant is compared with the production description-first variant over the same
input. Existing stored summaries remain unchanged until a user explicitly
regenerates them.
