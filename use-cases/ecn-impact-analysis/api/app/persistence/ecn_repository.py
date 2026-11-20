"""Stable historical ECN occurrence reads with evidence isolation.

This module implements ``EcnRepository``, which provides read-only access to
historical ECN occurrences stored in HANA.  All queries are assembled from
fixed SQL fragments; user-supplied filter values are always bound parameters.
The only text interpolated into SQL is a table identifier validated by
``quote_identifier``.

Design decisions
----------------
* Sort expressions are drawn exclusively from a closed ``SORT_EXPRESSIONS`` dict.
  ``_build_order`` indexes this dict with the validated ``query.sort`` value so
  the raw string is never interpolated into SQL.
* Stable default tie-breakers (``PUBLICATION_ORDER DESC, SOURCE_ORDINAL ASC,
  ECN_ROW_ID ASC``) ensure pagination is deterministic even when two ECNs share
  the same primary sort key.
* The derived ``COMPLETED_WITH_GAPS`` state is a CASE expression on the joined
  analysis table.  The job table stores only ``JobState.COMPLETED``; the API
  projection is computed at read time and never persisted.
* Evidence isolation: ``get_evidence`` SELECTs only the ``EVIDENCE_JSON`` for
  the exact requested ``ECN_ROW_ID``, decodes the bundle in Python, then checks
  membership.  An evidence ID that belongs to another ECN is never returned,
  regardless of whether it happens to be a valid UUID.
* Short transactions: every public method opens a fresh ``engine.connect()``
  context and closes it before returning.  No ``Connection`` is retained as
  instance state.

Usage::

    from app.persistence.ecn_repository import EcnRepository
    from app.persistence.hana import create_hana_engine

    engine = create_hana_engine()
    repo = EcnRepository(engine)
    response = repo.list_ecns(EcnListQuery(page=1, page_size=25))
"""

from __future__ import annotations

import math
from collections.abc import Mapping

from sqlalchemy import text
from sqlalchemy.engine import Engine

from app.ecn.analysis_models import AnalysisCompleteness
from app.ecn.api_models import (
    AnalysisFilterGroup,
    AnalysisFilterState,
    AnalysisJobView,
    EcnListQuery,
    EcnListResponse,
    ImpactCounts,
    LinkedEcnReference,
    StoredAnalysisView,
)
from app.ecn.models import (
    AnalysisSectionKey,
    ArtifactKind,
    EvidenceBundle,
    EvidenceItem,
    JobState,
    canonical_uuid,
)
from app.ecn.linked_ecns import extract_linked_ecn_numbers, normalize_ecn_number
from app.persistence.codec import (
    analysis_response_from_json,
    ecn_list_item_from_row,
    evidence_bundle_from_json,
    normalize_row_keys,
    read_lob_text,
)
from app.persistence.hana import from_hana_timestamp
from app.persistence.records import EcnDetailRecord
from app.persistence.records import ArtifactMetadataRecord, StoredEcnRecord
from app.persistence.table_contracts import PersistenceTableNames, quote_identifier


# ---------------------------------------------------------------------------
# Allowlisted sort expressions
# ---------------------------------------------------------------------------

# Scalar expression that reads the source-system ResolutionDate (the date an ECN
# was released) out of the already-stored raw ECN object. The ECN alias is always
# ``E`` in the list and detail queries, so this fragment is reused verbatim for the
# projected column and the sort key.
#
# ponytail: ResolutionDate is pulled from the existing SOURCE_JSON NCLOB instead of a
# dedicated column, so this needs no schema-contract change, no ALTER migration, and
# no backfill — it works immediately for rows imported before this feature. Ceiling:
# JSON_VALUE parses the NCLOB per row when sorting (fine for hundreds/low-thousands of
# ECNs) and orders the ISO-8601 string lexically (chronological while the source tz
# offset stays consistent). Upgrade path if the table grows large or offsets vary:
# promote ResolutionDate to an indexed UTC TIMESTAMP column via the additive-ALTER
# mechanism already in schema.py.
RESOLUTION_DATE_EXPR = "JSON_VALUE(E.\"SOURCE_JSON\", '$.ResolutionDate')"

# Bounded source facts for the detail overview. These expressions deliberately
# read individual scalar values rather than hydrating the raw source document.
SOURCE_STATUS_EXPR = (
    "COALESCE(JSON_VALUE(E.\"SOURCE_JSON\", '$.State.Display'), "
    "JSON_VALUE(E.\"SOURCE_JSON\", '$.State.Value'), "
    "JSON_VALUE(E.\"SOURCE_JSON\", '$.State'))"
)
REQUESTOR_NAME_EXPR = (
    "COALESCE(JSON_VALUE(E.\"SOURCE_JSON\", '$.REQUESTER_NAME.Display'), "
    "JSON_VALUE(E.\"SOURCE_JSON\", '$.REQUESTER_NAME.Value'), "
    "JSON_VALUE(E.\"SOURCE_JSON\", '$.REQUESTER_NAME.Name'), "
    "JSON_VALUE(E.\"SOURCE_JSON\", '$.REQUESTER_NAME'))"
)
AFFECTED_PRODUCTS_EXPR = "JSON_VALUE(E.\"SOURCE_JSON\", '$.AFFECTED_PRODUCTS')"

# This dict is the ONLY source of sort expressions. _build_order indexes it
# with query.sort (already validated by Pydantic as a Literal) so the raw
# user-supplied value is never interpolated directly into SQL.
SORT_EXPRESSIONS: dict[str, str] = {
    "uploaded_at": 'U."CREATED_AT"',
    "resolution_date": RESOLUTION_DATE_EXPR,
    # JSON text such as '["PL10 Plant A"]' orders by the first plant label;
    # rows not yet assessed (NULL) always sort last.
    "affected_plants": 'E."AFFECTED_PLANTS_JSON"',
    "ecn_number": 'E."ECN_NUMBER"',
    "description": 'E."DESCRIPTION_SUMMARY"',
    "affected_item_count": 'E."AFFECTED_ITEM_COUNT"',
    "ebom_match_count": 'E."EBOM_MATCH_COUNT"',
    "mbom_match_count": 'E."MBOM_MATCH_COUNT"',
    "warning_count": 'E."WARNING_COUNT"',
    "analysis_state": "DERIVED_ANALYSIS_STATE",
}

# Sort keys whose SQL expressions may return NULL values require NULLS LAST.
# resolution_date is nullable because some ECNs carry no ResolutionDate in source.
_NULLS_LAST_SORT_KEYS: frozenset[str] = frozenset(
    {"resolution_date", "affected_plants", "ecn_number", "description", "analysis_state"}
)

# Stable tie-breaker appended after every primary sort to make pagination
# deterministic when the primary key is not unique across the result set.
_TIE_BREAKER = 'U."PUBLICATION_ORDER" DESC, E."SOURCE_ORDINAL" ASC, E."ECN_ROW_ID" ASC'

# Derived CASE expression for the AnalysisFilterState projection.
# A completed job with no analysis row yet shows as COMPLETED (not a gap).
_DERIVED_STATE_CASE = (
    'CASE '
    "WHEN J.\"STATE\" = 'COMPLETED' AND A.\"COMPLETENESS_STATE\" = 'COMPLETED_WITH_GAPS' "
    "THEN 'COMPLETED_WITH_GAPS' "
    "WHEN J.\"STATE\" = 'COMPLETED' "
    "THEN 'COMPLETED' "
    "ELSE J.\"STATE\" "
    'END AS "DERIVED_ANALYSIS_STATE"'
)


# ---------------------------------------------------------------------------
# SQL fragment builders
# ---------------------------------------------------------------------------


def _build_filters(query: EcnListQuery) -> tuple[str, dict[str, object]]:
    """Construct WHERE clause fragments and bound parameters from a list query.

    Every user-supplied value becomes a bound parameter; no value is
    interpolated into the SQL text.  The WHERE keyword is omitted when the
    query has no active filters so that the string can be embedded in both
    the count and list SQL without extra manipulation.

    Args:
        query: A validated ``EcnListQuery`` carrying optional filter fields.

    Returns:
        A two-tuple of ``(where_sql, parameters)`` where ``where_sql`` is
        either an empty string (no filters) or ``"WHERE <conditions>"`` and
        ``parameters`` is a dict of bound values.
    """
    conditions: list[str] = []
    parameters: dict[str, object] = {}

    # Free-text search: case-insensitive LIKE over three ECN columns.
    if query.query is not None:
        conditions.append(
            '(UPPER(E."ECN_NUMBER") LIKE :query_pattern '
            'OR UPPER(E."TITLE") LIKE :query_pattern '
            'OR UPPER(E."DESCRIPTION_SUMMARY") LIKE :query_pattern)'
        )
        parameters["query_pattern"] = f"%{query.query.upper()}%"

    # Exact ECN number: equality, no case-folding, trimmed by Pydantic.
    if query.ecn_number is not None:
        conditions.append('E."ECN_NUMBER" = :ecn_number')
        parameters["ecn_number"] = query.ecn_number

    # Group filters share the same membership for list, count and bulk selections.
    if query.analysis_state is not None:
        state = query.analysis_state
        if state is AnalysisFilterGroup.SUMMARIES_AVAILABLE:
            # Match the two available-summary counters, requiring stored results.
            conditions.append(
                'J."STATE" = :completed_job_state AND '
                'A."COMPLETENESS_STATE" IN (:completed_analysis_state, :gap_state)'
            )
            parameters["completed_job_state"] = JobState.COMPLETED.value
            parameters["completed_analysis_state"] = AnalysisCompleteness.COMPLETED.value
            parameters["gap_state"] = AnalysisCompleteness.COMPLETED_WITH_GAPS.value
        elif state is AnalysisFilterGroup.IN_PROGRESS:
            conditions.append('J."STATE" IN (:queued_job_state, :running_job_state)')
            parameters["queued_job_state"] = JobState.QUEUED.value
            parameters["running_job_state"] = JobState.RUNNING.value
        elif state is AnalysisFilterState.COMPLETED_WITH_GAPS:
            # Must match both the job terminal state AND the gap completeness.
            conditions.append(
                'J."STATE" = :completed_job_state AND '
                'A."COMPLETENESS_STATE" = :gap_state'
            )
            parameters["completed_job_state"] = JobState.COMPLETED.value
            parameters["gap_state"] = AnalysisCompleteness.COMPLETED_WITH_GAPS.value
        elif state is AnalysisFilterState.COMPLETED:
            # Completed without gaps: job COMPLETED and completeness is either
            # COMPLETED or absent (no analysis row yet always shows as COMPLETED).
            conditions.append(
                'J."STATE" = :completed_job_state AND '
                '(A."COMPLETENESS_STATE" IS NULL OR '
                'A."COMPLETENESS_STATE" = :completed_analysis_state)'
            )
            parameters["completed_job_state"] = JobState.COMPLETED.value
            parameters["completed_analysis_state"] = AnalysisCompleteness.COMPLETED.value
        else:
            # QUEUED, RUNNING, FAILED — map directly to the job STATE column.
            conditions.append('J."STATE" = :analysis_state_value')
            parameters["analysis_state_value"] = state.value

    # Warning count filter.
    if query.has_warnings is True:
        conditions.append('E."WARNING_COUNT" > 0')
    elif query.has_warnings is False:
        conditions.append('E."WARNING_COUNT" = 0')

    # Plant: case-insensitive substring over the stored plant-label JSON array.
    # The array holds only labels (no keys), so the pattern cannot match JSON
    # field names. NULL (not yet assessed) never matches, by design.
    if query.plant is not None:
        conditions.append('UPPER(E."AFFECTED_PLANTS_JSON") LIKE :plant_pattern')
        parameters["plant_pattern"] = f"%{query.plant.upper()}%"

    if not conditions:
        return "", parameters
    return "WHERE " + " AND ".join(conditions), parameters


def _build_order(query: EcnListQuery) -> str:
    """Build a deterministic ORDER BY clause from the allowlisted sort map.

    Only ``SORT_EXPRESSIONS`` values are ever placed in the ORDER BY; the raw
    ``query.sort`` string is used exclusively as a dict key, never interpolated
    into the SQL text.

    Nullable sort keys (``ecn_number``, ``description``, ``analysis_state``)
    append ``NULLS LAST`` so that rows without a value sort consistently to the
    end of the result set regardless of direction.

    Args:
        query: A validated ``EcnListQuery`` with ``sort`` and ``order`` fields.

    Returns:
        A complete ``ORDER BY ...`` clause string including tie-breakers.
    """
    # Index the CLOSED map — never interpolate query.sort directly.
    expr = SORT_EXPRESSIONS[query.sort]
    direction = query.order.upper()
    nulls_clause = " NULLS LAST" if query.sort in _NULLS_LAST_SORT_KEYS else ""
    primary = f"{expr} {direction}{nulls_clause}"
    return f"ORDER BY {primary}, {_TIE_BREAKER}"


def _count_sql(tables: PersistenceTableNames, where_sql: str) -> str:
    """Return the parameterized COUNT(*) query for the given filter.

    Joins ECN, UPLOAD, JOB, and LEFT JOIN ANALYSIS so that the WHERE clause
    can reference any of those tables.  The SOURCE_ARTIFACT join is omitted
    because no filter condition references it.

    Args:
        tables: Physical table name set for interpolation via ``quote_identifier``.
        where_sql: Pre-built WHERE clause (may be empty string).

    Returns:
        A SQL string suitable for use with SQLAlchemy ``text()``.
    """
    e = quote_identifier(tables.ecn)
    j = quote_identifier(tables.analysis_job)
    a = quote_identifier(tables.analysis)
    return (
        f"SELECT COUNT(*) "
        f"FROM {e} E "
        f'JOIN {j} J ON J."ECN_ROW_ID" = E."ECN_ROW_ID" '
        f'LEFT JOIN {a} A ON A."ECN_ROW_ID" = E."ECN_ROW_ID" '
        f"{where_sql}"
    ).strip()


def _matching_ids_sql(tables: PersistenceTableNames, where_sql: str) -> str:
    """Return all ECN identities matching the same predicates as the list endpoint.

    Args:
        tables: Physical table name set for validated identifier interpolation.
        where_sql: Pre-built WHERE clause shared with list and count reads.

    Returns:
        A parameterized SQL query with no pagination or presentation sorting.
    """
    e = quote_identifier(tables.ecn)
    j = quote_identifier(tables.analysis_job)
    a = quote_identifier(tables.analysis)
    return (
        f'SELECT E."ECN_ROW_ID" FROM {e} E '
        f'JOIN {j} J ON J."ECN_ROW_ID" = E."ECN_ROW_ID" '
        f'LEFT JOIN {a} A ON A."ECN_ROW_ID" = E."ECN_ROW_ID" '
        f'{where_sql} '
        'ORDER BY E."ECN_ROW_ID"'
    ).strip()


def _list_sql(
    tables: PersistenceTableNames, where_sql: str, order_sql: str
) -> str:
    """Return the paginated SELECT query for the ECN list endpoint.

    Joins ECN, UPLOAD, SOURCE_ARTIFACT (for file name), JOB, and LEFT JOIN
    ANALYSIS, plus the newest REVIEW outcome per ECN (ranked derived table).  The DERIVED_ANALYSIS_STATE CASE expression maps the job state
    and analysis completeness into the five-value ``AnalysisFilterState``
    projection.  Pagination uses bound ``LIMIT :limit OFFSET :offset``.

    Args:
        tables: Physical table name set.
        where_sql: Pre-built WHERE clause (may be empty string).
        order_sql: Pre-built ORDER BY clause.

    Returns:
        A SQL string suitable for use with SQLAlchemy ``text()``.
    """
    e = quote_identifier(tables.ecn)
    u = quote_identifier(tables.upload)
    sa = quote_identifier(tables.source_artifact)
    j = quote_identifier(tables.analysis_job)
    a = quote_identifier(tables.analysis)
    stage = quote_identifier(tables.analysis_stage)
    attempt = quote_identifier(tables.llm_attempt)
    review = quote_identifier(tables.review)
    return (
        f'SELECT '
        f'E."ECN_ROW_ID", E."ECN_NUMBER", E."TITLE", E."DESCRIPTION_SUMMARY", '
        f'E."SOURCE_ORDINAL", E."AFFECTED_ITEM_COUNT", E."RAW_ECN_MATCH_COUNT", '
        f'E."EBOM_MATCH_COUNT", E."EBOM_CONTEXT_COUNT", E."MBOM_MATCH_COUNT", '
        f'E."WARNING_COUNT", '
        f'{RESOLUTION_DATE_EXPR} AS "RESOLUTION_DATE", '
        f'U."CREATED_AT" AS "UPLOADED_AT", '
        f'SA."FILE_NAME" AS "SOURCE_FILE_NAME", '
        f'{_DERIVED_STATE_CASE}, '
        f'CASE WHEN J."STATE" = \'FAILED\' AND EXISTS ('
        f'SELECT 1 FROM {stage} S '
        f'JOIN {attempt} LA ON LA."ATTEMPT_ID" = S."ATTEMPT_ID" '
        f'LEFT JOIN {attempt} BA ON BA."ATTEMPT_ID" = A."SUCCESSFUL_ATTEMPT_ID" '
        'WHERE S."JOB_ID" = J."JOB_ID" AND S."EVIDENCE_HASH" = E."EVIDENCE_HASH" '
        'AND S."PROMPT_VERSION" = J."PROMPT_VERSION" '
        'AND S."OUTPUT_SCHEMA_VERSION" = J."OUTPUT_SCHEMA_VERSION" '
        'AND LA."ATTEMPT_NUMBER" > COALESCE(BA."ATTEMPT_NUMBER", 0)'
        ') THEN 1 ELSE 0 END AS "HAS_PARTIAL_ANALYSIS", '
        f'A."COMPLETENESS_STATE", '
        f'E."AFFECTED_PLANTS_JSON", '
        f'RV."OUTCOME" AS "REVIEW_OUTCOME", '
        f'A."CONTENT_JSON" AS "ANALYSIS_CONTENT_JSON" '
        f'FROM {e} E '
        f'JOIN {u} U ON U."UPLOAD_ID" = E."UPLOAD_ID" '
        f'JOIN {sa} SA ON SA."ARTIFACT_ID" = U."ECN_ARTIFACT_ID" '
        f'JOIN {j} J ON J."ECN_ROW_ID" = E."ECN_ROW_ID" '
        f'LEFT JOIN {a} A ON A."ECN_ROW_ID" = E."ECN_ROW_ID" '
        # Newest engineering review decision of this exact assessment (NULL = not reviewed).
        # HANA rejects ORDER BY in correlated subqueries, so rank reviews per ECN instead.
        f'LEFT JOIN (SELECT R."ECN_ROW_ID", R."OUTCOME", ROW_NUMBER() OVER ('
        'PARTITION BY R."ECN_ROW_ID" ORDER BY R."CREATED_AT" DESC, R."REVIEW_ID" DESC) AS "REVIEW_RANK" '
        f'FROM {review} R) RV ON RV."ECN_ROW_ID" = E."ECN_ROW_ID" AND RV."REVIEW_RANK" = 1 '
        f'{where_sql} '
        f'{order_sql} '
        f'LIMIT :limit OFFSET :offset'
    ).strip()


def _evidence_bundle_sql(ecn_table: str) -> str:
    """Return the SELECT for the evidence JSON of one specific ECN row.

    Retrieves only the ``EVIDENCE_JSON`` column for the requested ECN row.
    The evidence_id lookup value is intentionally excluded from the SQL; it
    is matched in Python after the bundle is decoded.

    Args:
        ecn_table: Physical ECN table name (validated by ``quote_identifier``
            at construction time).

    Returns:
        A SQL string suitable for use with SQLAlchemy ``text()``.
    """
    t = quote_identifier(ecn_table)
    return f'SELECT E."EVIDENCE_JSON" FROM {t} E WHERE E."ECN_ROW_ID" = :ecn_row_id'


def _queued_jobs_sql(tables: PersistenceTableNames) -> str:
    """Return the SELECT for queued job IDs under one upload.

    Joins JOB to ECN so that filtering and ordering by source ordinal is
    possible without a subquery.  Results are ordered by ``SOURCE_ORDINAL ASC,
    JOB_ID ASC`` for a deterministic canonical sequence.

    Args:
        tables: Physical table name set.

    Returns:
        A SQL string suitable for use with SQLAlchemy ``text()``.
    """
    e = quote_identifier(tables.ecn)
    j = quote_identifier(tables.analysis_job)
    return (
        f'SELECT J."JOB_ID" '
        f'FROM {j} J '
        f'JOIN {e} E ON E."ECN_ROW_ID" = J."ECN_ROW_ID" '
        f'WHERE E."UPLOAD_ID" = :upload_id AND J."STATE" = :queued_state '
        f'ORDER BY E."SOURCE_ORDINAL" ASC, J."JOB_ID" ASC'
    )


def _detail_sql(tables: PersistenceTableNames) -> str:
    """Return the one-shot joined SELECT for the ECN detail page.

    Executes a single query that joins:
    - ECN table (all columns, no prefix)
    - UPLOAD table (U_ prefix to avoid clashes with ECN columns)
    - SOURCE_ARTIFACT for ECN JSON (ecn_artifact. prefix)
    - SOURCE_ARTIFACT for BOM workbook (bom_artifact. prefix)
    - SOURCE_ARTIFACT for rule workbook (rule_artifact. prefix, LEFT JOIN)
    - ANALYSIS_JOB (J_ prefix)
    - ANALYSIS (A_ prefix, LEFT JOIN)

    The three artifact blocks use the prefix convention supported by
    the metadata-only artifact decoder.

    Args:
        tables: Physical table name set.

    Returns:
        A SQL string suitable for use with SQLAlchemy ``text()``.
    """
    e = quote_identifier(tables.ecn)
    u = quote_identifier(tables.upload)
    sa = quote_identifier(tables.source_artifact)
    j = quote_identifier(tables.analysis_job)
    a = quote_identifier(tables.analysis)

    artifact_cols = '"ARTIFACT_ID", "ARTIFACT_KIND", "FILE_NAME", "SHA256"'

    def _prefix_artifact_cols(alias: str, prefix: str) -> str:
        """Return comma-separated alias expressions for all artifact columns."""
        return ", ".join(
            f'{alias}."{c.strip().strip(chr(34))}" AS "{prefix}{c.strip().strip(chr(34))}"'
            for c in artifact_cols.split(",")
        )

    ea_cols = _prefix_artifact_cols("EA", "ecn_artifact.")
    ba_cols = _prefix_artifact_cols("BA", "bom_artifact.")
    ra_cols = _prefix_artifact_cols("RA", "rule_artifact.")

    return (
        # ECN display columns — no raw source or evidence payloads.
        'SELECT '
        'E."ECN_ROW_ID", E."UPLOAD_ID", E."SOURCE_ORDINAL", E."ECN_NUMBER", E."TITLE", '
        'E."DESCRIPTION_SUMMARY", E."DESCRIPTION_TEXT", E."AFFECTED_ITEM_COUNT", '
        'E."RAW_ECN_MATCH_COUNT", E."EBOM_MATCH_COUNT", E."EBOM_CONTEXT_COUNT", '
        'E."MBOM_MATCH_COUNT", E."WARNING_COUNT", E."EVIDENCE_HASH", '
        'E."PARSER_VERSION", E."DETERMINISTIC_LOGIC_VERSION", '
        'E."RULE_EVALUATOR_VERSION", E."CREATED_AT", '
        # Source facts are bounded scalar reads from the stored raw ECN object.
        f'{RESOLUTION_DATE_EXPR} AS "RESOLUTION_DATE", '
        f'{SOURCE_STATUS_EXPR} AS "SOURCE_STATUS", '
        f'{REQUESTOR_NAME_EXPR} AS "REQUESTOR_NAME", '
        f'{AFFECTED_PRODUCTS_EXPR} AS "AFFECTED_PRODUCTS", '
        # Upload columns — U_ prefix to disambiguate from ECN columns.
        'U."UPLOAD_ID" AS "U_UPLOAD_ID", '
        'U."REQUEST_ID" AS "U_REQUEST_ID", '
        'U."PUBLICATION_ORDER" AS "U_PUBLICATION_ORDER", '
        'U."ECN_ARTIFACT_ID" AS "U_ECN_ARTIFACT_ID", '
        'U."BOM_ARTIFACT_ID" AS "U_BOM_ARTIFACT_ID", '
        'U."RULE_ARTIFACT_ID" AS "U_RULE_ARTIFACT_ID", '
        'U."BOM_REUSED" AS "U_BOM_REUSED", '
        'U."IMPORTED_ECN_COUNT" AS "U_IMPORTED_ECN_COUNT", '
        'U."CREATED_AT" AS "U_CREATED_AT", '
        # Artifact metadata blocks — no bytes or normalized content.
        f'{ea_cols}, '
        f'{ba_cols}, '
        f'{ra_cols}, '
        # Job columns — J_ prefix.
        'J."JOB_ID" AS "J_JOB_ID", '
        'J."STATE" AS "J_STATE", '
        'J."ATTEMPT_COUNT" AS "J_ATTEMPT_COUNT", '
        'J."ERROR_CODE" AS "J_ERROR_CODE", '
        'J."ERROR_MESSAGE" AS "J_ERROR_MESSAGE", '
        # Analysis columns — A_ prefix, nullable (LEFT JOIN).
        'A."ANALYSIS_ID" AS "A_ANALYSIS_ID", '
        'A."COMPLETENESS_STATE" AS "A_COMPLETENESS_STATE", '
        'A."CONTENT_JSON" AS "A_CONTENT_JSON", '
        'A."EVIDENCE_HASH" AS "A_EVIDENCE_HASH", '
        'A."PROMPT_VERSION" AS "A_PROMPT_VERSION", '
        'A."MODEL_NAME" AS "A_MODEL_NAME", '
        'A."MODEL_DEPLOYMENT" AS "A_MODEL_DEPLOYMENT", '
        'A."OUTPUT_SCHEMA_VERSION" AS "A_OUTPUT_SCHEMA_VERSION", '
        'A."CREATED_AT" AS "A_CREATED_AT" '
        f'FROM {e} E '
        f'JOIN {u} U ON U."UPLOAD_ID" = E."UPLOAD_ID" '
        f'JOIN {sa} EA ON EA."ARTIFACT_ID" = U."ECN_ARTIFACT_ID" '
        f'JOIN {sa} BA ON BA."ARTIFACT_ID" = U."BOM_ARTIFACT_ID" '
        f'LEFT JOIN {sa} RA ON RA."ARTIFACT_ID" = U."RULE_ARTIFACT_ID" '
        f'JOIN {j} J ON J."ECN_ROW_ID" = E."ECN_ROW_ID" '
        f'LEFT JOIN {a} A ON A."ECN_ROW_ID" = E."ECN_ROW_ID" '
        'WHERE E."ECN_ROW_ID" = :ecn_row_id'
    )


# ---------------------------------------------------------------------------
# Private row decoders for the detail joined query
# ---------------------------------------------------------------------------


def _upload_from_detail_row(row: Mapping[str, object]) -> object:
    """Reconstruct an ``UploadRecord`` from prefixed (U_) detail query columns.

    The upload columns in the detail JOIN are aliased with a ``U_`` prefix to
    avoid name collisions with ECN columns that share the same names
    (``UPLOAD_ID``, ``CREATED_AT``, ``PARSER_VERSION``, etc.).

    The row is normalized to uppercase keys at entry so that dialect casing
    differences are invisible to the rest of this function.

    Args:
        row: The full detail query result row.

    Returns:
        A frozen ``UploadRecord`` instance.
    """
    from app.persistence.records import UploadRecord

    row = normalize_row_keys(row)
    return UploadRecord(
        upload_id=str(row["U_UPLOAD_ID"]),
        request_id=str(row["U_REQUEST_ID"]),
        publication_order=int(row["U_PUBLICATION_ORDER"]),  # type: ignore[arg-type]
        ecn_artifact_id=str(row["U_ECN_ARTIFACT_ID"]),
        bom_artifact_id=str(row["U_BOM_ARTIFACT_ID"]),
        rule_artifact_id=(
            str(row["U_RULE_ARTIFACT_ID"])
            if row["U_RULE_ARTIFACT_ID"] is not None
            else None
        ),
        bom_reused=bool(row["U_BOM_REUSED"]),
        imported_ecn_count=int(row["U_IMPORTED_ECN_COUNT"]),  # type: ignore[arg-type]
        created_at=from_hana_timestamp(row["U_CREATED_AT"]),  # type: ignore[arg-type]
    )


def _artifact_metadata_from_detail_row(
    row: Mapping[str, object],
    prefix: str,
) -> ArtifactMetadataRecord:
    """Decode safe artifact metadata from a prefixed detail-query block.

    Args:
        row: Joined detail row with normalized or dialect-specific key casing.
        prefix: Artifact alias prefix such as ``ecn_artifact.``.

    Returns:
        Metadata-only artifact record with no content bytes or normalized text.
    """
    normalized = normalize_row_keys(row)
    key = prefix.upper()
    return ArtifactMetadataRecord(
        artifact_id=str(normalized[f"{key}ARTIFACT_ID"]),
        artifact_kind=ArtifactKind(str(normalized[f"{key}ARTIFACT_KIND"])),
        file_name=str(normalized[f"{key}FILE_NAME"]),
        sha256=str(normalized[f"{key}SHA256"]),
    )


def _stored_ecn_metadata_from_detail_row(
    row: Mapping[str, object],
) -> StoredEcnRecord:
    """Decode display metadata without reading raw JSON or the evidence NCLOB.

    Args:
        row: Joined detail-query result.

    Returns:
        Stored ECN record whose non-display source collections are empty because
        the initial detail request intentionally does not hydrate them.
    """
    normalized = normalize_row_keys(row)
    description_raw = normalized["DESCRIPTION_TEXT"]

    def source_text(column: str) -> str | None:
        value = normalized.get(column)
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    affected_products = tuple(
        product.strip()
        for product in (source_text("AFFECTED_PRODUCTS") or "").split(",")
        if product.strip()
    )
    return StoredEcnRecord(
        ecn_row_id=str(normalized["ECN_ROW_ID"]),
        upload_id=str(normalized["UPLOAD_ID"]),
        source_ordinal=int(normalized["SOURCE_ORDINAL"]),  # type: ignore[arg-type]
        ecn_number=(
            str(normalized["ECN_NUMBER"])
            if normalized["ECN_NUMBER"] is not None
            else None
        ),
        title=str(normalized["TITLE"]) if normalized["TITLE"] is not None else None,
        description_summary=(
            str(normalized["DESCRIPTION_SUMMARY"])
            if normalized["DESCRIPTION_SUMMARY"] is not None
            else None
        ),
        description_text=(
            read_lob_text(description_raw) if description_raw is not None else None
        ),
        counts=ImpactCounts(
            affected_item_count=int(normalized["AFFECTED_ITEM_COUNT"]),
            raw_ecn_match_count=int(normalized["RAW_ECN_MATCH_COUNT"]),
            ebom_match_count=int(normalized["EBOM_MATCH_COUNT"]),
            ebom_context_count=int(normalized["EBOM_CONTEXT_COUNT"]),
            mbom_match_count=int(normalized["MBOM_MATCH_COUNT"]),
            warning_count=int(normalized["WARNING_COUNT"]),
        ),
        source_json={},
        change_candidates=(),
        source_status=source_text("SOURCE_STATUS"),
        requestor_name=source_text("REQUESTOR_NAME"),
        affected_products=affected_products,
        resolution_date=(
            str(normalized["RESOLUTION_DATE"])
            if normalized.get("RESOLUTION_DATE") is not None
            else None
        ),
        evidence_hash=str(normalized["EVIDENCE_HASH"]),
        parser_version=str(normalized["PARSER_VERSION"]),
        deterministic_logic_version=str(normalized["DETERMINISTIC_LOGIC_VERSION"]),
        rule_evaluator_version=str(normalized["RULE_EVALUATOR_VERSION"]),
        created_at=from_hana_timestamp(normalized["CREATED_AT"]),  # type: ignore[arg-type]
    )


def _job_view_from_detail_row(row: Mapping[str, object]) -> AnalysisJobView:
    """Reconstruct an ``AnalysisJobView`` from prefixed (J_) detail query columns.

    The row is normalized to uppercase keys at entry so that dialect casing
    differences are invisible to the rest of this function.

    Args:
        row: The full detail query result row.

    Returns:
        A validated ``AnalysisJobView`` Pydantic model.
    """
    row = normalize_row_keys(row)
    return AnalysisJobView(
        job_id=str(row["J_JOB_ID"]),
        state=JobState(str(row["J_STATE"])),
        attempt_count=int(row["J_ATTEMPT_COUNT"]),  # type: ignore[arg-type]
        error_code=(
            str(row["J_ERROR_CODE"]) if row["J_ERROR_CODE"] is not None else None
        ),
        error_message=(
            str(row["J_ERROR_MESSAGE"]) if row["J_ERROR_MESSAGE"] is not None else None
        ),
    )


def _stored_analysis_from_detail_row(row: Mapping[str, object]) -> StoredAnalysisView | None:
    """Reconstruct an optional ``StoredAnalysisView`` from prefixed (A_) columns.

    Returns ``None`` when the LEFT JOIN produced no analysis row (i.e.,
    ``A_ANALYSIS_ID`` is ``NULL``).

    The row is normalized to uppercase keys at entry so that dialect casing
    differences are invisible to the rest of this function.

    Args:
        row: The full detail query result row.

    Returns:
        A validated ``StoredAnalysisView`` or ``None`` when absent.
    """
    row = normalize_row_keys(row)
    if row["A_ANALYSIS_ID"] is None:
        return None
    content_json = read_lob_text(row["A_CONTENT_JSON"])
    analysis_response = analysis_response_from_json(content_json)
    return StoredAnalysisView(
        analysis_id=str(row["A_ANALYSIS_ID"]),
        completeness=AnalysisCompleteness(str(row["A_COMPLETENESS_STATE"])),
        change_brief=analysis_response.change_brief,
        sections=analysis_response.sections,
        impact_factors=analysis_response.impact_factors,
        overall_impact=analysis_response.overall_impact,
        data_readiness_descriptions=analysis_response.data_readiness_descriptions,
        additional_next_steps=analysis_response.additional_next_steps,
        created_at=from_hana_timestamp(row["A_CREATED_AT"]),  # type: ignore[arg-type]
        evidence_hash=str(row["A_EVIDENCE_HASH"]),
        prompt_version=str(row["A_PROMPT_VERSION"]),
        output_schema_version=str(row["A_OUTPUT_SCHEMA_VERSION"]),
        model_name=str(row["A_MODEL_NAME"]),
        model_deployment=(
            str(row["A_MODEL_DEPLOYMENT"]) if row["A_MODEL_DEPLOYMENT"] is not None else None
        ),
    )


# ---------------------------------------------------------------------------
# Repository
# ---------------------------------------------------------------------------


class EcnRepository:
    """Read historical ECN occurrences with stable filters and evidence isolation.

    All public methods open a short-lived ``engine.connect()`` context per
    call.  No ``Connection`` is retained as instance state between calls.

    Args:
        engine: A SQLAlchemy ``Engine`` connected to the HANA instance.
        table_names: Optional ``PersistenceTableNames`` override; defaults to
            the production names.  Pass a suffixed instance for isolated live
            tests.

    Example::

        repo = EcnRepository(engine)
        response = repo.list_ecns(EcnListQuery(page=1, page_size=10))
        for item in response.items:
            print(item.display_ecn_number, item.analysis_state)
    """

    def __init__(
        self,
        engine: Engine,
        table_names: PersistenceTableNames = PersistenceTableNames(),
    ) -> None:
        """Bind the engine and table-name set.

        Args:
            engine: SQLAlchemy ``Engine``; never stored beyond this reference.
            table_names: Physical table identifiers already validated by
                ``PersistenceTableNames.__post_init__``.
        """
        self._engine = engine
        self._tables = table_names

    def list_ecns(self, query: EcnListQuery) -> EcnListResponse:
        """Return one stable historical page of immutable ECN occurrences.

        Executes a count query then a paginated list query, both sharing the
        identical WHERE clause and bound parameters so that ``total_items``
        exactly corresponds to the filtered result set.  ``total_pages`` is
        ``ceil(total_items / page_size)`` or ``0`` when ``total_items`` is
        ``0``.

        Args:
            query: Validated ``EcnListQuery`` carrying optional filter, sort,
                order, page, and page_size fields.

        Returns:
            An ``EcnListResponse`` with the requested page and correct totals.
        """
        where_sql, parameters = _build_filters(query)
        order_sql = _build_order(query)
        with self._engine.connect() as connection:
            total = int(
                connection.execute(
                    text(_count_sql(self._tables, where_sql)), parameters
                ).scalar_one()
            )
            rows = connection.execute(
                text(_list_sql(self._tables, where_sql, order_sql)),
                {
                    **parameters,
                    "limit": query.page_size,
                    "offset": (query.page - 1) * query.page_size,
                },
            ).mappings().all()
        items = [ecn_list_item_from_row(row) for row in rows]
        pages = 0 if total == 0 else math.ceil(total / query.page_size)
        return EcnListResponse(
            items=items,
            page=query.page,
            page_size=query.page_size,
            total_items=total,
            total_pages=pages,
        )

    def list_matching_ecn_ids(self, query: EcnListQuery) -> list[str]:
        """Resolve complete filter membership without pagination or sort effects.

        Args:
            query: Validated list query whose filter fields define membership.

        Returns:
            Matching ECN row IDs in deterministic database order.
        """
        where_sql, parameters = _build_filters(query)
        with self._engine.connect() as connection:
            rows = connection.execute(
                text(_matching_ids_sql(self._tables, where_sql)), parameters
            ).mappings().all()
        return [str(normalize_row_keys(row)["ECN_ROW_ID"]) for row in rows]

    def get_linked_ecns(self, record: EcnDetailRecord) -> list[LinkedEcnReference]:
        """Resolve outgoing raw-description references against all stored ECNs.

        Read only the target identities, with no evidence, snapshot, or provider
        work. One query covers every reference and avoids inbox pagination.
        Repeated imports resolve to the same upload first, then newest upload,
        with stable occurrence tie-breakers. Missing targets remain explicit.
        """
        numbers = extract_linked_ecn_numbers(
            record.ecn.description_text,
            record.ecn.description_summary,
            current_ecn_number=record.ecn.ecn_number,
        )
        if not numbers:
            return []
        # Narrow candidates by prefix/numeric suffix in SQL; the shared regex
        # below verifies the entire identity, including significant zeroes.
        # This keeps case/spacing/punctuation semantics in one Python parser.
        parameters: dict[str, object] = {
            "current_row_id": record.ecn.ecn_row_id,
            "current_upload_id": record.upload.upload_id,
            **{f"linked_{i}": f"ECN%{number[3:]}" for i, number in enumerate(numbers)},
        }
        filters = " OR ".join(
            f'UPPER(TRIM(E."ECN_NUMBER")) LIKE :linked_{i}' for i in range(len(numbers))
        )
        sql = (
            'SELECT E."ECN_ROW_ID", E."ECN_NUMBER" '
            f"FROM {quote_identifier(self._tables.ecn)} E "
            f"JOIN {quote_identifier(self._tables.upload)} U "
            'ON E."UPLOAD_ID" = U."UPLOAD_ID" '
            f'WHERE ({filters}) '
            'AND E."ECN_ROW_ID" <> :current_row_id '
            'ORDER BY CASE WHEN E."UPLOAD_ID" = :current_upload_id THEN 0 ELSE 1 END, '
            + _TIE_BREAKER
        )
        with self._engine.connect() as connection:
            rows = connection.execute(text(sql), parameters).mappings().all()
        targets: dict[str, str] = {}
        for row in rows:
            row = normalize_row_keys(row)
            number = normalize_ecn_number(str(row["ECN_NUMBER"]))
            if number in numbers:
                targets.setdefault(number, str(row["ECN_ROW_ID"]))
        return [
            LinkedEcnReference(ecn_number=number, ecn_row_id=targets.get(number))
            for number in numbers
        ]

    def get_ecn_detail(self, ecn_row_id: str) -> EcnDetailRecord | None:
        """Return the full provenance and evidence record for one ECN occurrence.

        Executes a single bounded JOIN query that fetches the ECN row, its
        upload, all three source artifacts (ECN JSON, BOM, optional rule), the
        current analysis job state, and the optional stored analysis result.
        Evidence is decoded from the ECN row's ``EVIDENCE_JSON`` column.

        Returns ``None`` when no row matches ``ecn_row_id``.

        Args:
            ecn_row_id: Canonical UUID string identifying the ECN occurrence.

        Returns:
            A fully populated ``EcnDetailRecord`` or ``None`` when not found.

        Raises:
            ValueError: When ``ecn_row_id`` is not a canonical UUID string.
        """
        canonical_uuid(ecn_row_id)
        with self._engine.connect() as connection:
            row = connection.execute(
                text(_detail_sql(self._tables)),
                {"ecn_row_id": ecn_row_id},
            ).mappings().one_or_none()
        if row is None:
            return None

        # Normalize once at the boundary; all sub-functions re-normalize (idempotent).
        row = normalize_row_keys(row)
        ecn = _stored_ecn_metadata_from_detail_row(row)
        upload = _upload_from_detail_row(row)
        ecn_artifact = _artifact_metadata_from_detail_row(row, "ecn_artifact.")
        bom_artifact = _artifact_metadata_from_detail_row(row, "bom_artifact.")
        # Rule artifact is only present when U.RULE_ARTIFACT_ID is not NULL.
        rule_artifact: ArtifactMetadataRecord | None = None
        if row["U_RULE_ARTIFACT_ID"] is not None:
            rule_artifact = _artifact_metadata_from_detail_row(row, "rule_artifact.")
        job = _job_view_from_detail_row(row)
        analysis = _stored_analysis_from_detail_row(row)
        return EcnDetailRecord(
            ecn=ecn,
            upload=upload,
            ecn_artifact=ecn_artifact,
            bom_artifact=bom_artifact,
            rule_artifact=rule_artifact,
            job=job,
            analysis=analysis,
        )

    def get_evidence(self, ecn_row_id: str, evidence_id: str) -> EvidenceItem | None:
        """Return an evidence item only when it belongs to the requested ECN bundle.

        Selects ONLY the ``EVIDENCE_JSON`` for the exact ``ecn_row_id`` and
        then searches the decoded bundle in Python.  The ``evidence_id`` value
        is never sent to HANA — it is matched entirely in application code
        after the bundle is decoded.  An ID that exists under another ECN's
        bundle will therefore always return ``None``.

        Args:
            ecn_row_id: Canonical UUID string of the ECN occurrence to scope to.
            evidence_id: Identifier of the evidence item to retrieve (≤200 chars).

        Returns:
            The matching ``EvidenceItem`` or ``None`` when absent or out of scope.

        Raises:
            ValueError: When ``ecn_row_id`` is not a canonical UUID string.
        """
        canonical_uuid(ecn_row_id)
        # Reject obviously invalid evidence IDs before any DB access.
        if not evidence_id or len(evidence_id) > 200:
            return None
        with self._engine.connect() as connection:
            row = connection.execute(
                text(_evidence_bundle_sql(self._tables.ecn)),
                {"ecn_row_id": ecn_row_id},
            ).mappings().one_or_none()
        if row is None:
            return None
        row = normalize_row_keys(row)
        bundle = evidence_bundle_from_json(read_lob_text(row["EVIDENCE_JSON"]))
        # Python-side membership check — evidence_id never reaches SQL.
        for section_key in AnalysisSectionKey:
            for item in getattr(bundle.sections, section_key.value):
                if item.evidence_id == evidence_id:
                    return item
        return None

    def get_evidence_bundle(self, ecn_row_id: str) -> EvidenceBundle | None:
        """Return the stored audit bundle only for an explicit evidence request.

        The ECN detail endpoint deliberately avoids this NCLOB.  This method is
        reserved for paginated evidence reads so loading the full stored bundle
        is isolated from initial page rendering.

        Args:
            ecn_row_id: Canonical UUID string of the owning ECN occurrence.

        Returns:
            Decoded immutable evidence bundle, or ``None`` when the ECN is absent.
        """
        canonical_uuid(ecn_row_id)
        with self._engine.connect() as connection:
            row = connection.execute(
                text(_evidence_bundle_sql(self._tables.ecn)),
                {"ecn_row_id": ecn_row_id},
            ).mappings().one_or_none()
        if row is None:
            return None
        normalized = normalize_row_keys(row)
        return evidence_bundle_from_json(read_lob_text(normalized["EVIDENCE_JSON"]))

    def list_queued_job_ids(self, upload_id: str) -> list[str]:
        """Return the canonical job IDs for all QUEUED jobs under one upload.

        Results are ordered by ``SOURCE_ORDINAL ASC, JOB_ID ASC`` so that
        the list is deterministic and stable across calls.  The upload must
        exist; when no queued jobs remain an empty list is returned.

        Args:
            upload_id: Canonical UUID string of the owning upload batch.

        Returns:
            A list of canonical job ID strings in source-ordinal order.
        """
        canonical_uuid(upload_id)
        with self._engine.connect() as connection:
            rows = connection.execute(
                text(_queued_jobs_sql(self._tables)),
                {
                    "upload_id": upload_id,
                    "queued_state": JobState.QUEUED.value,
                },
            ).mappings().all()
        return [str(normalize_row_keys(row)["JOB_ID"]) for row in rows]
