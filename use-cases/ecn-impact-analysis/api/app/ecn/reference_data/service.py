"""Validate and atomically publish Settings-managed reference datasets."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from threading import RLock
from uuid import NAMESPACE_URL, uuid5

from app.ecn.errors import SourceValidationError
from app.ecn.intake.workbook_parser import parse_bom_workbook
from app.ecn.models import (
    PROCUREMENT_TYPE_LABELS,
    ArtifactKind,
    ParseState,
    SourceArtifact,
)
from app.ecn.reference_data.models import (
    NormalizedReferenceRow,
    ReferenceDatasetPublication,
    ReferenceDatasetRecord,
    ReferenceDatasetType,
)
from app.ecn.reference_data.parsers import (
    parse_purchase_info_price_csv,
    parse_reference_workbook,
)
from app.ecn.rules.importer import normalize_rule_workbook
from app.ecn.rules.models import RuleArtifactMetadata
from app.ecn.services.lifecycle import HistoryMutationGuard
from app.ecn.settings import EcnSettings
from app.persistence.reference_dataset_repository import ReferenceDatasetRepository

PROCUREMENT_STATUS_LEGEND_VERSION = "sap-procurement-status-v1"
"""Version of the raw-code display legend exposed by Settings and evidence."""

PROCUREMENT_STATUS_LEGEND: dict[str, str] = dict(PROCUREMENT_TYPE_LABELS)
"""Display labels for known source codes; unknown codes remain unchanged.

Derived from the canonical ``PROCUREMENT_TYPE_LABELS`` in ``app.ecn.models`` so
the Settings legend published here and the evidence snapshot label cannot diverge
on the maintained E/F/X wording (code ``X`` reads "In-house and external
procurement allowed"). ``display_source_code`` still returns the raw code for any
value outside this map, so the legend contract (known codes only) is unchanged.
"""


@dataclass(frozen=True)
class ReferenceUploadRequest:
    """Hold one bounded Settings upload after transport validation.

    Args:
        dataset_type: Source contract selected by the operator.
        file_name: Sanitized original file name.
        media_type: Validated XLSX media type.
        content: Complete bounded workbook bytes.
        actor_id: Optional opaque actor identifier.
        context: Source metadata (model range, data group, environment,
            extraction date, coverage, comparison baseline). Plants are not
            declared at upload: they are derived from conditional rules.
    """

    dataset_type: ReferenceDatasetType
    file_name: str
    media_type: str
    content: bytes
    actor_id: str | None = None
    context: dict = field(default_factory=dict)


_ARTIFACT_KIND_BY_TYPE: dict[ReferenceDatasetType, ArtifactKind] = {
    ReferenceDatasetType.STOCK: ArtifactKind.STOCK_WORKBOOK,
    ReferenceDatasetType.BOM: ArtifactKind.BOM_WORKBOOK,
    ReferenceDatasetType.CONDITIONAL_RULES: ArtifactKind.CONDITIONAL_RULE_WORKBOOK,
    ReferenceDatasetType.MATERIAL_MASTER: ArtifactKind.MATERIAL_MASTER_WORKBOOK,
    ReferenceDatasetType.PURCHASE_INFO_RECORDS: ArtifactKind.PURCHASE_INFO_WORKBOOK,
    ReferenceDatasetType.PURCHASE_INFO_PRICES: ArtifactKind.PURCHASE_INFO_PRICE_CSV,
}


def artifact_kind_for_dataset(dataset_type: ReferenceDatasetType) -> ArtifactKind:
    """Return the upload metadata contract for one reference dataset type."""
    return _ARTIFACT_KIND_BY_TYPE[dataset_type]


def display_source_code(code: str | None) -> str | None:
    """Return a known business label while preserving unknown source codes."""
    if code is None:
        return None
    return PROCUREMENT_STATUS_LEGEND.get(code, code)


def _artifact(
    request: ReferenceUploadRequest,
    artifact_id: str,
    normalized_content: str,
    created_at: datetime,
) -> SourceArtifact:
    """Build an immutable valid source artifact after parsing succeeds."""
    return SourceArtifact(
        artifact_id=artifact_id,
        artifact_kind=_ARTIFACT_KIND_BY_TYPE[request.dataset_type],
        file_name=request.file_name,
        media_type=request.media_type,
        byte_size=len(request.content),
        sha256=hashlib.sha256(request.content).hexdigest(),
        content_bytes=request.content,
        normalized_content=normalized_content,
        parse_state=ParseState.VALID,
        parse_error_code=None,
        created_at=created_at,
    )


def _rule_rows(
    artifact_id: str, rules: tuple[object, ...]
) -> tuple[NormalizedReferenceRow, ...]:
    """Convert parsed conditional rules into source-specific HANA rows."""
    converted: list[NormalizedReferenceRow] = []
    for rule in rules:
        source_row_number = getattr(rule, "source_row_number")
        if not isinstance(source_row_number, int):
            continue
        row_uuid = str(
            uuid5(NAMESPACE_URL, f"ecn-reference-rule:{artifact_id}:{source_row_number}")
        )
        converted.append(
            NormalizedReferenceRow(
                row_id=row_uuid,
                artifact_id=artifact_id,
                sheet_name="Default",
                source_row_number=source_row_number,
                values={
                    "rule_number": getattr(rule, "rule_number"),
                    "parse_state": getattr(rule, "parse_state").value,
                    "lifecycle_state": getattr(rule, "lifecycle_state"),
                    "source_text": getattr(rule, "source_text"),
                },
            )
        )
    return tuple(converted)


class ReferenceDatasetService:
    """Coordinate pure validation and atomic HANA dataset publication."""

    def __init__(
        self,
        repository: ReferenceDatasetRepository,
        settings: EcnSettings,
        clock: Callable[[], datetime],
        uuid_factory: Callable[[], str],
        mutation_guard: HistoryMutationGuard | None = None,
    ) -> None:
        """Bind persistence, parser limits, identity sources, and mutation guard."""
        self._repository = repository
        self._settings = settings
        self._clock = clock
        self._uuid = uuid_factory
        self._guard = mutation_guard or HistoryMutationGuard()
        self._reference_lock = RLock()

    def delete(self, dataset_id: str) -> bool:
        """Delete one unused publication under the shared upload mutation guard.

        Args:
            dataset_id: UUID of the Settings publication to remove.

        Returns:
            Whether a publication with that ID was removed.
        """
        with self._guard.hold():
            with self._reference_lock:
                return self._repository.delete(dataset_id)

    def publish(self, request: ReferenceUploadRequest) -> ReferenceDatasetRecord:
        """Parse first and publish only a fully valid immutable snapshot.

        Parsing happens before the repository transaction. Therefore invalid
        workbooks leave the previously active dataset untouched.
        """
        created_at = self._clock()
        artifact_id = self._uuid()
        dataset_id = self._uuid()
        normalized_rows: tuple[NormalizedReferenceRow, ...] = ()
        bom_rows = ()
        warning_counts: dict[str, int] = {}
        if request.dataset_type is ReferenceDatasetType.BOM:
            parsed_bom = parse_bom_workbook(
                request.content,
                artifact_id,
                self._settings.max_workbook_rows,
                self._settings.max_workbook_cells,
            )
            bom_rows = parsed_bom.rows
            warning_counts = dict(Counter(parsed_bom.quality_qualifiers))
            normalized = json.dumps(
                {
                    "format_version": "ecn-bom-manifest-v2",
                    "row_count": parsed_bom.row_count,
                    "cell_count": parsed_bom.cell_count,
                    "sheet_roles": {
                        role.value: name for role, name in parsed_bom.sheet_roles.items()
                    },
                    "quality_qualifiers": list(parsed_bom.quality_qualifiers),
                },
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            artifact = _artifact(request, artifact_id, normalized, created_at)
            row_count = parsed_bom.row_count
        elif request.dataset_type is ReferenceDatasetType.CONDITIONAL_RULES:
            normalization = normalize_rule_workbook(
                request.content,
                RuleArtifactMetadata(
                    artifact_id=artifact_id,
                    file_name=request.file_name,
                    media_type=request.media_type,
                    created_at=created_at,
                ),
            )
            artifact = normalization.artifact
            normalized_rows = _rule_rows(artifact_id, normalization.rules)
            warning_counts = {
                str(key): int(value)
                for key, value in normalization.warning_counts.items()
            }
            row_count = len(normalization.rules)
        elif request.dataset_type is ReferenceDatasetType.PURCHASE_INFO_PRICES:
            parsed = parse_purchase_info_price_csv(request.content, artifact_id)
            normalized_rows = parsed.rows
            warning_counts = parsed.warning_counts
            normalized = json.dumps(
                {
                    "format_version": "ecn-purchase-info-price-csv-v1",
                    "dataset_type": request.dataset_type.value,
                    "row_count": parsed.row_count,
                    "warning_counts": warning_counts,
                },
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            artifact = _artifact(request, artifact_id, normalized, created_at)
            row_count = parsed.row_count
        else:
            parsed = parse_reference_workbook(
                request.dataset_type,
                request.content,
                artifact_id,
            )
            normalized_rows = parsed.rows
            warning_counts = parsed.warning_counts
            normalized = json.dumps(
                {
                    "format_version": "ecn-reference-manifest-v1",
                    "dataset_type": request.dataset_type.value,
                    "row_count": parsed.row_count,
                    "warning_counts": warning_counts,
                    "status_legend_version": PROCUREMENT_STATUS_LEGEND_VERSION,
                },
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            artifact = _artifact(request, artifact_id, normalized, created_at)
            row_count = parsed.row_count
        # Keep comparison selection and publication together without holding
        # the upload/history guard during a potentially large workbook insert.
        with self._reference_lock:
            from app.ecn.reference_data.scopes import DatasetContext

            context = DatasetContext.model_validate(request.context).model_dump(mode="json")
            previous = [
                record for record in self._repository.list()
                if str(record.dataset_type) == str(request.dataset_type)
                and record.context.get("model_range") == context["model_range"]
                and record.context.get("data_group", "customer") == context["data_group"]
                and record.context.get("environment", "UNKNOWN") == context["environment"]
            ]
            if context["comparison_dataset_id"]:
                if not any(record.dataset_id == context["comparison_dataset_id"] for record in previous):
                    raise SourceValidationError("COMPARISON_SCOPE_INVALID", "Select a previous dataset in the same source scope.", status_code=400)
            elif previous:
                prior = max(previous, key=lambda record: record.publication_order)
                context["comparison_dataset_id"] = prior.context.get("comparison_dataset_id") or prior.dataset_id
            publication = ReferenceDatasetPublication(
                dataset_id=dataset_id,
                dataset_type=request.dataset_type,
                artifact=artifact,
                rows=normalized_rows,
                row_count=row_count,
                warning_counts=warning_counts,
                scope_code=None,
                scope_name=None,
                created_by=request.actor_id,
                created_at=created_at,
                bom_rows=bom_rows,
                context=context,
            )
            return self._repository.publish(publication)
