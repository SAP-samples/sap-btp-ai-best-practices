"""Error and warning aggregation primitives for benchmark validation."""

from __future__ import annotations

from typing import Any

from .constants import MAX_SAMPLED_ERRORS, MAX_WARNING_SAMPLES
from .models import BenchmarkRowError, BenchmarkWarning


class RowValueError(ValueError):
    """Carry a stable validation code for one invalid source row value.

    Inputs:
        code: Machine-readable row validation category.
        message: Human-readable value failure description.

    Outputs:
        Exception used internally to aggregate sampled row errors.
    """

    def __init__(self, code: str, message: str) -> None:
        """Initialize a coded source row value failure.

        Inputs:
            code: Stable validation category.
            message: Human-readable row failure description.

        Outputs:
            None. ``ValueError`` is initialized with the same message.
        """

        self.code = code
        super().__init__(message)


class ValidationCollector:
    """Accumulate bounded error samples and unique rejected worksheet rows.

    Inputs:
        None. Errors are appended through ``add`` while parsing.

    Outputs:
        Collector exposing sampled errors and rejected-row identities.
    """

    def __init__(self) -> None:
        """Initialize empty sampled errors and rejected-row tracking.

        Inputs:
            None.

        Outputs:
            None. Mutable internal collections are ready for parser use.
        """

        self.errors: list[BenchmarkRowError] = []
        self.rejected_rows: set[int] = set()

    def add(self, row_number: int, code: str, message: str) -> None:
        """Record a rejected row and retain a bounded representative error.

        Inputs:
            row_number: One-based worksheet row number.
            code: Stable machine-readable validation category.
            message: Human-readable explanation of the rejection.

        Outputs:
            None. The row is counted once even when it has multiple errors.
        """

        self.rejected_rows.add(row_number)
        if len(self.errors) < MAX_SAMPLED_ERRORS:
            self.errors.append(
                BenchmarkRowError(
                    row_number=row_number,
                    code=code,
                    message=message,
                )
            )


class WarningAccumulator:
    """Aggregate non-blocking quality issues by stable warning code.

    Inputs:
        None. Warning occurrences are added during parsing and reconciliation.

    Outputs:
        Bounded JSON-safe aggregate warnings sorted by code.
    """

    def __init__(self) -> None:
        """Initialize empty aggregate and optional deduplication state.

        Inputs:
            None.

        Outputs:
            None. Internal warning dictionaries are ready for parser use.
        """

        self._warnings: dict[str, dict[str, Any]] = {}
        self._tokens: set[tuple[str, str]] = set()

    def add(
        self,
        code: str,
        message: str,
        sample: str,
        *,
        token: str | None = None,
    ) -> None:
        """Add one warning occurrence, optionally deduplicated by source token.

        Inputs:
            code: Stable aggregate warning category.
            message: Human-readable description shared by the category.
            sample: Representative row, entity, or value description.
            token: Optional identity that prevents repeat counting across rows.

        Outputs:
            None. Aggregate count and bounded samples are updated in place.
        """

        if token is not None:
            identity = (code, token)
            if identity in self._tokens:
                return
            self._tokens.add(identity)
        warning = self._warnings.setdefault(
            code,
            {"message": message, "count": 0, "samples": []},
        )
        warning["count"] += 1
        if sample and len(warning["samples"]) < MAX_WARNING_SAMPLES:
            warning["samples"].append(sample)

    def build(self) -> list[BenchmarkWarning]:
        """Return sorted JSON-safe warning models from accumulated occurrences.

        Inputs:
            None.

        Outputs:
            list[BenchmarkWarning]: Aggregate warnings ordered by stable code.
        """

        return [
            BenchmarkWarning(
                code=code,
                message=payload["message"],
                count=payload["count"],
                samples=payload["samples"],
            )
            for code, payload in sorted(self._warnings.items())
        ]
