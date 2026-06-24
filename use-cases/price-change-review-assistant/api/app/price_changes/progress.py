from __future__ import annotations

from typing import Any

from .repositories import PriceChangeRepository


class NullProgressReporter:
    """No-op progress reporter used when the UI did not request polling."""

    def event(
        self,
        stage: str,
        message: str,
        level: str = "info",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Ignore one progress event.

        Args:
            stage: Sanitized stage key.
            message: User-facing progress message.
            level: Event severity.
            metadata: Optional sanitized metadata.

        Returns:
            None.
        """
        return None


class HanaProgressReporter:
    """Persist processing progress events to HANA for UI polling."""

    def __init__(self, repository: PriceChangeRepository, processing_run_id: str) -> None:
        """Create a HANA-backed progress reporter.

        Args:
            repository: Repository used for progress persistence.
            processing_run_id: Run id that owns emitted events.

        Returns:
            None.
        """
        self.repository = repository
        self.processing_run_id = processing_run_id
        self._last_stage: str | None = None
        self._last_message: str | None = None

    def event(
        self,
        stage: str,
        message: str,
        level: str = "info",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Persist one sanitized progress event.

        Args:
            stage: Sanitized stage key.
            message: User-facing progress message.
            level: Event severity.
            metadata: Optional sanitized metadata.

        Returns:
            None.
        """
        if stage == self._last_stage and message == self._last_message:
            return
        self.repository.append_processing_event(
            self.processing_run_id,
            stage=stage,
            message=message,
            level=level,
            metadata=metadata,
        )
        self._last_stage = stage
        self._last_message = message


class ProgressReportingTools:
    """Tool facade wrapper that emits sanitized progress around S/4 calls."""

    def __init__(self, tools: Any, reporter: Any) -> None:
        """Wrap an existing agent tool facade.

        Args:
            tools: Tool facade used by the agent.
            reporter: Progress reporter receiving sanitized events.

        Returns:
            None.
        """
        self._tools = tools
        self._reporter = reporter

    def __getattr__(self, name: str) -> Any:
        """Delegate unknown attributes to the wrapped tool facade.

        Args:
            name: Attribute name requested by caller.

        Returns:
            Attribute from the wrapped tool facade.
        """
        return getattr(self._tools, name)

    def _report(self, stage: str, message: str, result: dict[str, Any] | None = None) -> None:
        """Emit a sanitized event with coarse result status.

        Args:
            stage: Sanitized stage key.
            message: User-facing progress message.
            result: Optional tool result used only for a coarse status.

        Returns:
            None.
        """
        metadata = {}
        if isinstance(result, dict) and result.get("status") is not None:
            metadata["status"] = str(result.get("status"))
        self._reporter.event(stage, message, metadata=metadata)

    def find_supplier_by_email(self, email: str | None) -> dict[str, Any]:
        """Find supplier by email while reporting a sanitized S/4 stage."""
        result = self._tools.find_supplier_by_email(email)
        self._report("resolving_supplier", "Resolving supplier in S/4", result)
        return result

    def find_supplier_by_id(self, supplier_id: str | None) -> dict[str, Any]:
        """Find supplier by id while reporting a sanitized S/4 stage."""
        result = self._tools.find_supplier_by_id(supplier_id)
        self._report("resolving_supplier", "Resolving supplier in S/4", result)
        return result

    def find_supplier_by_name(self, name_or_company: str | None) -> dict[str, Any]:
        """Find supplier by name while reporting a sanitized S/4 stage."""
        result = self._tools.find_supplier_by_name(name_or_company)
        self._report("resolving_supplier", "Resolving supplier in S/4", result)
        return result

    def find_material_by_number(self, material_number: str | None) -> dict[str, Any]:
        """Find material by number while reporting a sanitized S/4 stage."""
        result = self._tools.find_material_by_number(material_number)
        self._report("checking_material", "Checking material in S/4", result)
        return result

    def search_materials_by_description(
        self,
        query: str | None,
        supplier_id: str | None = None,
    ) -> dict[str, Any]:
        """Search materials by description while reporting a sanitized S/4 stage."""
        result = self._tools.search_materials_by_description(query, supplier_id=supplier_id)
        self._report("checking_material", "Checking material in S/4", result)
        return result

    def get_current_supplier_material_price(
        self,
        supplier_id: str | None,
        material_number: str | None,
    ) -> dict[str, Any]:
        """Read current supplier price while reporting a sanitized S/4 stage."""
        result = self._tools.get_current_supplier_material_price(supplier_id, material_number)
        self._report("reading_current_price", "Reading current supplier price", result)
        return result

    def persist_price_change_draft(
        self,
        proposal: Any,
        extraction_id: str,
        item_index: int,
        raw_agent_output: dict[str, Any],
    ) -> str:
        """Persist a draft and report that it is available for review.

        Args:
            proposal: Agent proposal to persist.
            extraction_id: Extraction row id.
            item_index: Extracted item index.
            raw_agent_output: Audit payload stored with the draft.

        Returns:
            Persisted draft id.
        """
        draft_id = self._tools.persist_price_change_draft(
            proposal=proposal,
            extraction_id=extraction_id,
            item_index=item_index,
            raw_agent_output=raw_agent_output,
        )
        self._reporter.event("draft_saved", "Draft saved for review", metadata={"draft_id": draft_id})
        return draft_id


def with_progress_reporting(tools: Any, reporter: Any | None) -> Any:
    """Return a progress-reporting wrapper when a reporter is provided.

    Args:
        tools: Original agent tool facade.
        reporter: Optional progress reporter.

    Returns:
        Original tools when reporter is missing, otherwise a wrapped facade.
    """
    if reporter is None:
        return tools
    return ProgressReportingTools(tools, reporter)
