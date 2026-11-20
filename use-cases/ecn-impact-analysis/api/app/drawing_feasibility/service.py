"""Application-scoped service for one transient PS4 drawing assessment."""

from __future__ import annotations

import asyncio
from uuid import uuid4

from app.drawing_feasibility.errors import DrawingAnalysisError
from app.drawing_feasibility.graph import create_drawing_assessment_graph
from app.drawing_feasibility.models import (
    DrawingFeasibilityReport,
    DrawingInput,
    TelemetryContext,
)
from app.drawing_feasibility.policy import PolicyBundle, load_policy_bundle
from app.drawing_feasibility.provider import (
    DrawingExtractionProvider,
    NativeDrawingExtractionProvider,
)
from app.drawing_feasibility.settings import DrawingFeasibilitySettings


class DrawingFeasibilityService:
    """Own the bounded graph, fixed policy prompt, concurrency, and provider lifecycle."""

    def __init__(
        self,
        settings: DrawingFeasibilitySettings,
        provider: DrawingExtractionProvider | None = None,
        policy: PolicyBundle | None = None,
    ) -> None:
        """Create the provider and finite graph for every deployed application instance."""
        self._settings = settings
        self._provider = provider or NativeDrawingExtractionProvider(settings)
        self._policy = policy or load_policy_bundle()
        self._graph = create_drawing_assessment_graph(self._provider)
        self._capacity = asyncio.Semaphore(settings.max_concurrency)

    @property
    def settings(self) -> DrawingFeasibilitySettings:
        """Expose immutable upload limits to the transport layer."""
        return self._settings

    async def assess(
        self,
        drawing: DrawingInput,
        *,
        user_id: str | None = None,
        client_host: str | None = None,
    ) -> DrawingFeasibilityReport:
        """Return a transient report through the configured shared LLM runtime.

        The service persists nothing. Uploaded drawing bytes are sent to SAP
        Generative AI Hub only while this assessment is executing.
        """
        telemetry = TelemetryContext(
            correlation_id=str(uuid4()), user_id=user_id, client_host=client_host
        )
        async with self._capacity:
            try:
                async with asyncio.timeout(self._settings.request_timeout_seconds):
                    state = await self._graph.ainvoke(
                        {"drawing": drawing, "policy": self._policy, "telemetry": telemetry},
                        {"recursion_limit": 8},
                    )
            except TimeoutError as exc:
                raise DrawingAnalysisError(
                    "DRAWING_ANALYSIS_TIMEOUT", status_code=504
                ) from exc
        return DrawingFeasibilityReport.model_validate(state["report"])

    async def close(self) -> None:
        """Close the isolated provider when the application stops."""
        await self._provider.close()
