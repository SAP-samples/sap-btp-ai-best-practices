from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from app.models.nbo import BatchArtifactsResponse, BatchRunResponse, BatchSummaryResponse
from app.nbo.config import OUTPUT_DIR
from app.nbo.engine import run_decision_batch
from app.nbo.output import write_excel, write_json


class BatchService:
    def __init__(self) -> None:
        self.runs_dir = OUTPUT_DIR / "batch_runs"
        self.runs_dir.mkdir(parents=True, exist_ok=True)

    def run_batch(self) -> BatchRunResponse:
        run_id = str(uuid4())
        run_dir = self.runs_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        results = run_decision_batch()
        excel_path = write_excel(results, path=run_dir / "nbo_recommendations.xlsx")
        json_path = write_json(results, path=run_dir / "nbo_recommendations.json")

        summary = BatchSummaryResponse(
            total_accounts=len(results),
            residential_accounts=sum(1 for result in results if result.customer_type == "RESIDENTIAL"),
            commercial_accounts=sum(1 for result in results if result.customer_type == "COMMERCIAL"),
            accounts_with_final_offer=sum(1 for result in results if result.final_offer is not None),
        )
        response = BatchRunResponse(
            run_id=run_id,
            created_at=datetime.now(timezone.utc).isoformat(),
            summary=summary,
            artifacts=BatchArtifactsResponse(
                excel_path=str(excel_path),
                json_path=str(json_path),
            ),
        )

        metadata_path = run_dir / "metadata.json"
        metadata_path.write_text(
            json.dumps(response.model_dump(), indent=2),
            encoding="utf-8",
        )
        return response

    def get_run(self, run_id: str) -> BatchRunResponse:
        metadata_path = self.runs_dir / run_id / "metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(run_id)
        return BatchRunResponse.model_validate_json(metadata_path.read_text(encoding="utf-8"))

    def get_artifact_path(self, run_id: str, artifact_name: str) -> Path:
        artifact_path = self.runs_dir / run_id / artifact_name
        if not artifact_path.exists():
            raise FileNotFoundError(artifact_name)
        return artifact_path


batch_service = BatchService()
