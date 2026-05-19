import uuid
import time
import logging
import asyncio
import os
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException

from ..models.matching import MatchRequest, MatchJobResponse, MatchStatusResponse, MatchProgress
from ..services.ai_matching_service import run_ai_matching

OUTPUT_BASE = Path(__file__).resolve().parent.parent.parent / "output"

logger = logging.getLogger(__name__)

router = APIRouter()

jobs: dict[str, dict[str, Any]] = {}
active_job: str | None = None

JOB_EXPIRY_S = 30 * 60


@router.post("/", response_model=MatchJobResponse, status_code=202)
async def start_matching(body: MatchRequest):
    global active_job

    if active_job and active_job in jobs:
        active = jobs[active_job]
        if active["status"] == "processing":
            raise HTTPException(status_code=409, detail="A matching job is already in progress")

    job_id = str(uuid.uuid4())
    job_state = {
        "status": "processing",
        "progress": {"phase": "", "message": "Starting...", "totalPayers": 0, "completedPayers": 0, "matchCount": 0},
        "results": None,
        "error": None,
        "createdAt": time.time(),
    }

    jobs[job_id] = job_state
    active_job = job_id

    col_config = {
        "invoiceNumber": body.colConfig.invoiceNumber if body.colConfig else "Invoice#",
        "invoiceAmount": body.colConfig.invoiceAmount if body.colConfig else "Invoice Amt",
        "textCols": body.colConfig.textCols if body.colConfig else ["BY_ORD_OF_NAME", "BY_ORD_OF_ADDR", "REMIT_NAME"],
        "tolerance": body.tolerance,
    }

    unmatched_invoices = body.ruleMatchedInvoices

    logger.info(
        f"[AI Match Route] Job {job_id} started. "
        f"Invoice CSV: {len(body.invoiceCSV) / 1024:.1f}KB, "
        f"Payment CSV: {len(body.paymentCSV) / 1024:.1f}KB, "
        f"Unmatched: {len(unmatched_invoices)}, Tolerance: {col_config['tolerance']}"
    )

    async def run_job():
        job_start = time.time()
        try:
            job_output_dir = OUTPUT_BASE / job_id
            job_output_dir.mkdir(parents=True, exist_ok=True)

            def on_progress(progress):
                job_state["progress"] = progress

            results = await run_ai_matching(
                body.invoiceCSV,
                body.paymentCSV,
                unmatched_invoices,
                col_config,
                on_progress,
                output_dir=str(job_output_dir),
            )
            job_state["status"] = "complete"
            job_state["results"] = results
            job_state["progress"]["message"] = f"Complete: {len(results)} invoices AI-matched"
            logger.info(f"[AI Match Route] Job {job_id} complete: {len(results)} matches in {time.time() - job_start:.1f}s")
        except Exception as err:
            job_state["status"] = "error"
            job_state["error"] = str(err)
            job_state["progress"]["message"] = f"Error: {err}"
            logger.error(f"[AI Match Route] Job {job_id} failed: {err}")

    asyncio.create_task(run_job())

    return MatchJobResponse(jobId=job_id)


@router.get("/status/{job_id}", response_model=MatchStatusResponse)
async def get_job_status(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = jobs[job_id]
    response = MatchStatusResponse(
        status=job["status"],
        progress=MatchProgress(**job["progress"]) if job["progress"] else None,
    )

    if job["status"] == "complete":
        response.results = job["results"]
    if job["status"] == "error":
        response.error = job["error"]

    return response


async def cleanup_expired_jobs():
    while True:
        await asyncio.sleep(60)
        now = time.time()
        global active_job
        expired = [jid for jid, j in jobs.items() if now - j["createdAt"] > JOB_EXPIRY_S]
        for jid in expired:
            del jobs[jid]
            if active_job == jid:
                active_job = None
