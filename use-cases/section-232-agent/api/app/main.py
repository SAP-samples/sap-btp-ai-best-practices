import asyncio
import os
import logging
import socket
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

load_dotenv()

from .routers import metal_composition, metal_composition_admin
from .models.common import HealthResponse
from .services.metal_composition.config import MetalCompositionSettings, get_settings
from .services.metal_composition.service import (
    get_metal_composition_service,
    warm_metal_composition_service,
)


# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


async def _wait_for_stop(stop_event: asyncio.Event, timeout_seconds: float) -> bool:
    try:
        await asyncio.wait_for(stop_event.wait(), timeout=timeout_seconds)
        return True
    except asyncio.TimeoutError:
        return False


async def _classification_job_worker_loop(
    stop_event: asyncio.Event,
    *,
    settings: MetalCompositionSettings,
) -> None:
    worker_id = f"{socket.gethostname()}:{os.getpid()}:{uuid.uuid4()}"
    running_tasks: set[asyncio.Task] = set()
    max_concurrency = max(1, int(settings.classification_job_worker_max_concurrency))
    poll_interval_seconds = max(0.1, float(settings.classification_job_poll_interval_seconds))

    logger.info(
        "Starting classification job worker loop with concurrency=%s poll_interval=%ss worker_id=%s",
        max_concurrency,
        poll_interval_seconds,
        worker_id,
    )

    while not stop_event.is_set():
        finished_tasks = {task for task in running_tasks if task.done()}
        for task in finished_tasks:
            running_tasks.discard(task)
            try:
                await task
            except Exception:
                logger.exception("Classification job worker task crashed")

        try:
            while len(running_tasks) < max_concurrency and not stop_event.is_set():
                service = get_metal_composition_service()
                job = await asyncio.to_thread(service.claim_next_classification_job, worker_id)
                if job is None:
                    break
                logger.info("Claimed classification job %s (%s)", job.job_id, job.job_type)
                running_tasks.add(
                    asyncio.create_task(
                        asyncio.to_thread(service.process_claimed_classification_job, job.job_id)
                    )
                )
        except Exception:
            logger.exception(
                "Worker loop failed while claiming jobs (running_tasks=%d) -- will retry next cycle",
                len(running_tasks),
            )

        if running_tasks:
            await _wait_for_stop(stop_event, timeout_seconds=poll_interval_seconds)
            continue

        await _wait_for_stop(stop_event, timeout_seconds=poll_interval_seconds)

    if running_tasks:
        await asyncio.gather(*running_tasks, return_exceptions=True)

    logger.info("Classification job worker loop stopped")

@asynccontextmanager
async def lifespan(_: FastAPI):
    settings = get_settings()
    stop_event = asyncio.Event()
    worker_task: asyncio.Task | None = None
    if settings.prewarm_on_startup:
        logger.info("Prewarming metal composition service")
        warm_metal_composition_service()
    worker_task = asyncio.create_task(
        _classification_job_worker_loop(stop_event, settings=settings)
    )
    try:
        yield
    finally:
        stop_event.set()
        if worker_task is not None:
            await worker_task


# Create FastAPI app
app = FastAPI(
    title="Metal Composition Classification API",
    description=(
        "FastAPI backend for product metal composition classification, HTS lookup, "
        "and Section 232 analysis."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# Define allowed origins for CORS
origins = []

# Add production UI URL from environment variable if it exists
prod_origin = os.getenv("ALLOWED_ORIGIN")
if prod_origin:
    origins.append(prod_origin)

# Allow localhost for development
if os.getenv("APP_ENV") != "production":
    origins.extend(
        [
            "http://localhost:4173",
            "http://localhost:4174",
            "http://localhost:4175",
            "http://localhost:4176",
            "http://localhost:4177",
            "http://localhost:5173",
            "http://localhost:5174",
            "http://localhost:5175",
            "http://localhost:5176",
            "http://localhost:5177",
            "http://127.0.0.1:4173",
            "http://127.0.0.1:4174",
            "http://127.0.0.1:4175",
            "http://127.0.0.1:4176",
            "http://127.0.0.1:4177",
            "http://127.0.0.1:5173",
            "http://127.0.0.1:5174",
            "http://127.0.0.1:5175",
            "http://127.0.0.1:5176",
            "http://127.0.0.1:5177",
            # "*"
        ]
    )

# Add CORS middleware for cross-origin requests from UI
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
async def health() -> HealthResponse:
    """Health check endpoint for monitoring service availability.

    Used by load balancers, monitoring systems, and health checks to verify
    that the API server is running and responsive.

    Returns:
        HealthResponse: Health check response containing:
            - status: Service health status ("healthy")
            - timestamp: Current Unix timestamp
            - service: Service identifier ("api")
    """
    return HealthResponse.healthy("api")


# Include routers
app.include_router(
    metal_composition.router,
    prefix="/api/metal-composition",
    tags=["metal-composition"],
)
app.include_router(
    metal_composition_admin.router,
    prefix="/api/metal-composition/admin",
    tags=["metal-composition-admin"],
)
@app.get("/")
def read_root():
    return {"message": "Metal Composition Classification API"}


if __name__ == "__main__":
    # Get port from environment (Cloud Foundry sets this)
    port = int(os.getenv("PORT", "8000"))
    host = "0.0.0.0"

    logger.info(f"Starting API server on {host}:{port}")

    # Run server
    import uvicorn

    uvicorn.run("main:app", host=host, port=port, log_level="info")
