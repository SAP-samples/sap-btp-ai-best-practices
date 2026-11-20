"""FastAPI entry point of the purchase-order intake API.

Start-up (``lifespan``):
    1. Verify/create every HANA table and sequence the app needs (``db.schema.ensure_schema``).
       If HANA is not configured or unreachable the API still starts and logs the problem,
       so health checks and the UI shell keep working.
    2. Start the order pipeline (agent workers + sweeper), unless ORDER_PIPELINE_ENABLED=false.
       Shut-down cancels the workers; their orders are requeued at the next start.

Run locally:
    cd api && ../.venv/bin/python -m uvicorn app.main:app --reload --port 8000
"""

import asyncio
import logging
import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()

from .db.engine import get_engine, hana_configured  # noqa: E402  (after load_dotenv)
from .db.schema import ensure_schema  # noqa: E402
from .models.common import HealthResponse  # noqa: E402
from .orders.pipeline import OrderPipeline  # noqa: E402
from .routers import master_data, orders  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application start-up / shut-down hook.

    Args:
        app: The FastAPI application; ``app.state.hana_ready`` records whether the
            schema bootstrap succeeded.

    Yields:
        Control to FastAPI while the application serves requests.
    """
    app.state.hana_ready = False
    pipeline = None
    if not hana_configured():
        logger.warning("HANA_* variables not set: database features are disabled")
    else:
        try:
            # Blocking DDL runs in a worker thread, never on the event loop.
            await asyncio.to_thread(ensure_schema, get_engine())
            app.state.hana_ready = True
        except Exception:  # noqa: BLE001 - start anyway, surface the cause in the logs
            logger.exception("HANA schema bootstrap failed")
    if app.state.hana_ready and os.getenv("ORDER_PIPELINE_ENABLED", "true").lower() != "false":
        pipeline = OrderPipeline()
        app.state.order_queue_event = pipeline.event
        await pipeline.start()
    yield
    if pipeline is not None:
        await pipeline.stop()


app = FastAPI(
    title="Purchase Order Intake API",
    description="Agentic intake of customer purchase orders into (mocked) SAP ECC sales orders",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS: production UI origin from ALLOWED_ORIGIN, plus local dev servers outside production.
origins = [origin for origin in [os.getenv("ALLOWED_ORIGIN")] if origin]
if os.getenv("APP_ENV") != "production":
    origins.extend(
        [f"http://{host}:{port}" for host in ("localhost", "127.0.0.1") for port in (4173, 5173, 5174, 5175, 5176, 5177)]
    )

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.include_router(orders.router, prefix="/api", tags=["orders"])
app.include_router(master_data.router, prefix="/api/master-data", tags=["master-data"])


@app.get("/api/health")
async def health() -> HealthResponse:
    """Health check for load balancers and the UI.

    Returns:
        HealthResponse with status "healthy", or "degraded" when the HANA bootstrap failed.
    """
    response = HealthResponse.healthy("api")
    if not getattr(app.state, "hana_ready", False):
        response.status = "degraded"
    return response


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8000"))
    logger.info("Starting API server on 0.0.0.0:%s", port)
    uvicorn.run("app.main:app", host="0.0.0.0", port=port, log_level="info")
