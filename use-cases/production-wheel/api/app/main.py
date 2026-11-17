import os
import time
import logging
from typing import Dict, Any

from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

load_dotenv()

from .models.common import HealthResponse
from .workspace.worker_runtime import workspace_lifespan


# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    lifespan=workspace_lifespan,
    title="Production Wheel Optimizer API",
    description=(
        "API surface for the production-wheel optimizer and its orchestrating "
        "agent. The agent runs via `python -m app.agent`; long solves run as async "
        "jobs tracked in HANA (PRODUCTION_WHEEL_RUNS)."
    ),
    version="2.0.0",
)

# Every workspace route uses the same server-side API-key boundary.
from .security import get_api_key
from .observability import (
    RequestUsageContext,
    extract_client_host_from_request,
    extract_user_id_from_request,
    request_usage_context,
)
from .routers.workspace import router as workspace_router
from .routers.workspace_chat import router as chat_router
from .routers.constraint_code import router as constraint_code_router
app.include_router(workspace_router, dependencies=[Depends(get_api_key)])
app.include_router(chat_router, dependencies=[Depends(get_api_key)])
# Control endpoints authenticate normally; task callbacks accept only a scoped,
# expiring capability so runner containers never receive the application key.
app.include_router(constraint_code_router)

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
            "http://localhost:5173",
            "http://localhost:5174",
            "http://localhost:5175",
            "http://localhost:5176",
            "http://localhost:5177",
            "http://127.0.0.1:5173",
            "http://127.0.0.1:5174",
            "http://127.0.0.1:5175",
            "http://127.0.0.1:5176",
            "http://127.0.0.1:5177",
            # "*"
        ]
    )

@app.middleware("http")
async def attach_llm_usage_context(request, call_next):
    """Tag LLM token-usage events with the request route, caller and correlation id.

    Stores a ``RequestUsageContext`` in a ContextVar before the endpoint runs, so
    every model call made while serving this request is attributed to it.
    """
    request_usage_context.set(
        RequestUsageContext(
            route=request.url.path,
            method=request.method,
            user_id=extract_user_id_from_request(request),
            client_host=extract_client_host_from_request(request),
            correlation_id=request.headers.get("x-correlation-id"),
        )
    )
    return await call_next(request)


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


@app.get("/")
def read_root():
    return {"message": "Production Wheel Optimizer API. Health at /api/health."}


if __name__ == "__main__":
    # Get port from environment (Cloud Foundry sets this)
    port = int(os.getenv("PORT", "8000"))
    host = "0.0.0.0"

    logger.info(f"Starting API server on {host}:{port}")

    # Run server
    import uvicorn

    uvicorn.run("app.main:app", host=host, port=port, log_level="info")
