import os
import time
import logging
from typing import Dict, Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

load_dotenv()

from .routers import spa_quick_lookup, spa_agent_chat, spa_summary, spa_material_hierarchy, spa_onboarding, gap, rfm, margin, account_manager, customer_summary, potential_breakdown
from .models.common import HealthResponse

try:
    from .routers import llm_usage_test
except ImportError as exc:
    llm_usage_test = None
    logging.getLogger(__name__).warning(
        "LLM usage test router is unavailable; /api/test/log-llm-usage will not be registered: %s",
        exc,
    )


# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="SPA Gap Analysis API",
    description="FastAPI server for SPA Gap Analysis with Quick Lookup and Agent Chat",
    version="1.0.0",
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


# Include SPA Gap Analysis routers
# Register summary routes before dynamic SPA detail routes so fixed paths like
# /api/spa/summary-stats are not captured by /api/spa/{spa_id}.
app.include_router(spa_summary.router, prefix="/api", tags=["SPA Summary"])
app.include_router(spa_quick_lookup.router, prefix="/api", tags=["SPA Analysis"])
app.include_router(spa_agent_chat.router, prefix="/api", tags=["SPA Agent Chat"])
app.include_router(spa_material_hierarchy.router, prefix="/api/spa", tags=["Material Hierarchy"])
app.include_router(spa_onboarding.router, prefix="/api", tags=["Customer Onboarding"])
if llm_usage_test is not None:
    app.include_router(llm_usage_test.router, prefix="/api", tags=["Observability"])

# NEW: Include routers for updated data (04.13.2026)
app.include_router(gap.router, tags=["Gap Analysis"])
app.include_router(rfm.router, tags=["RFM"])
app.include_router(margin.router, tags=["Margin"])
app.include_router(account_manager.router, tags=["Account Manager"])
app.include_router(customer_summary.router, tags=["Customer Summary (NEW)"])
app.include_router(potential_breakdown.router, tags=["Potential Breakdown"])


@app.get("/")
def read_root():
    return {"message": "Welcome to the FastAPI backend!"}


if __name__ == "__main__":
    # Get port from environment (Cloud Foundry sets this)
    port = int(os.getenv("PORT", "8000"))
    host = "0.0.0.0"

    logger.info(f"Starting API server on {host}:{port}")

    # Run server
    import uvicorn

    uvicorn.run("main:app", host=host, port=port, log_level="info")
