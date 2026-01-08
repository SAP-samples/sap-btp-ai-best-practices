import os
import time
import logging
from typing import Dict, Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

load_dotenv()

from .routers import document_intelligence
from .routers import rate_mapping
from .routers import prompts
from .models.common import HealthResponse


# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="SAP PoC Template",
    description="FastAPI server providing integration endpoints for SAP PoC Template",
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


# Include routers
app.include_router(
    document_intelligence.router,
    prefix="/api/document-intel",
    tags=["document-intelligence"],
)
app.include_router(
    rate_mapping.router,
    prefix="/api/rate-mapping",
    tags=["rate-mapping"],
)
app.include_router(
    prompts.router,
    prefix="/api/prompts",
    tags=["prompts"],
)


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
