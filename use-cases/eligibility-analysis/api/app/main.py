import os
import logging
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

ENV_PATH = Path(__file__).resolve().parents[1] / ".env"
load_dotenv(dotenv_path=ENV_PATH)

from .routers import eligibility, optimizer
from .a2a import a2a_router
from .models.common import HealthResponse


# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Invoice Eligibility Analysis API",
    description="FastAPI server for analyzing invoice eligibility based on configured criteria",
    version="1.0.0",
)

# Define allowed origins for CORS
origins = []
is_production = os.getenv("APP_ENV") == "production"

# Add production UI URL from environment variable if it exists
prod_origin = os.getenv("ALLOWED_ORIGIN")
if prod_origin:
    origins.append(prod_origin)

# Add CORS middleware for cross-origin requests from UI
# In development, allow all origins for easier testing
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins if is_production else ["*"],
    allow_credentials=True if is_production else False,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"],
    allow_headers=["*"],
)


@app.get("/api/health")
async def health() -> HealthResponse:
    """Health check endpoint for monitoring service availability."""
    return HealthResponse.healthy("eligibility-api")


# Include eligibility router
app.include_router(eligibility.router, prefix="/api", tags=["eligibility"])
app.include_router(optimizer.router, prefix="/api", tags=["optimizer"])
app.include_router(a2a_router, prefix="/api/a2a", tags=["a2a"])


@app.get("/")
def read_root():
    return {"message": "Invoice Eligibility Analysis API"}


if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    host = "0.0.0.0"

    logger.info(f"Starting API server on {host}:{port}")

    import uvicorn
    uvicorn.run("main:app", host=host, port=port, log_level="info")
