import logging
import os
from pathlib import Path
from typing import List

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .models.common import HealthResponse
from .routers import extraction


# Basic logging setup mirrors the template service for consistency.
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def _load_environment() -> None:
    """Load environment variables from the single .env file at api/.env."""

    env_path = Path(__file__).resolve().parents[1] / ".env"
    if env_path.exists():
        load_dotenv(env_path, override=False)
        logger.info("Loaded environment variables from %s", env_path)
    else:
        logger.warning("No .env file found at %s", env_path)


_load_environment()

app = FastAPI(
    title="Commodity Code Pipeline API",
    description="FastAPI wrapper around the document extraction and reference classification pipeline",
    version="1.0.0",
)

# CORS configuration follows the relaxed defaults from the template api.
app_env = os.getenv("APP_ENV", "")
allowed_origin = os.getenv("ALLOWED_ORIGIN")

if app_env == "production":
    origins: List[str] = []
    if allowed_origin:
        origins.append(allowed_origin)
else:
    origins = [
        "http://localhost:8501",
        "http://127.0.0.1:8501",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "*",
    ]

allow_credentials_flag = False if "*" in origins else True

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=allow_credentials_flag,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)


@app.get("/api/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Simple health probe."""

    return HealthResponse.healthy("api")


app.include_router(extraction.router, prefix="/api/extraction", tags=["extraction"])


@app.get("/")
async def read_root():
    return {"message": "Welcome to the Commodity Code Pipeline FastAPI backend"}


if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    host = "0.0.0.0"
    logger.info("Starting API server on %s:%s", host, port)

    import uvicorn

    uvicorn.run("app.main:app", host=host, port=port, log_level="info")
