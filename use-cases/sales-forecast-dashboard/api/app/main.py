import os
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv


from .routers import dma, stores, timeseries, chatbot, regressor, admin
from .models.common import HealthResponse

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Sales Forecast Dashboard API",
    description="FastAPI server providing data endpoints for the Sales Forecast Dashboard",
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
            "http://localhost:4173",
            "http://localhost:5173",
            "http://localhost:5174",
            "http://localhost:5175",
            "http://localhost:5176",
            "http://localhost:5177",
            "http://127.0.0.1:4173",
            "http://127.0.0.1:5173",
            "http://127.0.0.1:5174",
            "http://127.0.0.1:5175",
            "http://127.0.0.1:5176",
            "http://127.0.0.1:5177",
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


def _env_flag(name: str, default: bool = False) -> bool:
    """Return a boolean environment flag from common truthy string values."""
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    return raw_value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _preload_channels_from_env() -> list[str]:
    """Return configured forecast model channels for startup preload logging."""
    return [
        channel.strip()
        for channel in os.getenv("INFERENCE_PRELOAD_CHANNELS", "B&M").split(",")
        if channel.strip()
    ]


@app.on_event("startup")
async def preload_forecast_models() -> None:
    """Preload forecast models at API startup when configured to do so.

    Cloud Foundry sets PRELOAD_FORECAST_MODELS=true so model checkpoint issues
    fail the app fast instead of surfacing later during an agent workflow.
    Local developers can set PRELOAD_FORECAST_MODELS=false to defer loading.
    """
    if not _env_flag("PRELOAD_FORECAST_MODELS", default=False):
        logger.info("Forecast model preload skipped")
        return

    from app.services.inference_cache import warm_inference_cache

    channels = _preload_channels_from_env()
    logger.info("Preloading forecast inference cache for channels=%s", channels)
    warm_inference_cache(channels=channels, run_explainability=False)
    logger.info("Forecast inference cache preload complete")


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
    return HealthResponse.healthy("sales-forecast-api", version="1.0.0")


# Include routers
app.include_router(dma.router, prefix="/api", tags=["dma"])
app.include_router(stores.router, prefix="/api", tags=["stores"])
app.include_router(timeseries.router, prefix="/api", tags=["timeseries"])
app.include_router(chatbot.router, prefix="/api", tags=["chatbot"])
app.include_router(chatbot.public_router, prefix="/api", tags=["chatbot"])
app.include_router(regressor.router, prefix="/api", tags=["regressor"])
app.include_router(admin.router, prefix="/api", tags=["admin"])


@app.get("/")
def read_root():
    return {"message": "Welcome to the Sales Forecast Dashboard API!"}


if __name__ == "__main__":
    # Get port from environment (Cloud Foundry sets this)
    port = int(os.getenv("PORT", "8000"))
    host = "0.0.0.0"

    logger.info(f"Starting API server on {host}:{port}")

    # Run server
    import uvicorn

    uvicorn.run("main:app", host=host, port=port, log_level="info")
