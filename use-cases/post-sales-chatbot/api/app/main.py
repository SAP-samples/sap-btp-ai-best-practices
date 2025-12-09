"""
Apex Automotive Services API - FastAPI Application

Main entry point for the Apex Assistant chatbot API.
"""

import os
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from .routers import apex_chat
from .models.common import HealthResponse
from .services.data_handler import data_handler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler for startup/shutdown events."""
    # Startup
    logger.info("Starting Apex Automotive Services API...")
    logger.info("Loading data...")
    try:
        data_handler.load_data()
        logger.info("Data loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load data: {e}")

    yield

    # Shutdown
    logger.info("Shutting down Apex Automotive Services API...")


app = FastAPI(
    title="Apex Automotive Services API",
    description="FastAPI backend for Apex Assistant chatbot - Customer service for automotive services",
    version="1.0.0",
    lifespan=lifespan
)

# CORS configuration
origins = []

# Production origin from environment
prod_origin = os.getenv("ALLOWED_ORIGIN")
if prod_origin:
    origins.append(prod_origin)

# Development origins
if os.getenv("APP_ENV") != "production":
    origins.extend([
        "http://localhost:5173",
        "http://localhost:5174",
        "http://localhost:5175",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:5174",
        "http://127.0.0.1:5175",
    ])

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins if origins else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Session-Id"],
)


# Health check endpoint
@app.get("/api/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Health check endpoint."""
    return HealthResponse.healthy("apex-api", version="1.0.0")


# Include routers
app.include_router(apex_chat.router, prefix="/api", tags=["apex-chat"])


# Root endpoint
@app.get("/")
def read_root():
    """Root endpoint with API information."""
    return {
        "message": "Welcome to Apex Automotive Services API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/api/health"
    }


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("app.main:app", host="0.0.0.0", port=port, reload=True)
