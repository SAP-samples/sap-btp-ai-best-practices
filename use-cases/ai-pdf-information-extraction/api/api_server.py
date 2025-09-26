#!/usr/bin/env python3
"""
FastAPI server for Document Extraction Service.

This server provides REST API endpoints for extracting information from PDF documents
using AI-powered processing. It supports single document and batch processing with
configurable extraction schemas.

Endpoints:
    GET  /api/health: Health check endpoint
    POST /api/extraction/single: Process single document
    POST /api/extraction/batch: Process multiple documents
    GET  /api/extraction/schemas: Get document schemas
    POST /api/extraction/schemas: Update document schemas
    GET  /api/extraction/status/{task_id}: Check processing status

Environment Variables:
    PORT: Server port (optional, defaults to 8000)
    OPENAI_API_KEY: OpenAI API key for GPT-4 vision

Example:
    python api_server.py
    export PORT=8080 && python api_server.py
"""

import os
import time
import logging
from typing import Dict, Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

load_dotenv()

from routers import extraction

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Document Extraction API",
    description="FastAPI server for AI-powered document information extraction",
    version="1.0.0",
)

# Configure CORS for cross-origin requests from Streamlit UI
# Get allowed origin from environment variable
allowed_origin = os.getenv("ALLOWED_ORIGIN", "https://data_extraction.cfapps.eu10-004.hana.ondemand.com")

# In production, use specific origins; in development, allow all
if os.getenv("APP_ENV") == "production":
    allowed_origins = [
        allowed_origin,
        "https://data_extraction.cfapps.eu10-004.hana.ondemand.com"  # Ensure production UI is always allowed
    ]
else:
    # For development, allow common local ports
    allowed_origins = [
        "http://localhost:8501",
        "http://localhost:3000",
        "http://127.0.0.1:8501",
        "http://127.0.0.1:3000",
        "*"  # Allow all in development
    ]

allow_credentails_flag = False if "*" in allowed_origins else True

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=allow_credentails_flag,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)


@app.get("/api/health")
async def health() -> Dict[str, Any]:
    """
    Health check endpoint for monitoring service availability.

    Returns:
        dict: Health check response containing:
            - status: Service health status ("healthy")
            - timestamp: Current Unix timestamp
            - service: Service identifier ("extraction-api")
    """
    return {"status": "healthy", "timestamp": time.time(), "service": "extraction-api"}


# Include routers
app.include_router(extraction.router, prefix="/api/extraction", tags=["extraction"])

if __name__ == "__main__":
    # Get port from environment (Cloud Foundry sets this)
    port = int(os.getenv("PORT", "8000"))
    host = "0.0.0.0"

    logger.info(f"Starting Document Extraction API server on {host}:{port}")

    # Run server
    import uvicorn

    uvicorn.run("api_server:app", host=host, port=port, log_level="info", reload=True)