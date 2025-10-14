import os
import time
import logging
from typing import Dict, Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

load_dotenv()

from .routers import chat
from .routers import emails


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
            "http://127.0.0.1:5173",
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


# Include routers
app.include_router(chat.router, prefix="/api/chat", tags=["chat"])
app.include_router(emails.router, prefix="/api", tags=["emails"])


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
