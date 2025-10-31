"""FastAPI Application for Procurement Assistant"""

import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

try:
    from .config import settings
    from .routers import optimization, jobs, configuration
except ImportError:
    # Handle direct execution
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from api.config import settings
    from api.routers import optimization, jobs, configuration

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format=settings.LOG_FORMAT
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    logger.info("Starting Procurement Assistant API...")
    settings.ensure_directories()
    logger.info(f"Job storage path: {settings.JOB_STORAGE_PATH}")
    logger.info(f"Temp directory: {settings.TEMP_DIR}")
    logger.info(f"Results directory: {settings.RESULTS_DIR}")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Procurement Assistant API...")


# Create FastAPI app
app = FastAPI(
    title=settings.API_TITLE,
    version=settings.API_VERSION,
    description=settings.API_DESCRIPTION,
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ALLOW_ORIGINS,
    allow_credentials=settings.CORS_ALLOW_CREDENTIALS,
    allow_methods=settings.CORS_ALLOW_METHODS,
    allow_headers=settings.CORS_ALLOW_HEADERS,
)


# Custom exception handlers
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle validation errors"""
    return JSONResponse(
        status_code=422,
        content={
            "status": "error",
            "error_code": "VALIDATION_ERROR",
            "message": "Invalid request data",
            "details": exc.errors()
        }
    )


@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    """Handle HTTP exceptions"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "status": "error",
            "error_code": f"HTTP_{exc.status_code}",
            "message": exc.detail,
            "details": {}
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions"""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "status": "error",
            "error_code": "INTERNAL_SERVER_ERROR",
            "message": "An unexpected error occurred",
            "details": {"error": str(exc)} if settings.LOG_LEVEL == "DEBUG" else {}
        }
    )


# Include routers
app.include_router(optimization.router, prefix="/api/optimization", tags=["optimization"])
app.include_router(jobs.router, prefix="/api/jobs", tags=["jobs"])
app.include_router(configuration.router, prefix="/api/configuration", tags=["configuration"])


# Root endpoint
@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "name": settings.API_TITLE,
        "version": settings.API_VERSION,
        "status": "operational",
        "endpoints": {
            "optimization": "/api/optimization",
            "jobs": "/api/jobs",
            "configuration": "/api/configuration",
            "docs": "/docs",
            "redoc": "/redoc"
        }
    }


# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "version": settings.API_VERSION
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)