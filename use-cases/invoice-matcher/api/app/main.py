import os
import logging
import asyncio

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

load_dotenv()

from .routers import matching
from .routers.matching import cleanup_expired_jobs
from .models.common import HealthResponse

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Invoice-Payment Matcher API",
    description="FastAPI backend for AI-powered invoice-to-payment matching",
    version="2.0.0",
)

origins = []

prod_origin = os.getenv("ALLOWED_ORIGIN")
if prod_origin:
    origins.append(prod_origin)

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
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    asyncio.create_task(cleanup_expired_jobs())


@app.get("/api/health")
async def health() -> HealthResponse:
    return HealthResponse.healthy("api")


app.include_router(matching.router, prefix="/api/ai-match", tags=["matching"])


@app.get("/")
def read_root():
    return {"message": "Invoice-Payment Matcher API"}


if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=port, log_level="info")
