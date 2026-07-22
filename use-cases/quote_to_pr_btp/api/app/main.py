import os
import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

load_dotenv()

from .models.common import HealthResponse
from .routers import purchase_requisition


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Quote-to-PR BTP AI API",
    description=(
        "FastAPI backend for quote-to-purchase-requisition extraction. "
        "The service exposes upload, processing, saved extraction sessions, "
        "and per-document review results."
    ),
    version="1.0.0",
)

origins: list[str] = []
prod_origin = os.getenv("ALLOWED_ORIGIN")
if prod_origin:
    origins.append(prod_origin)

if os.getenv("APP_ENV") != "production":
    origins.extend(
        [
            "http://localhost:4173",
            "http://localhost:5173",
            "http://localhost:5174",
            "http://localhost:5175",
            "http://localhost:5176",
            "http://localhost:5177",
            "http://localhost:5178",
            "http://localhost:5188",
            "http://localhost:5189",
            "http://127.0.0.1:4173",
            "http://127.0.0.1:5173",
            "http://127.0.0.1:5174",
            "http://127.0.0.1:5175",
            "http://127.0.0.1:5176",
            "http://127.0.0.1:5177",
            "http://127.0.0.1:5178",
            "http://127.0.0.1:5188",
            "http://127.0.0.1:5189",
        ]
    )

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
async def health() -> HealthResponse:
    return HealthResponse.healthy("api")


app.include_router(purchase_requisition.router, prefix="/api/purchase-requisition", tags=["purchase-requisition"])


@app.get("/")
def read_root() -> dict[str, str]:
    return {"message": "Quote-to-PR BTP AI API"}


if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    host = "0.0.0.0"
    logger.info("Starting API server on %s:%s", host, port)

    import uvicorn

    uvicorn.run("app.main:app", host=host, port=port, log_level="info")
