import os
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Anomaly Detection API",
    description="Backend for Sales Order Anomaly Detection",
    version="1.0.0",
)

# CORS configuration
origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "*"  # Allow all for development convenience, tighten for prod
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from routers import dashboard, orders, anomaly, fine_tuning

app.include_router(dashboard.router, prefix="/api/dashboard", tags=["dashboard"])
app.include_router(orders.router, prefix="/api/orders", tags=["orders"])
app.include_router(anomaly.router, prefix="/api/anomaly", tags=["anomaly"])
app.include_router(fine_tuning.router, prefix="/api/fine-tuning", tags=["fine-tuning"])

@app.get("/api/health")
def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)

