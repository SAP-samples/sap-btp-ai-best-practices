import os
import logging
from typing import Dict, Any

import httpx
from fastapi import Depends, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from dotenv import load_dotenv

load_dotenv()

from .auth import UserIdentity, get_current_user, _get_xsuaa_config, cdc_enabled
from .services.cdc import store_user_login_data
from .routers import (
    admin,
    canvas,
    evaluation,
    events,
    handover,
    ideation_agent,
    ideation_upload,
    leaderboard,
    sharing,
    submissions,
    version_check,
)
from .models.common import HealthResponse
from .services.version_check import LOCAL_VERSION


# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="AI4U Ideation Agent",
    description="AI4U Ideation Agent is an AI-powered ideation tool that helps you ideate and create new AI use cases.",
    version=LOCAL_VERSION,
)

# Define allowed origins for CORS
origins = []

# Add production UI URL from environment variable if it exists
prod_origin = os.getenv("ALLOWED_ORIGIN")
if prod_origin:
    origins.extend([o.strip() for o in prod_origin.split(",") if o.strip()])

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
            "http://127.0.0.1:5173",
            "http://127.0.0.1:5174",
            "http://127.0.0.1:5175",
            "http://127.0.0.1:5176",
            "http://127.0.0.1:5177",
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
    return HealthResponse.healthy("api", version=LOCAL_VERSION)


@app.get("/api/user/me")
async def user_me(user: UserIdentity = Depends(get_current_user)) -> Dict[str, Any]:
    return {
        "userId": user.user_id,
        "email": user.email,
        "givenName": user.given_name,
        "familyName": user.family_name,
        "authProvider": user.auth_provider,
    }


@app.post("/api/auth/token")
async def auth_token_exchange(request: Request) -> JSONResponse:
    """Proxy token exchange to XSUAA (adds client_secret server-side)."""
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "invalid_request"}, status_code=400)

    config = _get_xsuaa_config()
    client_secret = os.getenv("XSUAA_CLIENT_SECRET", "")
    if not client_secret:
        import json
        vcap = os.getenv("VCAP_SERVICES", "")
        if vcap:
            try:
                services = json.loads(vcap)
                client_secret = services.get("xsuaa", [{}])[0].get("credentials", {}).get("clientsecret", "")
            except (KeyError, IndexError, json.JSONDecodeError):
                pass

    grant_type = body.get("grant_type", "authorization_code")
    form_data = {
        "grant_type": grant_type,
        "client_id": config.client_id,
        "client_secret": client_secret,
    }

    if grant_type == "authorization_code":
        form_data["code"] = body.get("code", "")
        form_data["redirect_uri"] = body.get("redirect_uri", "")
        form_data["code_verifier"] = body.get("code_verifier", "")
    elif grant_type == "refresh_token":
        form_data["refresh_token"] = body.get("refresh_token", "")

    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.post(
            f"{config.xsuaa_url}/oauth/token",
            data=form_data,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )

    return JSONResponse(resp.json(), status_code=resp.status_code)


@app.post("/api/auth/track-login")
async def track_login(user: UserIdentity = Depends(get_current_user)):
    """Track user login in CDC or HANA (consent + timestamps).

    Non-blocking: always returns 200, even if tracking fails.
    Uses CDC when configured, otherwise falls back to HANA users table.
    """
    try:
        if cdc_enabled():
            store_user_login_data(user.email)
        else:
            from .services.hana_users import track_login as hana_track_login

            display_name = f"{user.given_name} {user.family_name}".strip()
            await hana_track_login(user.email, display_name)
    except Exception as exc:
        logger.warning("Login tracking failed for %s: %s", user.email, exc)

    return {"status": "ok"}


# Include routers
app.include_router(admin.router, prefix="/api/admin", tags=["admin"])
app.include_router(canvas.router, prefix="/api/canvas", tags=["canvas"])
app.include_router(
    ideation_agent.router,
    prefix="/api/ideation-agent",
    tags=["ideation-agent"],
)
app.include_router(
    ideation_upload.router,
    prefix="/api/ideation-agent",
    tags=["ideation-upload"],
)
app.include_router(
    evaluation.router,
    prefix="/api/ideation-agent",
    tags=["evaluation"],
)
app.include_router(
    submissions.router,
    prefix="/api/ideation-agent",
    tags=["submissions"],
)
app.include_router(
    leaderboard.router,
    prefix="/api/ideation-agent",
    tags=["leaderboard"],
)
app.include_router(
    events.router,
    prefix="/api/events",
    tags=["events"],
)
app.include_router(
    sharing.router,
    prefix="/api/ideation-agent",
    tags=["sharing"],
)
app.include_router(version_check.router, prefix="/api", tags=["version"])
app.include_router(handover.router, prefix="/api/handover", tags=["handover"])

# Telemetry receiver — only on SAP-hosted instance
if os.getenv("INSTALLATION_ID", "").strip() == "sap-hosted-primary":
    from .routers import telemetry

    app.include_router(telemetry.router, prefix="/api/telemetry", tags=["telemetry"])


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
