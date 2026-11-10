import logging
import asyncio
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator, Dict

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()

from .routers import joule, payment_advice, email_ingestion, s4
from .models.common import HealthResponse

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Absolute path to the UC-02 agent YAML config, used by the lifespan and CLI.
_AGENT_CONFIG_PATH = Path(__file__).resolve().parent / "deduction_agent" / "config" / "agent.yaml"
# UC-02 rules-authoring chat config (memory-enabled) for the /rules-chat endpoint.
_CHAT_CONFIG_PATH = Path(__file__).resolve().parent / "deduction_agent" / "config" / "chat.yaml"


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """FastAPI lifespan: build HANA engine and deduction agent runtime on startup.

    Stores ``app.state.engine`` and ``app.state.runtime`` so the /interpret
    route can access them without rebuilding on each request.  On shutdown,
    closes the agent runtime (MCP clients + memory store).

    The HANA engine and AgentRuntime are only created here (at server startup),
    NOT at import time, so ``import app.main`` succeeds without live SAP
    credentials.
    """
    # ------------------------------------------------------------------ startup
    logger.info("lifespan startup: building HANA engine and deduction runtime")
    app.state.engine = None
    try:
        from .payment_advice.db import bootstrap, get_engine
        from .deduction_agent.template_agent.runtime import AgentRuntime
        from .deduction_agent.tools.uc02_tools import build_uc02_tools

        engine = get_engine()
        bootstrap(engine)
        app.state.engine = engine
        tools = build_uc02_tools(engine)
        runtime = await AgentRuntime.create(str(_AGENT_CONFIG_PATH), extra_tools=tools)
        app.state.engine = engine
        app.state.runtime = runtime
        logger.info("lifespan startup complete: engine and runtime ready")
    except Exception as exc:
        # Log but do not crash the process: existing UC-01 routes must stay up
        # even if HANA / AI Core credentials are absent (e.g. in staging without
        # the full secret set).
        logger.error("lifespan startup: could not initialize UC-02 runtime", exc_info=True)
        app.state.runtime = None

    # UC-02 rules-authoring chat runtime (memory-enabled) for /rules-chat. Depends on
    # the HANA engine; built separately so a chat-config problem cannot take down the
    # interpret route. Given both the rule-source readers and the playbook tools so one
    # conversation can look up, ingest, and persist customer deduction rules.
    app.state.chat_runtime = None
    try:
        engine = getattr(app.state, "engine", None)
        if engine is not None:
            from .deduction_agent.rule_sources import build_rule_source_tools
            from .deduction_agent.template_agent.runtime import AgentRuntime
            from .deduction_agent.tools.uc02_tools import build_uc02_tools

            from .deduction_agent.rule_sources import get_bound_source_path
            from .deduction_agent.tools.customer_admin import build_customer_admin_tools, current_request
            from .deduction_agent.tools.schema_tools import build_schema_tools
            # Document AI client/settings are created further below: the schema tools read them at call time.
            schema_tools = build_schema_tools(engine, lambda: getattr(app.state, "dox", None),
                                              lambda: getattr(app.state, "settings", None),
                                              current_request.get, get_bound_source_path)
            chat_tools = [*build_rule_source_tools(), *build_uc02_tools(engine), *build_customer_admin_tools(engine),
                          *schema_tools]
            app.state.chat_runtime = await AgentRuntime.create(
                str(_CHAT_CONFIG_PATH), extra_tools=chat_tools
            )
            logger.info("lifespan startup: rules-chat runtime ready")
    except Exception:
        logger.error("lifespan startup: could not initialize rules-chat runtime", exc_info=True)
        app.state.chat_runtime = None

    # SAP Document AI client + settings for the UC-01 /extract endpoint. Built in a
    # separate try so that missing Document AI credentials do not disable the
    # already-working UC-02 interpret route (and vice versa).
    try:
        from .payment_advice.config import PaymentAdviceSettings
        from .payment_advice.extract import build_client

        settings = PaymentAdviceSettings.from_env()
        app.state.settings = settings
        app.state.dox = build_client(settings.service_key_data)
        logger.info("lifespan startup: Document AI client ready for /extract")
    except Exception:
        logger.error("lifespan startup: could not initialize Document AI client", exc_info=True)
        app.state.settings = None
        app.state.dox = None

    app.state.workspace = None
    worker = None
    try:
        if app.state.engine is not None:
            from .email_ingestion.store import Store
            from .email_ingestion.worker import run_worker
            workspace = Store(app.state.engine)
            await asyncio.to_thread(workspace.ensure)
            app.state.workspace = workspace
            worker = asyncio.create_task(run_worker(app.state))
    except Exception:
        logger.error('Workspace storage startup failed', exc_info=True)

    yield  # server serves requests here
    if worker:
        worker.cancel()
        await asyncio.gather(worker, return_exceptions=True)

    # ----------------------------------------------------------------- shutdown
    logger.info("lifespan shutdown: closing deduction runtimes")
    for attr in ("runtime", "chat_runtime"):
        runtime_to_close = getattr(app.state, attr, None)
        if runtime_to_close is not None:
            try:
                await runtime_to_close.aclose()
            except Exception as exc:
                logger.warning("lifespan shutdown: error closing %s: %s", attr, exc)


# Create FastAPI app with the UC-02 lifespan wired in.
app = FastAPI(
    title="Payment Advice Extractor",
    description="FastAPI backend for the payment-advice extraction and deduction-interpretation PoC",
    version="1.0.0",
    lifespan=lifespan,
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
            "http://127.0.0.1:5173",
            "http://127.0.0.1:5174",
            "http://127.0.0.1:5175",
            "http://127.0.0.1:5176",
            "http://127.0.0.1:5177",
            # "*"
        ]
    )

# Add CORS middleware for cross-origin requests from UI
app.add_middleware(email_ingestion.IntakeSizeLimit)
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
    return HealthResponse.healthy("api")


# Include routers
app.include_router(joule.router, prefix="/api/joule", tags=["joule"])
app.include_router(payment_advice.router, prefix="/api/payment-advice", tags=["payment-advice"])
app.include_router(email_ingestion.router, prefix="/api/email-ingestion", tags=["email-ingestion"])
app.include_router(email_ingestion.advices, prefix="/api/payment-advice/advices", tags=["advice-review"])
app.include_router(s4.router, prefix="/api/payment-advice/advices", tags=["s4"])


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

    uvicorn.run("app.main:app", host=host, port=port, log_level="info")
