"""A2A server adapter for the assessment knowledge LangGraph agent."""

from __future__ import annotations

import json
import logging
import os
from uuid import uuid4

from starlette.responses import JSONResponse

from app.services.joule_knowledge_agent import (
    JouleKnowledgeGraphAgent,
    build_joule_message_repository,
)

logger = logging.getLogger(__name__)

HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))


def validate_a2a_api_key(api_key_header: str | None) -> tuple[int, str] | None:
    """Validate an ``X-API-Key`` header for mounted A2A traffic.

    Inputs:
        api_key_header: Header value supplied by the A2A caller.

    Outputs:
        tuple[int, str] | None: ``None`` when the key is valid, otherwise an
        HTTP status code and response detail.
    """

    api_key = os.getenv("API_KEY")
    if not api_key:
        return (500, "API key not configured on server")
    if api_key_header != api_key:
        return (403, "Could not validate credentials")
    return None


class A2AApiKeyMiddleware:
    """ASGI middleware that protects the mounted A2A app with ``X-API-Key``.

    Inputs:
        app: ASGI application returned by the A2A SDK.

    Outputs:
        Middleware object that forwards authorized requests and rejects missing
        or invalid API keys.
    """

    def __init__(self, app) -> None:
        """Store the wrapped ASGI app.

        Inputs:
            app: ASGI application to protect.

        Outputs:
            None.
        """

        self.app = app

    async def __call__(self, scope, receive, send) -> None:
        """Validate A2A HTTP requests before forwarding them.

        Inputs:
            scope: ASGI connection scope.
            receive: ASGI receive callable.
            send: ASGI send callable.

        Outputs:
            None. The wrapped app or an error response writes to ``send``.
        """

        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return

        headers = {
            key.decode("latin-1").lower(): value.decode("latin-1")
            for key, value in scope.get("headers", [])
        }
        validation_error = validate_a2a_api_key(headers.get("x-api-key"))
        if validation_error is not None:
            status_code, detail = validation_error
            response = JSONResponse({"detail": detail}, status_code=status_code)
            await response(scope, receive, send)
            return

        await self.app(scope, receive, send)


def resolve_public_a2a_url(host: str = HOST, port: int = PORT) -> str:
    """Resolve the public A2A endpoint URL for the agent card.

    Inputs:
        host: Local host used when no public URL can be inferred.
        port: Local port used when no public URL can be inferred.

    Outputs:
        str: URL ending with ``/a2a/`` for local or Cloud Foundry deployments.
    """

    public_url = os.getenv("AGENT_PUBLIC_URL")
    if public_url:
        return public_url.rstrip("/") + "/a2a/"

    api_base_url = os.getenv("API_BASE_URL")
    if api_base_url:
        return api_base_url.rstrip("/") + "/a2a/"

    vcap_application = os.getenv("VCAP_APPLICATION")
    if vcap_application:
        try:
            uris = json.loads(vcap_application).get("application_uris", [])
            if uris:
                return f"https://{uris[0].rstrip('/')}/a2a/"
        except (TypeError, ValueError):
            logger.warning("Unable to parse VCAP_APPLICATION for A2A route discovery")

    return f"http://{host}:{port}/a2a/"


def build_a2a_app(host: str = HOST, port: int = PORT):
    """Build the mounted A2A Starlette application.

    Inputs:
        host: Host used for local agent-card fallback URL generation.
        port: Port used for local agent-card fallback URL generation.

    Outputs:
        Starlette: A2A application suitable for mounting at ``/a2a``.

    Raises:
        ImportError: Raised when ``a2a-sdk`` is not installed. The FastAPI app
        catches this in local development so non-A2A tests can still run.
    """

    from a2a.server.agent_execution import AgentExecutor as BaseAgentExecutor
    from a2a.server.apps import A2AStarletteApplication
    from a2a.server.events import EventQueue
    from a2a.server.request_handlers import DefaultRequestHandler
    from a2a.server.tasks import InMemoryTaskStore, TaskUpdater
    from a2a.types import (
        AgentCapabilities,
        AgentCard,
        AgentSkill,
        InternalError,
        Part,
        TaskState,
        TextPart,
        UnsupportedOperationError,
    )
    from a2a.utils import new_agent_text_message, new_task
    from a2a.utils.errors import ServerError

    class AssessmentKnowledgeAgentExecutor(BaseAgentExecutor):
        """A2A executor that delegates each turn to the LangGraph runtime."""

        def __init__(self) -> None:
            """Initialize the executor without opening HANA connections.

            Inputs:
                None.

            Outputs:
                None. The wrapped agent is created lazily on first execution so
                importing the FastAPI app does not connect to HANA.
            """

            self.agent: JouleKnowledgeGraphAgent | None = None

        def _agent(self) -> JouleKnowledgeGraphAgent:
            """Return the lazily initialized LangGraph agent.

            Inputs:
                None.

            Outputs:
                JouleKnowledgeGraphAgent: Agent with HANA-backed message
                persistence configured for A2A execution.
            """

            if self.agent is None:
                self.agent = JouleKnowledgeGraphAgent(
                    message_repository=build_joule_message_repository()
                )
            return self.agent

        async def execute(self, context, event_queue: EventQueue) -> None:
            """Execute one A2A message/send request.

            Inputs:
                context: A2A request context from the SDK.
                event_queue: Queue used to emit task status and artifacts.

            Outputs:
                None. Responses are sent through the event queue.
            """

            query = context.get_user_input()
            task = context.current_task
            if not task:
                task = new_task(context.message)
                await event_queue.enqueue_event(task)

            context_id = task.context_id or uuid4().hex
            updater = TaskUpdater(event_queue, task.id, context_id)
            try:
                await updater.update_status(
                    TaskState.working,
                    new_agent_text_message(
                        "Searching assessment knowledge resources...",
                        context_id,
                        task.id,
                    ),
                )
                response = await self._agent().answer(query, context_id)
                await updater.add_artifact(
                    [Part(root=TextPart(text=response.message))],
                    name="agent_result",
                )
                await updater.complete()
            except Exception as exc:
                logger.exception("A2A assessment knowledge agent execution failed")
                raise ServerError(error=InternalError()) from exc

        async def cancel(self, context, event_queue: EventQueue) -> None:
            """Reject A2A cancellation because this PoC runs synchronously.

            Inputs:
                context: A2A request context.
                event_queue: A2A event queue.

            Outputs:
                None.

            Raises:
                ServerError: Always raised with unsupported operation.
            """

            raise ServerError(error=UnsupportedOperationError())

    skill = AgentSkill(
        id="assessment-knowledge-agent",
        name="assessment-knowledge-agent",
        description=(
            "Explains assessment glossary terms, questions, and dimensions "
            "from HANA-backed knowledge resources."
        ),
        tags=["glossary", "assessment", "knowledge"],
        examples=[
            "What does AI mean?",
            "Why is Q.STR.01.01 important?",
            "Explain the Strategy dimension.",
            "Che cosa significa AI?",
        ],
    )
    agent_card = AgentCard(
        name="assessment-knowledge-agent",
        description=(
            "HANA-backed bilingual assessment knowledge agent for glossary terms, "
            "assessment question explanations, and dimension explanations."
        ),
        url=resolve_public_a2a_url(host, port),
        version="1.0.0",
        defaultInputModes=["text", "text/plain"],
        defaultOutputModes=["text", "text/plain"],
        capabilities=AgentCapabilities(streaming=True, pushNotifications=False),
        skills=[skill],
    )
    server = A2AStarletteApplication(
        agent_card=agent_card,
        http_handler=DefaultRequestHandler(
            agent_executor=AssessmentKnowledgeAgentExecutor(),
            task_store=InMemoryTaskStore(),
        ),
    )
    return A2AApiKeyMiddleware(server.build())
