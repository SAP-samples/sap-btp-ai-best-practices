"""Expose the existing agent runtime over A2A 1.0 and Joule-compatible 0.3.

Examples:
    .venv/bin/python -m template_agent serve
    curl -s http://localhost:8080/.well-known/agent.json
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Protocol, cast

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.jsonrpc_models import JSONRPCError
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.request_handlers.request_handler import RequestHandler
from a2a.server.routes import create_agent_card_routes
from a2a.server.routes.jsonrpc_dispatcher import JsonRpcDispatcher
from a2a.server.tasks import InMemoryTaskStore, TaskUpdater
from a2a.compat.v0_3.jsonrpc_adapter import JSONRPC03Adapter
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentInterface,
    AgentSkill,
    Part,
    Task,
    TaskState,
    TaskStatus,
)
from a2a.utils.errors import A2AError, JSON_RPC_ERROR_CODE_MAP, InvalidParamsError
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route

from .config import A2ASettings, load_config


class RuntimeResult(Protocol):
    """Describe the final text returned by an agent runtime invocation."""

    output_text: str


class A2ARuntime(Protocol):
    """Describe the runtime operations required by the A2A adapter."""

    async def ainvoke(
        self,
        text: str,
        context_id: str,
        attachments: tuple[Any, ...] = (),
        response_model: type[Any] | dict[str, Any] | None = None,
    ) -> RuntimeResult:
        """Run one request and return final response text."""

        ...

    async def aclose(self) -> None:
        """Release resources owned by the runtime."""

        ...


RuntimeFactory = Callable[[str | Path], Awaitable[A2ARuntime]]


class _JouleJSONRPC03Adapter(JSONRPC03Adapter):
    """Preserve A2A error codes across the SDK's v0.3 compatibility bridge."""

    async def _process_non_streaming_request(
        self,
        request_id: str | int | None,
        request_obj: Any,
        context: Any,
    ) -> JSONResponse:
        """Translate v1 A2A exceptions before the SDK collapses them to -32603."""

        try:
            return await super()._process_non_streaming_request(
                request_id,
                request_obj,
                context,
            )
        except A2AError as error:
            jsonrpc_error = JSONRPCError(
                code=JSON_RPC_ERROR_CODE_MAP.get(type(error), -32603),
                message=str(error),
                data=error.data,
            )
            return self._generate_error_response(request_id, jsonrpc_error)


class _DualProtocolJsonRpcDispatcher(JsonRpcDispatcher):
    """Use the current dispatcher with corrected Joule v0.3 error mapping."""

    def __init__(self, request_handler: RequestHandler) -> None:
        """Initialize both protocol paths against one request handler."""

        super().__init__(request_handler, enable_v0_3_compat=True)
        self._v03_adapter = _JouleJSONRPC03Adapter(
            http_handler=request_handler,
            context_builder=self._context_builder,
        )


class AgentExecutorAdapter(AgentExecutor):
    """Translate A2A task events to one existing `AgentRuntime.ainvoke` call."""

    def __init__(self, runtime_getter: Callable[[], A2ARuntime]) -> None:
        """Store a getter for the lifespan-owned runtime.

        Args:
            runtime_getter: Callback returning the initialized agent runtime.
        """

        self._runtime_getter = runtime_getter

    async def execute(
        self, context: RequestContext, event_queue: EventQueue
    ) -> None:
        """Execute one text request using the SDK-required task event order."""

        message = context.message
        task_id = context.task_id
        context_id = context.context_id
        text = context.get_user_input().strip()
        if not message or not task_id or not context_id:
            raise InvalidParamsError("A2A request is missing message or task context")
        if any(not part.HasField("text") for part in message.parts):
            raise InvalidParamsError("A2A boundary accepts text parts only")
        if not text:
            raise InvalidParamsError("A2A message must contain non-empty text")

        task = context.current_task or Task(
            id=task_id,
            context_id=context_id,
            status=TaskStatus(state=TaskState.TASK_STATE_SUBMITTED),
            history=[message],
        )
        # A2A 1.0 requires the Task to be the first event in a task stream.
        await event_queue.enqueue_event(task)
        updater = TaskUpdater(event_queue, task_id, context_id)
        await updater.start_work(
            message=updater.new_agent_message(parts=[Part(text="Processing request")])
        )
        try:
            result = await self._runtime_getter().ainvoke(text, context_id)
        except Exception:
            failure = updater.new_agent_message(
                parts=[Part(text="Agent request failed")]
            )
            await updater.failed(message=failure)
            return

        await updater.add_artifact(
            parts=[Part(text=result.output_text)],
            name="agent_result",
            last_chunk=True,
        )
        await updater.complete()

    async def cancel(
        self, context: RequestContext, event_queue: EventQueue
    ) -> None:
        """Publish cancellation for the task selected by the request handler."""

        if not context.task_id or not context.context_id:
            raise InvalidParamsError("A2A cancellation is missing task context")
        updater = TaskUpdater(event_queue, context.task_id, context.context_id)
        await updater.cancel()


def create_agent_card(settings: A2ASettings) -> AgentCard:
    """Build one card advertising current and Joule-compatible JSON-RPC."""

    if not settings.enabled or not settings.public_url:
        raise ValueError("A2A must be enabled with a public URL")
    interfaces = [
        AgentInterface(
            protocol_binding="JSONRPC",
            protocol_version="1.0",
            url=settings.public_url,
        ),
        AgentInterface(
            protocol_binding="JSONRPC",
            protocol_version="0.3",
            url=settings.public_url,
        ),
    ]
    return AgentCard(
        name=settings.name,
        description=settings.description,
        version=settings.version,
        capabilities=AgentCapabilities(streaming=False, push_notifications=False),
        default_input_modes=["text/plain"],
        default_output_modes=["text/plain"],
        skills=[
            AgentSkill(
                id="chat",
                name=settings.name,
                description=settings.description,
                tags=["langgraph", "mcp"],
                examples=["What can you help me with?"],
                input_modes=["text/plain"],
                output_modes=["text/plain"],
            )
        ],
        supported_interfaces=interfaces,
    )


async def _create_default_runtime(config_path: str | Path) -> A2ARuntime:
    """Create the concrete runtime lazily to keep this adapter testable."""

    from .runtime import AgentRuntime

    return cast(A2ARuntime, await AgentRuntime.create(config_path))


def create_a2a_app(
    config_path: str | Path,
    runtime_factory: RuntimeFactory | None = None,
) -> Starlette:
    """Create a lifespan-managed dual-protocol Starlette application.

    Args:
        config_path: YAML configuration passed to the agent runtime.
        runtime_factory: Optional async factory used by tests or custom runtimes.

    Returns:
        An ASGI app serving both agent-card paths and JSON-RPC at `/`.
    """

    config = load_config(config_path)
    if not config.a2a.enabled:
        raise ValueError("A2A is disabled in the agent configuration")
    card = create_agent_card(config.a2a)
    factory = runtime_factory or _create_default_runtime
    runtime_holder: dict[str, A2ARuntime] = {}

    def get_runtime() -> A2ARuntime:
        """Return the runtime initialized by the application lifespan."""

        try:
            return runtime_holder["runtime"]
        except KeyError as exc:
            raise RuntimeError("A2A runtime is not initialized") from exc

    executor = AgentExecutorAdapter(get_runtime)
    handler = DefaultRequestHandler(
        agent_executor=executor,
        task_store=InMemoryTaskStore(),
        agent_card=card,
    )

    @asynccontextmanager
    async def lifespan(app: Starlette) -> AsyncIterator[None]:
        """Create and close the shared runtime with the ASGI application."""

        runtime = await factory(config_path)
        runtime_holder["runtime"] = runtime
        app.state.agent_runtime = runtime
        try:
            yield
        finally:
            await handler.aclose()
            await runtime.aclose()
            runtime_holder.clear()

    dispatcher = _DualProtocolJsonRpcDispatcher(handler)
    routes = [
        *create_agent_card_routes(card),
        *create_agent_card_routes(card, card_url="/.well-known/agent.json"),
        Route(path="/", endpoint=dispatcher.handle_requests, methods=["POST"]),
    ]
    return Starlette(routes=routes, lifespan=lifespan)
