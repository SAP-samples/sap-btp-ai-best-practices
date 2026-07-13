import logging
import time

from fastapi import APIRouter, Depends, Request

# LangGraph / LangChain imports for Agent Demo
from langgraph.graph import MessagesState, StateGraph, START
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import HumanMessage, SystemMessage as LCSystemMessage
from ..utils.langgraph.tools import calculator_tool
from ..utils.langgraph.common import make_llm

from ..models.llm import LLMRequest, LLMResponse
from ..observability.llm_usage_logging import (
    TokenUsage,
    emit_llm_usage_event,
    extract_client_host_from_request,
    extract_token_usage,
    extract_user_id_from_request,
    usage_dict_from_tokens,
)
from ..security import get_api_key

# Setup logging
logger = logging.getLogger(__name__)

router = APIRouter(dependencies=[Depends(get_api_key)])

MODEL_NAME = "gpt-4.1"  # Using GPT-4 as default for this demo


# -------------------------------------------------------------------------
# Agent Demo (Calculator)
# -------------------------------------------------------------------------


def _sum_token_usage(usages: list[TokenUsage]) -> TokenUsage:
    """Return aggregate token usage for all LLM calls in one agent run.

    Args:
        usages: Per-call token usage values collected by the assistant node.

    Returns:
        TokenUsage: Sum of input, output, and total token values.
    """
    return TokenUsage(
        input_tokens=sum(usage.input_tokens for usage in usages),
        output_tokens=sum(usage.output_tokens for usage in usages),
        total_tokens=sum(usage.total_tokens for usage in usages),
    )


def _build_agent_graph(http_request: Request, usage_events: list[TokenUsage]):
    """Build a minimal ReAct-style graph with just the calculator tool.

    Args:
        http_request: FastAPI request used to enrich each LLM usage event.
        usage_events: Mutable list where each assistant LLM call records tokens.

    Returns:
        CompiledStateGraph: LangGraph application ready for invocation.
    """

    # Create LLM using the project's common utility (uses SAP GenAI Hub proxy)
    llm = make_llm(provider="openai", model_name=MODEL_NAME, temperature=0.0)

    tools = [calculator_tool]
    llm_with_tools = llm.bind_tools(tools)

    graph = StateGraph(MessagesState)

    sys_msg = LCSystemMessage(
        content=(
            "You are a helpful assistant that can use a calculator tool. "
            "If the user asks a math question, use the calculator. "
            "Otherwise, just answer."
        )
    )

    async def assistant(state: MessagesState):
        """Call the bound LLM and emit one usage event for this graph node."""
        usage = TokenUsage()
        start_time = time.perf_counter()
        route = str(http_request.url.path)
        method = http_request.method
        user_id = extract_user_id_from_request(http_request)
        client_host = extract_client_host_from_request(http_request)
        correlation_id = http_request.headers.get("x-correlation-id")

        try:
            response = await llm_with_tools.ainvoke([sys_msg] + state["messages"])
            usage = extract_token_usage(response)
            usage_events.append(usage)
            emit_llm_usage_event(
                route=route,
                method=method,
                user_id=user_id,
                client_host=client_host,
                provider="sap-ai-core",
                model=MODEL_NAME,
                llm_endpoint="chat.completions",
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
                outcome="success",
                latency_ms=int((time.perf_counter() - start_time) * 1000),
                correlation_id=correlation_id,
            )
            return {"messages": [response]}
        except Exception:
            usage_events.append(usage)
            emit_llm_usage_event(
                route=route,
                method=method,
                user_id=user_id,
                client_host=client_host,
                provider="sap-ai-core",
                model=MODEL_NAME,
                llm_endpoint="chat.completions",
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
                outcome="error",
                latency_ms=int((time.perf_counter() - start_time) * 1000),
                correlation_id=correlation_id,
            )
            raise

    graph.add_node("assistant", assistant)
    graph.add_node("tools", ToolNode(tools=tools))

    graph.add_edge(START, "assistant")
    graph.add_conditional_edges("assistant", tools_condition)
    graph.add_edge("tools", "assistant")

    return graph.compile()


@router.post("/chat", response_model=LLMResponse)
async def agent_chat(http_request: Request, request: LLMRequest) -> LLMResponse:
    """Chat endpoint that uses an agent with a calculator tool.

    Args:
        http_request: FastAPI request used to enrich Cloud Logging events.
        request: User prompt and generation settings for the agent.

    Returns:
        LLMResponse: Final agent answer, model name, success flag, and tokens.
    """
    usage_events: list[TokenUsage] = []
    try:
        logger.info(f"Agent demo request: {request.message[:50]}...")

        app = _build_agent_graph(http_request, usage_events)

        # Invoke the graph with the user message
        # LangGraph expects a list of LangChain messages
        messages = [HumanMessage(content=request.message)]

        result = await app.ainvoke({"messages": messages})

        # Extract the final response
        # The result["messages"] contains the full conversation history
        # We want the last message content
        final_message = result["messages"][-1]
        text = final_message.content
        usage = _sum_token_usage(usage_events)

        return LLMResponse(
            text=text,
            model=MODEL_NAME,
            success=True,
            usage=usage_dict_from_tokens(usage),
        )

    except Exception as e:
        logger.error(f"Agent demo error: {e}")
        usage = _sum_token_usage(usage_events)
        return LLMResponse(
            text="",
            model=MODEL_NAME,
            success=False,
            usage=usage_dict_from_tokens(usage),
            error=str(e),
        )
