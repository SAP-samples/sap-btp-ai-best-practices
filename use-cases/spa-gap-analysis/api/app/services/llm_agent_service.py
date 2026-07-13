"""
LLM Agent Service for SPA Gap Analysis.

Orchestrates LLM-powered conversations using LangChain tool calling.
Uses SAP AI Core / GenAI Hub models only: Gemini 2.5 Flash primary with
Gemini 2.5 Pro fallback.
"""
import logging
import re
import time
from typing import Any, List, Dict, Optional
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from app.utils.langgraph.common import make_llm
from app.observability.llm_usage_logging import emit_llm_usage_event, extract_token_counts
from app.services.spa_tools import (
    analyze_customer_tool,
    search_customers_tool,
    find_similar_customers_tool,
    get_rfm_distribution_tool,
    research_customer_onboarding_tool
)

logger = logging.getLogger(__name__)


def _is_proxy_credentials_error(exc: Exception) -> bool:
    """Detect SAP GenAI Hub proxy wrapper credential failures."""
    text = str(exc)
    return (
        "GenAIHubProxyClient" in text
        or "No credentials found in any source" in text
        or "ChatBedrock" in text and "No credentials found" in text
    )


def _extract_customer_id_from_message(user_message: str, conversation_context: Dict) -> Optional[str]:
    """Extract a customer ID from the user message or fall back to conversation context."""
    match = re.search(r"\bcustomer\s+(\d{2,})\b", user_message, flags=re.IGNORECASE)
    if not match:
        match = re.search(r"\b(\d{2,})\b", user_message)
    if match:
        return match.group(1)
    return conversation_context.get("last_analyzed_customer_id")


def _process_with_deterministic_tools(
    user_message: str,
    conversation_context: Dict,
) -> Dict:
    """
    Fallback router for AI Core proxy credential failures.

    It keeps Agent Chat useful for canonical SPA tasks even when the optional
    LangChain proxy tool-calling layer cannot initialize in Cloud Foundry.
    """
    text = (user_message or "").strip()
    text_lower = text.lower()
    tool_calls = []
    tool_results = []

    if "rfm" in text_lower and "customer" not in text_lower:
        args = {}
        result = get_rfm_distribution_tool.invoke(args)
        tool_calls.append({"name": "get_rfm_distribution_tool", "args": args})
        tool_results.append(result)
        message = format_tool_results(tool_results, tool_calls[0])
        return {
            "message": message,
            "tool_calls": tool_calls,
            "tool_results": tool_results,
            "context": conversation_context,
        }

    customer_id = _extract_customer_id_from_message(text, conversation_context)

    if customer_id and (
        "similar" in text_lower
        or "peer" in text_lower
        or "compare" in text_lower and "missing" not in text_lower
    ):
        args = {
            "customer_id": customer_id,
            "top_n": 10,
            "exclude_unknown": bool(conversation_context.get("exclude_unknown", False)),
        }
        result = find_similar_customers_tool.invoke(args)
        tool_calls.append({"name": "find_similar_customers_tool", "args": args})
        tool_results.append(result)
        message = format_tool_results(tool_results, tool_calls[0])
        return {
            "message": message,
            "tool_calls": tool_calls,
            "tool_results": tool_results,
            "context": conversation_context,
        }

    if customer_id and any(
        marker in text_lower
        for marker in ["analyze", "analysis", "missing", "spa", "customer", "their"]
    ):
        args = {"customer_id": customer_id}
        result = analyze_customer_tool.invoke(args)
        if "error" not in result:
            conversation_context["last_analyzed_customer_id"] = customer_id
            conversation_context["last_analyzed_customer_name"] = result.get("customer_name", "Unknown")
        tool_calls.append({"name": "analyze_customer_tool", "args": args})
        tool_results.append(result)
        message = format_tool_results(tool_results, tool_calls[0])
        return {
            "message": message,
            "tool_calls": tool_calls,
            "tool_results": tool_results,
            "context": conversation_context,
        }

    if any(marker in text_lower for marker in ["find", "search", "lookup", "look up"]):
        query = re.sub(r"\b(find|search|lookup|look up|customer|customers|for)\b", " ", text, flags=re.IGNORECASE)
        query = re.sub(r"\s+", " ", query).strip(" :,-")
        args = {"query": query or text, "limit": 10}
        result = search_customers_tool.invoke(args)
        tool_calls.append({"name": "search_customers_tool", "args": args})
        tool_results.append(result)
        message = format_tool_results(tool_results, tool_calls[0])
        return {
            "message": message,
            "tool_calls": tool_calls,
            "tool_results": tool_results,
            "context": conversation_context,
        }

    return {
        "message": (
            "The live LLM routing layer is unavailable, but I can still run canonical SPA tools. "
            "Please ask for a specific action such as 'Analyze customer <customer_id>', "
            "'Show similar customers to <customer_id>', or 'Show RFM distribution'."
        ),
        "tool_calls": [],
        "tool_results": [],
        "context": conversation_context,
    }

# System prompt for SPA Gap Analysis Agent
SYSTEM_PROMPT = """You are a helpful SPA (Special Pricing Agreement) Gap Analysis assistant.

Your role is to help users:
1. Find customers by name or location (supports fuzzy matching)
2. Analyze customers for missing SPAs and recommend agreements they should have
3. Find similar customers to understand peer patterns
4. Get RFM segment distribution and customer insights

**Available Tools:**
- analyze_customer_tool: Analyze a customer for SPA gaps and missing opportunities
- search_customers_tool: Search customers by name (fuzzy matching), state, RFM segment, or sales office
- find_similar_customers_tool: Find customers similar to target customer
- get_rfm_distribution_tool: Get RFM segment distribution across all customers
- research_customer_onboarding_tool: Research new customer profile using Sonar-Pro and find similar existing customers

**Context Awareness:**
{context_info}

**Instructions:**
- Use tools when users ask for customer analysis, search, or data retrieval
- For follow-up questions like "What are their missing SPAs?" or "Show me their profile", use the customer_id from context
- Extract parameters from natural language:
  * "Utah" or "UT" → state="UT"
  * "Arizona" or "AZ" → state="AZ"
  * "high RFM" or "Champions" → rfm_segment="Champions"
  * "TEST ELECTRIC" → query="TEST ELECTRIC" (fuzzy match will find similar names)
- **For "Compare" requests:** Call both analyze_customer_tool AND find_similar_customers_tool, then provide comparative insights (what SPAs they have vs similar customers, key differences)
- **For onboarding/research requests:** When calling research_customer_onboarding_tool or find_similar_customers_tool for new customer research, ALWAYS set exclude_unknown=True to filter out customers without proper names
- **For specific questions about segments:** When asked "How many Champions?" focus answer on Champions segment, not full distribution
- Format responses clearly with markdown:
  * Use **bold** for emphasis
  * Use bullet points for lists
  * Use emojis for visual appeal (📍 location, 📊 stats, ⚠️ warnings, ✅ success)
- If no customer context exists for follow-up questions, politely ask for customer ID
- When reporting missing SPAs, highlight confidence scores and why they're recommended

Be concise, helpful, and professional!"""


def build_context_info(conversation_context: Dict) -> str:
    """
    Build context info string for system prompt injection.

    Args:
        conversation_context: Dict with last_analyzed_customer_id, exclude_unknown, etc.

    Returns:
        Context info string for system prompt
    """
    context_parts = []

    if 'last_analyzed_customer_id' in conversation_context:
        cust_id = conversation_context['last_analyzed_customer_id']
        cust_name = conversation_context.get('last_analyzed_customer_name', 'Unknown')
        context_parts.append(f"Currently discussing Customer {cust_id} ({cust_name}). Use this customer for follow-up questions.")
    else:
        context_parts.append("No customer context yet. Ask user for customer ID if they reference 'they/their/this customer'.")

    # Add exclude_unknown filter setting
    exclude_unknown = conversation_context.get('exclude_unknown', False)
    logger.info(f"[FILTER DEBUG] Context exclude_unknown: {exclude_unknown}")
    if exclude_unknown:
        context_parts.append("User has enabled 'Only show customers with names' filter. IMPORTANT: Pass exclude_unknown=True to ALL find_similar_customers_tool and research_customer_onboarding_tool calls to filter out customers with 'Unknown' names.")
        logger.info(f"[FILTER DEBUG] Added filter instruction to system prompt")

    return " ".join(context_parts)


def create_llm_with_tools(
    model_name: str = "gemini-2.5-flash",
    temperature: float = 0.2
):
    """
    Create LLM with tools bound.

    Args:
        model_name: "gemini-2.5-flash", "gemini-2.5-pro", "sonar-pro"
        temperature: LLM temperature (0.0-1.0)

    Returns:
        LLM with tools bound
    """
    # Determine provider based on model name
    if "gemini" in model_name:
        provider = "google"
    elif "anthropic" in model_name or "claude" in model_name:
        raise ValueError(
            f"Model {model_name} is not supported in this AI Core-only deployment. "
            "Use Gemini, GPT, or Sonar models configured through SAP AI Core / GenAI Hub."
        )
    elif "gpt" in model_name or "sonar" in model_name:
        provider = "openai"
    else:
        raise ValueError(f"Cannot determine provider for model: {model_name}")

    # Create LLM
    llm = make_llm(
        provider=provider,
        model_name=model_name,
        temperature=temperature,
        max_tokens=2048
    )

    # Bind tools
    tools = [
        analyze_customer_tool,
        search_customers_tool,
        find_similar_customers_tool,
        get_rfm_distribution_tool,
        research_customer_onboarding_tool
    ]

    llm_with_tools = llm.bind_tools(tools)
    return llm_with_tools


async def process_with_llm(
    user_message: str,
    conversation_history: List[Dict],
    conversation_context: Dict,
    model_name: str = "gemini-2.5-flash",
    request: Optional[Any] = None,
    route: str = "internal:llm_agent_service.process_with_llm",
    method: str = "INTERNAL",
    correlation_id: Optional[str] = None,
    actor_type: Optional[str] = None,
) -> Dict:
    """
    Process user message with LLM and execute tool calls.

    Args:
        user_message: User's input message
        conversation_history: List of previous messages [{"role": "user/assistant", "message": str}]
        conversation_context: Context dict with last_analyzed_customer_id, etc.
        model_name: LLM model to use ("gemini-2.5-flash" or "gemini-2.5-pro")

    Returns:
        Dict with:
        - message: Response message text
        - tool_calls: List of tool calls made
        - tool_results: List of tool execution results
        - context: Updated conversation context
    """
    try:
        # Build messages
        context_info = build_context_info(conversation_context)
        system_msg = SystemMessage(content=SYSTEM_PROMPT.format(context_info=context_info))

        messages = [system_msg]

        # Add conversation history (exclude current user message)
        for msg in conversation_history:
            if msg['role'] == 'user':
                messages.append(HumanMessage(content=msg['message']))
            elif msg['role'] == 'assistant':
                messages.append(AIMessage(content=msg['message']))

        # Add current user message
        messages.append(HumanMessage(content=user_message))

        # Get LLM with tools
        llm_with_tools = create_llm_with_tools(model_name)

        # Invoke LLM
        logger.info(f"Invoking LLM {model_name} with {len(messages)} messages")
        start = time.perf_counter()
        try:
            response = await llm_with_tools.ainvoke(messages)
            latency_ms = int((time.perf_counter() - start) * 1000)
            input_tokens, output_tokens, total_tokens = extract_token_counts(response)
            emit_llm_usage_event(
                route=route,
                method=method,
                request=request,
                actor_type=actor_type,
                correlation_id=correlation_id,
                provider="sap-ai-core",
                model=model_name,
                llm_endpoint="ainvoke",
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=total_tokens,
                outcome="success",
                latency_ms=latency_ms,
            )
        except Exception:
            latency_ms = int((time.perf_counter() - start) * 1000)
            emit_llm_usage_event(
                route=route,
                method=method,
                request=request,
                actor_type=actor_type,
                correlation_id=correlation_id,
                provider="sap-ai-core",
                model=model_name,
                llm_endpoint="ainvoke",
                input_tokens=0,
                output_tokens=0,
                total_tokens=0,
                outcome="error",
                latency_ms=latency_ms,
            )
            raise

        # Check if tool calls requested
        if hasattr(response, 'tool_calls') and response.tool_calls:
            logger.info(f"LLM requested {len(response.tool_calls)} tool calls")
            tool_results = []

            for tool_call in response.tool_calls:
                tool_name = tool_call['name']
                tool_args = tool_call['args']
                logger.info(f"Executing tool: {tool_name} with args: {tool_args}")

                # Execute tool
                if tool_name == 'analyze_customer_tool':
                    result = analyze_customer_tool.invoke(tool_args)
                    # Update context with analyzed customer
                    if 'error' not in result:
                        conversation_context['last_analyzed_customer_id'] = tool_args['customer_id']
                        conversation_context['last_analyzed_customer_name'] = result.get('customer_name', 'Unknown')
                    tool_results.append(result)

                elif tool_name == 'search_customers_tool':
                    result = search_customers_tool.invoke(tool_args)
                    tool_results.append(result)

                elif tool_name == 'find_similar_customers_tool':
                    logger.info(f"[FILTER DEBUG] find_similar_customers_tool args: {tool_args}")
                    # Auto-inject exclude_unknown from context if not set by LLM
                    if 'exclude_unknown' not in tool_args and conversation_context.get('exclude_unknown', False):
                        tool_args['exclude_unknown'] = True
                        logger.info(f"[FILTER DEBUG] Auto-injected exclude_unknown=True from context")
                    logger.info(f"[FILTER DEBUG] exclude_unknown in args: {tool_args.get('exclude_unknown', 'NOT SET')}")
                    result = find_similar_customers_tool.invoke(tool_args)
                    tool_results.append(result)

                elif tool_name == 'get_rfm_distribution_tool':
                    result = get_rfm_distribution_tool.invoke(tool_args)
                    tool_results.append(result)

                elif tool_name == 'research_customer_onboarding_tool':
                    logger.info(f"[FILTER DEBUG] research_customer_onboarding_tool args: {tool_args}")
                    # Auto-inject exclude_unknown from context if not set by LLM
                    if 'exclude_unknown' not in tool_args and conversation_context.get('exclude_unknown', False):
                        tool_args['exclude_unknown'] = True
                        logger.info(f"[FILTER DEBUG] Auto-injected exclude_unknown=True from context")
                    logger.info(f"[FILTER DEBUG] exclude_unknown in args: {tool_args.get('exclude_unknown', 'NOT SET')}")
                    result = research_customer_onboarding_tool.invoke(tool_args)
                    tool_results.append(result)

                else:
                    logger.warning(f"Unknown tool requested: {tool_name}")
                    tool_results.append({"error": f"Unknown tool: {tool_name}"})

            # Format results into response message
            response_message = format_tool_results(tool_results, tool_call=response.tool_calls[0])

            return {
                "message": response_message,
                "tool_calls": response.tool_calls,
                "tool_results": tool_results,
                "context": conversation_context
            }

        # No tool calls - direct response from LLM
        logger.info("LLM provided direct response (no tool calls)")

        # Ensure message is a string (response.content might be a list of content blocks)
        if isinstance(response.content, list):
            # Extract text from content blocks
            message_text = ""
            for block in response.content:
                if isinstance(block, dict) and 'text' in block:
                    message_text += block['text']
                elif hasattr(block, 'text'):
                    message_text += block.text
                else:
                    message_text += str(block)
        else:
            message_text = str(response.content)

        return {
            "message": message_text,
            "tool_calls": [],
            "tool_results": [],
            "context": conversation_context
        }

    except Exception as e:
        logger.error(f"LLM processing error with {model_name}: {e}", exc_info=True)

        if _is_proxy_credentials_error(e):
            logger.warning(
                "GenAI Hub proxy credentials unavailable for %s; using deterministic SPA tool routing",
                model_name,
            )
            return _process_with_deterministic_tools(user_message, conversation_context)

        # Fallback stays inside SAP AI Core / GenAI Hub. Do not fall back to
        # Bedrock/Claude here because this environment has no AWS Bedrock
        # credentials and all model access is routed through AI Core.
        if model_name == "gemini-2.5-flash":
            logger.info("Falling back to Gemini 2.5 Pro via SAP AI Core due to Gemini Flash error")
            return await process_with_llm(
                user_message,
                conversation_history,
                conversation_context,
                model_name="gemini-2.5-pro",
                request=request,
                route=route,
                method=method,
                correlation_id=correlation_id,
                actor_type=actor_type,
            )

        # If fallback also fails, return error
        raise


def format_multiple_tool_results(tool_results: List[Dict]) -> str:
    """
    Format multiple tool results (e.g., analyze + find_similar for compare).

    Args:
        tool_results: List of tool execution results

    Returns:
        Combined formatted message
    """
    lines = []

    # Check if we have analyze + similar (comparison case)
    has_analysis = any('customer_id' in r and 'missing_spas' in r for r in tool_results)
    has_similar = any('similar_customers' in r for r in tool_results)

    if has_analysis and has_similar:
        # Extract results
        analysis = next((r for r in tool_results if 'missing_spas' in r), None)
        similar = next((r for r in tool_results if 'similar_customers' in r), None)

        if analysis and similar:
            # Format comparison
            lines.append(f"**📊 Comparison: Customer {analysis['customer_id']} vs Similar Customers**\n")
            lines.append(f"**Target Customer:** {analysis['customer_name']}")
            lines.append(f"📍 Location: {analysis.get('city', 'N/A')}, {analysis.get('state', 'N/A')}")
            lines.append(f"🏢 Sales Office: {analysis['sales_office']}")
            lines.append(f"📂 Type: {analysis['customer_type']}")
            lines.append(f"📊 Current SPAs: {analysis['current_spa_count']}\n")

            # Similar customers stats
            similar_count = similar.get('count', 0)
            lines.append(f"**🔍 Analyzed {similar_count} Similar Customers:**")

            top_5_similar = similar.get('similar_customers', [])[:5]
            for i, cust in enumerate(top_5_similar, 1):
                name = cust.get('customer_name', 'Unknown')
                similarity = cust.get('similarity_score', 0)
                lines.append(f"{i}. {name} (Similarity: {similarity:.1f}%)")

            # Gap analysis
            missing_count = analysis['missing_spa_count']
            if missing_count > 0:
                lines.append(f"\n⚠️ **Gap Analysis:** Found {missing_count} missing SPA(s)\n")
                lines.append("**Why these SPAs are recommended:**")

                for i, spa in enumerate(analysis['missing_spas'][:3], 1):
                    spa_id = spa.get('sales_deal', 'Unknown')
                    spa_desc = spa.get('description', 'Unknown SPA')
                    percentage = spa.get('percentage_in_similar', 0)
                    count_similar = spa.get('count_in_similar', 0)

                    if percentage >= 80:
                        confidence_emoji = "🟢"
                        confidence = "High"
                    elif percentage >= 50:
                        confidence_emoji = "🟡"
                        confidence = "Medium"
                    else:
                        confidence_emoji = "🔴"
                        confidence = "Low"

                    lines.append(f"\n{i}. **SPA {spa_id}** - {spa_desc}")
                    lines.append(f"   {confidence_emoji} {percentage:.1f}% of similar customers have this ({count_similar}/{similar_count})")
                    lines.append(f"   Confidence: **{confidence}**")
            else:
                lines.append(f"\n✅ **No gaps detected** - this customer has optimal SPA coverage!")

            return "\n".join(lines)

    # Default: format each result separately
    formatted_parts = []
    for result in tool_results:
        if 'customer_id' in result and 'missing_spas' in result:
            formatted_parts.append(format_analysis_result(result))
        elif 'similar_customers' in result:
            formatted_parts.append(format_similar_customers_result(result))
        elif 'customers' in result:
            formatted_parts.append(format_search_result(result))
        elif 'rfm_distribution' in result:
            formatted_parts.append(format_rfm_distribution_result(result))
        elif 'profile' in result and 'business_type' in result:
            formatted_parts.append(format_onboarding_result(result))

    return "\n\n---\n\n".join(formatted_parts)


def format_tool_results(tool_results: List[Dict], tool_call: Dict) -> str:
    """
    Format tool results into user-friendly message.

    Args:
        tool_results: List of tool execution results
        tool_call: Original tool call info

    Returns:
        Formatted message string
    """
    if not tool_results:
        return "No results found."

    # Check for multiple tool results (e.g., analyze + find_similar for compare)
    if len(tool_results) > 1:
        return format_multiple_tool_results(tool_results)

    # Get first result (single tool case)
    result = tool_results[0]
    tool_name = tool_call['name']

    # Check for errors
    if 'error' in result:
        return f"❌ Error: {result['error']}\n\nPlease try again or rephrase your query."

    # Format based on tool type
    if tool_name == 'analyze_customer_tool':
        return format_analysis_result(result)
    elif tool_name == 'search_customers_tool':
        return format_search_result(result)
    elif tool_name == 'find_similar_customers_tool':
        return format_similar_customers_result(result)
    elif tool_name == 'get_rfm_distribution_tool':
        return format_rfm_distribution_result(result)
    elif tool_name == 'research_customer_onboarding_tool':
        return format_onboarding_result(result)
    else:
        return str(result)


def format_analysis_result(result: Dict) -> str:
    """Format customer analysis result"""
    lines = []
    lines.append(f"**Analysis for Customer {result['customer_id']}**")
    lines.append(f"**{result['customer_name']}**\n")
    lines.append(f"📍 Sales Office: {result['sales_office']}")
    lines.append(f"📂 Customer Type: {result['customer_type']}")
    lines.append(f"🎯 RFM Segment: {result.get('rfm_segment', 'N/A')}")
    lines.append(f"📊 Current SPAs: {result['current_spa_count']}")
    lines.append(f"🔍 Analyzed {result['similar_customers_count']} similar customers")

    missing_count = result['missing_spa_count']
    if missing_count > 0:
        lines.append(f"\n⚠️ Found **{missing_count}** missing SPA(s)\n")
        lines.append("**Recommended SPAs:**")
        for i, spa in enumerate(result['missing_spas'][:5], 1):  # Top 5
            # Use correct field names from detect_spa_gaps
            spa_id = spa.get('sales_deal', 'Unknown')
            spa_description = spa.get('description', 'Unknown SPA')
            count_in_similar = spa.get('count_in_similar', 0)
            percentage = spa.get('percentage_in_similar', 0)
            similar_total = result['similar_customers_count']
            vendor = spa.get('vendor', 'N/A')
            spa_type = spa.get('grouping', 'N/A')

            # Calculate confidence level based on percentage
            if percentage >= 80:
                confidence_level = "High"
                confidence_emoji = "🟢"
            elif percentage >= 50:
                confidence_level = "Medium"
                confidence_emoji = "🟡"
            else:
                confidence_level = "Low"
                confidence_emoji = "🔴"

            lines.append(f"\n{i}. **SPA {spa_id}** - {spa_description}")
            lines.append(f"   {confidence_emoji} Confidence: **{confidence_level}** ({percentage:.1f}% of similar customers)")
            lines.append(f"   📊 {count_in_similar}/{similar_total} similar customers have this SPA")
            lines.append(f"   🏢 Vendor: {vendor}")

            # Simplify SPA type display
            if "Blanket" in spa_type:
                lines.append(f"   🏷️ Type: Blanket SPA (easy to add)")
            elif "Customer Specific" in spa_type:
                lines.append(f"   🏷️ Type: Customer-Specific SPA")
            else:
                lines.append(f"   🏷️ Type: {spa_type}")
    else:
        lines.append(f"\n✅ No missing SPAs detected - this customer is well-optimized!")

    return "\n".join(lines)


def format_search_result(result: Dict) -> str:
    """Format customer search result"""
    count = result['count']
    customers = result['customers']

    if count == 0:
        return "No customers found matching your criteria. Try adjusting your search terms."

    lines = []
    lines.append(f"**Found {count} customer(s):**\n")

    for i, cust in enumerate(customers[:10], 1):  # Top 10
        name = cust.get('customer_name', 'Unknown')
        cust_id = cust.get('customer_id', 'N/A')
        city = cust.get('city', 'N/A')
        state = cust.get('state', 'N/A')
        similarity = cust.get('similarity_score')

        line = f"{i}. **{name}** (ID: {cust_id})"
        if city and state:
            line += f" - {city}, {state}"
        if similarity:
            line += f" [Match: {similarity}%]"

        lines.append(line)

    if count > 10:
        lines.append(f"\n_...and {count - 10} more customers_")

    return "\n".join(lines)


def format_similar_customers_result(result: Dict) -> str:
    """Format similar customers result"""
    count = result['count']
    similar = result['similar_customers']

    if count == 0:
        return "No similar customers found."

    lines = []
    lines.append(f"**Found {count} similar customer(s):**\n")

    for i, cust in enumerate(similar[:10], 1):
        name = cust.get('customer_name', 'Unknown')
        cust_id = cust.get('customer_id', 'N/A')
        similarity = cust.get('similarity_score', 0)

        lines.append(f"{i}. **{name}** (ID: {cust_id}) - Similarity: {similarity:.1f}%")

    return "\n".join(lines)


def format_rfm_distribution_result(result: Dict) -> str:
    """Format RFM distribution result"""
    distribution = result['rfm_distribution']
    total = result['total_customers']

    lines = []
    lines.append(f"**RFM Segment Distribution** (Total: {total} customers)\n")

    for segment, data in distribution.items():
        # Skip non-segment keys like 'total_customers'
        if segment == 'total_customers':
            continue

        # Handle both dict format and direct int format
        if isinstance(data, dict):
            count = data.get('count', 0)
        else:
            count = int(data) if data else 0

        pct = (count / total * 100) if total > 0 else 0
        lines.append(f"• **{segment}**: {count} customers ({pct:.1f}%)")

    return "\n".join(lines)


def format_onboarding_result(result: Dict) -> str:
    """Format customer onboarding research result"""
    logger.info(f"[format_onboarding_result] Called with keys: {result.keys()}")
    logger.info(f"[format_onboarding_result] Has 'sources' key: {'sources' in result}")

    if 'error' in result:
        return f"❌ Onboarding research failed: {result['error']}"

    lines = []

    # Header with customer name
    customer_name = result.get('customer_name', 'Unknown Customer')
    location = result.get('location', '')
    location_str = f" in {location}" if location else ""
    lines.append(f"**🎯 New Customer Onboarding Research**")
    lines.append(f"**Company:** {customer_name}{location_str}\n")

    # Expanded Summary stats box
    confidence = result.get('confidence', 0)
    business_type = result.get('business_type', 'unknown')
    materials = result.get('materials_likely_needed', [])
    similar_customers = result.get('similar_customers', [])
    profile = result.get('profile', '')

    # Confidence indicator
    if confidence >= 70:
        confidence_emoji = "🟢"
        confidence_label = "High Confidence"
    elif confidence >= 40:
        confidence_emoji = "🟡"
        confidence_label = "Medium Confidence"
    else:
        confidence_emoji = "🔴"
        confidence_label = "Low Confidence"

    # Extract key info from profile for summary
    lines.append("**📊 Executive Summary:**")
    lines.append(f"• **Status:** {confidence_emoji} {confidence_label} ({confidence}%) - Research quality indicator")
    lines.append(f"• **Industry:** {business_type.replace('_', ' ').title()} electrical contracting")
    lines.append(f"• **Service Focus:** Electrical installation & contracting services")
    lines.append(f"• **Key Material Categories:** {', '.join(materials) if materials else 'TBD - needs manual review'}")
    lines.append(f"• **Similar Customers in Database:** {len(similar_customers)} matching profiles found")

    # Add geographical info if available
    if location:
        lines.append(f"• **Location:** {location}")
    lines.append("")

    # Detailed Business Profile from Sonar
    lines.append("**🔍 Detailed Business Profile** (from web research):")
    # Format profile with proper line breaks and citations
    profile_lines = profile.split('\n')
    for line in profile_lines:
        if line.strip():
            lines.append(line)
    lines.append("")

    # Add sources/references if available
    sources = result.get('sources', [])
    logger.info(f"[format_onboarding_result] Sources count: {len(sources)}")
    logger.info(f"[format_onboarding_result] Sources preview: {sources[:2] if sources else 'None'}")
    if sources and len(sources) > 0:
        lines.append("**📚 Sources & References:**")
        for i, source in enumerate(sources[:10], 1):  # Limit to 10 sources
            if isinstance(source, dict):
                url = source.get('url', source.get('link', ''))
                title = source.get('title', f'Source {i}')
                if url:
                    lines.append(f"[{i}] {title}: {url}")
                else:
                    lines.append(f"[{i}] {title}")
            elif isinstance(source, str):
                lines.append(f"[{i}] {source}")
        lines.append("")
    else:
        lines.append("*Note: Citations [1][2]... reference public web sources (URLs not available via API)*")
        lines.append("")

    # Material recommendations
    if materials:
        lines.append("**📦 Expected Material Purchases:**")
        material_names = {
            '1A': 'Automation & Controls',
            '1G': 'Conduit, Fittings & Bodies',
            '1K': 'Wiring Devices',
            '1L': 'Lighting Equipment',
            '1N': 'Wire & Cable'
        }
        for mat in materials:
            mat_name = material_names.get(mat, mat)
            lines.append(f"• **{mat}** - {mat_name}")
        lines.append("")

    # Similar customers section with explicit clickable formatting
    if similar_customers:
        lines.append(f"**👥 Peer Analysis: {len(similar_customers)} Similar Customers Found**\n")
        lines.append("*Click Customer IDs below to view their full profiles and SPA agreements:*\n")

        for i, cust in enumerate(similar_customers[:5], 1):
            cust_id = cust.get('customer_id', 'N/A')
            cust_name = cust.get('customer_name', 'Unknown')
            similarity = cust.get('similarity_score', 0)
            matching_cats = cust.get('matching_categories', [])
            rfm_segment = cust.get('rfm_segment', 'N/A')

            # Format with patterns that match UI clickable detection
            lines.append(f"{i}. **{cust_name}** (ID: {cust_id})")
            lines.append(f"   📊 Match Score: **{similarity:.0f}%** similarity")
            lines.append(f"   🎯 RFM Segment: {rfm_segment}")
            if matching_cats:
                lines.append(f"   📦 Common Categories: {', '.join(matching_cats)}")
            lines.append("")

        lines.append("**💡 Recommended Next Steps:**")
        lines.append("1. **Click the Customer IDs above** to open their detailed profiles in Quick Lookup")
        lines.append("2. Review their current SPA agreements to identify applicable SPAs for new customer")
        lines.append("3. Analyze purchase volumes in matching material categories")
        lines.append("4. Contact similar customers' account managers for onboarding insights")
    else:
        lines.append("\n**⚠️ No Similar Customers Found**")
        lines.append("This appears to be a unique customer profile. Consider:")
        lines.append("• Manual review of material needs with sales team")
        lines.append("• Custom SPA evaluation based on business profile")
        lines.append("• Direct customer interview for requirements gathering")

    return "\n".join(lines)


__all__ = [
    'process_with_llm',
    'create_llm_with_tools',
    'format_tool_results'
]
