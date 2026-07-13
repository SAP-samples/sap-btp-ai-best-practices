"""
Agent Chat API Router

Conversational AI for SPA gap analysis
Tab 2 in UX: Natural language queries + Quick action buttons
"""

from fastapi import APIRouter, HTTPException, Depends, Request
import uuid
import logging
import numpy as np
import re
from typing import Any, Dict, List

from app.models import (
    AgentChatRequest,
    AgentChatResponse,
    AgentAction,
    ErrorResponse
)
from app.services import (
    get_customer_profile,
    search_customers,
    find_similar_customers,
    detect_spa_gaps,
    get_rfm_distribution
)
from app.services.llm_agent_service import process_with_llm, format_onboarding_result
from app.services.spa_tools import research_customer_onboarding_tool
from app.security import require_api_key

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/spa", tags=["SPA Agent Chat"])


def get_http_request(request: Request) -> Request:
    return request


def clean_numpy_types(obj: Any) -> Any:
    """
    Recursively convert numpy types to Python native types for JSON serialization
    """
    if isinstance(obj, dict):
        return {k: clean_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_numpy_types(item) for item in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj


# Simple conversation memory (in-memory for MVP)
# Phase 2+: Replace with LangGraph + SQLite persistence
conversation_memory = {}


def _parse_direct_onboarding_request(message: str) -> Dict[str, str] | None:
    """
    Detect explicit onboarding/research requests that do not need LLM tool routing.

    SAP AI Core Orchestration v2 already powers the onboarding service. Bypassing
    the LangChain proxy tool-calling hop avoids GenAIHubProxyClient credential
    failures for simple "Research onboarding..." requests.
    """
    text = (message or "").strip()
    if not text:
        return None

    normalized = re.sub(r"\s+", " ", text).strip()
    lower = normalized.lower()
    onboarding_markers = [
        "research onboarding for new customer",
        "research new customer",
        "onboard new customer",
        "onboarding for new customer",
    ]

    if not any(marker in lower for marker in onboarding_markers):
        return None

    customer_text = normalized
    for marker in onboarding_markers:
        match = re.search(re.escape(marker), customer_text, flags=re.IGNORECASE)
        if match:
            customer_text = customer_text[match.end():]
            break

    customer_text = customer_text.strip(" :-")
    if not customer_text:
        return None

    if "," in customer_text:
        customer_name, location = customer_text.rsplit(",", 1)
        return {
            "customer_name": customer_name.strip(),
            "location": location.strip(),
        }

    return {
        "customer_name": customer_text.strip(),
        "location": "",
    }


@router.post(
    "/agent-chat",
    response_model=AgentChatResponse,
    dependencies=[Depends(require_api_key)],
    summary="Agent Chat",
    description="Natural language chat interface for SPA analysis"
)
async def agent_chat(
    request: AgentChatRequest,
    http_request: Request = Depends(get_http_request),
) -> AgentChatResponse:
    """
    Agent Chat: Natural language SPA analysis

    LLM-powered with tool calling using LangChain.
    Supports fuzzy search, context memory, and natural language parameter extraction.

    Supported queries:
    - "Analyze customer 999001"
    - "Find TEST ELECTRIC" (fuzzy search)
    - "What are their missing SPAs?" (context-aware follow-up)
    - "Find customers in Utah with high RFM"
    - "Compare customer X with similar customers"
    - "Show RFM distribution"
    """
    logger.info(f"Agent chat message: {request.message[:100]}")
    logger.info(f"[FILTER DEBUG] exclude_unknown parameter received: {request.exclude_unknown}")

    # Generate or use conversation ID
    conversation_id = request.conversation_id or str(uuid.uuid4())

    # Initialize conversation memory if new
    if conversation_id not in conversation_memory:
        conversation_memory[conversation_id] = {
            'context': {},  # Store last_analyzed_customer_id, exclude_unknown, etc.
            'history': []
        }

    conv = conversation_memory[conversation_id]

    # Update context with exclude_unknown filter
    conv['context']['exclude_unknown'] = request.exclude_unknown
    logger.info(f"[FILTER DEBUG] Stored in context: exclude_unknown={conv['context']['exclude_unknown']}")

    # Add user message to history
    conv['history'].append({
        'role': 'user',
        'message': request.message
    })

    # Update context if customer_id provided in request
    if request.customer_id:
        conv['context']['last_analyzed_customer_id'] = request.customer_id

    try:
        onboarding_args = _parse_direct_onboarding_request(request.message)
        if onboarding_args:
            logger.info(
                "Routing explicit onboarding request directly through AI Core Orchestration service"
            )
            tool_args = {
                "customer_name": onboarding_args["customer_name"],
                "exclude_unknown": request.exclude_unknown,
            }
            if onboarding_args.get("location"):
                tool_args["location"] = onboarding_args["location"]

            tool_result = research_customer_onboarding_tool.invoke(tool_args)
            tool_result = clean_numpy_types(tool_result)
            message = format_onboarding_result(tool_result)

            conv['history'].append({
                'role': 'assistant',
                'message': message
            })
            conv['context']['last_onboarded_customer_name'] = onboarding_args["customer_name"]
            if onboarding_args.get("location"):
                conv['context']['last_onboarded_location'] = onboarding_args["location"]

            result = {
                "message": message,
                "tool_calls": [{"name": "research_customer_onboarding_tool", "args": tool_args}],
                "tool_results": [tool_result],
                "context": conv["context"],
            }
            entities = _extract_entities(result)

            return AgentChatResponse(
                message=message,
                conversation_id=conversation_id,
                actions=[
                    AgentAction(
                        tool="research_customer_onboarding_tool",
                        input=tool_args,
                        output=None,
                    )
                ],
                data={
                    "tool_results": [tool_result],
                    "entities": entities,
                    "routing": "direct_onboarding_orchestration",
                },
                quick_actions=[
                    "Research another customer",
                    "Search for similar customers",
                    "Help",
                ],
            )

        # Process with LLM (Gemini 2.5 Flash primary, Gemini 2.5 Pro fallback)
        result = await process_with_llm(
            user_message=request.message,
            conversation_history=conv['history'][:-1],  # Exclude current message (already in user_message)
            conversation_context=conv['context'],
            model_name="gemini-2.5-flash",
            request=http_request,
            route="/api/spa/agent-chat",
            method="POST",
        )

        # Add assistant response to history
        conv['history'].append({
            'role': 'assistant',
            'message': result['message']
        })

        # Update context with any changes
        conv['context'] = result['context']

        # Generate quick actions based on result
        quick_actions = _generate_quick_actions(result)

        # Extract entities for clickable links
        entities = _extract_entities(result)
        logger.info(f"Extracted {len(entities)} entities for clickable links")
        if entities:
            logger.debug(f"Entities: {entities}")

        # Return response
        return AgentChatResponse(
            message=result['message'],
            conversation_id=conversation_id,
            actions=[
                AgentAction(
                    tool=tc['name'],
                    input=tc['args'],
                    output=None  # Could store result if needed
                )
                for tc in result.get('tool_calls', [])
            ],
            data={
                "tool_results": result.get('tool_results', []),
                "entities": entities
            },
            quick_actions=quick_actions
        )

    except Exception as e:
        logger.error(f"Error in LLM agent chat: {e}", exc_info=True)
        return AgentChatResponse(
            message=f"I encountered an error: {str(e)}. Please try again or rephrase your query.",
            conversation_id=conversation_id,
            quick_actions=["Start over", "Help"]
        )




@router.delete(
    "/agent-chat/{conversation_id}",
    dependencies=[Depends(require_api_key)],
    summary="Delete Chat History",
    description="Clear conversation history for a specific conversation ID"
)
async def delete_chat_history(conversation_id: str) -> Dict[str, str]:
    """
    Delete chat history for a conversation ID

    Args:
        conversation_id: The conversation ID to delete

    Returns:
        Success message
    """
    if conversation_id in conversation_memory:
        del conversation_memory[conversation_id]
        logger.info(f"Deleted conversation history for {conversation_id}")
        return {"message": "Conversation history deleted successfully", "conversation_id": conversation_id}
    else:
        raise HTTPException(status_code=404, detail=f"Conversation {conversation_id} not found")


def _extract_entities(result: Dict) -> List[Dict]:
    """
    Extract clickable entities (customers, SPAs) from tool results.

    Args:
        result: LLM processing result with tool_results

    Returns:
        List of entity dicts with type, id, name/description
    """
    entities = []

    for tool_result in result.get('tool_results', []):
        if 'error' in tool_result:
            continue

        # Extract customer entity
        if 'customer_id' in tool_result and 'customer_name' in tool_result:
            entities.append({
                'type': 'customer',
                'id': str(tool_result['customer_id']),
                'name': tool_result['customer_name']
            })

        # Extract SPA entities from missing_spas
        if 'missing_spas' in tool_result:
            for spa in tool_result['missing_spas']:
                entities.append({
                    'type': 'spa',
                    'id': str(spa.get('sales_deal', '')),
                    'description': spa.get('description', 'Unknown SPA'),
                    'vendor': spa.get('vendor', 'N/A')
                })

        # Extract customer entities from search results
        if 'customers' in tool_result:
            for customer in tool_result['customers']:
                entities.append({
                    'type': 'customer',
                    'id': str(customer.get('customer_id', '')),
                    'name': customer.get('customer_name', 'Unknown')
                })

        # Extract similar customer entities
        if 'similar_customers' in tool_result:
            for customer in tool_result['similar_customers']:
                entities.append({
                    'type': 'customer',
                    'id': str(customer.get('customer_id', '')),
                    'name': customer.get('customer_name', 'Unknown')
                })

        # Extract customer entities from onboarding results
        if 'profile' in tool_result and 'business_type' in tool_result:
            # This is an onboarding result
            similar_customers = tool_result.get('similar_customers', [])
            logger.info(f"Found onboarding result with {len(similar_customers)} similar customers")
            for customer in similar_customers:
                cust_id = str(customer.get('customer_id', ''))
                cust_name = customer.get('customer_name', 'Unknown')
                logger.debug(f"Adding entity: customer_id={cust_id}, name={cust_name}")
                entities.append({
                    'type': 'customer',
                    'id': cust_id,
                    'name': cust_name
                })

    return entities


def _generate_quick_actions(result: Dict) -> List[str]:
    """
    Generate quick action buttons based on LLM result.

    Args:
        result: LLM processing result with tool_calls and tool_results

    Returns:
        List of quick action strings
    """
    quick_actions = []

    # Check if customer was analyzed
    if result.get('context', {}).get('last_analyzed_customer_id'):
        customer_id = result['context']['last_analyzed_customer_id']
        quick_actions.append(f"Show similar customers to {customer_id}")
        quick_actions.append("Analyze another customer")
    else:
        # Default actions
        quick_actions.append("Search for a customer")
        quick_actions.append("Show RFM distribution")

    # Add help as last option
    quick_actions.append("Help")

    return quick_actions[:3]  # Limit to 3 actions


# Keep legacy handler functions for backwards compatibility / testing
# These are no longer called by main agent_chat endpoint

    """Handle customer search intent"""

    # Extract search query
    # Simple extraction: take words after "find" or "search"
    words = message.split()
    query_parts = []
    found_trigger = False

    for word in words:
        if word in ['find', 'search', 'customer', 'look', 'up']:
            found_trigger = True
            continue
        if found_trigger and word.isalnum():
            query_parts.append(word)

    query = ' '.join(query_parts) if query_parts else None

    if not query and not customer_id:
        return AgentChatResponse(
            message="I can search for customers. What customer ID or name are you looking for?",
            conversation_id=conversation_id,
            quick_actions=[]
        )

    # Search customers
    results = search_customers(query=query or customer_id, limit=5)

    # Clean numpy types for JSON serialization
    results = clean_numpy_types(results)

    if not results:
        return AgentChatResponse(
            message=f"No customers found matching '{query or customer_id}'.",
            conversation_id=conversation_id,
            actions=[
                AgentAction(tool="search_customers", input={"query": query or customer_id}, output=results)
            ],
            quick_actions=["Try another search"]
        )

    # Format results
    message = f"Found {len(results)} customer(s):\n\n"
    for i, cust in enumerate(results, 1):
        message += f"{i}. **{cust['customer_id']}** - {cust.get('customer_name', 'N/A')}\n"
        message += f"   📍 {cust.get('city', 'N/A')}, {cust.get('state', 'N/A')}\n\n"

    return AgentChatResponse(
        message=message,
        conversation_id=conversation_id,
        actions=[
            AgentAction(tool="search_customers", input={"query": query or customer_id}, output=results)
        ],
        data={"customers": results},
        quick_actions=[
            f"Analyze {results[0]['customer_id']}",
            "Search again"
        ]
    )


async def _handle_analyze_customer(customer_id: str, conversation_id: str) -> AgentChatResponse:
    """Handle customer analysis intent"""

    try:
        # Get customer profile first to verify existence
        profile = get_customer_profile(customer_id)

        if not profile:
            return AgentChatResponse(
                message=f"❌ Customer {customer_id} not found in the database.",
                conversation_id=conversation_id,
                quick_actions=["Search for a customer", "Help"]
            )

        # Get gaps
        gaps = detect_spa_gaps(customer_id, top_n_similar=50, min_similar_count=2)

        # Clean numpy types for JSON serialization
        gaps = clean_numpy_types(gaps)

        if 'message' in gaps:
            # No similar customers
            return AgentChatResponse(
                message=f"Customer {customer_id}: {gaps['message']}",
                conversation_id=conversation_id,
                quick_actions=["Try another customer"]
            )

        # Format results
        missing_count = gaps.get('missing_spa_count', 0)
        target_spa_count = gaps.get('target_spa_count', 0)
        similar_count = gaps.get('similar_customers_count', 0)

        # Get customer info
        customer_name = profile.get('customer_name', 'N/A')
        sales_office = profile.get('sales_office', 'N/A')
        pl_type = profile.get('pl_type', 'N/A')

        message = f"**Analysis for Customer {customer_id}**\n"
        message += f"**{customer_name}**\n\n"
        message += f"📍 Sales Office: {sales_office}\n"
        message += f"📂 Customer Type: {pl_type}\n"
        message += f"📊 Current SPAs: {target_spa_count}\n"
        message += f"🔍 Analyzed {similar_count} similar customers\n"
        message += f"⚠️ Found {missing_count} missing SPAs\n\n"

        if missing_count > 0:
            message += "**Top Missing SPAs:**\n\n"
            for i, spa in enumerate(gaps.get('missing_spas', [])[:5], 1):
                confidence = spa.get('confidence_score', 0)
                message += f"{i}. **SPA {spa['sales_deal']}** - {spa.get('description', 'N/A')}\n"
                message += f"   Confidence: {confidence:.0f}% ({spa.get('confidence_level', 'N/A')})\n"
                message += f"   {spa['count_in_similar']}/{similar_count} similar customers have this ({spa['percentage_in_similar']:.1f}%)\n\n"
        else:
            message += "✅ No gaps found! Customer has all recommended SPAs."

        return AgentChatResponse(
            message=message,
            conversation_id=conversation_id,
            actions=[
                AgentAction(tool="detect_spa_gaps", input={"customer_id": customer_id}, output=gaps)
            ],
            data=gaps,
            quick_actions=[
                "Show similar customers",
                "Analyze another customer"
            ]
        )

    except Exception as e:
        logger.error(f"Error analyzing customer {customer_id}: {e}", exc_info=True)
        return AgentChatResponse(
            message=f"❌ Error analyzing customer {customer_id}: {str(e)}",
            conversation_id=conversation_id,
            quick_actions=["Try another customer", "Help"]
        )


async def _handle_similar_customers(customer_id: str, conversation_id: str) -> AgentChatResponse:
    """Handle similar customers intent"""

    try:
        # Verify customer exists
        profile = get_customer_profile(customer_id)

        if not profile:
            return AgentChatResponse(
                message=f"❌ Customer {customer_id} not found in the database.",
                conversation_id=conversation_id,
                quick_actions=["Search for a customer", "Help"]
            )

        similar = find_similar_customers(customer_id, top_n=10)

        # Clean numpy types for JSON serialization
        similar = clean_numpy_types(similar)

        if not similar:
            return AgentChatResponse(
                message=f"No similar customers found for {customer_id}.",
                conversation_id=conversation_id,
                quick_actions=["Try another customer"]
            )

        message = f"**Similar Customers to {customer_id}:**\n\n"
        for i, sim in enumerate(similar[:5], 1):
            message += f"{i}. **{sim['customer_id']}** - {sim.get('customer_name', 'N/A')}\n"
            message += f"   Similarity: {sim['similarity_score']:.1f}/140\n"
            message += f"   Sales Office: {sim.get('sales_office', 'N/A')}\n"
            message += f"   RFM: {sim.get('rfm_segment', 'N/A')}\n\n"

        message += f"📊 Total similar customers found: {len(similar)}"

        return AgentChatResponse(
            message=message,
            conversation_id=conversation_id,
            actions=[
                AgentAction(tool="find_similar_customers", input={"customer_id": customer_id}, output=similar)
            ],
            data={"similar_customers": similar},
            quick_actions=[
                f"Analyze {customer_id}",
                "Compare with similar"
            ]
        )

    except Exception as e:
        logger.error(f"Error finding similar customers for {customer_id}: {e}", exc_info=True)
        return AgentChatResponse(
            message=f"❌ Error finding similar customers for {customer_id}: {str(e)}",
            conversation_id=conversation_id,
            quick_actions=["Try another customer", "Help"]
        )


async def _handle_rfm_distribution(conversation_id: str) -> AgentChatResponse:
    """Handle RFM distribution intent"""

    try:
        distribution = get_rfm_distribution()

        # Clean numpy types for JSON serialization
        distribution = clean_numpy_types(distribution)

        if not distribution or 'total_customers' not in distribution:
            return AgentChatResponse(
                message="❌ Unable to retrieve RFM distribution data.",
                conversation_id=conversation_id,
                quick_actions=["Try again", "Help"]
            )

        message = "**RFM Segment Distribution:**\n\n"
        for segment, data in distribution.items():
            if segment != 'total_customers' and isinstance(data, dict):
                count = data.get('count', 0)
                percentage = data.get('percentage', 0)
                message += f"• **{segment}**: {count} customers ({percentage:.1f}%)\n"

        total = distribution.get('total_customers', 0)
        message += f"\n📊 Total: {total} customers"

        return AgentChatResponse(
            message=message,
            conversation_id=conversation_id,
            actions=[
                AgentAction(tool="get_rfm_distribution", input={}, output=distribution)
            ],
            data=distribution,
            quick_actions=[
                "Find Champions customers",
                "Analyze a customer"
            ]
        )

    except Exception as e:
        logger.error(f"Error getting RFM distribution: {e}", exc_info=True)
        return AgentChatResponse(
            message=f"❌ Error retrieving RFM distribution: {str(e)}",
            conversation_id=conversation_id,
            quick_actions=["Try again", "Help"]
        )


def _handle_help(conversation_id: str) -> AgentChatResponse:
    """Handle help intent"""

    message = """**SPA Gap Analysis Assistant**

I can help you with:

**Customer Analysis:**
• "Analyze customer 12345"
• "Find missing SPAs for customer X"

**Customer Search:**
• "Find customer ABC Corp"
• "Search for customer 12345"

**Similar Customers:**
• "Show similar customers to 12345"
• "Who is similar to customer X?"

**RFM Insights:**
• "Show RFM distribution"
• "Show Champions segment"

Just ask me a question or click a quick action button!
"""

    return AgentChatResponse(
        message=message,
        conversation_id=conversation_id,
        quick_actions=[
            "Search for a customer",
            "Show RFM distribution"
        ]
    )


@router.delete(
    "/agent-chat/{conversation_id}",
    dependencies=[Depends(require_api_key)],
    summary="Clear Conversation",
    description="Clear conversation memory"
)
async def clear_conversation(conversation_id: str):
    """Clear conversation memory"""
    if conversation_id in conversation_memory:
        del conversation_memory[conversation_id]
        return {"message": "Conversation cleared"}
    else:
        raise HTTPException(status_code=404, detail="Conversation not found")
