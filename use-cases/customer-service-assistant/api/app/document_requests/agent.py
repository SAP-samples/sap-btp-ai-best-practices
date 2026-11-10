from __future__ import annotations

from typing import Any, TypedDict

from langgraph.graph import END, StateGraph

from .gmail_service import build_mime_reply_with_attachments
from .models import DocumentRequestExtraction, DraftEmailProposal, GmailEmail
from .response_composer import TemplateResponseComposer
from .scenario_emails import normalize_line_identifier
from .tools import build_reply_body, build_reply_subject, choose_documents_for_request


class AgentState(TypedDict, total=False):
    """LangGraph state for document request resolution."""

    email: GmailEmail
    extraction: DocumentRequestExtraction
    extraction_id: str
    repository: Any
    gmail_client: Any
    composer: Any
    mailbox_id: str
    from_email: str | None
    proposal: DraftEmailProposal
    gmail_draft_id: str | None
    gmail_draft_message_id: str | None
    draft_id: str | None


def resolve_document_request(
    email: GmailEmail,
    extraction: DocumentRequestExtraction,
    extraction_id: str,
    repository: Any,
    composer: Any | None = None,
) -> DraftEmailProposal:
    """Resolve extracted request fields into a draft email proposal.

    Uses sales_order_id when available (preferred path). Falls back to
    country-based lookup for backward compatibility.

    Args:
        email: Normalized source Gmail message.
        extraction: Normalized document-request extraction.
        extraction_id: Persisted extraction id.
        repository: Repository exposing document lookup methods.
        composer: Optional response composer using retrieved HANA context.

    Returns:
        DraftEmailProposal for UI review.
    """
    order_status = None
    if extraction.sales_order_id:
        order_status = repository.get_order_status(extraction.sales_order_id)

    normalized_line = normalize_line_identifier(extraction.line_item_number)
    order_context: dict[str, Any] = {"status_lines": [], "item_details": []}
    if extraction.sales_order_id and hasattr(repository, "get_order_context"):
        order_context = repository.get_order_context(
            sales_order_id=extraction.sales_order_id,
            line_item_number=normalized_line,
            product_id=extraction.product_id,
            batch_number=extraction.batch_number,
        )

    requested_documents = [item.model_dump() for item in extraction.requested_documents]
    selected, errors = choose_documents_for_request(
        repository=repository,
        client_country=extraction.client_country,
        requested_documents=requested_documents,
        send_all=extraction.send_all,
        sales_order_id=extraction.sales_order_id,
        line_item_number=normalized_line,
        order_status=order_status,
    )

    if not email.sender_email:
        errors.append("sender_email is required")

    # Derive country: extraction → order status → HANA documents
    client_country = extraction.client_country
    if not client_country and order_status:
        client_country = order_status.client_country
    if not client_country and order_context.get("status_lines"):
        client_country = order_context["status_lines"][0].get("client_country")
    if not client_country and extraction.sales_order_id:
        country_rows = repository.list_documents_for_sales_order(extraction.sales_order_id)
        if country_rows:
            client_country = country_rows[0].get("client_country")

    subject = build_reply_subject(email.subject)
    response_context = {
        **order_context,
        "order_status": order_status,
        "client_country": client_country,
        "attachments": selected,
    }
    response_composer = composer or TemplateResponseComposer()
    body = response_composer.compose(email=email, extraction=extraction, context=response_context)
    if not body:
        body = build_reply_body(
            sales_order_id=extraction.sales_order_id,
            attachments=selected,
            order_status=order_status,
            client_country=client_country,
        )

    if selected and not errors:
        status = "ready_for_review"
    elif order_status and not errors:
        status = "ready_for_review"
    else:
        status = "needs_human_review"

    explanation_parts = []
    if order_status:
        explanation_parts.append(
            f"Order {extraction.sales_order_id}: PGI={order_status.pgi_status}, "
            f"shipped={order_status.shipped}"
        )
    if order_context.get("status_lines"):
        explanation_parts.append(f"Matched {len(order_context['status_lines'])} HANA status line(s).")
    explanation_parts.append(f"Selected {len(selected)} document(s).")
    explanation_parts.append(f"Reason: {extraction.reason}")

    return DraftEmailProposal(
        status=status,
        gmail_message_id=email.gmail_message_id,
        extraction_id=extraction_id,
        sales_order_id=extraction.sales_order_id,
        client_country=client_country,
        to_email=email.sender_email,
        subject=subject,
        body=body,
        selected_documents=selected,
        confidence=extraction.confidence,
        explanation=" | ".join(explanation_parts),
        validation_errors=errors,
    )


def build_failed_proposal(
    email: GmailEmail,
    extraction: DocumentRequestExtraction,
    extraction_id: str,
    error: Exception,
) -> DraftEmailProposal:
    """Build a failed proposal when agent processing raises.

    Args:
        email: Normalized source Gmail message.
        extraction: Normalized document-request extraction.
        extraction_id: Persisted extraction id.
        error: Exception raised by the agent workflow.

    Returns:
        Failed DraftEmailProposal preserving source email context.
    """
    return DraftEmailProposal(
        status="failed",
        gmail_message_id=email.gmail_message_id,
        extraction_id=extraction_id,
        sales_order_id=extraction.sales_order_id,
        client_country=extraction.client_country,
        to_email=email.sender_email,
        subject=build_reply_subject(email.subject),
        body="The automatic document draft could not be prepared.",
        selected_documents=[],
        confidence=0.0,
        explanation=f"Agent failed: {error}",
        validation_errors=[f"Agent failed: {error}"],
    )


def _resolve_node(state: AgentState) -> dict[str, DraftEmailProposal]:
    """Resolve the request into a draft proposal.

    Args:
        state: Current LangGraph workflow state.

    Returns:
        Partial state containing the draft proposal.
    """
    return {
        "proposal": resolve_document_request(
            email=state["email"],
            extraction=state["extraction"],
            extraction_id=state["extraction_id"],
            repository=state["repository"],
            composer=state.get("composer"),
        )
    }


def _gmail_draft_node(state: AgentState) -> dict[str, str | None]:
    """Create a Gmail draft when the proposal is ready for review.

    Args:
        state: Current LangGraph workflow state.

    Returns:
        Partial state containing Gmail draft identifiers when a draft was created.
    """
    proposal = state["proposal"]
    gmail_client = state.get("gmail_client")
    if proposal.status != "ready_for_review" or not proposal.to_email or not gmail_client:
        return {"gmail_draft_id": None, "gmail_draft_message_id": None}
    raw_message = build_mime_reply_with_attachments(
        source_email=state["email"],
        to_email=proposal.to_email,
        from_email=state.get("from_email"),
        subject=proposal.subject,
        body=proposal.body,
        attachments=proposal.selected_documents,
    )
    try:
        draft = state["gmail_client"].create_draft(
            user_id=state["mailbox_id"],
            raw_message=raw_message,
            thread_id=state["email"].thread_id,
        )
    except Exception:
        return {"gmail_draft_id": None, "gmail_draft_message_id": None}
    message = draft.get("message") or {}
    return {
        "gmail_draft_id": draft.get("id"),
        "gmail_draft_message_id": message.get("id"),
    }


def _persist_node(state: AgentState) -> dict[str, str]:
    """Persist the proposal for UI review.

    Args:
        state: Current LangGraph workflow state.

    Returns:
        Partial state containing the persisted review draft id.
    """
    proposal = state["proposal"]
    draft_id = state["repository"].insert_document_email_draft(
        proposal=proposal,
        gmail_draft_id=state.get("gmail_draft_id"),
        gmail_draft_message_id=state.get("gmail_draft_message_id"),
        raw_agent_output=proposal.model_dump(mode="json", exclude={"selected_documents"}),
    )
    return {"draft_id": draft_id}


def build_document_request_graph() -> Any:
    """Build the LangGraph workflow for document request resolution.

    Returns:
        Compiled LangGraph workflow.
    """
    graph = StateGraph(AgentState)
    graph.add_node("resolve", _resolve_node)
    graph.add_node("gmail_draft", _gmail_draft_node)
    graph.add_node("persist", _persist_node)
    graph.set_entry_point("resolve")
    graph.add_edge("resolve", "gmail_draft")
    graph.add_edge("gmail_draft", "persist")
    graph.add_edge("persist", END)
    return graph.compile()


class DocumentRequestAgent:
    """Orchestrate document selection, Gmail draft creation, and persistence."""

    def __init__(
        self,
        repository: Any,
        gmail_client: Any,
        mailbox_id: str,
        from_email: str | None,
        composer: Any | None = None,
    ) -> None:
        """Initialize the agent.

        Args:
            repository: Repository for document lookup and draft persistence.
            gmail_client: Gmail client for draft creation.
            mailbox_id: Gmail mailbox id, usually "me".
            from_email: Optional From header for created drafts.
            composer: Response composer for grounded email bodies.
        """
        self.repository = repository
        self.gmail_client = gmail_client
        self.mailbox_id = mailbox_id
        self.from_email = from_email
        self.composer = composer or TemplateResponseComposer()
        self.graph = build_document_request_graph()

    def resolve_email(
        self,
        email: GmailEmail,
        extraction: DocumentRequestExtraction,
        extraction_id: str,
    ) -> tuple[DraftEmailProposal, str | None]:
        """Resolve one email and return the proposal plus persisted draft id.

        Args:
            email: Normalized source Gmail message.
            extraction: Normalized document-request extraction.
            extraction_id: Persisted extraction id.

        Returns:
            Tuple containing the draft proposal and persisted draft id.
        """
        result = self.graph.invoke(
            {
                "email": email,
                "extraction": extraction,
                "extraction_id": extraction_id,
                "repository": self.repository,
                "gmail_client": self.gmail_client,
                "composer": self.composer,
                "mailbox_id": self.mailbox_id,
                "from_email": self.from_email,
            }
        )
        return result["proposal"], result.get("draft_id")
