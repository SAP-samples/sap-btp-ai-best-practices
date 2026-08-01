"""
Chat service — orchestrates entity extraction, dataset search,
prompt building, and LLM invocation for the AI Chat module.

Uses the shared FinancialProcessor singleton (passed in at construction)
so NO additional HANA queries are made.
"""

import logging
import time
import uuid
from typing import Any, Dict, Optional

from src.chat.context_manager import ContextManager
from src.chat.dataset_search import (
    COL_COST_CENTER,
    COL_GL_ACCOUNT,
    COL_PROFIT_CENTER,
    COL_SUPPLIER_NAME,
    build_cost_center_summary,
    build_flash_summary,
    build_gl_summary,
    build_je_text_summary,
    build_supplier_summary,
    build_variance_summary,
    extract_entities,
    search_dataset,
)
from src.chat.financial_explainer import (
    build_fallback_response,
    compute_totals,
)
from src.chat.prompt_builder import (
    build_chat_prompt,
    build_financial_context,
    build_no_data_prompt,
)

logger = logging.getLogger(__name__)

# Singleton context manager (shared across all requests in this process)
_context_manager = ContextManager()


class ChatService:
    """
    Main chat orchestration service.
    Receives the FinancialProcessor singleton from the FastAPI app
    so it reuses the already-cached in-memory dataset.
    """

    def __init__(self, processor) -> None:
        self._processor = processor

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def handle_message(
        self,
        message: str,
        conversation_id: Optional[str],
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Process a user chat message and return the assistant response.
        """
        start_time = time.time()

        # Ensure conversation ID
        if not conversation_id:
            conversation_id = str(uuid.uuid4())

        # Retrieve or create conversation context
        conv_ctx = _context_manager.get_or_create(conversation_id)

        # Determine fiscal years and global filters from context or processor defaults
        current_year, previous_year, company_code, segment, functional_area = \
            self._resolve_context(context)

        # Extract entities (merging with session context for follow-up resolution)
        entities = extract_entities(message, conv_ctx.entities)
        conv_ctx.update_entities(entities)

        # Diagnostic logging — extracted filters
        logger.info(
            "chat_service: Extracted filters: %s",
            {
                "segment":          entities.get("segment") or segment,
                "functional_area":  entities.get("functional_area") or functional_area,
                "gl_account":       entities.get("gl_account"),
                "cost_center":      entities.get("cost_center"),
                "profit_center":    entities.get("profit_center"),
                "quarter":          entities.get("quarter"),
                "month":            entities.get("month"),
                "years":            entities.get("years"),
                "je_type":          entities.get("je_type"),
                "flash_category":   entities.get("flash_category"),
                "current_year":     current_year,
                "previous_year":    previous_year,
            },
        )

        # Get the cached DataFrame from the processor
        df = self._get_cached_df()

        # Search dataset — filters applied in correct order BEFORE aggregation
        # Pass the hierarchy service so segment filtering uses the authoritative
        # Segment -> Profit Centers -> Cost Centers chain from SAP master data.
        hierarchy = getattr(self._processor, "hierarchy", None)
        filtered_df, search_meta = search_dataset(
            df=df,
            entities=entities,
            current_year=current_year,
            previous_year=previous_year,
            company_code=company_code,
            segment=segment,
            functional_area=functional_area,
            hierarchy=hierarchy,
        )

        matched_rows = search_meta.get("matched_rows", 0)
        filters_applied = search_meta.get("filters_applied", [])

        logger.info(
            "chat_service: search_dataset result — matched=%s, rows=%d, filters=%s",
            search_meta.get("matched"), matched_rows, filters_applied,
        )

        # Detect dimension of interest from the message
        dimension_col = self._detect_dimension(message, entities)

        # Compute financial summaries
        totals = compute_totals(filtered_df, current_year, previous_year)

        gl_summary       = build_gl_summary(filtered_df, current_year)
        cc_summary       = build_cost_center_summary(filtered_df, current_year)
        supplier_summary = build_supplier_summary(filtered_df, current_year)
        je_summary       = build_je_text_summary(filtered_df, current_year)
        flash_summary    = build_flash_summary(filtered_df, current_year)
        variance_summary = build_variance_summary(
            filtered_df, current_year, previous_year,
            group_col=dimension_col,
        ) if current_year and previous_year else []

        # Build financial context block
        fin_context = build_financial_context(
            matched_rows=matched_rows,
            filters_applied=filters_applied,
            gl_summary=gl_summary,
            cost_center_summary=cc_summary,
            supplier_summary=supplier_summary,
            je_text_summary=je_summary,
            variance_summary=variance_summary,
            current_year=current_year,
            previous_year=previous_year,
            total_current=totals.get("total_current"),
            total_previous=totals.get("total_previous"),
            flash_summary=flash_summary,
        )

        # Build prompt
        history_text = conv_ctx.get_history_text()
        if search_meta.get("matched"):
            prompt = build_chat_prompt(
                user_message=message,
                conversation_history=history_text,
                financial_context=fin_context,
                entities=entities,
                current_year=current_year,
                previous_year=previous_year,
            )
        else:
            prompt = build_no_data_prompt(
                user_message=message,
                conversation_history=history_text,
                entities=entities,
                current_year=current_year,
                previous_year=previous_year,
            )

        # Invoke LLM
        response_text, llm_meta = self._invoke_llm(prompt)

        # Update conversation history
        conv_ctx.add_turn("user", message)
        conv_ctx.add_turn("assistant", response_text)

        elapsed = round(time.time() - start_time, 3)

        metadata = {
            "conversation_id":  conversation_id,
            "matched_rows":     matched_rows,
            "filters_applied":  filters_applied,
            "entities_detected": {k: v for k, v in entities.items()
                                  if k not in ("years",)},
            "current_year":     current_year,
            "previous_year":    previous_year,
            "response_time_s":  elapsed,
            **llm_meta,
        }

        logger.info(
            "chat.service: conv=%s rows=%d filters=%s elapsed=%.3fs llm_ok=%s",
            conversation_id, matched_rows, filters_applied, elapsed,
            llm_meta.get("llm_ok", False),
        )

        return {
            "response":        response_text,
            "conversation_id": conversation_id,
            "metadata":        metadata,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_cached_df(self):
        """Return the cached DataFrame from the FinancialProcessor."""
        try:
            # FinancialProcessor exposes _get_data() (protected) or _cache
            # We call the internal method to avoid triggering a reload
            return self._processor._get_data()
        except Exception as exc:
            logger.error("chat.service: failed to get cached df: %s", exc)
            import pandas as pd
            return pd.DataFrame()

    def _detect_dimension(self, message: str, entities: Dict) -> str:
        """
        Detect which financial dimension the user is asking about.
        Returns the appropriate column name for grouping/variance analysis.
        """
        msg = message.lower()
        # If a specific entity was extracted, use its dimension
        if entities.get("cost_center"):
            return COL_COST_CENTER
        if entities.get("profit_center"):
            return COL_PROFIT_CENTER
        if entities.get("supplier_name"):
            return COL_SUPPLIER_NAME
        # Keyword-based dimension detection
        if any(kw in msg for kw in ["cost center", "cost centres", "cc "]):
            return COL_COST_CENTER
        if any(kw in msg for kw in ["profit center", "profit centre", "pc "]):
            return COL_PROFIT_CENTER
        if any(kw in msg for kw in ["supplier", "vendor", "vendors"]):
            return COL_SUPPLIER_NAME
        # Default: GL account
        return COL_GL_ACCOUNT

    def _resolve_context(self, context: Optional[Dict]) -> tuple:
        """
        Resolve current_year, previous_year, company_code, segment, functional_area
        from request context or fall back to processor defaults.
        Returns (current_year, previous_year, company_code, segment, functional_area).
        """
        current_year    = None
        previous_year   = None
        company_code    = None
        segment         = None
        functional_area = None

        if context:
            current_year    = context.get("currentYear")      or context.get("current_year")
            previous_year   = context.get("previousYear")     or context.get("previous_year")
            company_code    = context.get("companyCode")      or context.get("company_code")
            segment         = context.get("segment")          or None
            functional_area = context.get("functionalArea")   or context.get("functional_area") or None

        # Fall back to processor's available years
        if not current_year or not previous_year:
            try:
                available = self._processor.get_available_years(
                    company_code=company_code
                )
                if available and len(available) >= 2:
                    current_year  = current_year  or available[-1]
                    previous_year = previous_year or available[-2]
                elif available and len(available) == 1:
                    current_year  = current_year  or available[0]
            except Exception as exc:
                logger.warning("chat.service: could not resolve years: %s", exc)

        # Convert years to int if possible
        try:
            current_year  = int(current_year)  if current_year  else None
            previous_year = int(previous_year) if previous_year else None
        except (TypeError, ValueError):
            current_year  = None
            previous_year = None

        logger.info(
            "chat_service: resolved context — current_year=%s, previous_year=%s, "
            "company=%s, segment=%s, fa=%s",
            current_year, previous_year, company_code, segment, functional_area,
        )

        return current_year, previous_year, company_code, segment, functional_area

    def _invoke_llm(self, prompt: str) -> tuple:
        """
        Call SAP Gen AI Hub via gen_ai_hub + langchain.
        Returns (response_text, metadata_dict).
        Falls back gracefully if unavailable.
        """
        try:
            from dotenv import load_dotenv
            load_dotenv()

            from gen_ai_hub.proxy.core.proxy_clients import get_proxy_client
            from gen_ai_hub.proxy.langchain.openai import ChatOpenAI

            proxy_client = get_proxy_client("gen-ai-hub")
            llm = ChatOpenAI(
                proxy_model_name="gpt-5.4",
                proxy_client=proxy_client,
            )
            response = llm.invoke(prompt)
            content = response.content if hasattr(response, "content") else str(response)

            # Token usage if available
            token_meta = {}
            if hasattr(response, "usage_metadata") and response.usage_metadata:
                token_meta = {
                    "input_tokens":  response.usage_metadata.get("input_tokens"),
                    "output_tokens": response.usage_metadata.get("output_tokens"),
                }

            return content, {"llm_ok": True, **token_meta}

        except ImportError:
            logger.warning("chat.service: gen_ai_hub not installed — LLM unavailable")
            return (
                "⚠️ SAP Gen AI Hub is not available in this environment. "
                "Please ensure `gen-ai-hub-sdk` is installed and configured.",
                {"llm_ok": False, "error": "gen_ai_hub not installed"},
            )
        except Exception as exc:
            logger.error("chat.service: LLM invocation failed: %s", exc)
            return (
                f"⚠️ The AI assistant encountered an error: {str(exc)[:200]}. "
                "Please try again or rephrase your question.",
                {"llm_ok": False, "error": str(exc)[:200]},
            )