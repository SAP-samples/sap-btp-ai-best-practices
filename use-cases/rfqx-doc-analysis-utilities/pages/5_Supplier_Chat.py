"""
Supplier document chat page.

Enables interactive Q&A against processed supplier documents with streaming
responses where available.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from app_context import apply_template_theme, ensure_session_state, get_comparator

apply_template_theme()
ensure_session_state()

comparator = get_comparator()

st.title("Supplier Document Chat")
st.caption(
    "Ask targeted questions about processed supplier documents with optional comparative and summary responses."
)

completed_providers = [
    provider for provider in st.session_state.providers if provider.get("status") == "completed"
]

if not completed_providers:
    st.warning(
        "Process supplier documents before using the chat. Visit the Process Supplier Documents page."
    )
    pending_count = sum(1 for provider in st.session_state.providers if provider.get("status") == "pending")
    if pending_count > 0:
        st.info(f"You have {pending_count} suppliers ready to process.")
    st.stop()

provider_names = [provider["name"] for provider in completed_providers]
st.info(f"Available suppliers: {', '.join(provider_names)} ({len(completed_providers)} total)")

st.subheader("Chat Configuration")
col_providers, col_mode = st.columns(2)
with col_providers:
    selected_providers = st.multiselect(
        "Select suppliers to query:",
        options=provider_names,
        default=provider_names,
        help="Choose the suppliers whose documents will be used to answer the question.",
    )
with col_mode:
    query_mode = st.selectbox(
        "Query Mode:",
        options=["Individual Responses", "Comparative Analysis", "Summary Across All"],
        help="Choose how to present the responses.",
    )

st.subheader("Ask Questions About Your Suppliers")
st.write("**Common Questions:**")
common_questions = [
    "What is the submission deadline?",
    "What are the key technical requirements?",
    "What is the estimated contract value?",
    "What are the evaluation criteria?",
    "What insurance requirements must be met?",
    "What are the payment terms?",
    "What are the mandatory supplier requirements?",
]

question_cols = st.columns(3)
for index, question in enumerate(common_questions):
    with question_cols[index % 3]:
        if st.button(question, key=f"common_q_{index}", help=question):
            st.session_state.current_query = question

query = st.text_area(
    "Or enter your custom question:",
    value=st.session_state.get("current_query", ""),
    placeholder="e.g., Compare the environmental compliance requirements across suppliers...",
    height=100,
)

if st.button("Clear Query", key="clear_query"):
    st.session_state.current_query = ""
    st.rerun()


def _stream_individual_responses(selected: list[str], question: str) -> None:
    for provider_name in selected:
        provider = next(provider for provider in completed_providers if provider["name"] == provider_name)

        with st.expander(f"💼 Response from {provider_name}", expanded=True):
            try:
                extracted_data = provider["extracted_data"]
                raw_content = extracted_data.get("_raw_content", "")

                if not raw_content:
                    st.warning(
                        f"No raw content available for {provider_name}. Using extracted features as a fallback."
                    )
                    content_summary = f"Supplier: {provider_name}\n\n"
                    for category, fields in extracted_data.items():
                        if not category.startswith("_") and isinstance(fields, dict):
                            content_summary += f"{category.replace('_', ' ').title()}:\n"
                            for field, value in fields.items():
                                if value and value != "Not Found":
                                    content_summary += f"- {field.replace('_', ' ').title()}: {value}\n"
                            content_summary += "\n"
                    raw_content = content_summary

                from document_processor import check_context_limits, optimize_text_for_context

                temp_content = {
                    "full_text": raw_content,
                    "token_count": len(raw_content.split()),
                    "filename": f"{provider_name}_aggregated_documents",
                }

                limits_check = check_context_limits(temp_content, max_tokens=1_000_000)
                if not limits_check["within_limits"]:
                    st.info(f"Optimising large content for {provider_name} ({temp_content['token_count']:,} tokens).")
                    raw_content = optimize_text_for_context(raw_content, target_tokens=800_000)

                llm_data = {
                    "full_text": raw_content,
                    "token_count": len(raw_content.split()),
                    "filename": f"{provider_name}_aggregated_documents",
                }

                response_container = st.empty()
                accumulated_response = ""
                progress_message = st.info("Generating response...")

                try:
                    for chunk in comparator.client.answer_specific_query_streaming(llm_data, question):
                        accumulated_response += chunk
                        response_container.markdown(accumulated_response)

                    progress_message.success("Response completed!")
                except Exception as stream_error:
                    progress_message.error(f"Streaming error: {stream_error}")
                    st.warning("Falling back to standard response mode...")
                    response = comparator.client.answer_specific_query(llm_data, question)
                    response_container.markdown(response)
                    st.success(f"Analysis of {provider_name} completed (fallback mode).")

            except Exception as err:
                st.error(f"Error querying {provider_name}: {err}")


def _comparative_analysis(selected: list[str], question: str) -> None:
    st.write("**Comparative Analysis:**")
    all_raw_content = []

    for provider_name in selected:
        provider = next(provider for provider in completed_providers if provider["name"] == provider_name)
        extracted_data = provider["extracted_data"]
        raw_content = extracted_data.get("_raw_content", "")

        if raw_content:
            all_raw_content.append(f"=== {provider_name.upper()} ===\n{raw_content}")
        else:
            st.warning(f"No raw content available for {provider_name}. Using extracted features as fallback.")
            content_summary = f"Supplier: {provider_name}\n\n"
            for category, fields in extracted_data.items():
                if not category.startswith("_") and isinstance(fields, dict):
                    content_summary += f"{category.replace('_', ' ').title()}:\n"
                    for field, value in fields.items():
                        if value and value != "Not Found":
                            content_summary += f"- {field.replace('_', ' ').title()}: {value}\n"
                    content_summary += "\n"
            all_raw_content.append(f"=== {provider_name.upper()} ===\n{content_summary}")

    combined_content = "\n\n".join(all_raw_content)

    comparative_prompt = f"""
Based on the following supplier documents, answer the question: "{question}"

Suppliers: {', '.join(selected)}

Complete Document Content:
{combined_content}
"""

    from document_processor import check_context_limits, optimize_text_for_context

    temp_content = {
        "full_text": comparative_prompt,
        "token_count": len(comparative_prompt.split()),
        "filename": "comparative_analysis",
    }

    limits_check = check_context_limits(temp_content, max_tokens=1_000_000)
    if not limits_check["within_limits"]:
        st.info(f"Optimising comparative content ({temp_content['token_count']:,} tokens).")
        comparative_prompt = optimize_text_for_context(comparative_prompt, target_tokens=800_000)

    llm_data = {
        "full_text": comparative_prompt,
        "token_count": len(comparative_prompt.split()),
        "filename": "comparative_analysis",
    }

    response_container = st.empty()
    accumulated_response = ""
    progress_message = st.info("Generating comparative analysis...")

    try:
        for chunk in comparator.client.answer_specific_query_streaming(llm_data, question):
            accumulated_response += chunk
            response_container.markdown(accumulated_response)

        progress_message.success("Comparative analysis completed!")
    except Exception as stream_error:
        progress_message.error(f"Streaming error: {stream_error}")
        st.warning("Falling back to standard response mode...")
        response = comparator.client.answer_specific_query(llm_data, question)
        response_container.markdown(response)
        st.success("Comparative analysis completed (fallback mode).")


def _summary_across_all(selected: list[str], question: str) -> None:
    st.write("**Summary Across All Suppliers:**")

    all_raw_content = []
    for provider_name in selected:
        provider = next(provider for provider in completed_providers if provider["name"] == provider_name)
        extracted_data = provider["extracted_data"]
        raw_content = extracted_data.get("_raw_content", "")

        if raw_content:
            all_raw_content.append(f"=== {provider_name.upper()} ===\n{raw_content}")
        else:
            st.warning(f"No raw content available for {provider_name}. Using extracted features as fallback.")
            content_summary = f"Supplier: {provider_name}\n\n"
            for category, fields in extracted_data.items():
                if not category.startswith("_") and isinstance(fields, dict):
                    content_summary += f"{category.replace('_', ' ').title()}:\n"
                    for field, value in fields.items():
                        if value and value != "Not Found":
                            content_summary += f"- {field.replace('_', ' ').title()}: {value}\n"
                    content_summary += "\n"
            all_raw_content.append(f"=== {provider_name.upper()} ===\n{content_summary}")

    comprehensive_content = f"""
Based on complete document content from {len(selected)} suppliers, please provide a comprehensive summary answering: "{question}"

Suppliers: {', '.join(selected)}

Complete Document Content:
{'\n\n'.join(all_raw_content)}

Please synthesise the information and provide key insights, trends, and recommendations.
"""

    from document_processor import check_context_limits, optimize_text_for_context

    temp_content = {
        "full_text": comprehensive_content,
        "token_count": len(comprehensive_content.split()),
        "filename": "summary_analysis",
    }

    limits_check = check_context_limits(temp_content, max_tokens=1_000_000)
    if not limits_check["within_limits"]:
        st.info(f"Optimising large summary content ({temp_content['token_count']:,} tokens).")
        comprehensive_content = optimize_text_for_context(comprehensive_content, target_tokens=800_000)

    llm_data = {
        "full_text": comprehensive_content,
        "token_count": len(comprehensive_content.split()),
        "filename": "summary_analysis",
    }

    summary_container = st.empty()
    accumulated_summary = ""
    progress_message = st.info("Generating summary analysis...")

    try:
        for chunk in comparator.client.answer_specific_query_streaming(llm_data, question):
            accumulated_summary += chunk
            summary_container.markdown(accumulated_summary)

        progress_message.success("Summary analysis completed!")
    except Exception as stream_error:
        progress_message.error(f"Streaming error: {stream_error}")
        st.warning("Falling back to standard response...")
        summary_response = comparator.client.answer_specific_query(llm_data, question)
        summary_container.markdown(summary_response)
        st.success("Summary analysis completed (fallback mode).")


if st.button("Ask Question", type="primary", key="ask_question_main"):
    if not query:
        st.warning("Please enter a question.")
    elif not selected_providers:
        st.warning("Select at least one supplier.")
    else:
        st.subheader("Responses")
        try:
            if query_mode == "Individual Responses":
                _stream_individual_responses(selected_providers, query)
            elif query_mode == "Comparative Analysis":
                _comparative_analysis(selected_providers, query)
            else:
                _summary_across_all(selected_providers, query)

            st.session_state.chat_history.append(
                {
                    "query": query,
                    "providers": selected_providers,
                    "mode": query_mode,
                    "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
                }
            )
        except Exception as err:
            st.error(f"Error processing query: {err}")

if st.session_state.chat_history:
    st.divider()
    st.subheader("Recent Chat History")
    recent_chats = list(reversed(st.session_state.chat_history[-5:]))

    for chat in recent_chats:
        with st.expander(f"[{chat['timestamp']}] {chat['query'][:50]}...", expanded=False):
            st.write(f"**Question:** {chat['query']}")
            st.write(f"**Suppliers:** {', '.join(chat.get('providers', []))}")
            st.write(f"**Mode:** {chat.get('mode', 'Individual Responses')}")

    if st.button("Clear Chat History", key="clear_chat_history"):
        st.session_state.chat_history = []
        st.rerun()
