"""
Non-Catalog Purchase Module
Handles text-based purchase requests when products are not found in catalog.
Uses LLM to recommend commodity codes and vendors based on user text input.
"""

import os
import pandas as pd
import streamlit as st
import numpy as np
from typing import List, Dict, Any
import json
import random
from utils import safe_dataframe

from dotenv import load_dotenv


load_dotenv()


#############################################################################
# AI FUNCTIONS FOR TEXT ANALYSIS
#############################################################################
def analyze_text_with_llm(
    user_text: str,
    materials_df: pd.DataFrame,
    vendors_df: pd.DataFrame,
    use_sap_ai_hub: bool = False,
) -> Dict[str, Any]:
    """
    Analyze user text and recommend commodity code and vendor using LLM
    """

    # Get unique commodity codes from materials
    commodity_codes = materials_df["COMMODITY_CODE"].unique().tolist()

    # Get vendor information
    vendor_info = []
    for _, vendor in vendors_df.iterrows():
        vendor_info.append(
            {
                "name": vendor["VENDOR_NAME"],
                "commodity_code": vendor.get("COMMODITY_CODE", ""),
                "contract": vendor.get("CONTRACT_EXISTS", "NO"),
                "guaranteed_spend": vendor.get("GUARANTEED_SPEND", 0),
            }
        )

    # Create LLM prompt for analysis
    analysis_prompt = f"""
You are a procurement expert analyzing a purchase request for a product not in our catalog.

User Request: "{user_text}"

Available Commodity Codes in our system:
{json.dumps(commodity_codes, indent=2)}

Available Vendors:
{json.dumps(vendor_info, indent=2)}

Instructions:
1. Analyze the user's text to understand what they want to buy
2. Recommend the MOST APPROPRIATE commodity code from our available codes
3. Recommend the BEST vendor who can supply this type of product
4. Consider vendor contracts and specializations
5. Provide reasoning for your recommendations

Respond in JSON format:
{{
    "recommended_commodity_code": "selected_code",
    "commodity_reasoning": "why this commodity code fits",
    "recommended_vendor": "vendor_name", 
    "vendor_reasoning": "why this vendor is best",
    "estimated_price": estimated_price_number,
    "price_reasoning": "basis for price estimate"
}}
"""

    try:
        if use_sap_ai_hub:
            llm_response = ask_llm_real(analysis_prompt)
        else:
            llm_response = ask_llm_mock(analysis_prompt, user_text)

        # Parse JSON response
        try:
            analysis_result = json.loads(llm_response)
            return analysis_result
        except json.JSONDecodeError:
            # Fallback if JSON parsing fails
            return create_fallback_analysis(user_text, commodity_codes, vendor_info)

    except Exception as e:
        st.error(f"LLM analysis failed: {e}")
        return create_fallback_analysis(user_text, commodity_codes, vendor_info)


def ask_llm_real(prompt: str) -> str:
    """Ask LLM using SAP AI Hub"""
    try:
        from gen_ai_hub.proxy.core.proxy_clients import get_proxy_client
        from gen_ai_hub.proxy.langchain.openai import ChatOpenAI

        proxy_client = get_proxy_client("gen-ai-hub")
        llm = ChatOpenAI(
            proxy_model_name="gpt-4.1", proxy_client=proxy_client
        )  # Updated to use GPT-4o
        response = llm.invoke(prompt).content
        return response
    except Exception as e:
        raise e


def ask_llm_mock(prompt: str, user_text: str) -> str:
    """Smart mock LLM for text analysis"""
    user_text_lower = user_text.lower()

    # Simple logic for commodity code recommendation
    if any(
        word in user_text_lower for word in ["laptop", "computer", "notebook", "pc"]
    ):
        commodity_code = "43211503"
        vendor = "DELL MARKETING LP"
        price = 1200
    elif any(
        word in user_text_lower for word in ["chair", "desk", "furniture", "table"]
    ):
        commodity_code = "56101500"
        vendor = "CORPORATE FURNITURE CO"
        price = 400
    elif any(
        word in user_text_lower
        for word in ["paper", "supplies", "stationary", "office"]
    ):
        commodity_code = "14111700"
        vendor = "OFFICE DEPOT BUSINESS"
        price = 50
    elif any(word in user_text_lower for word in ["printer", "print", "toner", "ink"]):
        commodity_code = "43212100"
        vendor = "HP INC"
        price = 300
    else:
        # Generic recommendation
        commodity_code = "14111700"
        vendor = "BUSINESS SOLUTIONS INC"
        price = 100

    mock_response = {
        "recommended_commodity_code": commodity_code,
        "commodity_reasoning": f"Based on keywords in '{user_text}', this commodity code best matches the product category",
        "recommended_vendor": vendor,
        "vendor_reasoning": "This vendor has contracts and specializes in this commodity category",
        "estimated_price": price,
        "price_reasoning": "Estimated based on similar products in this category",
    }

    return json.dumps(mock_response, indent=2)


def create_fallback_analysis(
    user_text: str, commodity_codes: List, vendor_info: List
) -> Dict[str, Any]:
    """Create fallback analysis if LLM fails"""
    return {
        "recommended_commodity_code": (
            commodity_codes[0] if commodity_codes else "14111700"
        ),
        "commodity_reasoning": "Default commodity code selected",
        "recommended_vendor": (
            vendor_info[0]["name"] if vendor_info else "Default Vendor"
        ),
        "vendor_reasoning": "Default vendor selected",
        "estimated_price": 100,
        "price_reasoning": "Default price estimate",
    }


#############################################################################
# NON-CATALOG PURCHASE INTERFACE
#############################################################################
def show_non_catalog_purchase(
    user_query: str,
    materials_df: pd.DataFrame,
    vendors_df: pd.DataFrame,
    use_sap_ai_hub: bool = False,
):
    """
    Show the non-catalog purchase interface with LLM recommendations
    """

    st.markdown("---")
    st.markdown("### **Non-Catalog Purchase Order**")
    st.info(f'Creating purchase order for: *"{user_query}"*')

    # Get LLM analysis
    with st.spinner("Analyzing your order and making recommendations..."):
        analysis = analyze_text_with_llm(
            user_query, materials_df, vendors_df, use_sap_ai_hub
        )

    # Show AI recommendations
    st.markdown("#### **AI Recommendations**")

    col1, col2 = st.columns(2)

    with col1:
        st.success(
            f"**Recommended Commodity Code:** {analysis['recommended_commodity_code']}"
        )
        st.write(f"{analysis['commodity_reasoning']}")

    with col2:
        st.success(f"**Recommended Vendor:** {analysis['recommended_vendor']}")
        st.write(f"{analysis['vendor_reasoning']}")

    # Editable purchase details
    st.markdown("---")
    st.markdown("#### **Purchase Details (Editable)**")

    col1, col2 = st.columns(2)

    with col1:
        # Product description (editable)
        product_description = st.text_area(
            "**Product Description:**",
            value=user_query,
            height=100,
            key="non_catalog_description",
        )

        # Quantity (editable)
        quantity = st.number_input(
            "**Quantity:**", min_value=1, value=1, key="non_catalog_quantity"
        )

        # Price (editable)
        estimated_price = st.number_input(
            "**Unit Price (€):**",
            min_value=0.01,
            value=float(analysis["estimated_price"]),
            format="%.2f",
            key="non_catalog_price",
        )

    with col2:
        # Commodity Code (selectable)
        commodity_codes = materials_df["COMMODITY_CODE"].unique().tolist()
        try:
            default_index = commodity_codes.index(
                analysis["recommended_commodity_code"]
            )
        except ValueError:
            default_index = 0

        selected_commodity = st.selectbox(
            "**Commodity Code:**",
            commodity_codes,
            index=default_index,
            key="non_catalog_commodity",
        )

        # Vendor (selectable)
        vendor_names = vendors_df["VENDOR_NAME"].unique().tolist()
        try:
            default_vendor_index = vendor_names.index(analysis["recommended_vendor"])
        except ValueError:
            default_vendor_index = 0

        selected_vendor = st.selectbox(
            "**Vendor:**",
            vendor_names,
            index=default_vendor_index,
            key="non_catalog_vendor",
        )

        # Unit of measure
        unit_of_measure = st.selectbox(
            "**Unit of Measure:**",
            ["EA", "PK", "LB", "KG", "M", "FT"],
            index=0,
            key="non_catalog_uom",
        )

    # Calculate total
    total_price = quantity * estimated_price

    # Show selected vendor details
    selected_vendor_row = vendors_df[vendors_df["VENDOR_NAME"] == selected_vendor].iloc[
        0
    ]

    st.markdown("---")
    st.markdown("#### **Selected Vendor Details**")

    col1, col2, col3 = st.columns(3)

    with col1:
        contract_status = (
            "Yes"
            if str(selected_vendor_row.get("CONTRACT_EXISTS", "")).upper() == "YES"
            else "No"
        )
        st.write(f"**Contract:** {contract_status}")

    with col2:
        guaranteed = selected_vendor_row.get("GUARANTEED_SPEND", 0)
        try:
            guaranteed = float(guaranteed) if guaranteed else 0
            guaranteed_text = f"€{guaranteed:,.0f}" if guaranteed > 0 else "None"
        except:
            guaranteed_text = "None"
        st.write(f"**Guaranteed Spend:** {guaranteed_text}")

    with col3:
        st.write(
            f"**Payment Terms:** {selected_vendor_row.get('PAYMENT_TERMS', 'Net 30')}"
        )

    # Purchase summary
    st.markdown("---")
    st.markdown("### **Purchase Summary**")

    summary_data = {
        "Item": [product_description],
        "Commodity Code": [selected_commodity],
        "Quantity": [quantity],
        "Unit Price": [f"€{estimated_price:.2f}"],
        "UoM": [unit_of_measure],
        "Total": [f"€{total_price:.2f}"],
    }

    summary_df = pd.DataFrame(summary_data)
    # st.dataframe(summary_df, use_container_width=True, hide_index=True) original
    safe_dataframe(summary_df, use_container_width=True, hide_index=True)

    st.markdown(f"### **Total Amount: €{total_price:.2f}**")
    st.markdown(f"**Vendor: {selected_vendor}**")

    # Action buttons
    st.markdown("---")
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("Cancel", use_container_width=True):
            st.session_state.show_non_catalog = False
            st.rerun()

    with col2:
        if st.button("Save Draft", use_container_width=True):
            st.success("Draft saved.")

    with col3:
        if st.button("Create PO", use_container_width=True, type="primary"):
            # Create the non-catalog PO
            create_non_catalog_pr(
                description=product_description,
                commodity_code=selected_commodity,
                quantity=quantity,
                unit_price=estimated_price,
                unit_of_measure=unit_of_measure,
                vendor=selected_vendor,
                vendor_data=selected_vendor_row,
                total_amount=total_price,
            )


def create_non_catalog_pr(
    description: str,
    commodity_code: str,
    quantity: int,
    unit_price: float,
    unit_of_measure: str,
    vendor: str,
    vendor_data: pd.Series,
    total_amount: float,
):
    """Create a non-catalog Purchase Order"""

    st.markdown("---")
    st.success("Non-Catalog Purchase Order created successfully.")
    st.balloons()

    pr_number = f"NCR-{pd.Timestamp.now().strftime('%Y%m%d%H%M%S')}"

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.info(f"**PO Number:** {pr_number}")
        st.info(f"**Type:** Non-Catalog Purchase")
        st.info(f"**Total Amount:** €{total_amount:.2f}")
        st.info(f"**Vendor:** {vendor}")
        st.info(f"**Commodity Code:** {commodity_code}")

        if st.button("Create New Purchase", use_container_width=True, type="primary"):
            # Clear session state and start fresh
            for key in list(st.session_state.keys()):
                if key.startswith(
                    (
                        "materials_",
                        "search_",
                        "selected_",
                        "show_",
                        "vendors_",
                        "non_catalog",
                    )
                ):
                    del st.session_state[key]
            st.rerun()


#############################################################################
# MAIN FUNCTION TO CHECK IF PRODUCT NOT FOUND
#############################################################################
def should_show_non_catalog_option(
    search_results: pd.DataFrame, user_query: str
) -> bool:
    """
    Determine if we should show the non-catalog option
    Returns True if no good results found
    """
    # If no results at all
    if search_results.empty:
        return True

    # Check for obviously unrelated queries
    unrelated_terms = [
        "elephant",
        "dinosaur",
        "unicorn",
        "dragon",
        "spaceship",
        "castle",
        "rainbow",
        "magic",
        "alien",
        "monster",
    ]
    if any(term in user_query.lower() for term in unrelated_terms):
        return True

    # Check if results have very low relevance scores
    if "relevance_score" in search_results.columns:
        max_relevance = search_results["relevance_score"].max()
        if max_relevance < 0.3:  # Threshold for "not found"
            return True

    # If we have less than 3 results, probably not a good match
    if len(search_results) < 3:
        return True

    return False


def get_non_catalog_message(user_query: str) -> str:
    """Generate appropriate message when product not found in catalog"""
    return f'Product not found in catalog for "{user_query}". Try Non-Catalog Purchase below.'
