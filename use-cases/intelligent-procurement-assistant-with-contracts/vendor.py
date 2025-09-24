"""
Vendor Recommendation Module
Handles vendor data loading and intelligent vendor recommendations
based on contract status, guaranteed spend, and commodity matching.
"""

import os
import pandas as pd
import streamlit as st
import numpy as np
from typing import List, Dict, Any
import json
import random

from dotenv import load_dotenv
from utils import load_catalog_file, create_catalog_download_button

load_dotenv()

############################################################################
# AI CORE GEN IN access
############################################################################
from gen_ai_hub.proxy.core.proxy_clients import get_proxy_client
from gen_ai_hub.proxy.langchain.openai import ChatOpenAI

proxy_client = get_proxy_client("gen-ai-hub")
MODEL_NAME = "gpt-4.1"


def get_vendor_recommended_codes(vendor_row):
    code_list = vendor_row.get("Vendor Commodity Code List", "")
    codes = []
    for code in str(code_list).split(";"):
        code = code.strip()
        if not code:
            continue
        code_only = code.split()[0] if " " in code else code
        codes.append(code_only)
    return codes


def ask_llm_for_commodity_code(
    item_description: str, vendor_name: str, vendor_codes: list, all_codes: list
) -> str:
    codes_text = "\n".join(
        [
            f"{code['Commodity Code']} - {code['Commodity Description']}"
            for code in all_codes
        ]
    )
    prompt = f"""You are an expert in procurement classification.
You must determine the best Commodity Code for the given item description.

Item Description: "{item_description}"
Vendor: "{vendor_name if vendor_name else 'Unknown'}"
Vendor's Preferred Codes: {', '.join(vendor_codes) if vendor_codes else 'None'}

Here is the full list of available Commodity Codes:
{codes_text}

Instructions:
- If the vendor is known and has preferred codes, prioritize those codes unless there is a clear mismatch.
- For ambiguous names (like "notebook"), match based on vendor's industry. For example, DELL = computer notebook, OFFICE PRODUCTS = paper notebook.
- If you are not sure, choose the code that matches most closely based on both description and vendor profile.
- Reply with one line only in the format: <Commodity Code> - <Commodity Description>
"""
    llm = ChatOpenAI(proxy_model_name=MODEL_NAME, proxy_client=proxy_client)
    response = llm.invoke(prompt).content
    return response.strip().split("\n")[0]


def get_best_commodity_code_for_item(
    item_description: str,
    vendor_name: str = None,
    vendors_df: pd.DataFrame = None,
    all_codes_df: pd.DataFrame = None,
) -> str:
    if all_codes_df is None:
        all_codes_df = pd.read_csv("Ariba Material Commodity Codes.csv")
    all_codes = all_codes_df.to_dict(orient="records")
    vendor_codes = []
    if vendor_name and vendors_df is not None:
        row = vendors_df[vendors_df["VENDOR_NAME"].str.lower() == vendor_name.lower()]
        if not row.empty:
            vclist = row.iloc[0].get("Vendor Commodity Code List")
            if pd.notnull(vclist):
                vendor_codes = [x.strip() for x in vclist.split(";") if x.strip()]
    result = ask_llm_for_commodity_code(
        item_description, vendor_name, vendor_codes, all_codes
    )
    return result


def format_currency(value):
    try:
        return f"€{float(value):,.2f}"
    except (ValueError, TypeError):
        return f"€{0:,.2f}"


def format_number(value):
    try:
        return f"{float(value):,.2f}"
    except (ValueError, TypeError):
        return f"{0:,.2f}"


@st.cache_data
def load_vendors_data() -> pd.DataFrame:
    file_paths = ["vendors.csv", "data/vendors.csv"]
    df = pd.DataFrame()
    for file_path in file_paths:
        if os.path.exists(file_path):
            try:
                df = pd.read_csv(file_path)
                break
            except Exception as e:
                st.error(f"Error reading {file_path}: {e}")
                continue
    if df.empty:
        st.warning("vendors.csv not found, creating sample data...")
        sample_vendors = {
            "VENDOR_ID": ["VEN001", "VEN002", "VEN003", "VEN004", "VEN005", "VEN006"],
            "VENDOR_NAME": [
                "TechSource Pro",
                "Global IT Solutions",
                "Office Depot Business",
                "Corporate Furniture Co",
                "Premium Tech Suppliers",
                "Business Solutions Inc",
            ],
            "VENDOR_DESCRIPTION": [
                "Leading supplier of enterprise laptops, computers and IT hardware",
                "Comprehensive IT solutions provider specializing in business computers",
                "Office supplies and business essentials provider",
                "Professional office furniture and ergonomic solutions",
                "High-end technology equipment and premium IT hardware",
                "Complete business solutions including office supplies and equipment",
            ],
            "Vendor Commodity Code List": [
                "43211503.0 Laptop/Desktop; 43211900.0 Computer Monitors",
                "43210000.0 IT Hardware and Accessories; 43212100.0 Printers",
                "44120000.0 Office Supplies; 44122011.0 Paper notebooks",
                "56101500.0 Office Furniture; 56101703.0 Ergonomic Chairs",
                "43210000.0 IT Hardware and Accessories; 43212200.0 Computer data storage",
                "44120000.0 Office Supplies; 56101703.0 Office Furniture",
            ],
            "CONTRACT_EXISTS": ["YES", "YES", "NO", "YES", "NO", "YES"],
            "GUARANTEED_SPEND": [50000, 75000, 0, 25000, 0, 30000],
            "RATING": [4.8, 4.6, 4.2, 4.5, 4.9, 4.3],
            "PRICE_TIER": [
                "Premium",
                "Standard",
                "Budget",
                "Standard",
                "Premium",
                "Budget",
            ],
            "DELIVERY_TIME": [2, 3, 1, 5, 3, 2],
            "LOCATION": [
                "New York",
                "California",
                "Texas",
                "Illinois",
                "Washington",
                "Florida",
            ],
            "PAYMENT_TERMS": [
                "Net 30",
                "Net 45",
                "Net 15",
                "Net 30",
                "Net 60",
                "Net 30",
            ],
        }
        df = pd.DataFrame(sample_vendors)
    required_columns = ["VENDOR_NAME", "Vendor Commodity Code List", "CONTRACT_EXISTS"]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        st.error(f"Missing required vendor columns: {missing_columns}")
        st.info(f"Available columns: {list(df.columns)}")
        return pd.DataFrame()
    return df


def find_best_vendor(
    selected_materials: Dict, vendors_df: pd.DataFrame
) -> Dict[str, Any]:
    if vendors_df.empty:
        return None
    selected_commodities = set()
    for item in selected_materials.values():
        commodity = item.get("commodity", "")
        if commodity:
            selected_commodities.add(str(commodity))
    contract_vendors = vendors_df[
        vendors_df["CONTRACT_EXISTS"].str.upper() == "YES"
    ].copy()
    if contract_vendors.empty:
        contract_vendors = vendors_df.copy()
        st.warning("No vendors with contracts found, considering all vendors")
    vendor_scores = []
    for idx, vendor in contract_vendors.iterrows():
        score = 0
        reasons = []
        contract_exists = str(vendor.get("CONTRACT_EXISTS", "")).upper()
        if contract_exists == "YES":
            score += 100
            reasons.append("has active contract")
        guaranteed_spend = vendor.get("GUARANTEED_SPEND", 0)
        try:
            guaranteed_spend = float(guaranteed_spend) if guaranteed_spend else 0
        except (ValueError, TypeError):
            guaranteed_spend = 0
        if guaranteed_spend > 0:
            score += 50
            reasons.append(f"guaranteed spend of {format_currency(guaranteed_spend)}")
        vendor_commodities = str(vendor.get("Vendor Commodity Code List", "")).lower()
        commodity_matches = 0
        for commodity in selected_commodities:
            # теперь ищем совпадения по всем кодам из списка
            vendor_codes = [
                c.strip().split()[0] for c in vendor_commodities.split(";") if c.strip()
            ]
            if any(str(commodity).lower() == code.lower() for code in vendor_codes):
                commodity_matches += 1
        if commodity_matches > 0:
            score += commodity_matches * 20
            reasons.append(f"matches {commodity_matches} commodity categories")
        rating = vendor.get("RATING", 0)
        try:
            rating = float(rating) if rating else 0
        except (ValueError, TypeError):
            rating = 0
        if rating > 0:
            score += rating * 5
            reasons.append(f"rating of {rating}/5.0")
        vendor_scores.append(
            {
                "index": idx,
                "vendor": vendor,
                "score": score,
                "reasons": reasons,
                "guaranteed_spend": guaranteed_spend,
            }
        )
    vendor_scores.sort(key=lambda x: x["score"], reverse=True)
    if vendor_scores:
        best_vendor = vendor_scores[0]
        return {
            "vendor_data": best_vendor["vendor"],
            "reasons": best_vendor["reasons"],
            "score": best_vendor["score"],
            "guaranteed_spend": best_vendor["guaranteed_spend"],
            "index": best_vendor["index"],
        }
    return None


def get_vendor_recommendations(
    selected_materials: Dict, use_sap_ai_hub: bool = False
) -> bool:
    vendors_df = load_vendors_data()
    if vendors_df.empty:
        st.error("Could not load vendor data")
        return False
    st.session_state.vendors_df = vendors_df
    if st.session_state.get("vendor_manually_selected", False) and st.session_state.get(
        "selected_vendor_data"
    ):
        vendor_data = st.session_state.selected_vendor_data
        st.markdown("### **Selected Vendor**")
        st.success(f"**Vendor Selected:** {vendor_data['name']} (manually selected)")
        st.info("This vendor was manually selected by you")
    else:
        best_vendor_result = find_best_vendor(selected_materials, vendors_df)
        if not best_vendor_result:
            st.error("No suitable vendors found")
            return False
        st.markdown("### **Recommended Vendor**")
        vendor = best_vendor_result["vendor_data"]
        guaranteed_spend = best_vendor_result["guaranteed_spend"]
        vendor_name = vendor["VENDOR_NAME"]
        explanation_parts = []
        if vendor["CONTRACT_EXISTS"].upper() == "YES":
            explanation_parts.append("it has an active contract")
        # if guaranteed_spend and guaranteed_spend > 0:
        #     explanation_parts.append(
        #         f"guaranteed spend of {format_currency(guaranteed_spend)}"
        #     )
        explanation_parts.append(f"shortest delivery time of 2 days")
        explanation = " and ".join(explanation_parts)
        st.success(f"**Vendor Selected:** {vendor_name}")
        st.info(f"**Reason:** Selected because {explanation}")
        st.session_state.selected_vendor_data = {
            "name": vendor_name,
            "payment_terms": vendor.get("PAYMENT_TERMS", "Net 30"),
            "vendor_full_data": vendor,
        }
    if st.button("Select Another Vendor", key="change_vendor_simple"):
        st.session_state.show_vendor_list = True
    if st.session_state.get("show_vendor_list", False):
        st.markdown("---")
        st.markdown("### **Choose Different Vendor**")
        vendor_names = vendors_df["VENDOR_NAME"].tolist()
        current_vendor_name = st.session_state.selected_vendor_data.get(
            "name", vendor_names[0]
        )
        current_index = 0
        if current_vendor_name in vendor_names:
            current_index = vendor_names.index(current_vendor_name)
        selected_vendor_name = st.selectbox(
            "Select a vendor:",
            vendor_names,
            index=current_index,
            key="vendor_selectbox",
        )
        selected_vendor_row = vendors_df[
            vendors_df["VENDOR_NAME"] == selected_vendor_name
        ].iloc[0]
        col1, col2 = st.columns(2)
        with col1:
            contract_flag = (
                "Yes"
                if str(selected_vendor_row.get("CONTRACT_EXISTS", "")).upper() == "YES"
                else "No"
            )
            st.write(f"**Contract:** {contract_flag}")
            st.write(f"**Rating:** {selected_vendor_row.get('RATING', 'N/A')}/5.0")
        with col2:
            guaranteed = selected_vendor_row.get("GUARANTEED_SPEND", 0)
            try:
                guaranteed = float(guaranteed) if guaranteed else 0
                st.write(
                    f"**Guaranteed Spend:** {format_currency(guaranteed)}"
                    if guaranteed > 0
                    else "**Guaranteed Spend:** None"
                )
            except:
                st.write("**Guaranteed Spend:** None")
            st.write(
                f"**Payment Terms:** {selected_vendor_row.get('PAYMENT_TERMS', 'Net 30')}"
            )
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Confirm Selection", key="confirm_vendor_selection"):
                new_vendor_data = {
                    "name": selected_vendor_row["VENDOR_NAME"],
                    "payment_terms": selected_vendor_row.get("PAYMENT_TERMS", "Net 30"),
                    "vendor_full_data": selected_vendor_row,
                }
                st.session_state.selected_vendor_data = new_vendor_data
                st.session_state.vendor_manually_selected = True
                st.session_state.show_vendor_list = False
                if "vendor_selectbox" in st.session_state:
                    del st.session_state["vendor_selectbox"]
                st.success(f"Vendor changed to {selected_vendor_row['VENDOR_NAME']}!")
                st.rerun()
        with col2:
            if st.button("Cancel", key="cancel_vendor_selection"):
                st.session_state.show_vendor_list = False
                if "vendor_selectbox" in st.session_state:
                    del st.session_state["vendor_selectbox"]
                st.rerun()
    show_purchase_request_summary(
        selected_materials, st.session_state.selected_vendor_data
    )
    return True


def show_purchase_request_summary(selected_materials: Dict, vendor_data: Dict):
    st.markdown("---")
    st.markdown("### **Purchase Order Summary**")
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Company Code:** 1000 IPA")
        st.write("**Purchase Org:** 1000 - Intelligent Procurement Assistant")
        st.write("**Cost Center:** 0007001358")
    with col2:
        delivery_address = st.text_area(
            "**Delivery Address:**",
            value="Intelligent Procurement Assistant\n5 Street\nCincinnati, OH",
            height=100,
            key="delivery_address",
        )
    st.markdown("---")
    st.markdown("#### **Materials & Details**")
    total_amount = 0
    for item_idx, item in enumerate(selected_materials.values()):
        st.write(f"**Material:** {item['description']}")

        # Display Material Number or mock button if missing
        material_id = item.get("material_id", "")
        if not pd.isna(material_id) and str(material_id).strip():
            # Format as integer if it's a numeric value
            try:
                if float(material_id).is_integer():
                    st.write(f"**Material Number:** {int(float(material_id))}")
                else:
                    st.write(f"**Material Number:** {material_id}")
            except (ValueError, TypeError):
                st.write(f"**Material Number:** {material_id}")
        else:
            col1, col2 = st.columns([3, 1])
            with col1:
                st.write("**Material Number:** Not Available")
            with col2:
                if st.button(
                    "Add Material Number",
                    key=f"add_material_id_{item_idx}",
                    help="Click to add a Material Number for this item",
                ):
                    st.success(
                        "Material Number request submitted successfully! Workflow has been started."
                    )

        # Display Catalog information
        catalog = item.get("catalog", "")
        if not pd.isna(catalog) and str(catalog).strip():
            # Use the robust catalog loader
            success, pdf_data, error_msg = load_catalog_file(catalog, debug=False)

            if success:
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.write(f"**Catalog:** {catalog}")
                with col2:
                    # Use session-based download button to avoid multi-instance issues
                    create_catalog_download_button(
                        pdf_data,
                        catalog,
                        f"vendor_catalog_{item_idx}",
                    )
            else:
                st.write(f"**Catalog:** {catalog} (not available)")
                if error_msg:
                    print(f"Catalog download error for {catalog}: {error_msg}")
        else:
            st.write("**Catalog:** Not Available")

        if item.get("commodity_description"):
            st.write(f"**Category:** {item['commodity_description']}")
        st.write(f"**Quantity:** {format_number(item['quantity'])}")
        st.write(f"**Unit of Measure:** {item.get('unit_of_measure', 'EA')}")
        st.write(f"**Unit Price:** {format_currency(item['price'])}")
        st.write(f"**Total:** {format_currency(item['total'])}")
        total_amount += item["total"]
        st.markdown("---")
    st.markdown("#### **Vendor & Payment Details**")
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"**Vendor:** {vendor_data['name']}")
        st.write(f"**Payment Terms:** {vendor_data['payment_terms']}")
    with col2:
        st.write(f"**Total Amount:** {format_currency(total_amount)}")
        st.write(f"**Currency:** EUR")
    st.markdown("---")
    if st.button("Create Purchase Order", type="primary", key="create_pr_simple"):
        st.success("Purchase Order created successfully.")
        st.balloons()
        pr_number = f"PO-{pd.Timestamp.now().strftime('%Y%m%d%H%M%S')}"
        st.info(f"**PO Number:** {pr_number}")
        st.info(f"**Total Amount:** {format_currency(total_amount)}")
        st.info(f"**Vendor:** {vendor_data['name']}")
        if st.button("Create New Purchase Order", key="new_pr_simple"):
            for key in list(st.session_state.keys()):
                if key.startswith(
                    ("materials_", "search_", "selected_", "show_", "vendors_")
                ):
                    del st.session_state[key]
            st.rerun()


def get_vendor_info(vendors_df: pd.DataFrame) -> Dict[str, Any]:
    if vendors_df.empty:
        return {"count": 0, "status": "empty"}
    contract_count = len(vendors_df[vendors_df["CONTRACT_EXISTS"].str.upper() == "YES"])
    guaranteed_spend_count = len(vendors_df[vendors_df.get("GUARANTEED_SPEND", 0) > 0])
    return {
        "count": len(vendors_df),
        "with_contracts": contract_count,
        "with_guaranteed_spend": guaranteed_spend_count,
        "columns": list(vendors_df.columns),
        "status": "loaded",
    }
