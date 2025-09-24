import streamlit as st
import pandas as pd
from typing import Dict, Any
import os
from utils import (
    safe_dataframe,
    load_css_files,
    load_catalog_file,
    get_file_debug_info,
    create_catalog_download_button,
)
from dotenv import load_dotenv

# Constants
APP_TITLE = "Update SUMMA Product Catalog"

# Page Configuration
st.set_page_config(
    page_title=APP_TITLE, page_icon="static/images/SAP_logo_square.png", layout="wide"
)
st.logo("static/images/SAP_logo.svg")

# Header
st.title(APP_TITLE)

# Load custom CSS
css_files = [
    os.path.join(os.path.dirname(__file__), "static", "styles", "variables.css"),
    os.path.join(os.path.dirname(__file__), "static", "styles", "style.css"),
]
load_css_files(css_files)

# =====================================================================
# STREAMLIT COMPATIBILITY FIXES - MUST BE FIRST
# =====================================================================

# Fix 1: st.rerun() compatibility
if not hasattr(st, "rerun"):
    if hasattr(st, "experimental_rerun"):
        st.rerun = st.experimental_rerun
    else:

        def dummy_rerun():
            st.error("Streamlit rerun not available. Please update Streamlit.")

        st.rerun = dummy_rerun


# Fix 2: safe_dataframe function for hide_index compatibility
def safe_dataframe(df, use_container_width=True, hide_index=True, **kwargs):
    """
    Compatible dataframe function that works with both old and new Streamlit versions
    """
    try:
        # Try new method first (Streamlit 1.25+)
        return st.dataframe(
            df, use_container_width=use_container_width, hide_index=hide_index, **kwargs
        )
    except TypeError:
        # Fallback for old versions - use pandas styling
        if hide_index:
            try:
                styled_df = df.style.hide(axis="index")
                return st.dataframe(
                    styled_df, use_container_width=use_container_width, **kwargs
                )
            except:
                try:
                    styled_df = df.style.hide_index()
                    return st.dataframe(
                        styled_df, use_container_width=use_container_width, **kwargs
                    )
                except:
                    # Last resort: reset index
                    display_df = df.reset_index(drop=True)
                    return st.dataframe(
                        display_df, use_container_width=use_container_width, **kwargs
                    )
        else:
            return st.dataframe(df, use_container_width=use_container_width, **kwargs)


# =====================================================================

# Page config already set above with SAP logo and wide layout

# Load environment variables
load_dotenv()

# Import our modules
try:
    import materials
    import vendor
    import text
    import quotation  # New quotation module

    MODULES_AVAILABLE = True
except ImportError as e:
    st.error(f"Module import error: {e}")
    st.stop()

#############################################################################
# SAP AI HUB CONNECTION CHECK (Internal - No UI)
#############################################################################
USE_SAP_AI_HUB = False
sap_ai_hub_error = None

# Check required environment variables
required_vars = [
    "AICORE_AUTH_URL",
    "AICORE_CLIENT_ID",
    "AICORE_CLIENT_SECRET",
    "AICORE_BASE_URL",
    "AICORE_RESOURCE_GROUP",
]

missing_vars = [var for var in required_vars if not os.getenv(var)]

if missing_vars:
    print(f"DEBUG: Missing environment variables: {missing_vars}")
    sap_ai_hub_error = f"Missing environment variables: {missing_vars}"
else:
    print("DEBUG: All environment variables found")

    try:
        from gen_ai_hub.proxy.core.proxy_clients import get_proxy_client

        print("DEBUG: gen_ai_hub module imported successfully")

        # Set environment variables that gen_ai_hub expects
        os.environ["AICORE_AUTH_URL"] = os.getenv("AICORE_AUTH_URL")
        os.environ["AICORE_CLIENT_ID"] = os.getenv("AICORE_CLIENT_ID")
        os.environ["AICORE_CLIENT_SECRET"] = os.getenv("AICORE_CLIENT_SECRET")
        os.environ["AICORE_BASE_URL"] = os.getenv("AICORE_BASE_URL")
        os.environ["AICORE_RESOURCE_GROUP"] = os.getenv("AICORE_RESOURCE_GROUP")

        proxy_client = get_proxy_client("gen-ai-hub")
        print("DEBUG: Proxy client created successfully")

        # Test LLM connection
        try:
            from gen_ai_hub.proxy.langchain.openai import ChatOpenAI

            llm = ChatOpenAI(proxy_model_name="gpt-4.1", proxy_client=proxy_client)
            print("DEBUG: LLM client created and connected")
            USE_SAP_AI_HUB = True

        except Exception as llm_error:
            if "google.cloud" in str(llm_error):
                print("DEBUG: Google Cloud dependency missing")
            else:
                print(f"DEBUG: LLM test failed: {str(llm_error)}")
            USE_SAP_AI_HUB = False
            sap_ai_hub_error = f"LLM connection failed: {str(llm_error)}"

    except ImportError as e:
        sap_ai_hub_error = f"Import Error: gen_ai_hub not installed - {str(e)}"
        print("DEBUG: gen_ai_hub not installed")

    except Exception as e:
        sap_ai_hub_error = f"Connection Error: {str(e)}"
        print(f"DEBUG: SAP AI Hub connection failed: {str(e)}")

print(f"DEBUG: SAP AI Hub Status - Connected: {USE_SAP_AI_HUB}")


#############################################################################
# MAIN APPLICATION
#############################################################################
def main():
    # Initialize session state
    if "materials_df" not in st.session_state:
        st.session_state.materials_df = pd.DataFrame()
    if "search_results" not in st.session_state:
        st.session_state.search_results = pd.DataFrame()
    if "selected_items" not in st.session_state:
        st.session_state.selected_items = {}
    if "show_vendor_recommendations" not in st.session_state:
        st.session_state.show_vendor_recommendations = False
    if "show_non_catalog" not in st.session_state:
        st.session_state.show_non_catalog = False
    if "last_search_query" not in st.session_state:
        st.session_state.last_search_query = ""
    if "show_quotation_upload" not in st.session_state:
        st.session_state.show_quotation_upload = False

    # Auto-load materials data on startup
    if st.session_state.materials_df.empty:
        with st.spinner("Loading materials database..."):
            st.session_state.materials_df = materials.get_materials_data(USE_SAP_AI_HUB)

    # Main content check
    if st.session_state.materials_df.empty:
        st.error("Could not load materials database")
        st.info(
            "Please ensure materials.csv exists in the app directory or data/ folder"
        )
        return

    # Search interface
    st.header("Search Materials")

    col1, col2 = st.columns([4, 1])

    with col1:
        search_query = st.text_input(
            "What would you like to buy?",
            placeholder="e.g., I want to buy a laptop, office supplies, etc.",
            help="Enter your search query in natural language",
        )

    with col2:
        search_button = st.button("Search", type="primary", use_container_width=True)

    # Perform search
    if search_button and search_query:
        with st.spinner("Searching for materials..."):
            search_results = materials.search_materials(
                search_query, st.session_state.materials_df, USE_SAP_AI_HUB
            )
            st.session_state.search_results = search_results
            st.session_state.last_search_query = search_query

    # Display search results
    if not st.session_state.search_results.empty:
        st.markdown("### Available Materials")

        # Table headers - Updated to include contract information
        col_headers = st.columns([2.5, 1.5, 1, 1, 1, 1, 1, 0.8, 0.8])
        with col_headers[0]:
            st.write("**Material**")
        with col_headers[1]:
            st.write("**Material Number**")
        with col_headers[2]:
            st.write("**Price**")
        with col_headers[3]:
            st.write("**Contract #**")
        with col_headers[4]:
            st.write("**Contract Value**")
        with col_headers[5]:
            st.write("**Exhaustion Rate**")
        with col_headers[6]:
            st.write("**Catalog**")
        with col_headers[7]:
            st.write("**Select**")
        with col_headers[8]:
            st.write("**Qty**")

        st.markdown("---")

        # Display materials and collect selections
        selected_items = {}
        total_cost = 0

        for idx, row in st.session_state.search_results.iterrows():
            col1, col2, col3, col4, col5, col6, col7, col8, col9 = st.columns(
                [2.5, 1.5, 1, 1, 1, 1, 1, 0.8, 0.8]
            )

            with col1:
                st.write(f"{row['MATERIAL_DESCRIPTION']}")

            with col2:
                material_id = row.get("MATERIAL_ID", "")
                if not pd.isna(material_id) and str(material_id).strip():
                    # Format as integer if it's a numeric value
                    try:
                        if float(material_id).is_integer():
                            st.write(f"{int(float(material_id))}")
                        else:
                            st.write(f"{material_id}")
                    except (ValueError, TypeError):
                        st.write(f"{material_id}")
                else:
                    st.write("Not Available")

            with col3:
                st.write(f"**€{row['PRICE']:.2f}**")

            with col4:
                # Display contract number
                contract_number = row.get("CONTRACT_NUMBER", "")
                if pd.notna(contract_number) and str(contract_number).strip():
                    st.write(f"{contract_number}")
                else:
                    st.write("N/A")

            with col5:
                # Display contract value
                contract_value = row.get("CONTRACT_VALUE", "")
                if pd.notna(contract_value) and str(contract_value).strip():
                    try:
                        value = float(contract_value)
                        st.write(f"€{value:,.0f}")
                    except (ValueError, TypeError):
                        st.write("N/A")
                else:
                    st.write("N/A")

            with col6:
                # Display exhaustion rate
                exhaustion_rate = row.get("EXHAUSTION_RATE", "")
                if pd.notna(exhaustion_rate) and str(exhaustion_rate).strip():
                    try:
                        rate = float(exhaustion_rate)
                        st.write(f"{rate:.1f}%")
                    except (ValueError, TypeError):
                        st.write("N/A")
                else:
                    st.write("N/A")

            with col7:
                # Display catalog link if available
                if pd.notna(row.get("CATALOG")) and row.get("CATALOG"):
                    catalog_file = row["CATALOG"]

                    # Use the robust catalog loader
                    success, pdf_data, error_msg = load_catalog_file(
                        catalog_file, debug=False
                    )

                    if success:
                        # Use session-based download button to avoid multi-instance issues
                        create_catalog_download_button(
                            pdf_data,
                            catalog_file,
                            f"main_catalog_{idx}",
                        )
                    else:
                        st.write("N/A")
                        if error_msg:
                            print(
                                f"Catalog download error for {catalog_file}: {error_msg}"
                            )
                else:
                    st.write("N/A")

            with col8:
                selected = st.checkbox(
                    f"Select {row['MATERIAL_DESCRIPTION']}",
                    key=f"select_{idx}",
                    label_visibility="collapsed",
                )

            with col9:
                if selected:
                    quantity = st.number_input(
                        f"Quantity for {row['MATERIAL_DESCRIPTION']}",
                        min_value=1,
                        value=1,
                        key=f"qty_{idx}",
                        label_visibility="collapsed",
                    )

                    selected_items[idx] = {
                        "description": row["MATERIAL_DESCRIPTION"],
                        "commodity": row["COMMODITY_CODE"],
                        "price": row["PRICE"],
                        "quantity": quantity,
                        "total": row["PRICE"] * quantity,
                        "unit_of_measure": row.get("UNIT_OF_MEASURE", "EA"),
                        "material_id": row.get("MATERIAL_ID", ""),
                        "catalog": row.get("CATALOG", ""),
                        "commodity_description": row.get("COMMODITY_DESCRIPTION", ""),
                        "contract_number": row.get("CONTRACT_NUMBER", ""),
                        "contract_value": row.get("CONTRACT_VALUE", ""),
                        "exhaustion_rate": row.get("EXHAUSTION_RATE", ""),
                    }

                    total_cost += row["PRICE"] * quantity
                else:
                    st.write("")

            st.markdown("---")

        # Show selected items and actions
        if selected_items:
            st.markdown("### Selected Items")

            selected_data = []
            for item in selected_items.values():
                # Format Material Number properly
                material_id = item.get("material_id", "")
                if (
                    material_id
                    and not pd.isna(material_id)
                    and str(material_id).strip()
                ):
                    try:
                        if float(material_id).is_integer():
                            formatted_material_id = str(int(float(material_id)))
                        else:
                            formatted_material_id = str(material_id)
                    except (ValueError, TypeError):
                        formatted_material_id = str(material_id)
                else:
                    formatted_material_id = "Not Available"

                selected_data.append(
                    {
                        "Material": item["description"],
                        "Material Number": formatted_material_id,
                        "Commodity Code": item["commodity"],
                        "Price": f"€{item['price']:.2f}",
                        "Qty": item["quantity"],
                        "Total": f"€{item['total']:.2f}",
                    }
                )

            selected_df = pd.DataFrame(selected_data)
            # USE SAFE_DATAFRAME INSTEAD OF st.dataframe
            safe_dataframe(selected_df, use_container_width=True, hide_index=True)

            st.markdown(f"### **Total: €{total_cost:.2f}**")

            col1, col2, col3 = st.columns(3)

            with col1:
                if st.button("Clear Selection", use_container_width=True):
                    st.rerun()

            with col2:
                if st.button("Save Selection", use_container_width=True):
                    st.session_state.selected_items = selected_items
                    st.success("Selection saved.")

            with col3:
                if st.button(
                    "Get Vendor Recommendations",
                    use_container_width=True,
                    type="primary",
                ):
                    st.session_state.show_vendor_recommendations = True

        else:
            st.info("Select materials above to proceed with vendor recommendations.")

    elif search_query and search_button:
        st.warning("No materials found for your search. Try different keywords.")

    else:
        # Show example searches
        st.markdown("### Example Searches")
        examples = [
            "I want to buy a laptop",
            "I need office supplies",
            "Looking for office furniture",
            "Need computer hardware",
        ]

        cols = st.columns(len(examples))
        for i, example in enumerate(examples):
            with cols[i]:
                if st.button(example, key=f"example_{i}", use_container_width=True):
                    st.session_state.search_query = example
                    with st.spinner("Searching..."):
                        search_results = materials.search_materials(
                            example, st.session_state.materials_df, USE_SAP_AI_HUB
                        )
                        st.session_state.search_results = search_results
                        st.session_state.last_search_query = example
                    st.rerun()

    # Show vendor recommendations if requested
    if st.session_state.get("show_vendor_recommendations", False) and selected_items:
        with st.spinner("Finding best vendors..."):
            try:
                vendor_result = vendor.get_vendor_recommendations(
                    selected_items, USE_SAP_AI_HUB
                )
                if vendor_result:
                    st.success("Vendor recommendations ready.")
            except Exception as e:
                st.error(f"Vendor recommendation error: {e}")

    # Show Non-Catalog Purchase interface if requested
    if (
        st.session_state.get("show_non_catalog", False)
        and st.session_state.last_search_query
    ):
        try:
            vendors_df = vendor.load_vendors_data()
            text.show_non_catalog_purchase(
                user_query=st.session_state.last_search_query,
                materials_df=st.session_state.materials_df,
                vendors_df=vendors_df,
                use_sap_ai_hub=USE_SAP_AI_HUB,
            )
        except Exception as e:
            st.error(f"Non-catalog purchase error: {e}")

    # Show Quotation Upload interface if requested
    if st.session_state.get("show_quotation_upload", False):
        try:
            quotation.handle_quotation_upload(USE_SAP_AI_HUB)
        except Exception as e:
            st.error(f"Quotation upload error: {e}")
            st.info("Make sure quotation.py module is working correctly")

    # ALWAYS SHOW "NOT IN CATALOG" BUTTON AT THE END
    st.markdown("---")

    # Show warning if no results found
    if st.session_state.last_search_query and st.session_state.search_results.empty:
        st.warning(
            f'Product not found in catalog for "{st.session_state.last_search_query}". Try Non-Catalog Purchase below.'
        )

    # Always show the non-catalog button
    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        button_text = "Not in Catalog - Create Custom Purchase"
        if not st.session_state.last_search_query:
            button_text = "Create Custom Purchase Order"

        if st.button(
            button_text,
            use_container_width=True,
            key="non_catalog_button",
            help="Create a purchase order for items not in our catalog",
        ):
            if st.session_state.last_search_query:
                st.session_state.show_non_catalog = True
            else:
                st.session_state.last_search_query = "Custom item order"
                st.session_state.show_non_catalog = True
            st.rerun()

    with col3:
        if st.button(
            "Upload Quotation",
            use_container_width=True,
            key="upload_quotation_button",
            help="Upload a PDF quotation to extract vendor and pricing information",
        ):
            st.session_state.show_quotation_upload = True
            st.rerun()


if __name__ == "__main__":
    main()
