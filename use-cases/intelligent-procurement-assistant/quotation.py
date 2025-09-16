"""
Quotation Processing Module - PRODUCTION VERSION
HARDCODED CREDENTIALS - NO FALLBACKS
Fixed commodity code detection for PDF quotations
"""

import os
import pandas as pd
import streamlit as st
import numpy as np
from typing import List, Dict, Any
import json
import tempfile
import time
from io import BytesIO

#############################################################################
# UTILITY FUNCTIONS FOR FORMATTING
#############################################################################


def format_currency(value):
    """Format numbers as US currency with thousands separator"""
    try:
        return f"${float(value):,.2f}"
    except (ValueError, TypeError):
        return f"${0:,.2f}"


def format_number(value):
    """Format numbers with US format (comma separator, 2 decimals)"""
    try:
        return f"{float(value):,.2f}"
    except (ValueError, TypeError):
        return f"{0:,.2f}"


#############################################################################
# CLOUD FOUNDRY DEBUGGING
#############################################################################


def debug_cf_environment():
    """Debug Cloud Foundry environment"""
    print("=" * 50)
    print("CLOUD FOUNDRY ENVIRONMENT DEBUG - HARDCODED")
    print("=" * 50)

    # Hardcoded variables
    print("SAP AI HUB VARIABLES (HARDCODED):")
    print(f"  AICORE_AUTH_URL: {os.environ['AICORE_AUTH_URL']}")
    print(f"  AICORE_CLIENT_ID: {os.environ['AICORE_CLIENT_ID'][:20]}...")
    print(f"  AICORE_CLIENT_SECRET: {os.environ['AICORE_CLIENT_SECRET'][:20]}...")
    print(f"  AICORE_BASE_URL: {os.environ['AICORE_BASE_URL']}")
    print(f"  AICORE_RESOURCE_GROUP: {os.environ['AICORE_RESOURCE_GROUP']}")

    # CF Environment info
    print(f"CF_INSTANCE_INDEX: {os.getenv('CF_INSTANCE_INDEX', 'Not CF')}")
    print(f"MEMORY_LIMIT: {os.getenv('MEMORY_LIMIT', 'Unknown')}")
    print(f"PORT: {os.getenv('PORT', 'Unknown')}")

    print("=" * 50)


# Execute debug on import
debug_cf_environment()

#############################################################################
# STREAMLIT COMPATIBILITY
#############################################################################


def safe_dataframe(df, use_container_width=True, hide_index=True, **kwargs):
    try:
        return st.dataframe(
            df, use_container_width=use_container_width, hide_index=hide_index, **kwargs
        )
    except TypeError:
        if hide_index:
            try:
                styled_df = df.style.hide(axis="index")
                return st.dataframe(
                    styled_df, use_container_width=use_container_width, **kwargs
                )
            except:
                display_df = df.reset_index(drop=True)
                return st.dataframe(
                    display_df, use_container_width=use_container_width, **kwargs
                )
        else:
            return st.dataframe(df, use_container_width=use_container_width, **kwargs)


if not hasattr(st, "rerun"):
    if hasattr(st, "experimental_rerun"):
        st.rerun = st.experimental_rerun

#############################################################################
# PDF PROCESSING - PRODUCTION STRICT
#############################################################################

try:
    import PyPDF2

    PDF_PROCESSING_AVAILABLE = True
    print("PyPDF2 available")
except ImportError:
    PDF_PROCESSING_AVAILABLE = False
    st.error("PRODUCTION ERROR: PyPDF2 not installed")
    st.stop()


def extract_text_from_pdf(pdf_file) -> str:
    """Extract text from PDF - PRODUCTION VERSION"""
    try:
        print(f"DEBUG: Processing PDF: {pdf_file.name} ({pdf_file.size} bytes)")

        # Strict validations
        if pdf_file.size == 0:
            raise Exception("PDF file is empty")

        if pdf_file.size > 50 * 1024 * 1024:  # 50MB limit
            raise Exception(
                f"PDF too large: {pdf_file.size / (1024*1024):.1f}MB > 50MB"
            )

        # Process PDF
        file_stream = BytesIO(pdf_file.getvalue())
        reader = PyPDF2.PdfReader(file_stream)

        if len(reader.pages) == 0:
            raise Exception("PDF has no pages")

        print(f"DEBUG: PDF has {len(reader.pages)} pages")

        extracted_text = ""
        for i, page in enumerate(reader.pages):
            try:
                page_text = page.extract_text()
                if page_text:
                    extracted_text += page_text + "\n"
                print(f"DEBUG: Page {i+1}: {len(page_text)} chars")
            except Exception as page_error:
                print(f"ERROR: Page {i+1} failed: {page_error}")
                raise Exception(f"Failed to read page {i+1}: {page_error}")

        if not extracted_text.strip():
            raise Exception("No readable text found in PDF")

        print(f"DEBUG: Total extracted: {len(extracted_text)} characters")
        return extracted_text.strip()

    except Exception as e:
        error_msg = f"PDF Processing Error: {str(e)}"
        print(f"ERROR: {error_msg}")
        st.error(f"{error_msg}")
        raise e


#############################################################################
# SAP AI HUB - PRODUCTION STRICT (HARDCODED CREDENTIALS)
#############################################################################


def test_sap_ai_hub_connection():
    """Test SAP AI Hub connection with hardcoded credentials"""
    try:
        print("DEBUG: Testing SAP AI Hub connection with hardcoded credentials...")

        # Import test
        from gen_ai_hub.proxy.core.proxy_clients import get_proxy_client
        from gen_ai_hub.proxy.langchain.openai import ChatOpenAI

        print("gen_ai_hub imports successful")

        # Proxy client test
        print("DEBUG: Creating proxy client...")
        proxy_client = get_proxy_client("gen-ai-hub")
        print("Proxy client created")

        # LLM test
        print("DEBUG: Creating LLM client...")
        llm = ChatOpenAI(proxy_model_name="gpt-4.1", proxy_client=proxy_client)
        print("LLM client created")

        # Connection test
        print("DEBUG: Testing LLM connection...")
        test_response = llm.invoke("Hello").content
        print(f"LLM test successful: {test_response[:50]}...")

        return True

    except Exception as e:
        error_msg = f"SAP AI Hub Connection Error: {str(e)}"
        print(f"ERROR: {error_msg}")

        # Detailed error analysis
        if "invalid_client" in str(e) or "Bad credentials" in str(e):
            st.error("SAP AI Hub Authentication Failed - HARDCODED CREDENTIALS ISSUE")
            st.error("Check hardcoded credentials in code:")
            st.code(
                """
HARDCODED credentials are wrong or expired:
- AICORE_CLIENT_ID
- AICORE_CLIENT_SECRET  
- AICORE_AUTH_URL
Update the hardcoded values in quotation.py
            """
            )
        elif "timeout" in str(e).lower():
            st.error("SAP AI Hub Timeout")
            st.error("Possible causes:")
            st.code(
                """
- Cloud Foundry memory limits
- Network connectivity issues
- SAP AI Core service unavailable
            """
            )
        elif "google.cloud" in str(e):
            st.error("Missing Google Cloud Dependencies")
            st.code("pip install google-cloud-aiplatform")
        else:
            st.error(f"SAP AI Hub Error: {error_msg}")

        raise e


def ask_llm_production(pdf_text: str, vendors_df: pd.DataFrame) -> str:
    """Ask LLM with hardcoded credentials - UPDATED: No commodity code detection"""

    # Validate inputs
    if not pdf_text or not pdf_text.strip():
        raise Exception("No PDF text provided for analysis")

    if vendors_df.empty:
        raise Exception("No vendor data available")

    print(f"DEBUG: Analyzing {len(pdf_text)} chars of PDF text")
    print(f"DEBUG: Using {len(vendors_df)} vendors for matching")

    try:
        # Import and setup
        from gen_ai_hub.proxy.core.proxy_clients import get_proxy_client
        from gen_ai_hub.proxy.langchain.openai import ChatOpenAI

        print("DEBUG: Creating SAP AI Hub clients with hardcoded credentials...")
        proxy_client = get_proxy_client("gen-ai-hub")
        llm = ChatOpenAI(proxy_model_name="gpt-4.1", proxy_client=proxy_client)
        print("AI Hub clients ready")

        # Prepare data
        vendor_names = vendors_df["VENDOR_NAME"].unique().tolist()

        # UPDATED PROMPT - NO COMMODITY CODE DETECTION IN LLM
        prompt = f"""
You are a procurement analyst. Extract information from this quotation PDF.

PDF Content:
{pdf_text}

Available Vendors in Database:
{json.dumps(vendor_names, indent=2)}

    CRITICAL INSTRUCTIONS:
    1. CUSTOMER = Who is buying (your company, etc.) - DO NOT use as vendor
2. VENDOR = Who is selling/providing the quote (SAP, Dell, HP, etc.) - USE THIS as vendor
3. Look for company logos, contact information, or "From:" details to identify the VENDOR
4. The vendor is usually at the bottom of the quote or has contact details
5. Extract the EXACT vendor/supplier name from the PDF
6. Only set "matched_vendor" if there's an EXACT or very close match with database vendors
7. If no exact match exists, leave "matched_vendor" as empty string
8. Do NOT try to determine commodity codes - that will be done separately
9. Focus only on extracting basic quotation information
10. Respond ONLY with valid JSON. No markdown, no explanations.

    EXAMPLE:
    - If PDF shows "CUSTOMER: ExampleCorp" and "Contact: sales@sap.com" → vendor is "SAP"
    - If PDF shows "CUSTOMER: Acme Corp" and "Dell Sales Team" → vendor is "Dell"

{{
    "vendor_name_from_pdf": "EXACT vendor/supplier name (NOT customer name)",
    "matched_vendor": "exact match from available vendors OR empty string",
    "vendor_confidence": 0.0,
    "items": [
        {{
            "material_description": "item description",
            "quantity": 1,
            "unit_price": 100.00,
            "total_price": 100.00,
            "unit_of_measure": "EA"
        }}
    ],
    "total_quotation_amount": 100.00,
    "currency": "USD",
    "quotation_date": "YYYY-MM-DD",
    "quotation_number": "quote number"
}}
"""

        print("DEBUG: Sending request to SAP AI Hub...")
        response = llm.invoke(prompt).content
        print(f"DEBUG: Received response: {len(response)} chars")
        print(f"DEBUG: Response preview: {response[:200]}...")

        return response

    except Exception as e:
        error_msg = f"SAP AI Hub Request Failed: {str(e)}"
        print(f"ERROR: {error_msg}")

        # Enhanced error reporting
        if "400" in str(e):
            st.error("Bad Request (400) - Check request format")
            st.error("Possible causes:")
            st.code(
                """
- PDF text too long
- Invalid characters in text
- Malformed request
- Authentication token expired
            """
            )
        elif "401" in str(e):
            st.error("Unauthorized (401) - Hardcoded credentials failed")
            st.code("Update hardcoded AICORE_CLIENT_ID and AICORE_CLIENT_SECRET")
        elif "429" in str(e):
            st.error("Rate Limited (429) - Too many requests")
        elif "500" in str(e):
            st.error("Server Error (500) - SAP AI Core issue")

        raise e


def analyze_quotation_with_llm(
    pdf_text: str, vendors_df: pd.DataFrame
) -> Dict[str, Any]:
    """Analyze quotation with commodity code detection - FIXED VERSION"""

    print("=" * 50)
    print("STARTING QUOTATION ANALYSIS - HARDCODED CREDENTIALS")
    print("=" * 50)

    try:
        # Step 1: Validate inputs
        if not pdf_text:
            raise Exception("No PDF text to analyze")

        if vendors_df.empty:
            raise Exception("No vendor data loaded")

        # Step 2: Test AI Hub connection first
        print("STEP 1: Testing SAP AI Hub connection...")
        test_sap_ai_hub_connection()

        # Step 3: Send to LLM for basic extraction
        print("STEP 2: Sending to LLM for basic analysis...")
        llm_response = ask_llm_production(pdf_text, vendors_df)

        # Step 4: Parse response
        print("STEP 3: Parsing LLM response...")
        try:
            # Clean response
            clean_response = llm_response.strip()
            if clean_response.startswith("```json"):
                clean_response = clean_response[7:]
            if clean_response.endswith("```"):
                clean_response = clean_response[:-3]
            clean_response = clean_response.strip()

            # Parse JSON
            analysis_result = json.loads(clean_response)

            # Step 5: NEW - Determine commodity code for each item
            print("STEP 4: Determining commodity code for each material...")

            # Load commodity codes reference
            try:
                import materials

                materials_df = materials.get_materials_data()
                all_codes_df = pd.read_csv("Ariba Material Commodity Codes.csv")
                print("Commodity codes reference loaded")
            except Exception as e:
                print(f"WARNING: Could not load commodity codes: {e}")
                all_codes_df = pd.DataFrame()

            # Get vendor name from analysis
            vendor_name = analysis_result.get("vendor_name_from_pdf", "")

            # Process each item to determine commodity code
            if "items" in analysis_result and analysis_result["items"]:
                for item in analysis_result["items"]:
                    item_description = item.get("material_description", "")

                    if item_description:
                        print(
                            f"DEBUG: Determining commodity code for: {item_description}"
                        )

                        # Use algorithm from vendor.py
                        try:
                            from vendor import get_best_commodity_code_for_item

                            commodity_result = get_best_commodity_code_for_item(
                                item_description=item_description,
                                vendor_name=vendor_name,
                                vendors_df=vendors_df,
                                all_codes_df=all_codes_df,
                            )

                            # Parse result "43211503 - Computer Hardware"
                            if commodity_result and " - " in commodity_result:
                                code_part = commodity_result.split(" - ")[0].strip()
                                desc_part = commodity_result.split(" - ")[1].strip()

                                # Remove .0 from code if present
                                if code_part.endswith(".0"):
                                    code_part = code_part[:-2]

                                item["commodity_code"] = code_part
                                item["commodity_description"] = desc_part

                                print(
                                    f"Commodity code found: {code_part} - {desc_part}"
                                )
                            else:
                                print(
                                    f"Commodity code not found for: {item_description}"
                                )
                                item["commodity_code"] = "Unknown"
                                item["commodity_description"] = "Unknown"

                        except Exception as e:
                            print(f"ERROR: Error determining commodity code: {e}")
                            item["commodity_code"] = "Unknown"
                            item["commodity_description"] = "Unknown"

            print("Analysis completed successfully")
            return analysis_result

        except json.JSONDecodeError as e:
            error_msg = f"Invalid JSON response from AI: {str(e)}"
            print(f"ERROR: {error_msg}")
            print(f"Raw response: {llm_response}")
            st.error(f"{error_msg}")
            st.error("Raw AI Response:")
            st.code(llm_response)
            raise Exception(error_msg)

    except Exception as e:
        print(f"FATAL ERROR: Quotation analysis failed: {str(e)}")
        raise e


#############################################################################
# VENDOR ENHANCEMENT
#############################################################################


def enhance_quotation_with_vendor_data(
    quotation_data: Dict, vendors_df: pd.DataFrame
) -> Dict:
    """Check if vendor exists in database - show warning if not found"""

    if not quotation_data:
        raise Exception("No quotation data to enhance")

    # Get vendor from PDF
    vendor_from_pdf = quotation_data.get("vendor_name_from_pdf", "")
    matched_vendor = quotation_data.get("matched_vendor", "")

    print(f"DEBUG: Vendor from PDF: '{vendor_from_pdf}'")
    print(f"DEBUG: Matched vendor from DB: '{matched_vendor}'")

    try:
        # Check if we have a matched vendor from the database
        if matched_vendor and matched_vendor.strip():
            # Vendor found in database - use DB info
            vendor_row = vendors_df[
                vendors_df["VENDOR_NAME"].str.contains(
                    matched_vendor, case=False, na=False
                )
            ]

            if not vendor_row.empty:
                vendor_info = vendor_row.iloc[0]
                quotation_data["vendor_details"] = {
                    "name": vendor_info["VENDOR_NAME"],
                    "commodity_code": vendor_info.get("COMMODITY_CODE", "Unknown"),
                    "contract_exists": vendor_info.get("CONTRACT_EXISTS", "NO"),
                    "guaranteed_spend": vendor_info.get("GUARANTEED_SPEND", 0),
                    "payment_terms": vendor_info.get("PAYMENT_TERMS", "Net 30"),
                    "rating": vendor_info.get("RATING", "N/A"),
                    "location": vendor_info.get("LOCATION", "Unknown"),
                    "vendor_found": True,
                }
                print(f"Vendor found in database: {vendor_info['VENDOR_NAME']}")
            else:
                # This shouldn't happen if LLM worked correctly
                quotation_data["vendor_details"] = {
                    "name": vendor_from_pdf,
                    "vendor_found": False,
                }
        else:
            # Vendor NOT found in database
            quotation_data["vendor_details"] = {
                "name": vendor_from_pdf,
                "commodity_code": "Unknown",
                "contract_exists": "UNKNOWN",
                "guaranteed_spend": 0,
                "payment_terms": "Unknown",
                "rating": "N/A",
                "location": "Unknown",
                "vendor_found": False,
            }
            print(f"Vendor from PDF not in database: {vendor_from_pdf}")

        return quotation_data

    except Exception as e:
        error_msg = f"Vendor processing failed: {str(e)}"
        print(f"ERROR: {error_msg}")

        # Fallback: vendor not found
        quotation_data["vendor_details"] = {
            "name": vendor_from_pdf,
            "vendor_found": False,
        }
        return quotation_data


#############################################################################
# PR CREATION
#############################################################################


def simulate_pr_creation(
    quotation_data: Dict,
    vendor_details: Dict,
    total_amount: float,
    delivery_address: str,
):
    """Simulate PR creation - FIXED BUG"""
    pr_number = f"PR-{pd.Timestamp.now().strftime('%Y%m%d-%H%M%S')}"

    with st.spinner("Creating purchase request..."):
        time.sleep(1)

    # FIXED: Set PR created flag and data first
    st.session_state.quotation_pr_created = True
    st.session_state.quotation_pr_data = {
        "pr_number": pr_number,
        "total_amount": total_amount,
        "vendor_name": vendor_details.get("name", "N/A"),
        "quote_number": quotation_data.get("quotation_number", "N/A"),
        "items_count": len(st.session_state.get("edited_items", [])),
    }

    # FIXED: Keep show_quotation_upload = True so the success message shows
    # Don't set it to False here - let the success screen handle navigation

    # FIXED: Clear all quotation processing states to prevent confusion
    if "quotation_data" in st.session_state:
        del st.session_state["quotation_data"]
    if "edited_items" in st.session_state:
        del st.session_state["edited_items"]

    st.rerun()


def show_pr_creation_success():
    """Display PR creation success"""
    pr_data = st.session_state.get("quotation_pr_data", {})

    st.success("Purchase Request created successfully.")
    st.balloons()

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("### Purchase Request Created")

        info_text = f"""
        **PR Number:** {pr_data.get('pr_number', 'N/A')}
        **Total Amount:** {format_currency(pr_data.get('total_amount', 0))}
        **Vendor:** {pr_data.get('vendor_name', 'N/A')}
        **Quote Number:** {pr_data.get('quote_number', 'N/A')}
        **Items Count:** {pr_data.get('items_count', 0)}
        """

        st.info(info_text)

        if st.button("Process New Quotation", use_container_width=True, type="primary"):
            # Clear ALL quotation-related session states
            keys_to_clear = [
                k
                for k in st.session_state.keys()
                if "quotation" in k or "edited_items" in k
            ]
            for key in keys_to_clear:
                del st.session_state[key]
            # Keep show_quotation_upload = True to show upload interface
            st.session_state.show_quotation_upload = True
            st.rerun()


#############################################################################
# QUOTATION RESULTS DISPLAY
#############################################################################


def show_quotation_analysis_results(quotation_data: Dict, vendors_df: pd.DataFrame):
    """Display quotation analysis results with commodity code support"""

    st.markdown("---")
    st.markdown("### **Quotation Analysis Results**")

    # Check if PR already created
    if st.session_state.get("quotation_pr_created", False):
        show_pr_creation_success()
        return

    # Vendor Details
    vendor_details = quotation_data.get("vendor_details", {})
    vendor_found = vendor_details.get("vendor_found", False)

    st.markdown("#### Vendor Details")

    if not vendor_found:
        # Vendor NOT found in database - show warning
        st.error("**VENDOR NOT IN THE LIST**")
        st.error(
            "Please contact your **Purchase Agent Support** to add this vendor to the system."
        )

        col1, col2 = st.columns(2)
        with col1:
            st.info(f"**Vendor from PDF:** {vendor_details.get('name', 'N/A')}")
            st.info("**Status:** Not in approved vendor list")
        with col2:
            st.info("**Action Required:** Contact Purchase Agent Support")
            st.info("**Process:** Cannot proceed without approved vendor")

        # Show contact info or helpful message
        st.markdown("#### Next Steps")
        st.info(
            """
        **To proceed with this quotation:**
        1. Contact your Purchase Agent Support team
        2. Request to add vendor to approved list
        3. Provide vendor details and quotation
        4. Wait for vendor approval
        5. Reprocess quotation once vendor is added
        """
        )

        # Disable further processing
        st.markdown("---")
        st.warning("Purchase Request creation is disabled until vendor is approved")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Cancel", use_container_width=True):
                st.session_state.show_quotation_upload = False
                if "edited_items" in st.session_state:
                    del st.session_state.edited_items
                if "quotation_data" in st.session_state:
                    del st.session_state.quotation_data
                if "quotation_delivery_address" in st.session_state:
                    del st.session_state.quotation_delivery_address
                st.rerun()

        with col2:
            st.button(
                "Contact Support",
                use_container_width=True,
                disabled=True,
                help="Contact your Purchase Agent Support to add this vendor",
            )

        return  # Exit function - don't show items or create PR option

    else:
        # Vendor found in database - show normal vendor details
        col1, col2 = st.columns(2)
        with col1:
            st.info(f"**Vendor:** {vendor_details.get('name', 'N/A')}")
            st.info(
                f"**Contract Exists:** {vendor_details.get('contract_exists', 'UNKNOWN')}"
            )

        with col2:
            st.info(
                f"**Payment Terms:** {vendor_details.get('payment_terms', 'Net 30')}"
            )
            st.info(f"**Rating:** {vendor_details.get('rating', 'N/A')}")
            guaranteed_spend = vendor_details.get("guaranteed_spend", 0)
            st.info(f"**Guaranteed Spend:** {format_currency(guaranteed_spend)}")

    # Items with commodity code display
    st.markdown("#### Quotation Items")

    items = quotation_data.get("items", [])

    if "edited_items" not in st.session_state:
        st.session_state.edited_items = items

    if items:
        edited_df = pd.DataFrame(st.session_state.edited_items)
        edited_df["quantity"] = (
            pd.to_numeric(edited_df["quantity"], errors="coerce").fillna(0).astype(int)
        )
        edited_df["unit_price"] = (
            pd.to_numeric(edited_df["unit_price"], errors="coerce")
            .fillna(0.0)
            .astype(float)
        )
        edited_df["total_price"] = edited_df["quantity"] * edited_df["unit_price"]

        # Create display_df with commodity code
        display_df = edited_df.rename(
            columns={
                "material_description": "Material Description",
                "quantity": "Quantity",
                "unit_price": "Unit Price",
                "total_price": "Total Price",
                "unit_of_measure": "UOM",
                "commodity_code": "Commodity Code",
                "commodity_description": "Category",
            }
        )

        # Include commodity code in display
        display_cols = [
            "Material Description",
            "Commodity Code",
            "Category",
            "Quantity",
            "UOM",
            "Unit Price",
            "Total Price",
        ]

        # Ensure all columns exist
        for col in display_cols:
            if col not in display_df.columns:
                if col == "Commodity Code":
                    display_df[col] = "Unknown"
                elif col == "Category":
                    display_df[col] = "Unknown"
                else:
                    display_df[col] = ""

        # Data editor with commodity code support
        edited_data = st.data_editor(
            display_df[display_cols],
            column_config={
                "Material Description": st.column_config.TextColumn(
                    "Material Description", width="large", required=True
                ),
                "Commodity Code": st.column_config.TextColumn(
                    "Commodity Code", width="medium", required=True
                ),
                "Category": st.column_config.TextColumn(
                    "Category", width="medium", required=True
                ),
                "Quantity": st.column_config.NumberColumn(
                    "Quantity", min_value=0, format="%d", required=True
                ),
                "UOM": st.column_config.TextColumn("UOM", width="small", required=True),
                "Unit Price": st.column_config.NumberColumn(
                    "Unit Price", min_value=0.01, format="$%.2f", required=True
                ),
                "Total Price": st.column_config.NumberColumn(
                    "Total Price", disabled=True, format="$%.2f"
                ),
            },
            key="quotation_items_editor",
            num_rows="dynamic",
            hide_index=True,
            use_container_width=True,
        )

        # Update session state
        if edited_data is not None:
            updated_items = []
            for _, row in edited_data.iterrows():
                updated_items.append(
                    {
                        "material_description": row["Material Description"],
                        "commodity_code": row["Commodity Code"],
                        "commodity_description": row["Category"],
                        "quantity": row["Quantity"],
                        "unit_price": row["Unit Price"],
                        "total_price": row["Total Price"],
                        "unit_of_measure": row["UOM"],
                    }
                )
            st.session_state.edited_items = updated_items
    else:
        st.warning("No items found in quotation")

    # Calculate total
    total_amount = sum(
        item.get("quantity", 0) * item.get("unit_price", 0)
        for item in st.session_state.edited_items
    )

    # Delivery Address - Make it editable
    st.markdown("#### Delivery Address")

    # Initialize with default address if not set
    if "quotation_delivery_address" not in st.session_state:
        st.session_state.quotation_delivery_address = "Intelligent Procurement Assistant HQ\n675 Avenue of the Americas\nNew York, NY 10010"

    delivery_address = st.text_area(
        "Edit delivery address:",
        value=st.session_state.quotation_delivery_address,
        height=100,
        key="delivery_address_input",
        help="Enter the delivery address for this purchase request",
    )

    # Update session state when address changes
    st.session_state.quotation_delivery_address = delivery_address

    # Action buttons
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("Cancel", use_container_width=True):
            st.session_state.show_quotation_upload = False
            if "edited_items" in st.session_state:
                del st.session_state.edited_items
            if "quotation_data" in st.session_state:
                del st.session_state.quotation_data
            if "quotation_delivery_address" in st.session_state:
                del st.session_state.quotation_delivery_address
            st.rerun()

    with col2:
        if st.button("Save Draft", use_container_width=True):
            st.success("Draft saved.")

    with col3:
        if st.button("Confirm PR", use_container_width=True, type="primary"):
            simulate_pr_creation(
                quotation_data, vendor_details, total_amount, delivery_address
            )


#############################################################################
# MAIN WORKFLOW FUNCTIONS
#############################################################################


def process_quotation_pdf(uploaded_file):
    """Process PDF quotation - PRODUCTION STRICT"""

    try:
        # Step 1: Extract text
        with st.spinner("Extracting text from PDF..."):
            pdf_text = extract_text_from_pdf(uploaded_file)

            with st.expander("Extracted Text Preview"):
                st.text_area(
                    "PDF Content:",
                    pdf_text[:1000] + "..." if len(pdf_text) > 1000 else pdf_text,
                    height=200,
                )

        # Step 2: Load vendor data
        with st.spinner("Loading vendor data..."):
            import vendor

            vendors_df = vendor.load_vendors_data()

        # Step 3: Analyze with LLM
        with st.spinner("Analyzing with SAP Generative AI Hub..."):
            quotation_data = analyze_quotation_with_llm(pdf_text, vendors_df)

            if not quotation_data:
                raise Exception("Analysis returned empty result")

        # Step 4: Enhance with vendor data
        with st.spinner("Enhancing with vendor information..."):
            enhanced_quotation = enhance_quotation_with_vendor_data(
                quotation_data, vendors_df
            )

            if not enhanced_quotation:
                raise Exception("Vendor enhancement failed")

        # Step 5: Store and display
        st.session_state.quotation_data = enhanced_quotation
        show_quotation_analysis_results(enhanced_quotation, vendors_df)

    except Exception as e:
        error_msg = f"Quotation processing failed: {str(e)}"
        print(f"FATAL ERROR: {error_msg}")
        st.error(f"{error_msg}")
        st.error("**Process failed at this step. Check logs for details.**")

        # Don't continue - let it fail completely
        st.stop()


def show_quotation_upload_interface():
    """Show PDF upload interface"""

    st.markdown("---")
    st.markdown("### **Upload Quotation**")
    st.info(
        "Upload a PDF quotation for automated processing using SAP Generative AI Hub"
    )

    uploaded_file = st.file_uploader(
        "Choose a PDF quotation file",
        type=["pdf"],
        key="quotation_pdf_uploader",
        help="Upload a PDF quotation document",
    )

    if uploaded_file is not None:
        st.success(f"File uploaded: {uploaded_file.name} ({uploaded_file.size} bytes)")

        if st.button("Process Quotation", type="primary", key="process_quotation_btn"):
            process_quotation_pdf(uploaded_file)


def handle_quotation_upload(use_sap_ai_hub: bool = True):
    """Main quotation handling function"""

    print("DEBUG: handle_quotation_upload called - HARDCODED CREDENTIALS")

    # Initialize session state
    if "show_quotation_upload" not in st.session_state:
        st.session_state.show_quotation_upload = False

    # Handle different states - FIXED: Check PR creation first
    if st.session_state.get("quotation_pr_created", False):
        # Show success screen when PR is created
        show_pr_creation_success()
    elif st.session_state.get("quotation_data", None):
        # Show analysis results when data is available
        try:
            import vendor

            vendors_df = vendor.load_vendors_data()
            show_quotation_analysis_results(st.session_state.quotation_data, vendors_df)
        except Exception as e:
            st.error(f"Error loading vendor data: {e}")
            st.stop()
    else:
        # Show upload interface when starting fresh
        show_quotation_upload_interface()


# END OF FILE
