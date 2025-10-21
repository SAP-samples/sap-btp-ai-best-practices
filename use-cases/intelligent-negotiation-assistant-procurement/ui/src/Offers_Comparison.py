import streamlit as st
import os
import sys
from pathlib import Path
import pandas as pd
from utils import load_css_files
from data_loader import load_supplier_data, get_data_processor

# Enable import of shared components (floating chat) from prototype package
current_file_path = Path(__file__).resolve()
# Move up from src/ -> template_dashboard/ -> prototype/
sys.path.append(str(current_file_path.parent.parent.parent))
from dashboard_components import display_metric_pill, apply_metric_pill_styles, create_source_badge
import html

# Constants
APP_TITLE = "Comparison of Quotations Dashboard"
# LLM model selector for the dashboard; change to quickly test models.
# Defaults to Gemini 2.5 Pro if not overridden.
MODEL_NAME = os.getenv("LLM_MODEL", "gemini-2.5-pro")
# MODEL_NAME = os.getenv("LLM_MODEL", "gpt-5")

# Comparator model is pinned to Gemini 2.5 by default for its context window.
COMPARATOR_MODEL_NAME = os.getenv("COMPARATOR_MODEL", "gemini-2.5-pro")

CHAT_MODEL_NAME = os.getenv("CHAT_MODEL", "gemini-2.5-flash")

# Core part categories to guide LLM concentration of canonical parts.
# Business-editable list; when provided, the analyzer will strictly map into
# these categories plus an "Other" bucket. When None, the LLM proposes categories.
CORE_PART_CATEGORIES = [
    "Clutch Disk",
    "Clutch Cover",
    "Releaser",
]

# Seed categories into session state so other pages can inherit without passing explicitly
if "core_part_categories" not in st.session_state:
    st.session_state["core_part_categories"] = CORE_PART_CATEGORIES

# --- Page Configuration ---
# Use SAP square logo as page icon and apply SAP SVG as app logo
STATIC_IMAGES_DIR = os.path.join(os.path.dirname(__file__), "..", "static", "images")
PAGE_ICON_PATH = os.path.join(STATIC_IMAGES_DIR, "SAP_logo_square.png")
SAP_SVG_PATH = os.path.join(STATIC_IMAGES_DIR, "SAP_logo.svg")

st.set_page_config(
    page_title=APP_TITLE,
    page_icon=PAGE_ICON_PATH,
    layout="wide",
)

# Apply SAP logo (SVG) in header and sidebar; ignored if st.logo is unavailable
try:
    st.logo(SAP_SVG_PATH)
except Exception:
    pass

st.title(f"Welcome to {APP_TITLE}")

# --- CSS ---
css_files = [
    os.path.join(os.path.dirname(__file__), "..", "static", "styles", "variables.css"),
    os.path.join(os.path.dirname(__file__), "..", "static", "styles", "style.css"),
]
load_css_files(css_files)

# Inject metric pill styles for pill components
apply_metric_pill_styles()

# --- App ---

# Load data to check if it's available
supplier1_data, supplier2_data, comparison = load_supplier_data(
    model_name=MODEL_NAME,
    comparator_model_name=COMPARATOR_MODEL_NAME,
    core_part_categories=CORE_PART_CATEGORIES,
)

if supplier1_data and supplier2_data and comparison:
    # Display summary information
    # st.markdown("### Current Analysis Summary")

    # col1, col2, col3 = st.columns(3)

    # # Compute overall scores
    # avg1 = supplier1_data.get('tqdcs', {}).get('overall_assessment', {}).get('average_score')
    # avg2 = supplier2_data.get('tqdcs', {}).get('overall_assessment', {}).get('average_score')

    # # Supplier 1 pill
    # if avg1 is not None and avg2 is not None:
    #     if avg1 > avg2:
    #         delta_text_1 = "Leading"
    #         status_1 = "positive"
    #     elif avg1 < avg2:
    #         pct = ((avg2 - avg1) / avg2) * 100 if avg2 else 0
    #         delta_text_1 = f"-{pct:.1f}% lower"
    #         status_1 = "negative"
    #     else:
    #         delta_text_1 = "Tied"
    #         status_1 = "neutral"
    # else:
    #     delta_text_1 = None
    #     status_1 = "neutral"

    # with col1:
    #     display_metric_pill(
    #         label="Overall Score",
    #         value=f"{avg1:.1f}/5" if isinstance(avg1, (int, float)) else "N/A",
    #         vendor=supplier1_data.get('supplier_name', 'Supplier 1'),
    #         delta=delta_text_1,
    #         status=status_1,
    #         icon=None,
    #     )

    # # Supplier 2 pill
    # if avg1 is not None and avg2 is not None:
    #     if avg2 > avg1:
    #         delta_text_2 = "Leading"
    #         status_2 = "positive"
    #     elif avg2 < avg1:
    #         pct = ((avg1 - avg2) / avg1) * 100 if avg1 else 0
    #         delta_text_2 = f"-{pct:.1f}% lower"
    #         status_2 = "negative"
    #     else:
    #         delta_text_2 = "Tied"
    #         status_2 = "neutral"
    # else:
    #     delta_text_2 = None
    #     status_2 = "neutral"

    # with col2:
    #     display_metric_pill(
    #         label="Overall Score",
    #         value=f"{avg2:.1f}/5" if isinstance(avg2, (int, float)) else "N/A",
    #         vendor=supplier2_data.get('supplier_name', 'Supplier 2'),
    #         delta=delta_text_2,
    #         status=status_2,
    #         icon=None,
    #     )

    # # Recommendation pill
    # preferred = comparison.get('recommendation', {}).get('preferred_supplier')
    # confidence = comparison.get('recommendation', {}).get('confidence_level')
    # with col3:
    #     display_metric_pill(
    #         label="Recommended",
    #         value=preferred or "N/A",
    #         vendor=None,#f"Confidence: {confidence}" if confidence else None,
    #         delta=None,
    #         status="positive" if preferred else "neutral",
    #         icon=None,
    #     )

    # # Offers Overview (Homepage) Sections
    # st.markdown("---")
    # st.markdown("### Offers Overview")
    st.markdown('<h2 class="section-header">Offers Overview</h2>', unsafe_allow_html=True)


    homepage1 = supplier1_data.get('homepage', {}) if supplier1_data else {}
    homepage2 = supplier2_data.get('homepage', {}) if supplier2_data else {}

    def _safe_get_value_and_sources(section: dict, key: str) -> tuple[str, list[dict]]:
        """Return (value, sources) from field object or sensible defaults."""
        try:
            field = section.get(key, {}) if isinstance(section, dict) else {}
            if isinstance(field, dict):
                value = field.get("value", "Not found")
                if value is None or (isinstance(value, str) and not value.strip()):
                    value = "Not found"
                sources = field.get("sources", [])
                if not isinstance(sources, list):
                    sources = []
                return str(value), sources
            # Backward compatibility if field is a plain string
            value = field if isinstance(field, str) else "Not found"
            return (value or "Not found"), []
        except Exception:
            return "Not found", []

    def render_section(title: str, key: str, fields: list[tuple[str, str]]):
        section1 = homepage1.get(key, {}) if isinstance(homepage1, dict) else {}
        section2 = homepage2.get(key, {}) if isinstance(homepage2, dict) else {}
        
        # Prepare supplier display names for both the on-screen table header and CSV export
        s1_name = supplier1_data.get('supplier_name', 'Supplier 1')
        s2_name = supplier2_data.get('supplier_name', 'Supplier 2')

        # Build CSV rows for this section; each row corresponds to one metric/field
        # The CSV will include the information plus references (no emojis) within the supplier columns
        csv_rows: list[dict] = []

        # Helper to compose a plain-text cell for CSV: "value | fileA:chunkA | fileB:chunkB"
        def make_csv_cell(value: str, sources: list[dict]) -> str:
            try:
                safe_value = str(value).strip() if value is not None else "Not found"
                if not safe_value:
                    safe_value = "Not found"
                if sources:
                    parts: list[str] = []
                    for src in sources:
                        fname = str(src.get('filename', 'Unknown'))
                        chunk = str(src.get('chunk_id', ''))
                        parts.append(f"{fname}:{chunk}")
                    return f"{safe_value} | {' | '.join(parts)}"
                return safe_value
            except Exception:
                return "Not found"

        # Build compact HTML table similar to Detailed_Comparison
        table_css = """
        <style>
        table.rfq-table { width: 100%; border-collapse: collapse; table-layout: fixed; }
        table.rfq-table th, table.rfq-table td { 
            border-bottom: 1px solid #e5e7eb; padding: 8px 10px; vertical-align: top; word-wrap: break-word;
        }
        table.rfq-table thead th { background: #f8fafc; font-weight: 600; }
        table.rfq-table .caption { color: #6b7280; font-size: 12px; margin-top: 4px; }
        table.rfq-table col.metric { width: 25%; }
        table.rfq-table col.s1 { width: 37.5%; }
        table.rfq-table col.s2 { width: 37.5%; }
        </style>
        """

        # Compose rows
        body_rows: list[str] = []
        for field_key, label in fields:
            v1, s1 = _safe_get_value_and_sources(section1, field_key)
            v2, s2 = _safe_get_value_and_sources(section2, field_key)

            def make_cell(value: str, sources: list[dict]) -> str:
                is_missing = str(value).strip().lower() == "not found"
                safe_val = html.escape(str(value))
                val_html = f"<span style='color:#ef4444; font-weight:600'>{safe_val}</span>" if is_missing else safe_val
                if sources:
                    parts = []
                    for src in sources:
                        fname = html.escape(str(src.get('filename', 'Unknown')))
                        chunk = html.escape(str(src.get('chunk_id', '')))
                        parts.append(f"{fname}:{chunk}")
                    src_html = f"<div class='caption'>{' | '.join(parts)}</div>"
                else:
                    src_html = ""
                return f"<div>{val_html}</div>{src_html}"

            # Append the plain-text data to CSV rows (includes references without emojis)
            csv_rows.append({
                "Metric": label,
                s1_name: make_csv_cell(v1, s1),
                s2_name: make_csv_cell(v2, s2),
            })

            metric_html = html.escape(label)
            s1_cell = make_cell(v1, s1)
            s2_cell = make_cell(v2, s2)
            body_rows.extend([
                "<tr>",
                f"<td>{metric_html}</td>",
                f"<td>{s1_cell}</td>",
                f"<td>{s2_cell}</td>",
                "</tr>",
            ])

        # Create a DataFrame from the collected rows for CSV download
        df = pd.DataFrame(csv_rows)

        table_html = [
            table_css,
            '<table class="rfq-table">',
            '<colgroup><col class="metric"/><col class="s1"/><col class="s2"/></colgroup>',
            '<thead>',
            '<tr>',
            '<th>Metric</th>',
            f"<th>SupplierA</th>",
            f"<th>SupplierB</th>",
            '</tr>',
            '</thead>',
            '<tbody>',
            *body_rows,
            '</tbody>',
            '</table>'
        ]

        with st.expander(title, expanded=False):
            st.markdown("".join(table_html), unsafe_allow_html=True)
            # Provide a CSV download button for each section table
            st.download_button(
                label=f"Download {title} as CSV",
                data=df.to_csv(index=False),
                file_name=f"{key}_{title.lower().replace(' ', '_')}.csv",
                mime="text/csv",
                key=f"download_{key}_csv",
            )

    # Sections configuration (order preserved)
    render_section(
        "PROJECT INFORMATION",
        "project_information",
        [
            ("project_title", "Project Title"),
            ("project_description", "Project Description"),
            ("contracting_authority", "Contracting Authority"),
            ("estimated_contract_value", "Estimated Contract Value"),
            ("contract_duration", "Contract Duration"),
            ("location_of_works_services", "Location of Works/Services"),
            ("contact_person_details", "Contact Person Details"),
        ],
    )
    render_section(
        "KEY DATES & DEADLINES",
        "key_dates_and_deadlines",
        [
            ("rfq_issue_date", "RFQ Issue Date"),
            ("clarification_deadline", "Clarification Deadline"),
            ("submission_deadline", "Submission Deadline"),
            ("contract_award_date", "Contract Award Date"),
            ("contract_start_date", "Contract Start Date"),
            ("contract_completion_date", "Contract Completion Date"),
            ("rfq_validity_period", "RFQ Validity Period"),
        ],
    )
    render_section(
        "SCOPE & TECHNICAL REQUIREMENTS",
        "scope_and_technical_requirements",
        [
            ("scope_of_work", "Scope of Work"),
            ("methodology_requirements", "Methodology Requirements"),
            ("technical_specifications", "Technical Specifications"),
            ("required_outputs_deliverables", "Required Outputs/Deliverables"),
            ("equipment_or_materials", "Equipment or Materials"),
            ("site_access_requirements", "Site Access Requirements"),
            ("data_management_standards", "Data Management Standards"),
        ],
    )
    render_section(
        "SUPPLIER REQUIREMENTS",
        "supplier_requirements",
        [
            ("mandatory_supplier_information", "Mandatory Supplier Information"),
            ("financial_thresholds", "Financial Thresholds"),
            ("insurance_requirements", "Insurance Requirements"),
            ("health_and_safety_compliance", "Health & Safety Compliance"),
            ("certifications_and_accreditations", "Certifications & Accreditations"),
            ("references_and_experience", "References & Experience"),
            ("key_personnel_qualifications", "Key Personnel Qualifications"),
            ("subcontractor_requirements", "Subcontractor Requirements"),
            ("equality_diversity_inclusion", "Equality, Diversity & Inclusion"),
        ],
    )
    render_section(
        "EVALUATION CRITERIA",
        "evaluation_criteria",
        [
            ("evaluation_methodology", "Evaluation Methodology"),
            ("technical_criteria", "Technical Criteria"),
            ("commercial_criteria", "Commercial Criteria"),
            ("scoring_system", "Scoring System"),
            ("award_decision_basis", "Award Decision Basis"),
        ],
    )
    render_section(
        "PRICING & PAYMENT",
        "pricing_and_payment",
        [
            ("pricing_format", "Pricing Format"),
            ("price_inclusions", "Price Inclusions"),
            ("payment_terms", "Payment Terms"),
            ("invoicing_requirements", "Invoicing Requirements"),
        ],
    )
    render_section(
        "LEGAL & CONTRACTUAL",
        "legal_and_contractual",
        [
            ("terms_and_conditions", "Terms and Conditions"),
            ("acceptance_of_terms", "Acceptance of Terms"),
            ("confidentiality_requirements", "Confidentiality Requirements"),
            ("intellectual_property_rights", "Intellectual Property Rights"),
            ("freedom_of_information", "Freedom of Information"),
            ("anti_corruption_and_bribery", "Anti-Corruption and Bribery"),
            ("termination_clauses", "Termination Clauses"),
            ("dispute_resolution", "Dispute Resolution"),
        ],
    )
    render_section(
        "COMPLIANCE & EXCLUSION GROUNDS",
        "compliance_and_exclusion_grounds",
        [
            ("mandatory_exclusion_grounds", "Mandatory Exclusion Grounds"),
            ("discretionary_exclusion_grounds", "Discretionary Exclusion Grounds"),
            ("self_declaration_requirements", "Self-Declaration Requirements"),
            ("evidence_submission_timing", "Evidence Submission Timing"),
        ],
    )
    render_section(
        "SUSTAINABILITY & SOCIAL VALUE",
        "sustainability_and_social_value",
        [
            ("sustainability_commitments", "Sustainability Commitments"),
            ("social_value_requirements", "Social Value Requirements"),
            ("modern_slavery_compliance", "Modern Slavery Compliance"),
        ],
    )
    render_section(
        "CONTRACT MANAGEMENT & REPORTING",
        "contract_management_and_reporting",
        [
            ("contract_management_arrangements", "Contract Management Arrangements"),
            ("progress_reporting_requirements", "Progress Reporting Requirements"),
            ("key_project_milestones", "Key Project Milestones"),
            ("quality_assurance_measures", "Quality Assurance Measures"),
        ],
    )
# # Sidebar information
# with st.sidebar:
#     st.header("Dashboard Controls")
    
#     if st.button("Refresh Analysis", type="primary", use_container_width=True):
#         # Clear cache and reload
#         st.cache_data.clear()
#         st.rerun()
    
#     with st.expander("Advanced Options"):
#         processor = get_data_processor()
#         if st.button("Clear Cache"):
#             cleared = processor.clear_cache()
#             st.success(f"Cleared {cleared} cache files")
        
#         st.caption("Use this if you're experiencing data issues")
    
#     st.markdown("---")
#     st.markdown("### About")
#     st.markdown("""
#     This dashboard analyzes supplier proposals using structured LLM analysis to provide insights on:
#     - TQDCS scoring
#     - Cost optimization
#     - Risk assessment
#     - Parts comparison
#     - Sourcing strategy
#     """)
    

