# Financial Flux Analysis — System Documentation

## Overview

Enterprise financial analysis application for enterprise financial analysis.
Reads financial data directly from SAP HANA view `v_delec_fin`.

## Architecture

- **Backend**: Python 3.x + FastAPI
- **Frontend**: SAP UI5 1.120.0 / Fiori (Horizon theme) + Vite
- **AI**: SAP Generative AI Hub (`gen-ai-hub-sdk`)
- **Data Source**: SAP HANA / SAP Datasphere — view `v_delec_fin`

---

## Backend Modules

### `database/hana_connection.py`
SAP HANA connection encapsulation using `hdbcli`.
Credentials loaded from environment variables (`HANA_ADDRESS`, `HANA_PORT`, etc.).
Implements context manager protocol for safe connection lifecycle.

### `utils/currency_parser.py`
Robust parser for European-format currency strings from HANA:
- `"- 561.434,27 USD"` → `-561434.27`
- `"1.234,99 USD"` → `1234.99`
- Never raises exceptions; logs warnings and returns `0.0` on failure.

### `services/hana_financial_service.py`
Main data access layer. Queries `v_delec_fin` with parameterized SQL.
Methods:
- `get_financial_data(fiscal_year, fiscal_period, company_code)` → normalized DataFrame
- `get_available_periods(fiscal_year, company_code)` → `{periods, years}`
- `get_company_codes(fiscal_year)` → list of strings
- `get_grouped_analysis(group_by, ...)` → list with variance data
- `check_connectivity()` → `{connected, message}`

### `src/load_data.py`
Thin wrapper around `HANAFinancialService.get_financial_data()`.
Maintains the same function signature as the previous Excel-based implementation
for full backward compatibility with all callers.

### `src/load_wbs.py`
Loads WBS data from HANA table configured via `HANA_WBS_TABLE` env var.
Returns empty DataFrame if not configured (graceful degradation).

### `src/detect_variance.py`
Detects variances between fiscal periods. **Unchanged.**

### `src/classify_drivers.py`
Classifies variance drivers by magnitude. **Unchanged.**

### `src/explain_llm.py`
Generates LLM explanations via SAP Generative AI Hub. **Unchanged.**

### `src/drilldown_analysis.py`
Performs drilldown analysis for specific accounts/periods. **Unchanged.**

### `src/wbs_analysis.py`
WBS analysis by joining financial and WBS data. **Unchanged.**

### `src/align_documents.py`
Aligns financial documents with WBS structure. **Unchanged.**

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Health check + HANA connectivity status |
| POST | `/api/analysis` | Full variance analysis with AI explanation |
| POST | `/api/wbs-analysis` | WBS analysis |
| GET | `/api/financial-data` | Raw financial data with filters |
| GET | `/api/available-periods` | Distinct fiscal periods and years |
| GET | `/api/company-codes` | Distinct company codes |
| GET | `/api/comparison` | Period-over-period variance comparison |
| GET | `/api/grouped-analysis` | Aggregated data by G/L Account / Profit Center / Cost Center |
| GET | `/api/drilldown` | Detailed transaction records |
| GET | `/api/drilldown-analysis` | Alias for /api/drilldown |
| GET | `/api/wbs-data` | WBS data |

### `/api/grouped-analysis` Parameters
| Parameter | Type | Description |
|-----------|------|-------------|
| `group_by` | string | Required. One of: `G/L Account`, `Profit Center`, `Cost Center` |
| `fiscal_year` | string | Optional filter |
| `fiscal_period` | string | Current period |
| `previous_period` | string | Previous period for variance calculation |
| `company_code` | string | Optional filter |

---

## Frontend

- SAP UI5 1.120.0 with Horizon theme
- Vite dev server with proxy to backend (`/api` → `http://localhost:8000`)
- Views: App (main shell), Variances (grouped analysis), WBS
- Grouping selector: G/L Account, Profit Center, Cost Center
- Previous Period selector for variance comparison

---

## Environment Variables

```env
# SAP Generative AI Hub
AICORE_AUTH_URL=
AICORE_CLIENT_ID=
AICORE_CLIENT_SECRET=
AICORE_RESOURCE_GROUP=
AICORE_BASE_URL=

# SAP HANA Connection (required)
HANA_ADDRESS=
HANA_HOST=
HANA_PORT=443
HANA_USER=
HANA_PASSWORD=
HANA_SCHEMA=
HANA_ENCRYPT=True

# SAP HANA WBS table (optional)
HANA_WBS_TABLE=
```

---

## Setup Instructions

### Backend

```bash
cd backend
pip install -r requirements.txt
# Fill in backend/.env with HANA and AI Core credentials
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

### Frontend

```bash
cd fe-fiori
npm install
npm run dev
# Open http://localhost:3000
```

---

## Amount Parsing

The `Amount in Company Code Currency` column in `v_delec_fin` contains
European-format strings with currency codes:

```
"- 561.434,27 USD"  →  -561434.27
"1.234,99 USD"      →   1234.99
```

The `parse_currency_amount()` function in `utils/currency_parser.py` handles:
1. Null / NaN → `0.0`
2. Already-numeric values → `float(value)`
3. Trailing currency code removal
4. Negative sign with space (`"- 561"` → `-561`)
5. European thousand separator (`.`) removal
6. Decimal comma (`,`) → decimal point (`.`)

---

## Migration Notes (Excel → HANA)

- `openpyxl` dependency removed from `requirements.txt`
- `EXCEL_FILE_PATH` and `WBS_FILE_PATH` env vars no longer used
- All data now served from `v_delec_fin` HANA view
- All existing API contracts preserved
- All AI/LLM analysis logic unchanged
**Performance**: Dictionary-based lookups for O(1) access time

---

#### `src/drilldown_analysis.py`
**Purpose**: Progressive drill-down investigation following Rivian requirements

**Key Classes**:
- `DrillDownResult`: Data structure for analysis results
- `DrillDownAnalyzer`: Main analysis engine

**Investigation Hierarchy**:
```
GL Account
   ↓
WBS Element / Internal Order
   ↓
Project
   ↓
Program (High-Level → Detailed)
   ↓
Cost Center
   ↓
Vendor/Partner
   ↓
Text Descriptions
```

**Context-Aware Investigation Paths**:

1. **R&D GLs (630-633)**:
   - Path: WBS/Project → Program → Cost Center → Text → Vendor
   - Rationale: R&D spending is primarily project-based

2. **Payroll GLs (640-641)**:
   - Path: Cost Center → Org → Headcount
   - Rationale: Personnel costs follow organizational structure

3. **Facilities GLs (652-653)**:
   - Path: Vendor → Location → Cost Center → Text
   - Rationale: Facilities costs are vendor/location-centric

4. **Default Path**:
   - Path: WBS/Project → Cost Center → Vendor → Text
   - Rationale: General-purpose investigation

**Drill-Down Steps**:

1. `determine_investigation_path()`: Selects appropriate path based on GL type
2. `drill_into_wbs_project()`: Aggregates by WBS, enriches with descriptions/programs
3. `drill_into_cost_center()`: Aggregates by cost center
4. `drill_into_vendor()`: Aggregates by vendor/partner
5. `analyze_text_evidence()`: Extracts insights from unstructured text
6. `assess_evidence_sufficiency()`: Evaluates completeness (minimum 2 levels required)
7. `generate_root_cause_explanation()`: Creates business-readable narrative
8. `perform_drilldown()`: Orchestrates complete analysis

**Evidence Chain**:
Each drill-down level collects:
- Aggregated amounts
- Top contributors (top 5)
- Enrichment data (descriptions, programs)
- Missing data indicators

**Confidence Scoring**:
- **High**: 3+ evidence levels found
- **Medium**: 2 evidence levels found
- **Low**: <2 evidence levels found

**Escalation Criteria**:
- Evidence insufficiency (< 2 levels)
- Low confidence
- Ambiguous drivers without clear descriptions
- Large variances without business justification

---

### AI Analysis Module

#### `src/explain_llm.py`
**Purpose**: Comprehensive variance analysis with AI explanation

**Key Functions**:

1. **Analysis Functions**:
   - `summarize_variances()`: Aggregates to GL + Cost Center + Driver level
   - `classify_variances_by_magnitude()`: Splits into Critical/High/Medium/Low
   - `detect_patterns()`: Identifies concentration, trends, anomalies
   - `analyze_text_descriptions()`: Extracts insights from unstructured data
   - `trace_to_source()`: Adds source-level context (reversals, intercompany)
   - `identify_insufficient_evidence()`: Flags items needing escalation

2. **Pattern Detection**:
   - Driver distribution (counts & amounts)
   - GL account concentration (top 10)
   - Cost center concentration (top 10)
   - Sign analysis (increases vs decreases)
   - Pareto analysis (top 20% concentration)
   - Text frequency analysis

3. **LLM Integration**:
   - `build_variance_context()`: Structures data for LLM consumption
   - `build_explanation_prompt()`: Creates finance-controller persona prompt
   - `ask_llm_for_explanation()`: Calls SAP Gen AI Hub (gpt-4o)
   - `generate_insights_summary()`: Creates human-readable summary

4. **Main Pipeline** (`analyze_variances_with_llm()`):

```
Step 1: Load WBS lookup data
   ↓
Step 2: Trace to source (add context fields)
   ↓
Step 3: Summarize by GL + CC + Driver
   ↓
Step 4: Classify by magnitude
   ↓
Step 5: Detect patterns
   ↓
Step 6: Analyze text descriptions
   ↓
Step 7: Perform progressive drill-down (top 10)
   ↓
Step 8: Build enriched context (includes drill-down insights)
   ↓
Step 9: Generate LLM explanation
   ↓
Step 10: Identify escalation items
   ↓
Return: Complete analysis results
```

**LLM Prompt Structure**:
- **Persona**: Senior finance controller
- **Task**: Explain R&D cost variances (YTD Q3 2024 vs 2025)
- **Instructions**: 
  1. Identify main drivers
  2. Explain business implications
  3. Highlight anomalies
  4. Analyze transaction descriptions
  5. Provide actionable insights
  6. Use professional language
  7. Keep 300-500 words
- **Context**: Includes summary data, patterns, text analysis, and drill-down results

**Output Dictionary**:
```python
{
    "summary": DataFrame,              # GL + CC + Driver aggregation
    "classifications": Dict,            # Critical/High/Medium/Low splits
    "patterns": Dict,                   # Detected patterns & metrics
    "text_analysis": Dict,              # Unstructured data insights
    "insights_summary": str,            # Human-readable summary
    "llm_explanation": str,             # AI-generated narrative
    "escalations": List[Dict],          # Items requiring review
    "drilldown_results": List[Result],  # Progressive drill-down findings
    "traced_data": DataFrame,           # Enriched transaction data
    "context_used": str                 # Full context sent to LLM
}
```

---

## Application Layout

### Header Section
- **Title**: "R&D YTD Variance – SAP ACDOCU with AI Analysis"
- **Purpose**: Clear identification of application scope

### Sidebar Configuration
```
┌─────────────────────────────┐
│ Analysis Configuration      │
├─────────────────────────────┤
│ Materiality Threshold ($)   │
│ [    1000    ] ▲▼           │
│                             │
│ ☑ Generate AI Explanation   │
│                             │
│ ───────────────────────────│
│ Powered by SAP Gen AI Hub   │
└─────────────────────────────┘
```

### Main Content Flow

#### 1. Document-Level Variances
- **Display**: Full transaction detail table
- **Features**: 
  - Conditional styling (disabled if > 262K cells)
  - Sortable columns
  - Amount formatting ($XX,XXX.XX)
  - Download to CSV
- **Purpose**: Raw variance data at document level

#### 2. Variance Summary by GL & Cost Center
- **Display**: Aggregated table
- **Columns**: GL Account, Cost Center, Driver, Total Delta, Document Count
- **Features**: 
  - Formatted amounts
  - Sorted by absolute delta
  - Download to CSV
- **Purpose**: Executive-level summary

#### 3. Variance Classification by Magnitude
- **Display**: 4-column metric cards
- **Metrics**:
  - Critical (Top 10%)
  - High (10-30%)
  - Medium (30-60%)
  - Low (Bottom 40%)
- **Features**: 
  - Expandable Critical items table
  - Formatted amounts
- **Purpose**: Risk prioritization

#### 4. Pattern Detection & Insights
- **Display**: Formatted text output
- **Content**:
  - Top variance drivers
  - Document counts by driver
  - Top GL accounts
  - Top cost centers
  - Variance direction analysis
  - Concentration metrics
  - Text description analysis
- **Purpose**: Trend identification

#### 5. Transaction Description Analysis
- **Display**: Expandable section
- **Content**:
  - Unique description count
  - Coverage statistics
  - Top 10 descriptions table
- **Purpose**: Unstructured data insights

#### 6. AI-Generated Variance Explanation
- **Display**: Markdown-formatted text
- **Content**: 300-500 word narrative from GPT-4o
- **Purpose**: Business-readable explanation

#### 7. Progressive Drill-Down Analysis (NEW)
- **Display**: Summary table + detailed expanders
- **Summary Table Columns**:
  - GL Account
  - Cost Center
  - Driver
  - Amount
  - Investigation Path
  - Confidence
  - Escalation flag
- **Detailed View** (Top 5):
  - Root cause explanation
  - Evidence chain status (✅ Found / ⚠️ Missing)
  - Evidence details (expandable tables)
  - Escalation reasons
  - Missing data list
- **Purpose**: Deep-dive investigation results

#### 8. Items Requiring Escalation
- **Display**: Warning banner + table
- **Content**:
  - Account Number
  - Cost Center
  - Driver
  - Amount
  - Reasons (semicolon-separated)
  - Priority (High/Medium)
  - Missing Data
- **Features**: 
  - Color coding by priority
  - Download to CSV
- **Purpose**: Action item tracking

#### 9. Key Metrics Dashboard
- **Display**: 3-column metric cards
- **Metrics**:
  - Total Variance
  - Net Variance (with increase/decrease indicator)
  - Top 20% Concentration
- **Purpose**: High-level KPIs

---

## Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    User Interface (app.py)                   │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
         ┌─────────────────────────────┐
         │  Data Loading & Preparation  │
         │  (load_data.py)             │
         └──────────┬──────────────────┘
                    │
                    ▼
         ┌─────────────────────────────┐
         │  Period Alignment           │
         │  (align_documents.py)       │
         └──────────┬──────────────────┘
                    │
                    ▼
         ┌─────────────────────────────┐
         │  Variance Detection         │
         │  (detect_variance.py)       │
         └──────────┬──────────────────┘
                    │
                    ▼
         ┌─────────────────────────────┐
         │  Driver Classification      │
         │  (classify_drivers.py)      │
         └──────────┬──────────────────┘
                    │
                    ▼
         ┌─────────────────────────────┐
         │  WBS Enrichment             │
         │  (load_wbs.py)              │
         └──────────┬──────────────────┘
                    │
                    ▼
    ┌───────────────────────────────────────────┐
    │     Comprehensive Analysis                │
    │     (explain_llm.py)                      │
    ├───────────────────────────────────────────┤
    │  • Summarization                          │
    │  • Classification                         │
    │  • Pattern Detection                      │
    │  • Text Analysis                          │
    │  • Progressive Drill-Down                 │
    │    (drilldown_analysis.py)               │
    │  • LLM Explanation                        │
    │  • Escalation Identification              │
    └──────────┬────────────────────────────────┘
               │
               ▼
    ┌─────────────────────────────┐
    │  Results Presentation       │
    │  • Tables                   │
    │  • Charts                   │
    │  • Metrics                  │
    │  • Drill-Down Evidence      │
    │  • Escalations              │
    └─────────────────────────────┘
```

---

## Key Design Principles

### 1. Modularity
Each file has a single, well-defined responsibility. This enables:
- Easy testing
- Independent updates
- Clear dependencies
- Reusable components

### 2. Progressive Enhancement
The system works in layers:
- **Layer 1**: Basic variance detection (always runs)
- **Layer 2**: Statistical analysis and classification (always runs)
- **Layer 3**: WBS enrichment (runs if data available)
- **Layer 4**: Progressive drill-down (runs if WBS loaded)
- **Layer 5**: AI explanation (optional, user-controlled)

### 3. Error Resilience
The system gracefully handles:
- Missing WBS data
- Failed LLM calls
- Large datasets (>262K cells)
- Missing columns
- Data type mismatches

### 4. Performance Optimization
- Dictionary-based lookups (O(1) vs O(n))
- Conditional styling (disabled for large datasets)
- Top-N analysis (drill-down on top 10 only)
- Cached WBS lookups

### 5. Auditability
Every analysis decision is traceable:
- Evidence chain recorded
- Missing data documented
- Confidence levels explicit
- Investigation paths visible

---

## Configuration Files

### `.env`
**Purpose**: Environment variables for SAP Gen AI Hub
**Required Variables**:
- Gen AI Hub endpoint
- Authentication credentials
- Model selection (gpt-4o)

### `requirements.txt`
**Purpose**: Python dependencies
**Key Libraries**:
- `streamlit`: Web UI framework
- `pandas`: Data manipulation
- `openpyxl`: Excel file reading
- `numpy`: Numerical operations
- `gen-ai-hub`: SAP AI Core integration
- `langchain`: LLM orchestration

---

## Usage Workflow

### For End Users

1. **Launch Application**: `streamlit run app.py`
2. **Configure Analysis**:
   - Set materiality threshold (default: $1,000)
   - Enable/disable AI explanation
3. **Run Analysis**: Click "Run Analysis" button
4. **Review Results**:
   - Scan document-level variances
   - Review executive summary
   - Examine classification breakdown
   - Read AI explanation
   - Investigate drill-down findings
   - Download escalation list
5. **Take Action**:
   - Contact cost center owners for escalated items
   - Use evidence chain for audit trail
   - Export data for presentations

### For Developers

1. **Add New Data Sources**: Extend `load_data.py`
2. **Add Driver Types**: Modify `classify_drivers.py` logic
3. **Customize Investigation Paths**: Update `drilldown_analysis.py` GL patterns
4. **Enhance LLM Prompts**: Modify `explain_llm.py` prompt templates
5. **Add Visualizations**: Update `app.py` layout sections

---

## Technical Requirements

### System Requirements
- Python 3.8+
- 4GB+ RAM (for large Excel files)
- SAP Gen AI Hub access (for AI features)

### Data Requirements
- SAP ACDOCU Excel exports (both periods)
- WBS Master file (optional but recommended)
- WBS Descriptions file (optional but recommended)

### Network Requirements
- Internet access for SAP Gen AI Hub API
- HTTPS support
- Outbound connections allowed

---

## Future Enhancement Opportunities

### Short-Term
1. Add vendor enrichment from SAP master data
2. Implement PO-level drill-down
3. Add temporal trending (multi-period comparison)
4. Export drill-down results to PowerPoint

### Medium-Term
1. Real-time SAP HANA connectivity
2. Blackline journal entry integration
3. Automated notification system for escalations
4. Interactive drill-down visualization (tree maps)

### Long-Term
1. Predictive variance forecasting
2. Automated RCA (Root Cause Analysis)
3. Natural language query interface
4. Integration with SAP Analytics Cloud

---

## Support and Maintenance

### Logging
- Warnings printed to console for:
  - Failed WBS data loads
  - Failed drill-down analyses
  - LLM errors

### Error Handling
- Try-catch blocks around:
  - WBS data loading
  - Drill-down execution
  - LLM API calls
- Graceful degradation (system continues without failed components)

### Performance Monitoring
- Monitor for:
  - Excel file load times (>30s indicates size issue)
  - LLM response times (>30s indicates API issue)
  - Memory usage (DataFrame sizes)

---

## Conclusion

This system implements Rivian's variance analysis requirements using a modular, AI-enhanced architecture. It combines rule-based classification, progressive drill-down investigation, and natural language explanation to provide comprehensive insights into R&D cost fluctuations.

The progressive drill-down methodology traces variances from GL accounts through WBS elements, projects, programs, cost centers, and vendors, building an evidence chain that enables confident root cause identification or appropriate escalation when evidence is insufficient.

All analysis is auditable, traceable, and presented in a business-friendly format suitable for executive review and decision-making.
