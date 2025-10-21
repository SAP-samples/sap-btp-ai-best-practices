# Touchless Transactions AI Agents for GR and Invoice Workflows

A comprehensive Microsoft Teams bot for handling invoice validation, purchase order management, and goods receipt confirmation. The bot processes Excel data and provides interactive Adaptive Cards for users to confirm receipts and PO changes.

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Business Logic](#business-logic)
- [Scenarios](#scenarios)
- [Test Cases](#test-cases)
- [Installation & Setup](#installation--setup)
- [Usage](#usage)
- [API Reference](#api-reference)
- [Configuration](#configuration)

## Overview

This bot automates the invoice validation process by:
- Processing Excel data containing invoice, PO, and GR information
- Classifying invoice lines into 7 different scenarios based on business rules
- Providing interactive Adaptive Cards for user confirmation
- Supporting proactive notifications and follow-up alerts
- Integrating with AI/LLM for natural language queries

## Architecture

### Core Components

1. **app.py** - Main Flask application with bot logic
2. **api_client.py** - Data access layer for Excel processing
3. **cards.py** - Adaptive Card UI components
4. **send_after_delay.py** - Proactive messaging utility
5. **llm_client.py** - AI/LLM integration (optional)

### Technology Stack

- **Microsoft Bot Framework** - Teams integration
- **Flask** - Web application framework
- **pandas** - Excel data processing
- **Adaptive Cards** - Interactive UI components
- **Python asyncio** - Asynchronous operations

## Business Logic

### Data Model

The system processes Excel data with the following key columns:
- `Invoice_Number` - Unique invoice identifier
- `Line` - Line number within invoice
- `PO#` - Purchase order number
- `IR_amount` - Invoice Receipt amount
- `PO_amount` - Purchase Order amount
- `GR_amount` - Goods Receipt amount
- `Item_Name` - Item description
- `User` - Assigned user (e.g., "Marta")
- `Received_Flag` - Boolean indicating if item is received

### Classification Algorithm

Each invoice line is classified using the `classify_line()` function based on:

```python
def classify_line(ir: float, po: float, gr: float) -> Dict[str, Any]:
    diff = ir - po
    pct_thr = 0.05 * max(po, 0.0)  # 5% threshold
    amt_thr = 250.0                # €250 threshold
```

**Tolerance Rules:**
- **Within tolerance**: `diff > 0 AND diff ≤ 5% of PO AND diff ≤ €250`
- **Exceed percentage only**: `diff > 5% of PO AND diff ≤ €250`
- **Exceed amount only**: `diff > €250 AND diff ≤ 5% of PO`
- **Exceed both**: `diff > 5% of PO AND diff > €250`

## Scenarios

The bot automatically classifies each invoice line into one of seven scenarios based on the relationship between Invoice Receipt (IR), Purchase Order (PO), and Goods Receipt (GR) amounts. Each scenario triggers specific user actions and system responses.

### Scenario 1 (S1) - Full Receipt Confirmation
**Business Context:** Items have been invoiced but not yet marked as received in the system.

**Conditions:**
- `IR ≤ PO AND GR = 0` (Invoice within PO limit, nothing received yet)
- `IR > PO (within tolerance) AND GR = 0` (Invoice slightly over PO, within acceptable limits)

**User Experience:**
- Bot presents: *"Have you received this order?"*
- User action: Click **"✅ Confirm received"** button
- No manual input required

**System Response:**
- Sets `GR_amount = IR_amount`
- Sets `Received_Flag = True`
- Displays confirmation: *"Goods Receipt has been successfully submitted. View order status."*

**Example:**
```
IR: €1,000.00, PO: €1,000.00, GR: €0.00 → User confirms → GR: €1,000.00
```

### Scenario 2 (S2) - Partial Receipt
**Business Context:** Items have been partially received, requiring user to specify the exact amount.

**Conditions:**
- `IR ≤ PO AND IR > GR` (Invoice within PO, but more than currently received)
- `IR > PO (within tolerance) AND IR > GR` (Invoice slightly over PO, more than received)

**User Experience:**
- Bot presents: *"Have you received these items?"*
- Shows current GR amount as suggested value
- User action: Enter actual received amount or click **"✅ Confirm partial received"**

**System Response:**
- Updates `GR_amount` to user-specified value
- Sets `Received_Flag = True` if `GR ≥ IR`
- Intelligent interpretation of user input:
  - Empty/0 → Full receipt (GR = IR)
  - Value between current GR and IR → Absolute total
  - Value ≤ (IR-GR) → Delta addition
  - Confirmation of current GR → Full receipt

**Example:**
```
IR: €1,000.00, PO: €1,000.00, GR: €500.00 → User enters €750.00 → GR: €750.00
```

### Scenario 3 (S3) - Major PO Change Required
**Business Context:** Invoice significantly exceeds PO limits, requiring formal approval workflow.

**Conditions:**
- `IR > PO (exceeds both 5% AND €250)` (Major discrepancy requiring approval)

**User Experience:**
- Bot presents: *"Your open order has a PO amount of €X and an invoice amount of €Y. Would you like to conduct a PO amount change to €Y?"*
- User action: Click **"✅ Confirm change"** button
- System suggests IR amount as new PO value

**System Response:**
- Updates `PO_amount = IR_amount`
- Initiates approval workflow in Ariba
- Schedules follow-up alert after 5 seconds
- Displays: *"PO Amount has been updated to €X. New approval workflow has been initiated in Ariba, would you like to view the status of this order?"*

**Example:**
```
IR: €1,300.00, PO: €1,000.00 → Exceeds both 5% (€50) and €250 → PO updated to €1,300.00
```

### Scenario 6 (S6) - Percentage-Based PO Change
**Business Context:** Invoice exceeds PO by percentage but within absolute amount threshold.

**Conditions:**
- `IR > PO (exceeds 5% only) AND IR > GR` (Percentage overage only)

**User Experience:**
- Bot presents PO change option with suggested IR amount
- User action: Click **"✅ Confirm changes"** button

**System Response:**
- Updates `PO_amount = IR_amount`
- No approval workflow required (minor change)
- Displays confirmation message

**Example:**
```
IR: €1,100.00, PO: €1,000.00 → Exceeds 5% (€50) but under €250 → PO updated
```

### Scenario 7 (S7) - Amount-Based PO Change
**Business Context:** Invoice exceeds PO by absolute amount but within percentage threshold.

**Conditions:**
- `IR > PO (exceeds €250 only)` (Amount overage only)

**User Experience:**
- Bot presents PO change option
- User action: Click **"✅ Confirm change"** button

**System Response:**
- Updates `PO_amount = IR_amount`
- Displays confirmation message

**Example:**
```
IR: €5,300.00, PO: €5,000.00 → Exceeds €250 but under 5% → PO updated
```

### Scenario Classification Logic

The system uses precise business rules to determine scenarios:

```python
def classify_line(ir: float, po: float, gr: float):
    diff = ir - po
    pct_threshold = 0.05 * max(po, 0.0)  # 5% of PO amount
    amt_threshold = 250.0                # €250 absolute threshold
    
    # Tolerance checks
    within_tolerance = (diff > 0 and diff <= pct_threshold and diff <= amt_threshold)
    exceed_percent_only = (diff > pct_threshold and diff <= amt_threshold)
    exceed_amount_only = (diff > amt_threshold and diff <= pct_threshold)
    exceed_both = (diff > pct_threshold and diff > amt_threshold)
```

### Multi-Line Invoice Handling

For invoices with multiple line items, the bot:
1. **Analyzes each line individually** using the scenario classification
2. **Groups similar scenarios** for efficient user interaction
3. **Presents unified confirmation** with line-by-line details
4. **Processes all confirmations** in a single transaction
5. **Provides comprehensive results** showing all changes made

**Example Multi-Line Display:**
```
Hello Marta, there are 3 line items requiring your confirmation. Have you received these items?

Line 1 — Office Cables (Line 15)
Condition: IR ≤ PO, GR = 0
[Confirm received button]

Line 2 — Whiteboards (Line 16) 
Condition: IR > PO (within tolerance), IR > GR
[Partial receipt input field]

Line 3 — Monitors (Line 17)
Condition: IR > PO (exceed % and €250)
[PO change confirmation]
```

## Test Cases

The system includes comprehensive test cases covering all scenarios:

### Standard Test Cases
1. **List my POs** - Display user's purchase orders
2. **Which orders are received?** - Show completed orders
3. **Which orders are not received?** - Show pending orders
4. **Show totals IR vs PO** - Display financial summaries
5. **Items for invoice 5109058689** - Show invoice line items
6. **Invoice 5109058689** - Validate specific invoice
7. **All PO for Whiteboards** - Filter by item type
8. **ALL PO for office cable** - Filter by item description
9. **What is the status of invoice 5109058684** - Status inquiry
10. **Check invoice 5109058683** - Validation request

### Push Notification Test Cases

Four specific use cases for proactive alerts:

```bash
# Case 1: Invoice 5109058676, PO 4501366627
python send_after_delay.py --app-url http://localhost:3978/alerts --type not_received --po 4501366627 --invoice 5109058676 --recipient Marta --delay 1

# Case 2: Invoice 5109058677, PO 4501366626 (Special handling)
python send_after_delay.py --app-url http://localhost:3978/alerts --type not_received --po 4501366626 --invoice 5109058677 --recipient Marta --delay 1

# Case 3: Invoice 5109058679, PO 4501366634
python send_after_delay.py --app-url http://localhost:3978/alerts --type not_received --po 4501366634 --invoice 5109058679 --recipient Marta --delay 1

# Case 4: Invoice 5109058689, PO 4501366628
python send_after_delay.py --app-url http://localhost:3978/alerts --type not_received --po 4501366628 --invoice 5109058689 --recipient Marta --delay 1
```

**Special Case 2 Handling:**
Invoice 5109058677 requires special attention for received flag setting. The system properly handles partial receipt confirmation to ensure the received flag is set when users confirm the current GR amount.

## Installation & Setup

### Prerequisites
- Python 3.8+
- Microsoft Teams app registration
- Excel file with invoice data

### Installation Steps

1. **Clone the repository**
```bash
git clone <repository-url>
cd "Touchless Transactions AI Agents for GR and Invoice Workflows"
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Configure environment variables**
Create a `.env` file with the following variables:

```env
# Microsoft Teams Bot Configuration
MICROSOFT_APP_ID=your-app-id                    # Bot application ID from Azure Bot Service
MICROSOFT_APP_PASSWORD=your-app-password        # Bot application password/secret
MICROSOFT_APP_TENANT_ID=your-tenant-id          # Azure AD tenant ID for your organization

# SAP AI Core Configuration (for LLM integration)
AICORE_AUTH_URL="https://xxxxxxxxxx.authentication.eu10.hana.ondemand.com"  # SAP AI Core authentication endpoint
AICORE_CLIENT_ID="xxxxxx"                       # SAP AI Core client ID for OAuth
AICORE_CLIENT_SECRET="xxxxxx"                   # SAP AI Core client secret for OAuth
AICORE_BASE_URL="https://api.xxxxxxx.eu-central-1.aws.ml.hana.ondemand.com/v2"  # SAP AI Core API base URL
AICORE_RESOURCE_GROUP="default"                 # Resource group name in SAP AI Core

# Application Configuration
USER_NAME=Marta                                 # Default user name for filtering data
EXCEL_SOURCE=dataset_enriched.xlsx              # Path to Excel file containing invoice data
EXCEL_SHEET=POC scienarios                      # Sheet name within the Excel file
APP_PUBLIC_URL=http://localhost:3978            # Public URL for webhook callbacks and proactive messaging
```

### Microsoft Teams Setup

To set up the bot in Microsoft Teams, follow the comprehensive guide at: https://www.unpage.ai/guides/how-to-build-a-bot-for-microsoft-teams

**Required Steps:**

1. **Create Azure Bot Service**
   - Register a new bot in Azure Portal
   - Generate App ID and App Password
   - Configure messaging endpoint: `https://your-domain.com/api/messages`

2. **Configure Bot Channels**
   - Enable Microsoft Teams channel
   - Add bot to Teams app manifest
   - Configure permissions and scopes

3. **App Registration**
   - Create Azure AD app registration
   - Set redirect URIs for authentication
   - Configure API permissions for Teams

4. **Teams App Manifest**
   - Create manifest.json with bot configuration
   - Define supported commands and capabilities
   - Set up adaptive card permissions

5. **Required Credentials to Obtain:**
   - **MICROSOFT_APP_ID**: Bot application ID from Azure Bot Service
   - **MICROSOFT_APP_PASSWORD**: Bot application password/secret
   - **MICROSOFT_APP_TENANT_ID**: Azure AD tenant ID
   - **App Package**: Teams app package (.zip) for sideloading
   - **Messaging Endpoint**: Public HTTPS URL for bot communication

6. **Deployment Considerations**
   - Ensure HTTPS endpoint is accessible
   - Configure proper CORS settings
   - Set up SSL certificates for production
   - Test bot functionality in Teams

4. **Prepare Excel data**
Ensure your Excel file contains the required columns and is accessible to the application.

5. **Run the application**
```bash
flask --app=app run --host=0.0.0.0 --port=3978 --reload
```

## Usage

### Supported Commands and Questions

The bot supports natural language queries and specific commands. All interactions are case-insensitive and support various phrasings.

#### 1. Purchase Order Management

**List All Orders:**
- `"List my POs"`
- `"List my orders"`
- `"Show my purchase orders"`
- `"My POs"`

**Response:** Displays a comprehensive table with all user's purchase orders including Invoice numbers, Item names, IR/PO/GR amounts, and received status.

**Filter by Receipt Status:**
- `"Which orders are received?"`
- `"Show received orders"`
- `"Received orders"`

**Response:** Shows only orders where `GR_amount >= IR_amount` or `Received_Flag = True`.

- `"Which orders are not received?"`
- `"Show not received orders"`
- `"Pending orders"`
- `"Not received orders"`

**Response:** Shows orders where `IR_amount > GR_amount` and `Received_Flag = False`.

**List All with Status Breakdown:**
- `"List all PO"`
- `"List all POs"`
- `"Show all purchase orders"`

**Response:** Displays two separate tables - one for received orders and one for not received orders.

#### 2. Item-Specific Filtering

**Filter by Item Type:**
- `"All PO for Whiteboards"`
- `"All POs for office cable"`
- `"Show orders for [item name]"`
- `"POs containing [keyword]"`

**Response:** Filters and displays orders containing the specified item name or keyword in the `Item_Name` field.

**Examples:**
```
"All PO for Whiteboards" → Shows all orders with "Whiteboards" in item name
"All POs for office cable" → Shows all orders with "office cable" in item name
```

#### 3. Invoice Validation and Details

**Validate Specific Invoice:**
- `"Invoice 5109058689"`
- `"Validate invoice 5109058689"`
- `"Check invoice 5109058689"`
- `"validate 5109058689"` (command format)

**Response:** Launches the unified validation card with scenario-specific options for confirmation.

**Invoice Status Inquiry:**
- `"What is the status of invoice 5109058684"`
- `"Status of invoice 5109058684"`
- `"Check status 5109058684"`

**Response:** Displays invoice status card with overall delivery status, line counts, and amount summaries, followed by detailed line items table.

**Invoice Line Items:**
- `"Items for invoice 5109058689"`
- `"Show items for invoice 5109058689"`
- `"Line items 5109058689"`
- `"Invoice 5109058689 items"`

**Response:** Shows detailed table of all line items within the specified invoice.

#### 4. Financial Summaries

**Total Value Analysis:**
- `"Show totals IR vs PO"`
- `"What is the total value of my orders"`
- `"Show financial summary"`
- `"Totals"`

**Response:** Displays comprehensive financial breakdown including:
- Delivered orders count and IR total
- Not delivered orders count and IR total  
- Overall IR total across all orders

#### 5. Welcome and Help

**Welcome Interaction:**
- `"Hi"`
- `"Hello"`
- `"Hey"`
- `"Good morning"`
- `"Good afternoon"`
- `"Good evening"`
- `"Start"`
- `"Begin"`

**Response:** Displays welcome card with bot introduction and "Validate invoice" button.

**Help and Examples:**
- Any unrecognized command triggers help examples
- Bot provides sample commands and usage patterns

#### 6. AI-Powered Natural Language Queries

The bot includes optional AI/LLM integration for complex natural language questions:

**Supported Query Types:**
- `"How many invoices are pending for Marta?"`
- `"What's the total amount of undelivered orders?"`
- `"Which invoices need PO changes?"`
- `"Show me all orders with discrepancies"`
- `"What items are most frequently delayed?"`

**AI Response Features:**
- Context-aware responses based on user's data
- Procurement domain expertise
- Concise, actionable answers
- Fallback to structured data when AI unavailable

#### 7. Interactive Card Actions

**Validation Commands:**
- `"Validate invoice"` button → Prompts for invoice number input
- `"Show Help"` button → Displays command examples
- `"Ask AI"` button → Enables natural language queries

**Confirmation Actions:**
- `"✅ Confirm received"` → Full receipt confirmation (S1)
- `"✅ Confirm partial received"` → Partial receipt with amount input (S2)
- `"✅ Confirm change"` → PO amount change (S3, S7)
- `"✅ Confirm changes"` → Multiple changes or percentage-based PO change (S6)

#### 8. Domain-Specific Intelligence

**Keyword Recognition:**
The bot recognizes procurement-related terms and provides relevant responses:
- `invoice`, `po`, `purchase order`, `goods receipt`, `gr`, `order`

**Smart Filtering:**
- Automatically filters results by user (default: "Marta")
- Handles various invoice number formats
- Supports partial item name matching
- Case-insensitive search across all fields

#### 9. Error Handling and Guidance

**Invalid Queries:**
- Non-procurement questions → Domain guidance message
- Invalid invoice numbers → "Invoice not found" with suggestions
- Empty results → Helpful explanations and alternative suggestions

**Example Error Responses:**
```
"What's the weather?" → "This assistant only handles invoices, purchase orders (PO), and goods receipts (GR). Please provide an invoice number or ask about your orders."

"Invoice 999999999" → "No items found for invoice 999999999. Please check the number."
```

#### 10. Proactive Notifications

**Automated Alerts:**
The bot can send proactive notifications for:
- Overdue invoice validations
- PO approval workflow updates
- Receipt confirmations pending
- System-generated follow-ups

**Alert Types:**
- `not_received` → Items awaiting receipt confirmation
- `po_mismatch` → PO amount discrepancies requiring attention
- `generic` → General invoice validation reminders

### Command Processing Logic

The bot uses sophisticated pattern matching to understand user intent:

```python
# Example pattern matching
INVOICE_ID_RE = re.compile(r"\b(\d{7,12})\b")
MY_ORDERS_PATTERNS = re.compile(r"(?is)\b(my\s+orders|list\s+my\s+orders|list\s+my\s+po\b)")
RECEIVED_Q = re.compile(r"(?is)\b(which\s+orders\s+are\s+received|received\s+orders)\b")
STATUS_Q = re.compile(r"(?is)\bwhat\s+is\s+the\s+status\s+of\s+invoice\s+(\d{7,12})\b")
```

**Processing Priority:**
1. **Exact command matches** (e.g., "validate 5109058689")
2. **Pattern-based recognition** (e.g., "Which orders are received?")
3. **Keyword extraction** (e.g., invoice numbers, item names)
4. **AI/LLM fallback** for complex natural language
5. **Domain validation** to ensure procurement relevance
6. **Help guidance** for unrecognized inputs

### Interactive Features

1. **Adaptive Cards** - Rich interactive UI for confirmations
2. **AI Integration** - Natural language query processing
3. **Proactive Alerts** - Automated follow-up notifications
4. **Multi-line Processing** - Handle complex invoices with multiple items

### Confirmation Workflow

1. User receives invoice validation card
2. System displays scenario-specific options
3. User enters required information (if any)
4. User clicks confirmation button
5. System processes changes and updates data
6. Confirmation results displayed
7. Follow-up alerts scheduled (if applicable)

## API Reference

### Main Endpoints

- `POST /api/messages` - Bot Framework message endpoint
- `POST /alerts` - Proactive alert endpoint

### Key Functions

#### classify_line(ir, po, gr)
Classifies invoice line into appropriate scenario based on business rules.

#### build_unified_card(invoice_number)
Creates interactive Adaptive Card for invoice validation.

#### _compute_new_gr(ir, gr, entered)
Intelligently interprets user input for goods receipt amounts:
- Empty/0 → Full receipt (GR = IR)
- Value between GR and IR → Absolute total
- Value ≤ (IR-GR) → Delta addition
- Confirmation of current GR → Full receipt

### Data Operations

- `get_invoice_lines(invoice_number)` - Retrieve invoice data
- `update_po_amount(invoice, line, amount)` - Update PO amount
- `book_gr_amount(invoice, line, amount)` - Update GR amount
- `set_received_flag(invoice, line, value)` - Set received status

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `MICROSOFT_APP_ID` | Teams app ID | Required |
| `MICROSOFT_APP_PASSWORD` | Teams app password | Required |
| `MICROSOFT_APP_TENANT_ID` | Azure AD tenant ID | Required |
| `AICORE_AUTH_URL` | SAP AI Core authentication URL | Optional |
| `AICORE_CLIENT_ID` | SAP AI Core client ID | Optional |
| `AICORE_CLIENT_SECRET` | SAP AI Core client secret | Optional |
| `AICORE_BASE_URL` | SAP AI Core API base URL | Optional |
| `AICORE_RESOURCE_GROUP` | SAP AI Core resource group | "default" |
| `USER_NAME` | Default user name | "Marta" |
| `EXCEL_SOURCE` | Excel file path | "dataset_enriched.xlsx" |
| `EXCEL_SHEET` | Sheet name | "POC scienarios" |
| `APP_PUBLIC_URL` | Public URL for webhooks | "http://localhost:3978" |

### Business Rules Configuration

Thresholds can be modified in the `classify_line()` function:
- Percentage threshold: 5% of PO amount
- Amount threshold: €250
- Floating point tolerance: 1e-6

### Excel Data Requirements

The Excel file must contain these columns (case-insensitive):
- Invoice_Number / Invoice Number
- Line (auto-generated if missing)
- PO# / PO Number / PO
- IR_amount / IR Amount
- PO_amount / PO Amount
- GR_amount / GR Amount
- Item_Name / ItemName
- User
- Received_Flag / received_flag

## Features

### Data Processing
- **In-memory operations** - No disk writes, all changes in RAM
- **Column normalization** - Handles various Excel column naming conventions
- **Data cleaning** - Removes apostrophes and quotes from string fields
- **Floating point handling** - Robust currency parsing and calculations

### User Interface
- **Adaptive Cards 1.4** - Modern, interactive UI components
- **Multi-scenario support** - Different card layouts per scenario
- **Responsive design** - Works across different screen sizes
- **Accessibility** - Screen reader compatible
- **Advanced Table System** - Hybrid ColumnSet approach with precise alignment

## Table Display System

The bot uses an advanced table display system that combines the best of both ColumnSet and monospace approaches:

### ColumnSet Tables (Current Implementation)

**Features:**
- **Precise column weights** - Proportional sizing for optimal layout
- **No-wrap columns** - Prevents text wrapping in critical columns
- **Smart item clipping** - Long item names are truncated with "…"
- **Decimal-only numbers** - No thousands separators (e.g., "1000.00" not "1,000.00")
- **Responsive alignment** - Left/Right/Center alignment per column

**Column Configuration:**

*User Orders Table:*
```python
headers = ["Invoice", "Item", "IR", "PO", "GR", "Rec"]
weights = [21, 29, 18, 18, 18, 10]  # Proportional weights
aligns  = ["Left", "Left", "Right", "Right", "Right", "Center"]
```

*Invoice Items Table:*
```python
headers = ["Line", "Item", "IR", "PO", "GR", "Rec"]
weights = [12, 32, 18, 18, 18, 10]  # Proportional weights
aligns  = ["Left", "Left", "Right", "Right", "Right", "Center"]
```

### Key Functions

#### _col(text, width, align, wrap, header)
Creates individual column with TextBlock:
- `maxLines: 1` for no-wrap columns
- `weight: "Bolder"` for headers
- `size: "Small"` for compact display

#### _header_cs(headers, weights, aligns)
Creates table header with separator line:
- Bold text for all headers
- Consistent alignment with data rows
- Visual separator from data

#### _row_cs(values, weights, aligns, nowrap_idx)
Creates data rows with strict no-wrap policy:
- All columns set to `wrap: false`
- Prevents mid-line text wrapping
- Maintains column alignment

#### _clip_item(name, max_chars)
Smart text truncation:
- Preserves important information
- Adds "…" for truncated text
- Configurable character limits

#### _num_str(value)
Number formatting without thousands separators:
- Returns "1000.00" instead of "1,000.00"
- Consistent decimal places
- Right-aligned in columns

### Alternative Monospace System (Available)

The codebase also includes a complete monospace table system for scenarios requiring absolute precision:

**Features:**
- **Fixed character widths** - Exact column sizing
- **NBSP padding** - Non-breaking space alignment
- **Dynamic width calculation** - Adapts to data content
- **Container-based rendering** - Each line as separate TextBlock

**Key Functions:**
- `_compute_widths_for_orders()` - Calculates optimal column widths
- `_row_orders()` - Formats order table rows
- `_mono_table_block()` - Renders monospace container

### Table Selection Strategy

**Use ColumnSet when:**
- Standard Teams display requirements
- Nee[ERROR] Failed to process stream: aborted
