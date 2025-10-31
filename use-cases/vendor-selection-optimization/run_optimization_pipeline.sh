#!/bin/bash
# Run the procurement optimization pipeline with country-based tariffs

# Get profile parameter (default to profile_1 if not provided)
PROFILE_ID="${1:-profile_1}"

# Define directories
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
GLOBAL_TABLES_DIR="$SCRIPT_DIR/tables"
GLOBAL_CONFIG_DIR="$SCRIPT_DIR/config"
OPTIMIZATION_DIR="$SCRIPT_DIR/optimization"
SCRIPTS_DIR="$SCRIPT_DIR/scripts"

# Profile-specific directories
PROFILE_DIR="$SCRIPT_DIR/profiles/$PROFILE_ID"
PROFILE_TABLES_DIR="$PROFILE_DIR/tables"
PROFILE_CONFIG_DIR="$PROFILE_DIR/config"

# Create profile directories if they don't exist
mkdir -p "$PROFILE_TABLES_DIR"
mkdir -p "$PROFILE_CONFIG_DIR"

# Colors for console output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

# Function to print with colors
print_info() {
    echo -e "${GREEN}INFO:${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}WARNING:${NC} $1"
}

print_error() {
    echo -e "${RED}ERROR:${NC} $1"
}

# Check if required directories exist
if [ ! -d "$GLOBAL_TABLES_DIR" ]; then
    print_error "Global tables directory not found: $GLOBAL_TABLES_DIR"
    exit 1
fi

if [ ! -d "$OPTIMIZATION_DIR" ]; then
    print_error "Optimization directory not found: $OPTIMIZATION_DIR"
    exit 1
fi

if [ ! -d "$SCRIPTS_DIR" ]; then
    print_error "Scripts directory not found: $SCRIPTS_DIR"
    exit 1
fi

if [ ! -d "$GLOBAL_CONFIG_DIR" ]; then
    print_error "Global config directory not found: $GLOBAL_CONFIG_DIR"
    exit 1
fi

# Step 0: Ensure we're in the script directory
cd "$SCRIPT_DIR"
print_info "Working directory: $(pwd)"
print_info "Processing profile: $PROFILE_ID"
print_info "Profile tables directory: $PROFILE_TABLES_DIR"
print_info "Profile config directory: $PROFILE_CONFIG_DIR"

# Step 1: Use existing tariff_values.json or create it if it doesn't exist (profile-specific)
TARIFF_JSON="$PROFILE_TABLES_DIR/tariff_values.json"
if [ ! -f "$TARIFF_JSON" ]; then
    # Check if there's a global fallback
    GLOBAL_TARIFF_JSON="$GLOBAL_TABLES_DIR/tariff_values.json"
    if [ -f "$GLOBAL_TARIFF_JSON" ]; then
        print_info "Copying global tariff values to profile..."
        cp "$GLOBAL_TARIFF_JSON" "$TARIFF_JSON"
    else
        print_info "Tariff values file not found, generating default values..."
        # Generate to global location first, then copy to profile
        python -m scripts.generate_default_tariffs
        if [ -f "$GLOBAL_TARIFF_JSON" ]; then
            cp "$GLOBAL_TARIFF_JSON" "$TARIFF_JSON"
        else
            print_error "Failed to generate tariff values JSON file."
            exit 1
        fi
    fi
else
    print_info "Using existing profile-specific tariff values from $TARIFF_JSON"
fi

# Print current tariff values for key countries for debugging
if command -v jq >/dev/null 2>&1; then
    print_info "Current tariff values for key countries:" 
    jq '{CN, US, DE, IN, ID, MX, TH, HU, MY}' "$TARIFF_JSON"
else
    print_info "Current tariff values (jq not available for formatting)"
    cat "$TARIFF_JSON" | grep -E '"CN":|"US":|"DE":|"IN":|"ID":|"MX":|"TH":|"HU":|"MY":'
fi

# Step 1b: Generate logistics factors if they don't exist (profile-specific)
LOGISTICS_JSON="$PROFILE_TABLES_DIR/logistics_factors.json"
if [ ! -f "$LOGISTICS_JSON" ]; then
    # Check if there's a global fallback
    GLOBAL_LOGISTICS_JSON="$GLOBAL_TABLES_DIR/logistics_factors.json"
    if [ -f "$GLOBAL_LOGISTICS_JSON" ]; then
        print_info "Copying global logistics factors to profile..."
        cp "$GLOBAL_LOGISTICS_JSON" "$LOGISTICS_JSON"
    else
        print_info "Logistics factors file not found, generating values..."
        # Generate to global location first, then copy to profile
        python -m scripts.generate_logistics_factors
        if [ -f "$GLOBAL_LOGISTICS_JSON" ]; then
            cp "$GLOBAL_LOGISTICS_JSON" "$LOGISTICS_JSON"
        else
            print_error "Failed to generate logistics factors JSON file."
            exit 1
        fi
    fi
else
    print_info "Using existing profile-specific logistics factors from $LOGISTICS_JSON"
fi

# Step 2: Generate a temporary tariff CSV (placeholder for the tariff model output)
print_info "Generating temporary tariff CSV..."
TARIFF_CSV="$PROFILE_TABLES_DIR/tariff_values.csv"
echo "EBELN,EBELP,Cumulative_Tariff_Percent" > "$TARIFF_CSV"
echo "DummyPO,DummyItem,0.0" >> "$TARIFF_CSV"

# Step a: Ensure table_map.json exists in profile config directory
TABLE_MAP="$PROFILE_CONFIG_DIR/table_map.json"
if [ ! -f "$TABLE_MAP" ]; then
    print_info "Creating table_map.json..."
    cat > "$TABLE_MAP" << EOF
{
  "SAP_VLY_IL_PO_ITEMS.csv": "SAP_VLY_IL_PO_ITEMS.csv",
  "SAP_VLY_IL_PO_HEADER.csv": "SAP_VLY_IL_PO_HEADER.csv",
  "SAP_VLY_IL_PO_HISTORY.csv": "SAP_VLY_IL_PO_HISTORY.csv",
  "SAP_VLY_IL_PO_SCHEDULE_LINES.csv": "SAP_VLY_IL_PO_SCHEDULE_LINES.csv",
  "SAP_VLY_IL_SUPPLIER.csv": "SAP_VLY_IL_SUPPLIER.csv",
  "SAP_VLY_IL_MATERIAL_GROUP.csv": "SAP_VLY_IL_MATERIAL_GROUP.csv",
  "SAP_VLY_IL_MATERIAL.csv": "SAP_VLY_IL_MATERIAL.csv",
  "SAP_VLY_IL_COUNTRY.csv": "SAP_VLY_IL_COUNTRY.csv"
}
EOF
fi

# Step b: Ensure column_map.json exists in profile config directory
COLUMN_MAP="$PROFILE_CONFIG_DIR/column_map.json"
if [ ! -f "$COLUMN_MAP" ]; then
    print_info "Creating column_map.json..."
    cat > "$COLUMN_MAP" << EOF
{
  "PO_Number": "EBELN",
  "PO_Item_Number": "EBELP",
  "Material_ID": "MATNR",
  "Material_Group_Code": "MATKL",
  "Item_Ordered_Quantity": "MENGE",
  "Net_Price_Per_Unit": "NETPR",
  "Price_Unit_Quantity": "PEINH",
  "Item_Net_Order_Value": "NETWR",
  "Supplier_ID": "LIFNR",
  "PO_Creation_Date": "BEDAT",
  "PO_Currency_Code": "WAERS",
  "Order_Unit": "MEINS",
  "PO_History_Transaction_Type": "VGABE",
  "History_Posting_Date": "BUDAT",
  "History_Quantity_Base_UoM": "MENGE",
  "Plant_Code": "WERKS",
  "Schedule_Line_Number": "ETENR",
  "Scheduled_Delivery_Date": "EINDT",
  "Supplier_Name_1": "NAME1",
  "Material_Group_Description": "MATKL_DESC",
  "Material_Description": "MAKTX",
  "Supplier_Country": "LAND1"
}
EOF
fi

# Step c: Ensure metrics.json exists in profile config directory
METRICS_JSON="$PROFILE_CONFIG_DIR/metrics.json"
if [ ! -f "$METRICS_JSON" ]; then
    print_info "Creating metrics.json..."
    cat > "$METRICS_JSON" << EOF
{
  "AvgUnitPriceUSD_Norm": 0.20,
  "PriceVolatility_Norm": 0.15,
  "PriceTrend_Norm": 0.10,
  "TariffImpact_Norm": 0.15,
  "AvgLeadTimeDays_Norm": 0.10,
  "LeadTimeVariability_Norm": 0.10,
  "OnTimeRate_Norm": 0.10,
  "InFullRate_Norm": 0.10
}
EOF
fi

# Step d: Ensure costs.json exists in profile config directory
COSTS_JSON="$PROFILE_CONFIG_DIR/costs.json"
if [ ! -f "$COSTS_JSON" ]; then
    print_info "Creating costs.json..."
    cat > "$COSTS_JSON" << EOF
{
  "cost_components": {
    "cost_BasePrice": "True",
    "cost_Tariff": "True",
    "cost_Holding_LeadTime": "True",
    "cost_Holding_LTVariability": "True",
    "cost_Holding_Lateness": "True",
    "cost_Inefficiency_InFull": "False",
    "cost_Risk_PriceVolatility": "True",
    "cost_Impact_PriceTrend": "True"
  },
  "economic_impact_parameters": {
    "EIP_ANNUAL_HOLDING_COST_RATE": 0.18,
    "EIP_SafetyStockMultiplierForLTVar_Param": 1.65,
    "EIP_RiskPremiumFactorForPriceVolatility_Param": 0.25,
    "EIP_PlanningHorizonDaysForPriceTrend_Param": 90,
    "PRICE_TREND_CUTOFF_DAYS": 180
  }
}
EOF
fi

# Step 3: Run evaluate_vendor_material_with_country_tariffs.py
print_info "Running vendor evaluation with country-based tariffs..."
python -m optimization.evaluate_vendor_material_with_country_tariffs \
    --tariff-results-path "$TARIFF_CSV" \
    --tables-dir "$GLOBAL_TABLES_DIR" \
    --ranking-output-dir "$PROFILE_TABLES_DIR" \
    --table-map "$TABLE_MAP" \
    --column-map "$COLUMN_MAP" \
    --mode matnr \
    --metric-weights "$METRICS_JSON" \
    --costs-config-path "$COSTS_JSON" \
    --country-tariffs-path "$TARIFF_JSON"

# Check if the output file was generated
EVAL_OUTPUT="$PROFILE_TABLES_DIR/vendor_matnr_ranking_tariff_values.csv"
if [ ! -f "$EVAL_OUTPUT" ]; then
    print_error "Vendor evaluation failed to generate output file."
    exit 1
fi
print_info "Vendor evaluation completed successfully."

# Step 4: Run fix_country_mapping.py
print_info "Adding country data to vendor ranking file..."
python -m scripts.fix_country_mapping --profile-id "$PROFILE_ID"

# Step 5: Run optimize_procurement.py
print_info "Running procurement optimization..."
python -m optimization.optimize_procurement \
    --ranking-results-path "$EVAL_OUTPUT" \
    --tables-dir "$GLOBAL_TABLES_DIR" \
    --optimization-output-path "$PROFILE_TABLES_DIR/optimized_allocation_matnr_vendor_matnr_ranking_tariff_values.csv" \
    --table-map "$TABLE_MAP" \
    --column-map "$COLUMN_MAP" \
    --mode matnr

# Check if the output file was generated
OPT_OUTPUT="$PROFILE_TABLES_DIR/optimized_allocation_matnr_vendor_matnr_ranking_tariff_values.csv"
if [ ! -f "$OPT_OUTPUT" ]; then
    print_error "Optimization failed to generate output file."
    exit 1
fi
print_info "Optimization completed successfully."

# Step 6: Run compare_policies.py
print_info "Comparing procurement policies..."
python -m optimization.compare_policies \
    --ranking-results-path "$EVAL_OUTPUT" \
    --optimization-results-path "$OPT_OUTPUT" \
    --tables-dir "$GLOBAL_TABLES_DIR" \
    --comparison-output-path "$PROFILE_TABLES_DIR/comparison.csv" \
    --table-map "$TABLE_MAP" \
    --column-map "$COLUMN_MAP" \
    --mode matnr \
    --costs-config-path "$COSTS_JSON"

# Check if the output file was generated
COMP_OUTPUT="$PROFILE_TABLES_DIR/comparison.csv"
if [ ! -f "$COMP_OUTPUT" ]; then
    print_error "Policy comparison failed to generate output file."
    exit 1
fi
print_info "Policy comparison completed successfully."

# Final message
print_info "✅ Optimization pipeline with country-based tariffs completed successfully!"
print_info "You can now run the dashboard to view the results:"
print_info "   cd $SCRIPT_DIR && streamlit run Home.py"