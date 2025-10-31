# Procurement Analytics Dashboard

An interactive vendor performance and procurement analysis tool that helps identify optimal suppliers based on cost, lead time, and reliability metrics.

## Recent Updates

### Core Configuration Features
- **Economic Impact Parameters (EIP) Configuration**: New UI in Optimization Settings tab to configure EIP values that control cost calculations
- **Cost Component Management**: Added checkboxes to enable/disable individual cost components, including new logistics cost component
- **Enhanced Configuration Structure**: Migrated from flat to nested JSON structure in costs.json for better organization
- **Improved Data Pipeline**: Fixed file regeneration order to ensure parameter changes propagate correctly across all dashboard tabs
- **Demand Window Configuration**: Added configurable demand period setting (DEMAND_PERIOD_DAYS) with UI display

### New Cost Components
- **Logistics Cost Integration**: Added comprehensive logistics cost calculation based on country distance factors and material characteristics
- **Country-Distance Mapping**: Implemented distance-based cost factors (0.1 for domestic US to 1.0 for distant countries)
- **Material-Specific Logistics**: Different logistics multipliers for various material types (0.3 for small electronics to 0.9 for heavy machinery)

### Visualization & UI Improvements
- **Enhanced Graph Labels**: Improved vendor identification in charts with country codes (e.g., "WaveCrest Labs-24 (US)")
- **Top Materials Ranking**: Added "Top 5 materials with highest opportunity" feature showing materials with greatest cost reduction potential
- **Sankey Diagram Enhancements**: Fixed typography issues and improved flow visualization
- **Better Chart Positioning**: Optimized colorbar placement in dual scatter plots
- **Resolved UI Issues**: Fixed nested expander problems and improved cache management for better performance

## Overview

This procurement analytics dashboard provides comprehensive analysis of supplier performance across multiple dimensions:

- **Vendor Performance**: Compare vendors by cost, lead time, on-time rate, and in-full rate
- **Cost Analysis**: Detailed breakdown of cost components (base price, tariffs, holding costs, etc.)
- **Geographic Distribution**: Visualize supplier performance metrics by country
- **AI Recommendations**: Natural language query interface for material procurement
- **Optimization Insights**: AI-generated insights and recommendations based on optimization results
- **Configurable Parameters**: Adjust Economic Impact Parameters (EIPs) and cost components through UI

## Project Structure

The project is organized into the following directories:

```
/resources
├── README.md                     # Project documentation
├── TARIFF_CONFIGURATION.md       # Tariff configuration docs
├── Home.py                       # Multipage application entry point
├── requirements.txt              # Project dependencies
├── run_optimization_pipeline.sh  # Pipeline script
├── config/                       # Configuration files and settings
│   ├── settings.py               # App settings, paths, and configurations
│   ├── column_map.json           # SAP column name mappings
│   ├── table_map.json            # Table name mappings
│   ├── costs.json                # Cost component configuration
│   └── metrics.json              # Metric weights for vendor scoring
├── core/                         # Essential data handling and utilities
│   ├── data_loader.py            # Data loading functionality
│   ├── utils.py                  # Utility functions
│   └── models/                   # Data model definitions
├── optimization/                 # Procurement optimization logic
│   ├── optimize_procurement.py   # Optimization algorithm
│   ├── evaluate_vendor_material.py # Vendor evaluation
│   ├── evaluate_vendor_material_with_country_tariffs.py # Enhanced vendor evaluation with tariffs
│   ├── compare_policies.py       # Policy comparison
│   ├── tariff_configuration.py   # Tariff configuration UI
│   ├── optimization_settings.py  # EIP and cost component configuration UI
│   └── optimized_vendor_selection.py # Optimized selection UI
├── ui/                           # Streamlit UI components
│   ├── components.py             # UI components and dashboard tabs
│   ├── visualization.py          # Data visualization functions
│   ├── vendor_analysis.py        # Vendor analysis functionality
│   └── pages/                    # Individual dashboard pages
├── ai/                           # AI-related functionality
│   ├── genai_query_parser.py     # AI query parsing
│   └── insights_generator.py     # AI insights generation
├── scripts/                      # Utility scripts
│   ├── fix_country_mapping.py    # Data preprocessing script
│   ├── refresh_data.py           # Data refresh script
│   ├── generate_default_tariffs.py # Tariff generation script
│   └── get_country_tariff_stats.py # Country tariff statistics
└── tables/                       # Data tables
```

## Features

### 📊 Interactive Dashboard
- Multiple visualization tabs for different analysis perspectives
- Filter by material, vendor, country, lead time, and PO count
- Comparative metrics with industry benchmarks

### 📈 Performance Analysis
- Scatter plot analysis of lead time vs. OTIF performance
- Performance heatmap for normalized vendor comparisons
- Identification of top-performing vendors

### 🌎 Geographic Analysis
- Choropleth maps showing supplier performance by country
- Color-coded visualization of lead times, OTIF rates, and tariff impacts
- Country performance summary tables

### 💰 Cost Breakdown
- Detailed visualization of all cost components including logistics costs
- Configurable cost factors via costs.json with country-distance and material-type factors
- Comparative cost analysis across vendors with geographic impact visualization
- Logistics cost modeling based on supplier country distance and material characteristics

### 🤖 AI-Powered Procurement Assistant
- Natural language query processing for material procurement
- Automatic extraction of material and quantity requirements
- AI-generated vendor recommendations
- Uses SAP GenAI Hub (via OpenAI proxy) for intelligent query parsing

### 📊 Optimization Pipeline
- Optimized vendor allocation based on configurable cost factors
- Country-based tariff configuration for scenario planning
- Historical vs. optimized performance comparison
- AI-generated optimization insights and recommendations

## Getting Started

### Prerequisites

- Python 3.8 or higher
- Required Python packages: streamlit, pandas, plotly, matplotlib, generative-ai-hub-sdk, python-dotenv, pulp, scipy

### Installation

1. Clone the repository:
```
git clone [repository-url]
cd procurement_unified/test
```

2. Install dependencies:
```
pip install -r requirements.txt
```

3. Run the dashboard:
```
streamlit run Home.py
```

4. Optional utilities:
```
# Fix country mapping issues
python -m scripts.fix_country_mapping

# Force dashboard data refresh
python -m scripts.refresh_data --fix-mapping

# Run optimization pipeline with current tariff settings
./run_optimization_pipeline.sh
```

### Data Setup

The dashboard requires SAP data tables in the following format:
- `tables/SAP_VLY_IL_MATERIAL.csv`: Material master data
- `tables/SAP_VLY_IL_SUPPLIER.csv`: Supplier information with country codes
- `tables/SAP_VLY_IL_COUNTRY.csv`: Country code mappings
- `tables/SAP_VLY_IL_PO_HEADER.csv`: Purchase order headers
- `tables/SAP_VLY_IL_PO_ITEMS.csv`: Purchase order line items
- `tables/SAP_VLY_IL_DELIVERY_TOLERANCES.csv`: Delivery tolerance settings
- `tables/SAP_VLY_IL_LEADTIME_INDEX.csv`: Lead time index data
- `tables/SAP_VLY_IL_VPI_TARGET_VALUES.csv`: Vendor performance index target values

The application also uses these derived data files:
- `tables/vendor_with_direct_countries.csv`: Pre-processed vendor data with country information
- `tables/vendor_maktx_ranking_tariff_values.csv`: Vendor ranking with tariff and logistics impact
- `tables/tariff_values.json`: Country-specific tariff settings
- `tables/optimized_allocation_maktx_vendor_maktx_ranking_tariff_values.csv`: Optimization results
- `tables/comparison.csv`: Comparison between historical and optimized procurement

### Country Data Fix

If you're experiencing issues with country data:

1. Run the country mapping fix script:
```
python -m scripts.fix_country_mapping
```

2. Confirm that the `tables/vendor_with_direct_countries.csv` file was created

3. If you need to force a data refresh after fixing country data:
```
python -m scripts.refresh_data --fix-mapping
```

## Configuration

### Cost Components and Economic Impact Parameters

Cost components and Economic Impact Parameters (EIPs) can be configured in the `config/costs.json` file or through the Optimization Settings tab:

```json
{
    "cost_components": {
        "cost_BasePrice": "True",
        "cost_Tariff": "True",
        "cost_Holding_LeadTime": "True",
        "cost_Holding_LTVariability": "True",
        "cost_Holding_Lateness": "True",
        "cost_Inefficiency_InFull": "False",
        "cost_Risk_PriceVolatility": "True",
        "cost_Impact_PriceTrend": "True",
        "cost_Logistics": "True"
    },
    "economic_impact_parameters": {
        "EIP_ANNUAL_HOLDING_COST_RATE": 0.18,
        "EIP_SafetyStockMultiplierForLTVar_Param": 1.65,
        "EIP_RiskPremiumFactorForPriceVolatility_Param": 0.25,
        "EIP_PlanningHorizonDaysForPriceTrend_Param": 90,
        "PRICE_TREND_CUTOFF_DAYS": 180,
        "EIP_BaseLogisticsCostRate_Param": 0.15,
        "DEMAND_PERIOD_DAYS": 365
    }
}
```

#### Configuring via UI
1. Navigate to the "Optimization Settings" tab
2. Adjust EIP values using the input fields
3. Enable/disable cost components using checkboxes
4. Click "Update Configuration & Run Optimization" to apply changes

### Tariff Configuration

Country-specific tariffs can be configured in the Tariff Configuration tab. See [TARIFF_CONFIGURATION](resources/TARIFF_CONFIGURATION.md) for detailed documentation.

When updating tariff values, follow this process to ensure changes are properly reflected across all dashboard tabs:

1. Navigate to the "Tariff Configuration" tab
2. Select a country and update its tariff value
3. Click "Update Tariffs & Run Optimization" to rerun the optimization pipeline
4. This will:
   - Update the tariff values in tariff_values.json
   - Run the optimization pipeline
   - Run fix_country_mapping.py to update country mappings
   - Refresh data across dashboard components

### Troubleshooting Data Refresh Issues

If changes are not reflected in all dashboard tabs, you can force a data refresh using:

1. The "Refresh Dashboard" button in the Tariff Configuration tab
2. The "Force Reload All Data" button in the sidebar
3. Running the manual refresh utility: `python -m scripts.refresh_data --fix-mapping`

## Data Model

The dashboard uses the following data relationships:

- Each vendor (`LIFNR`) in `SAP_VLY_IL_SUPPLIER.csv` comes from a specific country (`LAND1`)
- Country codes (`LAND1`) map to country names (`LANDX`) in `SAP_VLY_IL_COUNTRY.csv`
- Materials (`MATNR`) in `SAP_VLY_IL_MATERIAL.csv` have descriptions (`MAKTX`)
- PO items in `SAP_VLY_IL_PO_ITEMS.csv` connect materials to PO headers
- PO headers in `SAP_VLY_IL_PO_HEADER.csv` link to vendors

## Using the Dashboard

### Dashboard Tab Structure
The dashboard includes the following tabs in this order:
1. **Vendor Selection Assistant** - AI-powered natural language query interface
2. **Optimized Vendor Comparison** - Side-by-side comparison of historical vs. optimized vendor allocations
3. **Scatter Analysis** - Lead time vs. OTIF performance visualization
4. **Performance Heatmap** - Normalized vendor performance metrics
5. **Geographic View** - Country-based performance visualization
6. **Data Table** - Detailed vendor data with cost components
7. **Tariff Configuration** - Country-specific tariff settings
8. **Optimization Settings** - Configure Economic Impact Parameters (EIPs) and cost components

### Visualization Improvements
The dashboard includes these visualization improvements:
- **Enhanced Vendor Identification**: Vendor IDs now include country codes (e.g., "WaveCrest Labs-24 (US)") in barplots and tables for better context
- **Improved Scatter Plot Hover Information**: Hover text now includes LIFNR code and country information for better data context
- **Optimized Colorbar Positioning**: The colorbar in dual scatter plots is now positioned between plots to prevent overlap
- **Top Materials Opportunity Ranking**: New feature displaying the top 5 materials with highest cost reduction potential
- **Enhanced Sankey Diagrams**: Improved typography and flow visualization for better readability
- **Logistics Cost Visualization**: Integrated logistics cost breakdown in cost component charts

### Material Selection
1. Use the sidebar filters to select materials, vendors, or countries
2. Or use the AI-powered assistant to search with natural language in the Vendor Selection Assistant tab

### Analyzing Vendor Performance
1. Navigate between the different tabs based on your analysis needs
2. View the best vendors for price, delivery time, on-time rate, and in-full rate
3. Use the Optimized Vendor Comparison tab to see AI-recommended vendor allocations
4. Examine detailed cost breakdowns of each vendor in the Data Table tab
5. Configure country-specific tariffs in the Tariff Configuration tab
6. Adjust Economic Impact Parameters in the Optimization Settings tab
7. Export data for further analysis

### AI-Powered Features

The dashboard includes several AI-powered features using the SAP GenAI Hub:

1. **Natural Language Query Parsing**: Automatically extracts material names and quantities from natural language queries
2. **Vendor Recommendation Generation**: Provides intelligent vendor suggestions based on extracted requirements
3. **Optimization Insights**: Automatically generates data-driven insights and recommendations based on optimization results
4. **Actionable To-Do Lists**: Creates prioritized task lists for implementing optimization recommendations

## Backup and Version Control

The system automatically creates backups of optimization results in the `tables/backup/` directory with timestamped folders containing:
- `comparison.csv`: Comparison data between historical and optimized allocations
- `optimized_allocation_maktx_vendor_maktx_ranking_tariff_values.csv`: Detailed optimization results
- `vendor_maktx_ranking_tariff_values.csv`: Vendor ranking data with tariff values

This allows for tracking changes over time and reverting to previous optimization states if needed.

## Troubleshooting

### EIP Changes Not Reflecting
If Economic Impact Parameter changes are not showing in all tabs:
1. Ensure you clicked "Update Configuration & Run Optimization" after making changes
2. Check that the optimization pipeline completed successfully
3. Use the "Force Reload All Data" button in the sidebar
4. Verify that vendor_with_direct_countries.csv was regenerated (check file timestamp)

### Configuration File Issues
If you encounter errors loading configuration:
1. Check that costs.json has valid JSON syntax
2. The system maintains backward compatibility with the old flat structure
3. If needed, manually update costs.json to the new nested structure format

### Optimization Pipeline Failures
If the optimization pipeline fails:
1. Check the console output for specific error messages
2. Ensure all required data files are present in the tables/ directory
3. Verify that Python dependencies are installed (especially PuLP for optimization)
4. Try running individual pipeline steps manually to isolate the issue
