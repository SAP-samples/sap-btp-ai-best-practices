"""
Optimization Settings functionality for the AI Supplier Sourcing Optimizer.
Displays and explains the metrics, parameters, and calculations used in optimization.
"""

import streamlit as st
import json
import os
import subprocess
from datetime import datetime
from config import settings
from optimization.profile_manager import ProfileManager


def load_eip_parameters(profile_id='profile_1'):
    """Load current EIP parameters from costs.json and optimization constants"""
    try:
        profile_manager = ProfileManager(".")
        costs_file = profile_manager.get_data_file_path(profile_id, 'costs.json')
        
        # If profile-specific file doesn't exist, fall back to global settings
        if not os.path.exists(costs_file):
            costs_file = settings.COSTS_CONFIG_FILE
            
        with open(costs_file, 'r') as f:
            config_data = json.load(f)
        
        # Handle both old and new format
        if 'economic_impact_parameters' in config_data:
            eip_params = config_data['economic_impact_parameters']
        else:
            # Return default values if using old format
            eip_params = {
                'EIP_ANNUAL_HOLDING_COST_RATE': 0.18,
                'EIP_SafetyStockMultiplierForLTVar_Param': 1.65,
                'EIP_RiskPremiumFactorForPriceVolatility_Param': 0.25,
                'EIP_PlanningHorizonDaysForPriceTrend_Param': 90,
                'PRICE_TREND_CUTOFF_DAYS': 180,
                'EIP_BaseLogisticsCostRate_Param': 0.15
            }
        
        # Add DEMAND_PERIOD_DAYS from settings with fallback
        try:
            demand_period_days = getattr(settings, 'DEMAND_PERIOD_DAYS', 365)
        except:
            demand_period_days = 365
        eip_params['DEMAND_PERIOD_DAYS'] = demand_period_days
        
        return eip_params
        
    except Exception as e:
        st.error(f"Error loading EIP parameters: {e}")
        # Return default values with demand period fallback
        try:
            demand_period_days = getattr(settings, 'DEMAND_PERIOD_DAYS', 365)
        except:
            demand_period_days = 365
            
        return {
            'EIP_ANNUAL_HOLDING_COST_RATE': 0.18,
            'EIP_SafetyStockMultiplierForLTVar_Param': 1.65,
            'EIP_RiskPremiumFactorForPriceVolatility_Param': 0.25,
            'EIP_PlanningHorizonDaysForPriceTrend_Param': 90,
            'PRICE_TREND_CUTOFF_DAYS': 180,
            'EIP_BaseLogisticsCostRate_Param': 0.15,
            'DEMAND_PERIOD_DAYS': demand_period_days
        }


def save_eip_parameters(new_eips, profile_id='profile_1'):
    """Save EIP parameters to costs.json"""
    try:
        profile_manager = ProfileManager(".")
        costs_file = profile_manager.get_data_file_path(profile_id, 'costs.json')
        
        # If profile-specific file doesn't exist, create it from global settings
        if not os.path.exists(costs_file):
            with open(settings.COSTS_CONFIG_FILE, 'r') as f:
                config_data = json.load(f)
        else:
            with open(costs_file, 'r') as f:
                config_data = json.load(f)
        
        # Handle both old and new format
        if 'cost_components' not in config_data:
            # Convert old format to new format
            old_costs = config_data.copy()
            config_data = {
                'cost_components': old_costs,
                'economic_impact_parameters': new_eips
            }
        else:
            # Update EIP parameters in new format
            config_data['economic_impact_parameters'] = new_eips
        
        # Ensure profile directory exists
        os.makedirs(os.path.dirname(costs_file), exist_ok=True)
        
        # Save updated config to profile-specific file
        with open(costs_file, 'w') as f:
            json.dump(config_data, f, indent=4)
        
        return True
    except Exception as e:
        st.error(f"Error saving EIP parameters: {e}")
        return False


def save_cost_components(new_costs, profile_id='profile_1'):
    """Save cost component configuration to costs.json"""
    try:
        profile_manager = ProfileManager(".")
        costs_file = profile_manager.get_data_file_path(profile_id, 'costs.json')
        
        # If profile-specific file doesn't exist, create it from global settings
        if not os.path.exists(costs_file):
            with open(settings.COSTS_CONFIG_FILE, 'r') as f:
                config_data = json.load(f)
        else:
            with open(costs_file, 'r') as f:
                config_data = json.load(f)
        
        # Handle both old and new format
        if 'cost_components' not in config_data:
            # Convert old format to new format
            config_data = {
                'cost_components': new_costs,
                'economic_impact_parameters': {
                    'EIP_ANNUAL_HOLDING_COST_RATE': 0.18,
                    'EIP_SafetyStockMultiplierForLTVar_Param': 1.65,
                    'EIP_RiskPremiumFactorForPriceVolatility_Param': 0.25,
                    'EIP_PlanningHorizonDaysForPriceTrend_Param': 90,
                    'PRICE_TREND_CUTOFF_DAYS': 180,
                    'EIP_BaseLogisticsCostRate_Param': 0.15
                }
            }
        else:
            # Update cost components in new format
            config_data['cost_components'] = new_costs
        
        # Ensure profile directory exists
        os.makedirs(os.path.dirname(costs_file), exist_ok=True)
        
        # Save updated config to profile-specific file
        with open(costs_file, 'w') as f:
            json.dump(config_data, f, indent=4)
        
        return True
    except Exception as e:
        st.error(f"Error saving cost components: {e}")
        return False


def run_optimization_with_eips(profile_id='profile_1'):
    """Run the optimization pipeline with updated EIP parameters"""
    try:
        # Get the path to the optimization script
        script_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
            'run_optimization_pipeline.sh'
        )
        
        if not os.path.exists(script_path):
            return False, f"Optimization script not found at {script_path}"
        
        # Make script executable
        os.chmod(script_path, 0o755)
        
        # Run the optimization pipeline with profile parameter
        process = subprocess.run(
            [script_path, profile_id], 
            capture_output=True, 
            text=True,
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
        
        if process.returncode != 0:
            return False, f"Optimization failed: {process.stderr}"
        
        # Check for country mapping
        mapping_file = os.path.join('tables', 'vendor_with_direct_countries.csv')
        if not os.path.exists(mapping_file):
            # Run fix_country_mapping.py
            fix_script = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                'scripts', 
                'fix_country_mapping.py'
            )
            if os.path.exists(fix_script):
                subprocess.run(["python", fix_script], capture_output=True, text=True)
        
        return True, "Optimization completed successfully with updated EIP parameters!"
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return False, f"Error running optimization: {str(e)}"


def render_optimization_settings_tab(tab, profile_id='profile_1'):
    """Render the optimization settings tab with detailed explanations"""
    with tab:
        st.subheader("Optimization Settings and Methodology")
        
        # Display current profile information
        st.info(f"Currently configuring: **{profile_id}**")
        
        st.markdown("""
        This section provides a detailed overview of the methodology used by the AI Supplier Sourcing Optimizer.
        It covers the calculation of raw performance metrics, the economic impact parameters that translate these metrics into costs,
        the formulation of individual cost components, and the structure of the optimization model.
        The system employs a comprehensive total cost of ownership (TCO) approach, moving beyond simple unit price
        to incorporate a variety of factors influencing procurement effectiveness.
        """)
        
        # Define all possible cost components and their details
        # This structure will hold all information needed for rendering
        all_cost_component_details = {
            "cost_BasePrice": {
                "display_name": "Base Price Cost",
                "latex": r"\text{cost\_BasePrice} = \text{AvgUnitPriceUSD\_raw}",
                "description": "The fundamental median unit price in USD, representing the direct acquisition cost before other factors."
            },
            "cost_Tariff": {
                "display_name": "Tariff Cost",
                "latex": r"\text{cost\_Tariff} = \text{AvgUnitPriceUSD\_raw} \times \left( \frac{\text{TariffImpact\_raw\_percent}}{100} \right)",
                "description": "The additional cost per unit incurred due to import duties, based on the supplier's country of origin and the item's base price."
            },
            "cost_Holding_LeadTime": {
                "display_name": "Lead Time Holding Cost",
                "latex": r"""
                \text{cost\_Holding\_LeadTime} = \text{AvgLeadTimeDays\_raw} \\ \times \text{EIP\_DailyHoldingCostRate} \\ \times \text{AvgUnitPriceUSD\_raw}
                """,
                "description": "The cost of capital tied up in inventory during the average lead time. It reflects the financial burden of inventory in transit and during initial processing/receipt."
            },
            "cost_Holding_LTVariability": {
                "display_name": "Lead Time Variability Holding Cost (Safety Stock Cost)",
                "latex": r"""
                \text{cost\_Holding\_LTVariability} = \text{LeadTimeVariability\_raw\_days} \\ \times \text{EIP\_SafetyStockMultiplierForLTVar} \\ \times \text{EIP\_DailyHoldingCostRate} \times \text{AvgUnitPriceUSD\_raw}
                """,
                "description": "The cost of holding additional safety stock required to buffer against inconsistencies in supplier lead times. This aims to maintain a target service level despite delivery uncertainties."
            },
            "cost_Holding_Lateness": {
                "display_name": "Lateness Holding Cost (Late Delivery Impact)",
                "latex": r"""
                \text{cost\_Holding\_Lateness} = (1 - \text{OnTimeRate\_raw}) \\ \times \text{EIP\_AvgDaysLateIfLate\_material} \\ \times \text{EIP\_DailyHoldingCostRate} \times \text{AvgUnitPriceUSD\_raw}
                """,
                "description": "The expected inventory holding cost per unit due to late deliveries. This is calculated by considering the probability of a delivery being late (`1 - OnTimeRate_raw`), the material-specific average extent of such delays, and the daily cost of holding the delayed item."
            },
            "cost_Risk_PriceVolatility": {
                "display_name": "Price Volatility Risk Cost",
                "latex": r"\text{cost\_Risk\_PriceVolatility} = \text{StdDevUnitPriceUSD\_raw} \times \text{EIP\_RiskPremiumFactorForPriceVolatility}",
                "description": "A risk premium added to the unit cost to account for historical price instability. This quantifies the financial buffer needed to mitigate risks from unpredictable price fluctuations and potential budget overruns."
            },
            "cost_Impact_PriceTrend": {
                "display_name": "Price Trend Impact Cost",
                "latex": r"\text{cost\_Impact\_PriceTrend} = \text{PriceTrend\_raw\_slope} \times \left( \frac{\text{EIP\_PlanningHorizonDaysForPriceTrend}}{2} \right)",
                "description": "The anticipated average change in unit cost over the defined planning horizon, based on recent historical price trends. A positive trend (increasing prices) adds to the cost, while a negative trend (decreasing prices) reduces it."
            },
            "cost_Logistics": {
                "display_name": "Logistics Cost",
                "latex": r"\text{cost\_Logistics} = \text{AvgUnitPriceUSD\_raw} \times \text{EIP\_BaseLogisticsCostRate} \times \text{LogisticsFactor}_{country,material}",
                "description": "The estimated shipping and logistics cost based on the supplier's country of origin and material characteristics. The LogisticsFactor combines distance from the US (0.1 for domestic to 1.0 for distant countries) and material-specific factors (0.3 for small electronics to 0.9 for heavy machinery), with some random variation to simulate real-world complexity."
            }
        }
        
        # Maintain an ordered list of component keys for consistent display order
        ordered_cost_component_keys = list(all_cost_component_details.keys())
        num_total_possible_components = len(ordered_cost_component_keys)

        # Load current cost configuration from profile
        costs_config = {} # Default to empty if file not found or error
        try:
            profile_manager = ProfileManager(".")
            costs_file = profile_manager.get_data_file_path(profile_id, 'costs.json')
            
            # If profile-specific file doesn't exist, fall back to global settings
            if not os.path.exists(costs_file):
                costs_file = settings.COSTS_CONFIG_FILE
            
            with open(costs_file, 'r') as f:
                config_data = json.load(f)
                # Handle both old and new format
                if 'cost_components' in config_data:
                    costs_config = config_data['cost_components']
                else:
                    costs_config = config_data
        except FileNotFoundError:
            st.error(f"Cost configuration file not found. Assuming all components are disabled for display.")
            # If file not found, treat all as "False" for display purposes.
            # The actual script will default to True if file is missing, but for settings page it's better to show what's configurable.
            costs_config = {key: "False" for key in ordered_cost_component_keys}
        except json.JSONDecodeError:
            st.error(f"Error decoding JSON from cost configuration file. Assuming all components are disabled for display.")
            costs_config = {key: "False" for key in ordered_cost_component_keys}
        except Exception as e:
            st.error(f"An unexpected error occurred while loading cost configuration: {e}. Assuming all components are disabled for display.")
            costs_config = {key: "False" for key in ordered_cost_component_keys}


        # Create expandable sections for each category
        with st.expander("Raw Metrics Calculations", expanded=True):
            st.markdown("### Raw Metrics Calculations")
            st.markdown("These fundamental measurements are calculated from historical purchase order data for each relevant supplier-material combination. They form the basis for cost calculations and performance assessment.")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("""
                **AvgUnitPriceUSD_raw**
                - **Description**: The typical historical unit price in USD for a material from a specific supplier.
                - **Calculation**: Calculated as the *median* of `UnitPriceUSD` values from past purchase orders. `UnitPriceUSD` is derived from `Net_Price_Per_Unit` (`NETPR`) divided by `Price_Unit_Quantity` (`PEINH`), then converted to USD using prevailing exchange rates.
                - **Purpose**: Represents the central tendency of the price paid per unit, reducing sensitivity to outliers.
                
                **StdDevUnitPriceUSD_raw**
                - **Description**: The absolute variability of historical USD unit prices.
                - **Calculation**: Standard deviation of the `UnitPriceUSD` values.
                - **Purpose**: Measures the dispersion of prices around the average; a higher value indicates greater price fluctuation.
                
                **PriceVolatility_raw**
                - **Description**: The relative variability of historical USD unit prices, normalized by the mean.
                - **Calculation**: Coefficient of Variation, calculated as `StdDevUnitPriceUSD_raw / MeanUnitPriceUSD_raw`.
                - **Purpose**: Provides a standardized measure of price uncertainty, allowing comparison across items with different price levels.
                
                **PriceTrend_raw_slope**
                - **Description**: The rate of change in USD unit prices over a recent period.
                - **Calculation**: The slope of a linear regression line fitted to `UnitPriceUSD` values against `PO_Creation_Date` (converted to days) for purchase orders within the last 180 days.
                - **Purpose**: Identifies whether prices are, on average, increasing (positive slope) or decreasing (negative slope), indicating future price direction.
                """)
            
            with col2:
                st.markdown("""
                **TariffImpact_raw_percent**
                - **Description**: The applicable import duty or tariff percentage.
                - **Calculation**: Determined by the supplier's country of origin (`Supplier_Country`) lookup in a pre-configured tariff schedule. The value used for a supplier-material combination is the average tariff if multiple POs exist with potentially varying (though unlikely for same country) tariff data.
                - **Purpose**: Captures the direct cost impact of customs duties on imported goods.
                
                **AvgLeadTimeDays_raw**
                - **Description**: The typical delivery time in days from PO creation to the first goods receipt.
                - **Calculation**: *Median* of `LeadTimeDays` for a supplier-material combination. `LeadTimeDays` is calculated as `FirstGRDate` (first `BUDAT` for history event type '1') minus `PO_Creation_Date` (`BEDAT`), floored at zero.
                - **Purpose**: Key input for calculating inventory holding costs associated with in-transit and pipeline inventory.
                
                **LeadTimeVariability_raw_days**
                - **Description**: The consistency or variability in supplier lead times.
                - **Calculation**: Standard deviation of `LeadTimeDays`.
                - **Purpose**: Measures the reliability of supplier delivery durations; higher variability necessitates more safety stock.

                **OnTimeRate_raw**
                - **Description**: The supplier's reliability in meeting scheduled delivery dates.
                - **Calculation**: Percentage of purchase order line items where the `FirstGRDate` is on or before the `EarliestEINDT` (earliest scheduled delivery date from PO schedule lines).
                - **Purpose**: Quantifiespunctuality, impacting production schedules and potential stockout risks.
                
                **InFullRate_raw**
                - **Description**: The supplier's ability to deliver the complete ordered quantity.
                - **Calculation**: Percentage of purchase order line items where the `TotalDeliveredQty` (sum of `MENGE` from goods receipt history) meets or exceeds the `Item_Ordered_Quantity`.
                - **Purpose**: Measures fulfillment accuracy; short shipments can lead to disruptions and reordering costs.
                """)
        
        with st.expander("Economic Impact Parameters (EIPs)", expanded=True):
            st.markdown("### Economic Impact Parameters (EIPs)")
            st.markdown("These configurable parameters are crucial for translating operational raw metrics into tangible economic costs. They reflect the business's financial policies and risk appetite.")
            
            # Load current EIP parameters from costs.json
            current_eips = load_eip_parameters(profile_id)
            
            st.subheader("Configure Economic Impact Parameters")
            
            # Create input fields for each EIP parameter
            col1, col2 = st.columns(2)
            
            with col1:
                annual_holding_rate = st.number_input(
                    "Annual Holding Cost Rate (%)",
                    min_value=0.0,
                    max_value=100.0,
                    value=current_eips.get('EIP_ANNUAL_HOLDING_COST_RATE', 0.18) * 100,
                    step=0.1,
                    format="%.1f",
                    help="Annual cost of holding inventory as a percentage of inventory value. Typically 10-30%."
                )
                
                safety_stock_multiplier = st.number_input(
                    "Safety Stock Multiplier (Z-score)",
                    min_value=0.0,
                    max_value=5.0,
                    value=current_eips.get('EIP_SafetyStockMultiplierForLTVar_Param', 1.65),
                    step=0.01,
                    format="%.2f",
                    help="Z-score for safety stock calculation. Common values: 1.28 (90% CSL), 1.65 (95% CSL), 2.33 (99% CSL)"
                )
                
                risk_premium_factor = st.number_input(
                    "Price Volatility Risk Premium (%)",
                    min_value=0.0,
                    max_value=100.0,
                    value=current_eips.get('EIP_RiskPremiumFactorForPriceVolatility_Param', 0.25) * 100,
                    step=1.0,
                    format="%.0f",
                    help="Risk premium applied to price standard deviation. Higher values penalize volatile suppliers more."
                )
            
            with col2:
                planning_horizon = st.number_input(
                    "Planning Horizon (days)",
                    min_value=1,
                    max_value=365,
                    value=current_eips.get('EIP_PlanningHorizonDaysForPriceTrend_Param', 90),
                    step=1,
                    help="Future period over which price trends are projected. Typically 30-180 days."
                )
                
                price_trend_cutoff = st.number_input(
                    "Price Trend Analysis Window (days)",
                    min_value=30,
                    max_value=365,
                    value=current_eips.get('PRICE_TREND_CUTOFF_DAYS', 180),
                    step=1,
                    help="Historical period used to calculate price trends. Longer periods smooth volatility."
                )
                
                base_logistics_rate = st.number_input(
                    "Base Logistics Cost Rate (%)",
                    min_value=0.0,
                    max_value=50.0,
                    value=current_eips.get('EIP_BaseLogisticsCostRate_Param', 0.15) * 100,
                    step=0.1,
                    format="%.1f",
                    help="Base logistics cost as percentage of unit price, before country and material factors. Typically 5-25%."
                )
            
            # Display calculated daily holding cost rate
            daily_rate = (annual_holding_rate / 100) / 365.0
            st.info(f"Daily Holding Cost Rate: {daily_rate*100:.4f}% (derived from {annual_holding_rate:.1f}% annual rate)")
            
            # Parameter descriptions table
            st.subheader("Parameter Descriptions")
            param_descriptions = {
                "Parameter": [
                    "Annual Holding Cost Rate",
                    "Safety Stock Multiplier",
                    "Price Volatility Risk Premium",
                    "Planning Horizon",
                    "Price Trend Analysis Window",
                    "Base Logistics Cost Rate"
                ],
                "Description": [
                    "Annual cost of holding inventory, including capital, storage, obsolescence, insurance, etc.",
                    "Z-score for calculating safety stock to achieve desired service level against lead time variability.",
                    "Multiplier applied to price standard deviation to quantify financial risk of price volatility.",
                    "Future period over which the impact of current price trends is projected and monetized.",
                    "Historical period used for calculating price trend slopes via linear regression.",
                    "Base percentage of unit price allocated to shipping and logistics costs, adjusted by country distance and material factors."
                ]
            }
            st.table(param_descriptions)
            
            # Update button
            if st.button("Save EIP Configuration", key="save_eips"):
                new_eips = {
                    'EIP_ANNUAL_HOLDING_COST_RATE': annual_holding_rate / 100,
                    'EIP_SafetyStockMultiplierForLTVar_Param': safety_stock_multiplier,
                    'EIP_RiskPremiumFactorForPriceVolatility_Param': risk_premium_factor / 100,
                    'EIP_PlanningHorizonDaysForPriceTrend_Param': planning_horizon,
                    'PRICE_TREND_CUTOFF_DAYS': price_trend_cutoff,
                    'EIP_BaseLogisticsCostRate_Param': base_logistics_rate / 100
                }
                if save_eip_parameters(new_eips, profile_id):
                    st.success("EIP configuration saved successfully!")
                    st.info("**Note**: To apply these changes to the vendor analysis, scroll down and click 'Update Configuration & Run Optimization'.")
                else:
                    st.error("Failed to save EIP configuration")
        
        with st.expander("Cost Component Calculations", expanded=True):
            st.markdown("### Cost Component Calculations")
            st.markdown("Configure which cost components are included in the total effective cost calculation. Each component represents a different aspect of the total cost of ownership.")
            
            st.subheader("Configure Cost Components")
            
            # Create columns for better layout
            col1, col2 = st.columns(2)
            
            # Dictionary to store the updated cost component states
            updated_costs = {}
            
            # First column - first 4 components
            with col1:
                for comp_key in ordered_cost_component_keys[:4]:
                    details = all_cost_component_details.get(comp_key, {})
                    current_state = str(costs_config.get(comp_key, "False")).lower() == "true"
                    
                    # Create checkbox with help text
                    new_state = st.checkbox(
                        details.get("display_name", comp_key),
                        value=current_state,
                        key=f"cost_comp_{comp_key}",
                        help=details.get("description", "")
                    )
                    updated_costs[comp_key] = "True" if new_state else "False"
            
            # Second column - remaining components
            with col2:
                for comp_key in ordered_cost_component_keys[4:]:
                    details = all_cost_component_details.get(comp_key, {})
                    current_state = str(costs_config.get(comp_key, "False")).lower() == "true"
                    
                    # Create checkbox with help text
                    new_state = st.checkbox(
                        details.get("display_name", comp_key),
                        value=current_state,
                        key=f"cost_comp_{comp_key}",
                        help=details.get("description", "")
                    )
                    updated_costs[comp_key] = "True" if new_state else "False"
            
            # Save button for cost components
            if st.button("Save Cost Component Configuration", key="save_cost_components"):
                if save_cost_components(updated_costs, profile_id):
                    st.success("Cost component configuration saved successfully!")
                    st.info("**Note**: To apply these changes to the vendor analysis, scroll down and click 'Update Configuration & Run Optimization'.")
                    # Force reload of costs_config
                    costs_config = updated_costs
                else:
                    st.error("Failed to save cost component configuration")
            
            st.markdown("---")
            
            # Display active components and their formulas
            st.subheader("Active Cost Component Details")
            
            active_component_display_names = []
            component_number = 1
            any_component_displayed = False
            
            for comp_key in ordered_cost_component_keys:
                is_active = str(updated_costs.get(comp_key, costs_config.get(comp_key, "False"))).lower() == "true"
                
                if is_active:
                    any_component_displayed = True
                    details = all_cost_component_details.get(comp_key)
                    if details:
                        active_component_display_names.append(details["display_name"])
                        
                        # Use container instead of expander to avoid nesting
                        st.markdown(f"**{component_number}. {details['display_name']}**")
                        with st.container():
                            st.latex(details['latex'])
                            st.markdown(f"_{details['description']}_")
                            st.markdown("")  # Add spacing
                        component_number += 1
            
            if active_component_display_names:
                st.success(f"**Active Components**: {', '.join(active_component_display_names)}")
            else:
                st.warning("**No cost components are currently active. EffectiveCostPerUnit_USD will be 0.**")
            
            if not any_component_displayed:
                st.info("Enable cost components above to see their calculation details.")

            st.markdown("---")
            st.markdown("#### Total Effective Cost Per Unit")
            st.latex(r"\text{EffectiveCostPerUnit\_USD} = \sum_{\text{active components}} \text{cost\_component\_value}")
            st.markdown("This comprehensive metric is the sum of all enabled cost components, allowing you to tailor the optimization to your specific business priorities.")
        
        with st.expander("Optimization Algorithm (Linear Programming)", expanded=False):
            st.markdown("### Optimization Algorithm Overview")
            
            st.markdown("""
            The optimization process employs **Linear Programming (LP)** to determine the optimal allocation of procurement quantities to vendors for each material/item. The primary goal is to:
            
            1.  **Minimize Total Effective Procurement Cost**: The objective function seeks to minimize the sum of (`EffectiveCostPerUnit_USD` × `Allocated_Quantity`) across all supplier-item combinations.
            
            2.  **Adhere to Operational Constraints**:
                *   **Demand Satisfaction**: For each material/item, the total quantity allocated across all eligible suppliers must precisely meet the target demand. Target demand is typically derived from historical consumption over a defined period (e.g., 365 days).
                  ```
                  Constraint: For each item 'i':
                  SUM(AllocatedQuantity[supplier 's', item 'i'] for all 's') = TargetDemand[item 'i']
                  ```
                *   **Supplier Capacity (Value-Based)**: For each supplier, the total monetary value of all items allocated to them must not exceed their estimated capacity. This capacity is calculated based on their historical peak periodic (e.g., monthly) sales value, scaled to the demand period, and augmented by a configurable buffer percentage. The value of allocated goods is determined using each item's `AvgUnitPriceUSD_raw`.
                  ```
                  Constraint: For each supplier 's':
                  SUM(AllocatedQuantity[supplier 's', item 'i'] * AvgUnitPriceUSD_raw[supplier 's', item 'i'] for all 'i') <= MaxCapacity_USD[supplier 's']
                  ```
                *   **Non-Negativity**: Allocated quantities cannot be negative.
                  ```
                  Constraint: For each supplier 's' and item 'i':
                  AllocatedQuantity[supplier 's', item 'i'] >= 0
                  ```
            
            The decision variables are the `AllocatedQuantity[supplier, item]`, which are continuous. A non-zero allocation to a supplier for an item implies that the supplier is selected for that portion of the demand.
            
            **Key Algorithm Parameters:**
            - **Demand Period for Quantity Calculation**: 365 days (historical consumption period).
            - **Supplier Capacity Calculation**: Based on historical peak sales value within a period (e.g., monthly: `'M'`), scaled, plus a `CAPACITY_BUFFER_PERCENT` (10%).
            - **Solver**: PuLP interface.
            """)
            
            st.code("""
# Conceptual PuLP Model Structure (Simplified)
# prob = pulp.LpProblem("Procurement_Cost_Minimization", pulp.LpMinimize)
#
# # Variables: AllocatedQuantity_vars[supplier, item] (Continuous)
#
# # Objective Function
# prob += pulp.lpSum(
#     EffectiveCostPerUnit_USD[s,i] * AllocatedQuantity_vars[s,i] for s,i
# )
#
# # Demand Constraint (for each item 'i')
# prob += pulp.lpSum(AllocatedQuantity_vars[s,i] for s) == TargetDemand[i]
#
# # Capacity Constraint (for each supplier 's')
# prob += pulp.lpSum(
#     AllocatedQuantity_vars[s,i] * AvgUnitPriceUSD_raw[s,i] for i
# ) <= MaxCapacity_USD[s]
            """, language="python")
        
      
        # Add a configuration status section
        st.markdown("---")
        st.markdown("### Current Configuration Status")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            active_count_from_config = 0
            if costs_config: # Check if costs_config is not empty
                 active_count_from_config = sum(1 for comp_key in ordered_cost_component_keys if str(costs_config.get(comp_key, "False")).lower() == "true")
            st.metric(f"Active Cost Components", f"{active_count_from_config}/{num_total_possible_components}")

        with col2:
            current_eips_display = load_eip_parameters(profile_id)
            st.metric("Demand Window", f"{current_eips_display.get('DEMAND_PERIOD_DAYS', 365)} days")

        with col3:
            st.metric("Price Trend Window", f"{current_eips_display.get('PRICE_TREND_CUTOFF_DAYS', 180)} days")
        
        with col4:
            st.metric("Planning Horizon (Price Trend)", f"{current_eips_display.get('EIP_PlanningHorizonDaysForPriceTrend_Param', 90)} days")
        
        st.info("""
        **Tip**: Both cost components and Economic Impact Parameters are now managed via the `costs.json` file. Use the button below to apply your changes and re-run the optimization.
        """)
        
        # Add optimization button section
        st.markdown("---")
        st.subheader("Apply Changes and Run Optimization")
        
        with st.expander("About the Optimization Process", expanded=False):
            st.markdown("""
            This process will:
            1. Save your current cost component and EIP parameter configuration
            2. Update vendor performance metrics with the new settings
            3. Run the optimization algorithm to select the best vendors
            4. Generate a comparison between historical and optimized procurement
            5. Update all dashboard visualizations with the new data
            
            **Note:** This process may take several minutes to complete.
            """)
        
        # Create the optimization button
        if st.button("Update Configuration & Run Optimization", type="primary", key="run_optimization_with_config"):
            # First save the current configuration if any changes were made
            st.info("Saving current configuration...")
            
            # Force delete optimization files to ensure regeneration
            files_to_clean = [
                os.path.join('tables', 'vendor_maktx_ranking_tariff_values.csv'),
                os.path.join('tables', 'vendor_with_direct_countries.csv'),  # Also clean this file
                os.path.join('tables', 'optimized_allocation_maktx_vendor_maktx_ranking_tariff_values.csv'),
                os.path.join('tables', 'comparison.csv')
            ]
            
            # Create backup directory
            backup_dir = os.path.join('tables', 'backup', datetime.now().strftime('%Y%m%d_%H%M%S'))
            os.makedirs(backup_dir, exist_ok=True)
            
            for file_path in files_to_clean:
                if os.path.exists(file_path):
                    try:
                        backup_path = os.path.join(backup_dir, os.path.basename(file_path))
                        os.rename(file_path, backup_path)
                        st.info(f"Backed up {os.path.basename(file_path)}")
                    except Exception as e:
                        st.warning(f"Could not backup {os.path.basename(file_path)}: {e}")
            
            # Run the optimization pipeline
            with st.spinner("Running optimization with updated EIP parameters..."):
                success, message = run_optimization_with_eips(profile_id)
                
                if success:
                    st.success(f"{message}")
                    
                    # Clear ALL caches to force complete reload
                    st.cache_data.clear()
                    
                    # Clear all session state related to data
                    keys_to_clear = [key for key in st.session_state.keys() if 
                                    'vendor' in key.lower() or 
                                    'data' in key.lower() or 
                                    'cache' in key.lower() or
                                    'mtime' in key.lower()]
                    for key in keys_to_clear:
                        if key in st.session_state:
                            del st.session_state[key]
                    
                    st.info("Dashboard data has been updated with the new EIP parameters.")
                    st.warning("**Important**: The page will now refresh to load the updated data. All tabs will reflect the new calculations.")
                    
                    # Force a complete page reload
                    st.rerun()
                else:
                    st.error(f"{message}")
                    with st.expander("Error Details"):
                        st.code(message)


def render_optimization_settings_tab_standalone():
    """Standalone version for lazy loading"""
    class DummyTab:
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
    render_optimization_settings_tab(DummyTab())