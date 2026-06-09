"""Typed fact definitions for the executable decision engine."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class AnswerOption:
    """Represents a possible answer option for a question."""

    label: str
    value: Any
    description: str | None = None


@dataclass(frozen=True)
class FactDefinition:
    fact_id: str
    value_type: str
    description: str
    question_prompt: str | None = None
    question_source: str = "customer"
    answer_options: tuple[AnswerOption, ...] | None = None


BOOLEAN_YES_NO_OPTIONS = (
    AnswerOption("Yes", True, "The customer confirms this is true"),
    AnswerOption("No", False, "The customer confirms this is not true"),
    AnswerOption("Not sure / Prefer not to answer", None, "The customer is unsure or prefers not to answer"),
)

HOME_OWNERSHIP_OPTIONS = (
    AnswerOption("Homeowner", "HOMEOWNER", "The customer owns the property"),
    AnswerOption("Renter", "RENTER", "The customer rents the property"),
    AnswerOption("Not sure", None, "The customer is unsure about the ownership status"),
)

DWELLING_TYPE_OPTIONS = (
    AnswerOption("Single-family home", "SINGLE_FAMILY", "Detached single-family residence"),
    AnswerOption("Townhome / condo / patio home", "ATTACHED_HOME", "Attached owner-occupied unit"),
    AnswerOption("Apartment or multifamily unit", "MULTIFAMILY_UNIT", "Unit within a multifamily property"),
    AnswerOption("Mobile home", "MOBILE_HOME", "Manufactured or mobile home"),
    AnswerOption("Not sure", None, "The customer is unsure about the dwelling type"),
)

COOLING_SYSTEM_OPTIONS = (
    AnswerOption("Central air conditioner", "CENTRAL_AC", "The home uses central AC"),
    AnswerOption("Heat pump", "HEAT_PUMP", "The home uses a heat pump"),
    AnswerOption("Other / Not sure", None, "Cooling system is unknown or another type"),
)

PROJECT_STAGE_OPTIONS = (
    AnswerOption("Early design", "EARLY_DESIGN", "Project is still in early design"),
    AnswerOption("Later design / construction", "LATE_DESIGN_OR_CONSTRUCTION", "Project is already advanced"),
    AnswerOption("Existing facility", "EXISTING_FACILITY", "This is not a new-construction project"),
    AnswerOption("Not sure", None, "Project stage is unknown"),
)

ACCOUNT_NAME_TYPE_OPTIONS = (
    AnswerOption("Personal / individual", "PERSONAL", "Account is held by an individual or household"),
    AnswerOption("Company / business", "COMPANY", "Account is held in a company or business name"),
    AnswerOption("Other / Not sure", None, "Account name type is unknown"),
)

PREPAY_ADVANCE_COUNT_OPTIONS = (
    AnswerOption("0 (None)", 0, "No offers this month"),
    AnswerOption("1", 1, "One offer this month"),
    AnswerOption("2 or more", 2, "Two or more offers this month"),
)


FACT_REGISTRY: dict[str, FactDefinition] = {
    "customer_type": FactDefinition(
        "customer_type",
        "string",
        "Residential or commercial account type.",
    ),
    "has_current_snapshot": FactDefinition(
        "has_current_snapshot",
        "boolean",
        "Whether the account has at least one row with a valid READ DATE.",
        "Retrieve a current billing snapshot for this account before making an offer decision.",
        question_source="system",
    ),
    "current_status": FactDefinition(
        "current_status",
        "string",
        "Current account status from the latest snapshot.",
    ),
    "current_rate_plan": FactDefinition(
        "current_rate_plan",
        "string",
        "Current price plan code from the latest snapshot.",
    ),
    "snapshot_read_date": FactDefinition(
        "snapshot_read_date",
        "date",
        "READ DATE from the latest dated billing snapshot.",
    ),
    "segment_name": FactDefinition(
        "segment_name",
        "string",
        "Persona segment from the segment workbook.",
    ),
    "current_program_codes": FactDefinition(
        "current_program_codes",
        "list[string]",
        "Program codes already enrolled on the account.",
    ),
    "residential_price_plan_customer": FactDefinition(
        "residential_price_plan_customer",
        "boolean",
        "Whether the account is a residential customer on an utility residential price plan.",
    ),
    "is_mpower_enrolled": FactDefinition(
        "is_mpower_enrolled",
        "boolean",
        "Whether the customer is currently enrolled in the Prepay price plan.",
    ),
    "service_charge_tier": FactDefinition(
        "service_charge_tier",
        "string",
        "Residential fixed-charge tier derived from dwelling type and service entrance amps.",
    ),
    "dwelling_type": FactDefinition(
        "dwelling_type",
        "string",
        "Dwelling type used for rebate eligibility and monthly service charge tiering.",
        "Which dwelling type best matches the service address?",
        answer_options=DWELLING_TYPE_OPTIONS,
    ),
    "service_entrance_amps": FactDefinition(
        "service_entrance_amps",
        "number",
        "Service entrance amperage for the residence.",
        question_source="system",
    ),
    "service_start_date": FactDefinition(
        "service_start_date",
        "date",
        "Service start date for the address or account profile.",
        question_source="system",
    ),
    "service_at_address_over_12_months": FactDefinition(
        "service_at_address_over_12_months",
        "boolean",
        "Whether utility service at the address has been active for at least 12 months.",
        "Has utility electric service been active at this address for at least 12 months?",
        answer_options=BOOLEAN_YES_NO_OPTIONS,
    ),
    "home_ownership_status": FactDefinition(
        "home_ownership_status",
        "string",
        "Whether the customer is a homeowner or renter.",
        "Is the customer a homeowner or a renter?",
        answer_options=HOME_OWNERSHIP_OPTIONS,
    ),
    "ownership_status": FactDefinition(
        "ownership_status",
        "string",
        "Ownership status from the profile table.",
        "Is the customer the property owner?",
        answer_options=HOME_OWNERSHIP_OPTIONS,
    ),
    "occupancy_status": FactDefinition(
        "occupancy_status",
        "string",
        "Occupancy status from the profile table.",
    ),
    "customer_of_record_on_site": FactDefinition(
        "customer_of_record_on_site",
        "boolean",
        "Whether the customer of record lives at the service address as their primary residence.",
        "Does the customer of record live at this service address as their primary residence?",
        answer_options=BOOLEAN_YES_NO_OPTIONS,
    ),
    "account_name_type": FactDefinition(
        "account_name_type",
        "string",
        "Whether the account is held by an individual or a company.",
        "Is the account held in an individual's name or in a company/business name?",
        answer_options=ACCOUNT_NAME_TYPE_OPTIONS,
    ),
    "is_company_name_account": FactDefinition(
        "is_company_name_account",
        "boolean",
        "Whether the account is held in a company or business name.",
    ),
    "cooling_system_type": FactDefinition(
        "cooling_system_type",
        "string",
        "Cooling system type used for residential rebate qualification.",
        "What type of cooling system does the home use?",
        answer_options=COOLING_SYSTEM_OPTIONS,
    ),
    "central_air_conditioning": FactDefinition(
        "central_air_conditioning",
        "boolean",
        "Whether the residence uses central air conditioning.",
        "Does the home have central air conditioning?",
        answer_options=BOOLEAN_YES_NO_OPTIONS,
    ),
    "thermostat_controls_central_ac": FactDefinition(
        "thermostat_controls_central_ac",
        "boolean",
        "Whether the thermostat controls central air conditioning at the service location.",
        "Does the thermostat control the home's central air conditioning?",
        answer_options=BOOLEAN_YES_NO_OPTIONS,
    ),
    "thermostat_provider_account": FactDefinition(
        "thermostat_provider_account",
        "boolean",
        "Whether the customer has the required thermostat provider account.",
        "Does the customer have the required account with the thermostat provider?",
        answer_options=BOOLEAN_YES_NO_OPTIONS,
    ),
    "thermostat_wifi_connected": FactDefinition(
        "thermostat_wifi_connected",
        "boolean",
        "Whether the thermostat is connected to Wi-Fi at the service address.",
        "Is the thermostat connected to Wi-Fi at the service address?",
        answer_options=BOOLEAN_YES_NO_OPTIONS,
    ),
    "thermostat_brand": FactDefinition(
        "thermostat_brand",
        "string",
        "Thermostat brand from the DER profile when available.",
    ),
    "smart_thermostat_purchase_eligible": FactDefinition(
        "smart_thermostat_purchase_eligible",
        "boolean",
        "Whether the customer is pursuing an eligible smart thermostat purchase/install path.",
        "Is the customer buying or installing an eligible smart thermostat through the utility program path?",
        answer_options=BOOLEAN_YES_NO_OPTIONS,
    ),
    "new_construction_flag": FactDefinition(
        "new_construction_flag",
        "boolean",
        "Whether the home or project is new construction.",
        "Is this a new-construction home or project?",
        answer_options=BOOLEAN_YES_NO_OPTIONS,
    ),
    "eligible_contractor": FactDefinition(
        "eligible_contractor",
        "boolean",
        "Whether the installation uses the contractor or installer type required by the program document.",
        "Will the work be completed by the contractor or installer type required by the program?",
        answer_options=BOOLEAN_YES_NO_OPTIONS,
    ),
    "air_conditioning_measure_qualified": FactDefinition(
        "air_conditioning_measure_qualified",
        "boolean",
        "Whether the air-conditioning project meets the document's qualifying equipment and replacement requirements.",
        "Does the air-conditioning project meet the program's qualifying equipment and replacement requirements?",
        answer_options=BOOLEAN_YES_NO_OPTIONS,
    ),
    "insulation_measure_qualified": FactDefinition(
        "insulation_measure_qualified",
        "boolean",
        "Whether the insulation project meets the qualifying measure requirements.",
        "Does the insulation project meet the program's qualifying insulation requirements?",
        answer_options=BOOLEAN_YES_NO_OPTIONS,
    ),
    "cool_roof_measure_qualified": FactDefinition(
        "cool_roof_measure_qualified",
        "boolean",
        "Whether the cool-roof project meets the qualifying material and application requirements.",
        "Does the cool-roof project meet the program's qualifying requirements?",
        answer_options=BOOLEAN_YES_NO_OPTIONS,
    ),
    "hpwh_measure_qualified": FactDefinition(
        "hpwh_measure_qualified",
        "boolean",
        "Whether the heat pump water heater project meets the qualifying equipment requirements.",
        "Does the heat pump water heater project meet the program's qualifying equipment requirements?",
        answer_options=BOOLEAN_YES_NO_OPTIONS,
    ),
    "window_measure_qualified": FactDefinition(
        "window_measure_qualified",
        "boolean",
        "Whether the window replacement project meets the qualifying product requirements.",
        "Does the window replacement project meet the program's qualifying window requirements?",
        answer_options=BOOLEAN_YES_NO_OPTIONS,
    ),
    "avg_on_peak_kwh_3m": FactDefinition(
        "avg_on_peak_kwh_3m",
        "number",
        "Average on-peak kWh over the latest three valid bills.",
    ),
    "avg_off_peak_kwh_3m": FactDefinition(
        "avg_off_peak_kwh_3m",
        "number",
        "Average off-peak kWh over the latest three valid bills.",
    ),
    "avg_super_off_peak_kwh_3m": FactDefinition(
        "avg_super_off_peak_kwh_3m",
        "number",
        "Estimated average super off-peak kWh over the latest three valid bills for super-off-peak plan comparisons.",
    ),
    "avg_total_usage_3m": FactDefinition(
        "avg_total_usage_3m",
        "number",
        "Average total meter usage over the latest three valid bills.",
    ),
    "avg_on_peak_summer": FactDefinition(
        "avg_on_peak_summer",
        "number",
        "Average on-peak kWh over summer months (May-October).",
    ),
    "avg_off_peak_summer": FactDefinition(
        "avg_off_peak_summer",
        "number",
        "Average off-peak kWh over summer months (May-October).",
    ),
    "avg_super_off_peak_summer": FactDefinition(
        "avg_super_off_peak_summer",
        "number",
        "Estimated average super off-peak kWh over summer months (May-October).",
    ),
    "avg_total_usage_summer": FactDefinition(
        "avg_total_usage_summer",
        "number",
        "Average total meter usage over summer months (May-October).",
    ),
    "avg_on_peak_daily_kw_3m": FactDefinition(
        "avg_on_peak_daily_kw_3m",
        "number",
        "Estimated average on-peak daily demand in kW over recent bills.",
    ),
    "latest_bill_shortfall": FactDefinition(
        "latest_bill_shortfall",
        "number",
        "Current unpaid amount on the latest bill.",
    ),
    "repeated_shortfall_count_6m": FactDefinition(
        "repeated_shortfall_count_6m",
        "integer",
        "Number of positive shortfalls across the latest six valid bills.",
    ),
    "payment_distress_disconnected": FactDefinition(
        "payment_distress_disconnected",
        "boolean",
        "Whether the account is currently disconnected.",
    ),
    "payment_distress_current_shortfall": FactDefinition(
        "payment_distress_current_shortfall",
        "boolean",
        "Whether the latest bill has a positive unpaid balance.",
    ),
    "payment_distress_repeat_shortfalls": FactDefinition(
        "payment_distress_repeat_shortfalls",
        "boolean",
        "Whether recent billing history shows repeated positive shortfalls.",
    ),
    "has_payment_distress_signal": FactDefinition(
        "has_payment_distress_signal",
        "boolean",
        "Composite payment distress signal: true if disconnected, current shortfall, or repeated shortfalls.",
    ),
    "bill_increase_yoy": FactDefinition(
        "bill_increase_yoy",
        "number",
        "Year-over-year bill increase ratio when history supports it.",
    ),
    "yoy_history_supported": FactDefinition(
        "yoy_history_supported",
        "boolean",
        "Whether the account has enough history to evaluate year-over-year metrics.",
    ),
    "high_usage_percentile": FactDefinition(
        "high_usage_percentile",
        "boolean",
        "Whether usage is above the configured percentile threshold.",
    ),
    "high_usage_yoy": FactDefinition(
        "high_usage_yoy",
        "boolean",
        "Whether usage increased materially year over year when supported.",
    ),
    "has_high_usage_signal": FactDefinition(
        "has_high_usage_signal",
        "boolean",
        "Composite high-usage signal based on percentile and year-over-year growth.",
    ),
    "bill_increase_or_high_usage": FactDefinition(
        "bill_increase_or_high_usage",
        "boolean",
        "Composite trigger for rising bills or high usage.",
    ),
    "payments_on_time": FactDefinition(
        "payments_on_time",
        "boolean",
        "Whether the customer pays within the due date.",
        "Are this customer's payments currently being made on time?",
        answer_options=BOOLEAN_YES_NO_OPTIONS,
    ),
    "prepay_advance_offers_this_month": FactDefinition(
        "prepay_advance_offers_this_month",
        "integer",
        "Count of Prepay Advance offers already made this month.",
        "How many Prepay Advance offers has this customer already received this month?",
        answer_options=PREPAY_ADVANCE_COUNT_OPTIONS,
    ),
    "three_day_usage_cost_estimate": FactDefinition(
        "three_day_usage_cost_estimate",
        "number",
        "Estimated cost of three days of usage based on the latest billing period.",
    ),
    "customer_wants_followup": FactDefinition(
        "customer_wants_followup",
        "boolean",
        "Whether the customer wants to continue with more eligibility questions.",
        "Would the customer like to answer more questions to explore additional programs?",
        answer_options=BOOLEAN_YES_NO_OPTIONS,
    ),
    "customer_requests_custom_due_date": FactDefinition(
        "customer_requests_custom_due_date",
        "boolean",
        "Whether the customer wants to enroll in Custom Due Date.",
        "Would the customer like to enroll in Custom Due Date?",
        answer_options=BOOLEAN_YES_NO_OPTIONS,
    ),
    "household_income_qualified": FactDefinition(
        "household_income_qualified",
        "boolean",
        "Whether the household income qualifies for household assistance discount.",
        "Does the household income meet the household assistance discount income guidelines?",
        answer_options=BOOLEAN_YES_NO_OPTIONS,
    ),
    "income_assistance_auto_qualifier": FactDefinition(
        "income_assistance_auto_qualifier",
        "boolean",
        "Whether the customer is already auto-qualified through LIHEAP or LIHSUP.",
        "Is the customer already qualified through LIHEAP or LIHSUP?",
        answer_options=BOOLEAN_YES_NO_OPTIONS,
    ),
    "has_60_day_arrears": FactDefinition(
        "has_60_day_arrears",
        "boolean",
        "Whether the account has 60-day arrears.",
        "Does this account currently have 60-day arrears?",
        question_source="system",
        answer_options=BOOLEAN_YES_NO_OPTIONS,
    ),
    "current_rate_supported_for_optimization": FactDefinition(
        "current_rate_supported_for_optimization",
        "boolean",
        "Whether the current rate plan is covered by the deterministic tariff simulator.",
    ),
    "ev_ownership": FactDefinition(
        "ev_ownership",
        "boolean",
        "Whether the customer owns or leases an electric vehicle.",
        "Does the customer own or lease an electric vehicle?",
        answer_options=BOOLEAN_YES_NO_OPTIONS,
    ),
    "tesla_vehicle_ownership": FactDefinition(
        "tesla_vehicle_ownership",
        "boolean",
        "Whether the customer owns or leases a Tesla vehicle.",
        "Does the customer own or lease a Tesla vehicle?",
        answer_options=BOOLEAN_YES_NO_OPTIONS,
    ),
    "eligible_home_ev_charger": FactDefinition(
        "eligible_home_ev_charger",
        "boolean",
        "Whether the customer has an eligible residential EV charger.",
        "Does the customer have an eligible home EV charger for this program?",
        answer_options=BOOLEAN_YES_NO_OPTIONS,
    ),
    "eligible_connected_devices": FactDefinition(
        "eligible_connected_devices",
        "boolean",
        "Whether the customer has an eligible connected thermostat or other required connected device.",
        "Does the customer have an eligible connected thermostat or device at this service address?",
        answer_options=BOOLEAN_YES_NO_OPTIONS,
    ),
    "solar_ownership": FactDefinition(
        "solar_ownership",
        "boolean",
        "Whether the customer owns or operates a qualifying rooftop solar or export facility.",
        "Does the customer own a qualifying rooftop solar or other export facility?",
        answer_options=BOOLEAN_YES_NO_OPTIONS,
    ),
    "new_rooftop_solar_installation": FactDefinition(
        "new_rooftop_solar_installation",
        "boolean",
        "Whether the customer is installing a new rooftop solar system.",
        "Is this a new rooftop solar installation?",
        answer_options=BOOLEAN_YES_NO_OPTIONS,
    ),
    "preferred_solar_installer": FactDefinition(
        "preferred_solar_installer",
        "boolean",
        "Whether the solar installer is on the required preferred-installer list.",
        "Is the solar installer on the program's preferred installer list?",
        answer_options=BOOLEAN_YES_NO_OPTIONS,
    ),
    "demand_management_system_included": FactDefinition(
        "demand_management_system_included",
        "boolean",
        "Whether the demand-management system is included in the solar installation/application package.",
        "Is the demand-management system included in the solar installation application package?",
        answer_options=BOOLEAN_YES_NO_OPTIONS,
    ),
    "qualifying_facility_kw_ac": FactDefinition(
        "qualifying_facility_kw_ac",
        "number",
        "Nameplate or qualifying-facility AC kW size for export programs.",
        question_source="system",
    ),
    "qualifying_facility_le_100kw": FactDefinition(
        "qualifying_facility_le_100kw",
        "boolean",
        "Whether the qualifying export facility is 100 kW AC or less.",
    ),
    "storage_only_configuration": FactDefinition(
        "storage_only_configuration",
        "boolean",
        "Whether the export/battery system is storage-only without qualifying generation.",
        "Is the system storage-only without qualifying solar generation?",
        answer_options=BOOLEAN_YES_NO_OPTIONS,
    ),
    "solar_export_rate_plan": FactDefinition(
        "solar_export_rate_plan",
        "boolean",
        "Whether the customer is currently on a residential solar export-compatible price plan.",
    ),
    "battery_ownership": FactDefinition(
        "battery_ownership",
        "boolean",
        "Whether the customer owns a compatible battery system.",
        "Does the customer own a compatible home battery system?",
        answer_options=BOOLEAN_YES_NO_OPTIONS,
    ),
    "compatible_battery_configuration": FactDefinition(
        "compatible_battery_configuration",
        "boolean",
        "Whether the battery configuration is compatible with the documented program requirements.",
        "Is the battery configuration compatible with the program requirements?",
        answer_options=BOOLEAN_YES_NO_OPTIONS,
    ),
    "rec_rights_owned": FactDefinition(
        "rec_rights_owned",
        "boolean",
        "Whether the customer still owns the renewable energy credit rights.",
        "Does the customer still own the renewable energy credit rights for this system?",
        answer_options=BOOLEAN_YES_NO_OPTIONS,
    ),
    "prior_rec_assignment_active": FactDefinition(
        "prior_rec_assignment_active",
        "boolean",
        "Whether the renewable energy credits have already been assigned elsewhere.",
        "Have the renewable energy credits already been assigned or sold elsewhere?",
        answer_options=BOOLEAN_YES_NO_OPTIONS,
    ),
    "battery_partner_brand": FactDefinition(
        "battery_partner_brand",
        "string",
        "Battery brand from the DER profile when available.",
    ),
    "battery_partner_brand_supported": FactDefinition(
        "battery_partner_brand_supported",
        "boolean",
        "Whether the battery brand is on the documented Battery Partner list.",
        "Is the battery brand on the supported Battery Partner list?",
        answer_options=BOOLEAN_YES_NO_OPTIONS,
    ),
    "legal_right_to_plant": FactDefinition(
        "legal_right_to_plant",
        "boolean",
        "Whether the customer can legally plant trees on the property.",
        "Does the customer have the legal right to plant trees on the property?",
        answer_options=BOOLEAN_YES_NO_OPTIONS,
    ),
    "completed_shade_tree_workshop": FactDefinition(
        "completed_shade_tree_workshop",
        "boolean",
        "Whether the customer completed the shade tree workshop.",
        "Has the customer completed the Shade Tree workshop?",
        answer_options=BOOLEAN_YES_NO_OPTIONS,
    ),
    "already_received_shade_tree": FactDefinition(
        "already_received_shade_tree",
        "boolean",
        "Whether the property previously received trees from the shade tree program.",
        "Has this property already received trees through the Shade Tree Program?",
        answer_options=BOOLEAN_YES_NO_OPTIONS,
    ),
    "annual_usage_estimate_kwh": FactDefinition(
        "annual_usage_estimate_kwh",
        "number",
        "Estimated annual usage based on billing history.",
    ),
    "service_history_days": FactDefinition(
        "service_history_days",
        "integer",
        "Span in days across the known billing history.",
    ),
    "commercial_naic_code": FactDefinition(
        "commercial_naic_code",
        "integer",
        "NAIC code from the commercial workbook.",
    ),
    "commercial_industry": FactDefinition(
        "commercial_industry",
        "string",
        "Industry label from the commercial workbook.",
    ),
    "business_taxonomy": FactDefinition(
        "business_taxonomy",
        "string",
        "Normalized business family for commercial program routing.",
    ),
    "small_business_rate_plan_eligible": FactDefinition(
        "small_business_rate_plan_eligible",
        "boolean",
        "Whether the current rate plan is one of the rate plans accepted by Small Business Solutions.",
    ),
    "same_account_holder_12_months": FactDefinition(
        "same_account_holder_12_months",
        "boolean",
        "Whether the same account holder has held the service for at least 12 months.",
        "Has the same account holder had this service account for at least 12 months?",
        answer_options=BOOLEAN_YES_NO_OPTIONS,
    ),
    "facility_age_over_4_years": FactDefinition(
        "facility_age_over_4_years",
        "boolean",
        "Whether the facility is more than four years old.",
        "Is the facility more than four years old?",
        answer_options=BOOLEAN_YES_NO_OPTIONS,
    ),
    "account_good_standing": FactDefinition(
        "account_good_standing",
        "boolean",
        "Whether the account is in good standing for commercial-program participation.",
        "Is the account in good standing for this program?",
        answer_options=BOOLEAN_YES_NO_OPTIONS,
    ),
    "connected_unit_count": FactDefinition(
        "connected_unit_count",
        "integer",
        "Connected unit count for multifamily programs.",
        question_source="system",
    ),
    "connected_units_ge_4": FactDefinition(
        "connected_units_ge_4",
        "boolean",
        "Whether a multifamily property has at least four connected units.",
        "Does the property have at least four connected units?",
        answer_options=BOOLEAN_YES_NO_OPTIONS,
    ),
    "new_construction_project": FactDefinition(
        "new_construction_project",
        "boolean",
        "Whether the commercial project is new construction.",
        "Is this a new-construction project?",
        answer_options=BOOLEAN_YES_NO_OPTIONS,
    ),
    "project_stage": FactDefinition(
        "project_stage",
        "string",
        "Project stage for multifamily or new-construction programs.",
        "What stage is the project currently in?",
        answer_options=PROJECT_STAGE_OPTIONS,
    ),
    "project_early_design": FactDefinition(
        "project_early_design",
        "boolean",
        "Whether the new-construction project is in the early design stage.",
        "Is the project still in the early stages of design?",
        answer_options=BOOLEAN_YES_NO_OPTIONS,
    ),
    "conditioned_sqft": FactDefinition(
        "conditioned_sqft",
        "number",
        "Conditioned square footage for project qualification.",
        question_source="system",
    ),
    "project_sqft_qualifies": FactDefinition(
        "project_sqft_qualifies",
        "boolean",
        "Whether the project meets the minimum square-footage requirement.",
        "Does the project meet the minimum square-footage requirement for the program track?",
        answer_options=BOOLEAN_YES_NO_OPTIONS,
    ),
    "project_exceeds_baseline_by_10": FactDefinition(
        "project_exceeds_baseline_by_10",
        "boolean",
        "Whether the project exceeds baseline performance by at least 10%.",
        "Is the project expected to exceed the baseline building performance by at least 10%?",
        answer_options=BOOLEAN_YES_NO_OPTIONS,
    ),
    "energy_star_multifamily_project": FactDefinition(
        "energy_star_multifamily_project",
        "boolean",
        "Whether the multifamily project is pursuing the ENERGY STAR MFNC path.",
        "Is the project using the ENERGY STAR Multifamily New Construction path?",
        answer_options=BOOLEAN_YES_NO_OPTIONS,
    ),
    "qualified_price_plan": FactDefinition(
        "qualified_price_plan",
        "boolean",
        "Whether the commercial project is enrolled in a qualified utility price plan.",
        "Is the project enrolled in a qualified utility price plan?",
        answer_options=BOOLEAN_YES_NO_OPTIONS,
    ),
    "large_energy_consumer": FactDefinition(
        "large_energy_consumer",
        "boolean",
        "Whether the site passes the large-energy-consumer screen used for demand response.",
        "Is the site a large energy consumer suitable for demand response participation?",
        answer_options=BOOLEAN_YES_NO_OPTIONS,
    ),
    "can_participate_in_demand_response": FactDefinition(
        "can_participate_in_demand_response",
        "boolean",
        "Whether the business can reduce load during dispatch events.",
        "Can the business reduce energy consumption during demand response events?",
        answer_options=BOOLEAN_YES_NO_OPTIONS,
    ),
    "site_can_install_ev_charging": FactDefinition(
        "site_can_install_ev_charging",
        "boolean",
        "Whether the site can support EV charger deployment.",
        "Can the business install and support EV charging infrastructure at this site?",
        answer_options=BOOLEAN_YES_NO_OPTIONS,
    ),
}


def get_fact_definition(fact_id: str) -> FactDefinition:
    return FACT_REGISTRY[fact_id]


def is_customer_answerable_fact(fact_id: str) -> bool:
    """Return whether a fact can be directly overlaid from customer answers."""
    definition = get_fact_definition(fact_id)
    return (
        definition.question_source == "customer"
        and definition.question_prompt is not None
        and definition.answer_options is not None
    )


def normalize_answer_value(fact_id: str, value: Any) -> Any:
    """Normalize customer answers to canonical option values where possible."""
    if value is None:
        return None

    if fact_id not in FACT_REGISTRY:
        return value

    definition = get_fact_definition(fact_id)
    if not definition.answer_options:
        return value

    for option in definition.answer_options:
        if value == option.value:
            return option.value

    if not isinstance(value, str):
        return None

    normalized = value.strip().casefold()
    for option in definition.answer_options:
        if str(option.value).strip().casefold() == normalized:
            return option.value
        if option.value is not None and isinstance(option.value, str):
            if option.value.casefold() == normalized:
                return option.value
        if option.label.casefold() == normalized:
            return option.value

    return None


def known_fact_ids() -> set[str]:
    return set(FACT_REGISTRY)
