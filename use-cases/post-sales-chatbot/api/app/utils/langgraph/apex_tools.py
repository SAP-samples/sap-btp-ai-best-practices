"""
LangChain tools for Apex Automotive Services chatbot.

Converts business logic from actions_chatbot.py to LangChain @tool decorated functions.
"""

from __future__ import annotations

import pandas as pd
from typing import Any, Dict, Optional, List
from datetime import datetime

from langchain_core.tools import tool

from ...services.data_handler import data_handler
from ...services.session_manager import session_manager


def _get_session_state(session_id: str):
    """Helper to get session state, creating if needed."""
    if not session_id:
        return None
    state = session_manager.get(session_id)
    return state


@tool("find_client")
def find_client_tool(
    identifier_type: str,
    identifier_value: str,
    session_id: str = ""
) -> Dict[str, Any]:
    """Find a client by their identifier and store in session.

    Args:
        identifier_type: Type of identifier - one of 'email', 'phone', 'vin', 'license_plate', 'full_name'
        identifier_value: The value to search for
        session_id: Current session ID for state management

    Returns:
        Dict with client_id and name if found, or error message if not found.
    """
    all_data = data_handler.data
    clientes_df = all_data.get('clientes')
    unidades_df = all_data.get('unidades')

    if clientes_df is None or identifier_value is None:
        return {"error": "Client data not available or no identifier provided"}

    client_id = None
    value_str = str(identifier_value)
    value_lower = value_str.lower()

    # Map external identifier types to internal column names
    type_mapping = {
        'email': 'Correo',
        'phone': 'NumeroTelefonico',
        'vin': 'Vin',
        'license_plate': 'Matricula',
        'full_name': 'NombreCompleto'
    }

    internal_type = type_mapping.get(identifier_type.lower(), identifier_type)

    # Search by VIN in unidades table
    if internal_type == 'Vin' and unidades_df is not None:
        unit = unidades_df[unidades_df['Vin'].str.lower() == value_lower]
        if not unit.empty:
            client_id = unit['NumeroDeCliente'].iloc[0]

    # Search by license plate in unidades table
    elif internal_type == 'Matricula' and unidades_df is not None:
        unit = unidades_df[unidades_df['Matricula'].str.lower() == value_lower]
        if not unit.empty:
            client_id = unit['NumeroDeCliente'].iloc[0]

    # Search by full name
    elif internal_type == 'NombreCompleto':
        parts = value_str.split()
        if len(parts) >= 2:
            first_name = parts[0].lower()
            last_name = parts[-1].lower()
            client = clientes_df[
                (clientes_df['Nombre1'].str.lower() == first_name) &
                (clientes_df['ApellidoPaterno'].str.lower() == last_name)
            ]
            if not client.empty:
                client_id = client['NumeroDeCliente'].iloc[0]

    # Search by direct column match (email, phone)
    elif internal_type in clientes_df.columns:
        if pd.api.types.is_string_dtype(clientes_df[internal_type]):
            client = clientes_df[clientes_df[internal_type].str.lower() == value_lower]
        else:
            try:
                col_type = clientes_df[internal_type].dtype
                client = clientes_df[clientes_df[internal_type] == col_type.type(identifier_value)]
            except Exception:
                client = pd.DataFrame()

        if not client.empty:
            client_id = client['NumeroDeCliente'].iloc[0]

    if client_id is None or pd.isna(client_id):
        return {
            "error": f"No client found with {identifier_type}: {identifier_value}",
            "found": False
        }

    client_id_int = int(client_id)

    # Get client name
    client_row = clientes_df[clientes_df['NumeroDeCliente'] == client_id_int]
    client_name = ""
    if not client_row.empty:
        first = client_row['Nombre1'].iloc[0] if 'Nombre1' in client_row.columns else ""
        last = client_row['ApellidoPaterno'].iloc[0] if 'ApellidoPaterno' in client_row.columns else ""
        client_name = f"{first} {last}".strip()

    # Update session state
    if session_id:
        state = _get_session_state(session_id)
        if state:
            state.client_id = client_id_int
            session_manager.update(session_id, state)

    return {
        "found": True,
        "client_id": client_id_int,
        "client_name": client_name,
        "message": f"Found client: {client_name} (ID: {client_id_int})"
    }


@tool("list_client_vehicles")
def list_client_vehicles_tool(session_id: str = "") -> Dict[str, Any]:
    """List all vehicles for the currently identified client.

    Args:
        session_id: Current session ID

    Returns:
        Dict with list of vehicles (VIN, make, model, year) or error.
    """
    state = _get_session_state(session_id)
    if not state or not state.client_id:
        return {
            "error": "No client identified yet. Please identify the client first using their email, phone, or name.",
            "vehicles": []
        }

    all_data = data_handler.data
    unidades_df = all_data.get('unidades')

    if unidades_df is None or unidades_df.empty:
        return {"error": "Vehicle data not available", "vehicles": []}

    client_id_int = int(state.client_id)
    client_units = unidades_df[unidades_df['NumeroDeCliente'] == client_id_int]

    if client_units.empty:
        return {
            "message": "No vehicles found for this client",
            "vehicles": []
        }

    vehicles = client_units[['Vin', 'Marca', 'Modelo', 'Año', 'Matricula']].to_dict('records')

    # Format for display
    formatted_vehicles = []
    for v in vehicles:
        formatted_vehicles.append({
            "vin": v.get('Vin', ''),
            "make": v.get('Marca', ''),
            "model": v.get('Modelo', ''),
            "year": v.get('Año', ''),
            "license_plate": v.get('Matricula', '')
        })

    return {
        "vehicles": formatted_vehicles,
        "count": len(formatted_vehicles),
        "message": f"Found {len(formatted_vehicles)} vehicle(s) for this client"
    }


@tool("select_vehicle")
def select_vehicle_tool(
    vin: Optional[str] = None,
    license_plate: Optional[str] = None,
    session_id: str = ""
) -> Dict[str, Any]:
    """Select a vehicle for subsequent queries by VIN or license plate.

    Args:
        vin: Vehicle Identification Number (optional if license_plate provided)
        license_plate: License plate number (optional if VIN provided)
        session_id: Current session ID

    Returns:
        Dict with selected vehicle details or error.
    """
    state = _get_session_state(session_id)
    if not state or not state.client_id:
        return {"error": "No client identified yet. Please identify the client first."}

    all_data = data_handler.data
    unidades_df = all_data.get('unidades')

    if unidades_df is None:
        return {"error": "Vehicle data not available"}

    # Resolve license plate to VIN if needed
    if not vin and license_plate:
        unit = unidades_df[unidades_df['Matricula'].str.lower() == str(license_plate).lower()]
        if not unit.empty:
            vin = unit['Vin'].iloc[0]
        else:
            return {"error": f"No vehicle found with license plate: {license_plate}"}

    if not vin:
        return {"error": "Please provide either a VIN or license plate"}

    # Validate VIN exists and belongs to client
    vin_lower = str(vin).lower()
    unit_data = unidades_df[
        (unidades_df['Vin'].str.lower() == vin_lower) &
        (unidades_df['NumeroDeCliente'] == state.client_id)
    ]

    if unit_data.empty:
        return {"error": f"Vehicle with VIN {vin} not found or does not belong to this client"}

    # Update session
    state.selected_vin = vin
    session_manager.update(session_id, state)

    unit = unit_data.iloc[0]
    return {
        "selected": True,
        "vin": unit.get('Vin', ''),
        "make": unit.get('Marca', ''),
        "model": unit.get('Modelo', ''),
        "year": unit.get('Año', ''),
        "license_plate": unit.get('Matricula', ''),
        "mileage": unit.get('Kilometraje', 0),
        "message": f"Selected vehicle: {unit.get('Año', '')} {unit.get('Marca', '')} {unit.get('Modelo', '')}"
    }


@tool("get_vehicle_details")
def get_vehicle_details_tool(
    vin: Optional[str] = None,
    session_id: str = ""
) -> Dict[str, Any]:
    """Get detailed information about a specific vehicle.

    Args:
        vin: Vehicle Identification Number (uses selected vehicle if not provided)
        session_id: Current session ID

    Returns:
        Dict with all vehicle details.
    """
    state = _get_session_state(session_id)

    # Use session VIN if not provided
    if not vin and state:
        vin = state.selected_vin

    if not vin:
        return {"error": "No vehicle selected. Please select a vehicle first or provide a VIN."}

    all_data = data_handler.data
    unidades_df = all_data.get('unidades')

    if unidades_df is None:
        return {"error": "Vehicle data not available"}

    unit_data = unidades_df[unidades_df['Vin'].str.lower() == str(vin).lower()]

    if unit_data.empty:
        return {"error": f"Vehicle with VIN {vin} not found"}

    unit = unit_data.iloc[0].to_dict()

    return {
        "vin": unit.get('Vin', ''),
        "license_plate": unit.get('Matricula', ''),
        "make": unit.get('Marca', ''),
        "model": unit.get('Modelo', ''),
        "year": unit.get('Año', ''),
        "mileage": unit.get('Kilometraje', 0),
        "client_id": unit.get('NumeroDeCliente', '')
    }


@tool("get_service_history")
def get_service_history_tool(
    vin: Optional[str] = None,
    session_id: str = ""
) -> Dict[str, Any]:
    """Get the last 3 service visits for a vehicle.

    Args:
        vin: Vehicle Identification Number (uses selected vehicle if not provided)
        session_id: Current session ID

    Returns:
        Dict with service history including dates, operations, and prices.
    """
    state = _get_session_state(session_id)

    if not vin and state:
        vin = state.selected_vin

    if not vin:
        return {"error": "No vehicle selected. Please select a vehicle first or provide a VIN."}

    all_data = data_handler.data
    servicios_df = all_data.get('servicios')
    kits_df = all_data.get('kits')
    operaciones_df = all_data.get('operaciones')

    if any(df is None or df.empty for df in [servicios_df, kits_df, operaciones_df]):
        return {"error": "Service data not available", "services": []}

    # Get last 3 services for VIN
    unit_services = servicios_df[
        servicios_df['Vin'].str.lower() == str(vin).lower()
    ].sort_values(by='FechaCierre', ascending=False).head(3).copy()

    if unit_services.empty:
        return {"message": "No service history found for this vehicle", "services": []}

    # Prepare operations map
    try:
        operaciones_df_copy = operaciones_df.copy()
        operaciones_df_copy['IdOperacion'] = pd.to_numeric(operaciones_df_copy['IdOperacion'], errors='coerce')
        operaciones_df_copy.dropna(subset=['IdOperacion'], inplace=True)
        operaciones_df_copy['IdOperacion'] = operaciones_df_copy['IdOperacion'].astype(int)
        operaciones_map = operaciones_df_copy.set_index('IdOperacion')['DescripcionOperacion'].to_dict()
    except Exception:
        operaciones_map = {}

    # Prepare kits map
    try:
        kits_map = kits_df.set_index('CodigoKit')[['IdOperacion', 'IdOperacion2', 'PrecioKit']].to_dict('index')
    except Exception:
        kits_map = {}

    def get_kit_details(kit_code):
        kit_info = kits_map.get(kit_code)
        op1_desc = "N/A"
        op2_desc = "N/A"
        price = 0.0

        if kit_info:
            price = kit_info.get('PrecioKit', 0.0)
            op1_id = kit_info.get('IdOperacion')
            op2_id = kit_info.get('IdOperacion2')

            if pd.notna(op1_id):
                try:
                    op1_desc = operaciones_map.get(int(op1_id), f"Op ID {op1_id}")
                except (ValueError, TypeError):
                    pass

            if pd.notna(op2_id):
                try:
                    op2_desc = operaciones_map.get(int(op2_id), f"Op ID {op2_id}")
                except (ValueError, TypeError):
                    pass

        return op1_desc, op2_desc, price

    services = []
    for _, row in unit_services.iterrows():
        kit_code = row.get('UltimoServicio')
        op1, op2, price = get_kit_details(kit_code) if pd.notna(kit_code) else ("N/A", "N/A", 0.0)

        # Format dates
        open_date = row.get('FechaApertura')
        close_date = row.get('FechaCierre')
        open_str = open_date.strftime('%d-%m-%Y') if pd.notna(open_date) else "N/A"
        close_str = close_date.strftime('%d-%m-%Y') if pd.notna(close_date) else "N/A"

        services.append({
            "order_number": row.get('Orden', ''),
            "open_date": open_str,
            "close_date": close_str,
            "service_code": kit_code if pd.notna(kit_code) else "N/A",
            "operation_1": op1,
            "operation_2": op2,
            "price": f"${price:,.2f}" if price else "N/A"
        })

    return {
        "services": services,
        "count": len(services),
        "message": f"Found {len(services)} service record(s)"
    }


@tool("get_next_service_recommendation")
def get_next_service_recommendation_tool(
    vin: Optional[str] = None,
    session_id: str = ""
) -> Dict[str, Any]:
    """Recommend the next service based on vehicle mileage.

    Args:
        vin: Vehicle Identification Number (uses selected vehicle if not provided)
        session_id: Current session ID

    Returns:
        Dict with recommended service kit, price, and included operations.
    """
    state = _get_session_state(session_id)

    if not vin and state:
        vin = state.selected_vin

    if not vin:
        return {"error": "No vehicle selected. Please select a vehicle first or provide a VIN."}

    all_data = data_handler.data
    unidades_df = all_data.get('unidades')
    kits_df = all_data.get('kits')
    operaciones_df = all_data.get('operaciones')
    servicios_df = all_data.get('servicios')

    if any(df is None or df.empty for df in [unidades_df, kits_df, operaciones_df]):
        return {"error": "Required data not available"}

    # Get vehicle details
    unit_data = unidades_df[unidades_df['Vin'].str.lower() == str(vin).lower()]
    if unit_data.empty:
        return {"error": f"Vehicle with VIN {vin} not found"}

    unit = unit_data.iloc[0]
    current_mileage = unit.get('Kilometraje', 0)

    result = {
        "vin": vin,
        "make": unit.get('Marca', ''),
        "model": unit.get('Modelo', ''),
        "year": unit.get('Año', ''),
        "current_mileage": current_mileage
    }

    # Get last service date
    if servicios_df is not None:
        last_service = servicios_df[
            servicios_df['Vin'].str.lower() == str(vin).lower()
        ].sort_values(by='FechaCierre', ascending=False).head(1)

        if not last_service.empty:
            close_date = last_service['FechaCierre'].iloc[0]
            if pd.notna(close_date):
                result["last_service_date"] = close_date.strftime('%d-%m-%Y')

    if current_mileage <= 0:
        result["message"] = "Could not determine current mileage to recommend a service."
        return result

    # Calculate next 10k milestone
    next_milestone = ((current_mileage // 10000) + 1) * 10000
    next_milestone = min(next_milestone, 100000)  # Cap at 100k

    recommended_kit_code = f"SERV{next_milestone // 1000}"

    # Find the kit
    recommended_kit_row = kits_df[kits_df['CodigoKit'] == recommended_kit_code]

    if recommended_kit_row.empty:
        result["message"] = f"Specific information for the recommended service ({recommended_kit_code}) was not found."
        return result

    kit_data = recommended_kit_row.iloc[0]

    # Prepare operations map
    try:
        operaciones_df_copy = operaciones_df.copy()
        operaciones_df_copy['IdOperacion'] = pd.to_numeric(operaciones_df_copy['IdOperacion'], errors='coerce')
        operaciones_df_copy.dropna(subset=['IdOperacion'], inplace=True)
        operaciones_df_copy['IdOperacion'] = operaciones_df_copy['IdOperacion'].astype(int)
        operaciones_map = operaciones_df_copy.set_index('IdOperacion')['DescripcionOperacion'].to_dict()
    except Exception:
        operaciones_map = {}

    op1_desc = "N/A"
    op2_desc = "N/A"
    op1_id = kit_data.get('IdOperacion')
    op2_id = kit_data.get('IdOperacion2')

    if pd.notna(op1_id):
        try:
            op1_desc = operaciones_map.get(int(op1_id), f"Op ID {op1_id}")
        except (ValueError, TypeError):
            pass

    if pd.notna(op2_id):
        try:
            op2_desc = operaciones_map.get(int(op2_id), f"Op ID {op2_id}")
        except (ValueError, TypeError):
            pass

    price = kit_data.get('PrecioKit', 0.0)

    result["recommendation"] = {
        "service_code": recommended_kit_code,
        "description": kit_data.get('DescripcionKit', ''),
        "price": f"${price:,.2f}" if price else "N/A",
        "operation_1": op1_desc,
        "operation_2": op2_desc,
        "next_milestone_km": next_milestone
    }
    result["message"] = f"Based on your current mileage of {current_mileage:,} km, we recommend the {recommended_kit_code} service at {next_milestone:,} km."

    return result


@tool("get_promotions")
def get_promotions_tool(
    vin: Optional[str] = None,
    session_id: str = ""
) -> Dict[str, Any]:
    """Find active promotions for a specific vehicle.

    Args:
        vin: Vehicle Identification Number (uses selected vehicle if not provided)
        session_id: Current session ID

    Returns:
        Dict with list of active promotions.
    """
    state = _get_session_state(session_id)

    if not vin and state:
        vin = state.selected_vin

    if not vin:
        return {"error": "No vehicle selected. Please select a vehicle first or provide a VIN."}

    all_data = data_handler.data
    campanas_df = all_data.get('campanas')
    operaciones_df = all_data.get('operaciones')

    if campanas_df is None or campanas_df.empty:
        return {"message": "No promotions available", "promotions": []}

    vin_lower = str(vin).lower()

    # Filter campaigns for VIN
    valid_campanas = campanas_df.dropna(subset=['FechaInicio', 'FechaFin'])
    active_campanas = valid_campanas[
        valid_campanas['Vin'].str.lower() == vin_lower
    ].copy()

    if active_campanas.empty:
        return {"message": "No active promotions for this vehicle", "promotions": []}

    # Merge with operations
    if operaciones_df is not None and not operaciones_df.empty:
        try:
            active_campanas['IdOperacion'] = pd.to_numeric(active_campanas['IdOperacion'], errors='coerce')
            operaciones_df_copy = operaciones_df.copy()
            operaciones_df_copy['IdOperacion'] = pd.to_numeric(operaciones_df_copy['IdOperacion'], errors='coerce')

            active_campanas = pd.merge(
                active_campanas,
                operaciones_df_copy[['IdOperacion', 'DescripcionOperacion']],
                on='IdOperacion',
                how='left'
            )
            active_campanas['DescripcionOperacion'] = active_campanas['DescripcionOperacion'].fillna('Details not available')
        except Exception:
            active_campanas['DescripcionOperacion'] = 'Details not available'

    promotions = []
    for _, row in active_campanas.iterrows():
        end_date = row.get('FechaFin')
        end_str = end_date.strftime('%d-%m-%Y') if pd.notna(end_date) else "N/A"

        promotions.append({
            "code": row.get('CodigoCampana', ''),
            "description": row.get('DescripcionCampana', ''),
            "operation": row.get('DescripcionOperacion', 'N/A'),
            "valid_until": end_str
        })

    return {
        "promotions": promotions,
        "count": len(promotions),
        "message": f"Found {len(promotions)} promotion(s) for this vehicle"
    }


@tool("schedule_appointment")
def schedule_appointment_tool(
    service_type: str,
    date: Optional[str] = None,
    time: Optional[str] = None,
    vin: Optional[str] = None,
    session_id: str = ""
) -> Dict[str, Any]:
    """Request to schedule a service appointment.

    Args:
        service_type: Type of service requested (e.g., 'oil change', 'inspection')
        date: Preferred date (optional)
        time: Preferred time (optional)
        vin: Vehicle Identification Number (uses selected vehicle if not provided)
        session_id: Current session ID

    Returns:
        Dict with appointment confirmation or request for more info.
    """
    state = _get_session_state(session_id)

    if not vin and state:
        vin = state.selected_vin

    if not vin:
        return {"error": "No vehicle selected. Please select a vehicle first."}

    # Get vehicle details for context
    all_data = data_handler.data
    unidades_df = all_data.get('unidades')

    vehicle_info = ""
    if unidades_df is not None:
        unit_data = unidades_df[unidades_df['Vin'].str.lower() == str(vin).lower()]
        if not unit_data.empty:
            unit = unit_data.iloc[0]
            vehicle_info = f"{unit.get('Año', '')} {unit.get('Marca', '')} {unit.get('Modelo', '')} (VIN: {vin})"

    if not vehicle_info:
        vehicle_info = f"VIN: {vin}"

    result = {
        "vehicle": vehicle_info,
        "service_type": service_type,
        "status": "pending"
    }

    if date or time:
        # User provided date/time
        result["requested_date"] = date or "Not specified"
        result["requested_time"] = time or "Not specified"
        result["status"] = "confirmation_needed"
        result["message"] = f"Appointment request received for {service_type} on {date or 'a date to be confirmed'}{' at ' + time if time else ''}. We will check availability and confirm shortly."
    else:
        # Need more info
        result["status"] = "more_info_needed"
        result["message"] = f"To schedule your {service_type} appointment for your {vehicle_info}, please provide your preferred date and time (e.g., 'tomorrow morning' or 'April 15th at 10:00 AM')."

    return result


# Export all tools
APEX_TOOLS = [
    find_client_tool,
    list_client_vehicles_tool,
    select_vehicle_tool,
    get_vehicle_details_tool,
    get_service_history_tool,
    get_next_service_recommendation_tool,
    get_promotions_tool,
    schedule_appointment_tool,
]

__all__ = [
    "find_client_tool",
    "list_client_vehicles_tool",
    "select_vehicle_tool",
    "get_vehicle_details_tool",
    "get_service_history_tool",
    "get_next_service_recommendation_tool",
    "get_promotions_tool",
    "schedule_appointment_tool",
    "APEX_TOOLS",
]
