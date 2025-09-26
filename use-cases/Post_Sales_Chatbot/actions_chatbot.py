import pandas as pd
import json
from datetime import datetime
# Adjust import for top-level structure
from llm_handler_chatbot import generate_response_llm

# --- Data Retrieval/Core Logic Functions (RAG Pattern) ---

def find_client(all_data, identifier_type, identifier_value):
    """Finds client ID directly from DataFrame based on identifier."""
    clientes_df = all_data.get('clientes')
    unidades_df = all_data.get('unidades')
    if clientes_df is None or identifier_value is None: return None

    client_id = None
    value_str = str(identifier_value) # Ensure value is string for comparison
    value_lower = value_str.lower()

    # Using internal DB column names ('Vin', 'Matricula', etc.)
    if identifier_type == 'Vin' and unidades_df is not None:
        unit = unidades_df[unidades_df['Vin'].str.lower() == value_lower]
        if not unit.empty: client_id = unit['NumeroDeCliente'].iloc[0]
    elif identifier_type == 'Matricula' and unidades_df is not None:
        unit = unidades_df[unidades_df['Matricula'].str.lower() == value_lower]
        if not unit.empty: client_id = unit['NumeroDeCliente'].iloc[0]
    elif identifier_type in clientes_df.columns:
        # This section handles direct lookups in the 'clientes' table
        # Use internal DB column names like 'Correo', 'NumeroTelefonico'
        if identifier_type not in clientes_df.columns: return None # Should not happen if check passes, but safe

        if pd.api.types.is_string_dtype(clientes_df[identifier_type]):
            client = clientes_df[clientes_df[identifier_type].str.lower() == value_lower]
        else:
             try:
                 # Attempt conversion to the column's specific type
                 col_type = clientes_df[identifier_type].dtype
                 client = clientes_df[clientes_df[identifier_type] == col_type.type(identifier_value)]
             except Exception as e:
                 
                 print(f"Warning: Could not convert identifier '{identifier_value}' to type {col_type} for column {identifier_type}. Error: {e}")
                 client = pd.DataFrame() # Ensure client is empty DataFrame on error
        if not client.empty: client_id = client['NumeroDeCliente'].iloc[0]

    elif identifier_type == 'NombreCompleto':
        # Assumes 'NombreCompleto' was derived from first/last names
        parts = value_str.split()
        if len(parts) >= 2:
            first_name = parts[0].lower()
            last_name = parts[-1].lower()
            # Uses internal DB column names 'Nombre1', 'ApellidoPaterno'
            client = clientes_df[
                (clientes_df['Nombre1'].str.lower() == first_name) &
                (clientes_df['ApellidoPaterno'].str.lower() == last_name)
            ]
            if not client.empty: client_id = client['NumeroDeCliente'].iloc[0]

    # Convert to int only if not NaN/None
    return int(client_id) if pd.notna(client_id) else None

def get_vin_from_matricula(all_data, matricula):
    """Looks up VIN based on Matricula (License Plate)."""
    unidades_df = all_data.get('unidades')
    # Uses internal DB column name 'Matricula'
    if unidades_df is None or unidades_df.empty or matricula is None: return None

    unit = unidades_df[unidades_df['Matricula'].str.lower() == str(matricula).lower()]
    # Returns VIN using internal column name 'Vin'
    return unit['Vin'].iloc[0] if not unit.empty else None

def get_client_units(all_data, client_id):
    """Retrieves units (vehicles) for a given client ID directly from DataFrame."""
    unidades_df = all_data.get('unidades')
    # Uses internal DB column name 'NumeroDeCliente'
    if unidades_df is None or unidades_df.empty or client_id is None: return []
    try:
        client_id_int = int(client_id)
        client_units = unidades_df[unidades_df['NumeroDeCliente'] == client_id_int]
        # Returns specific columns, assuming these names are consistent ('Marca', 'Modelo', 'Año')
        return client_units[['Vin', 'Marca', 'Modelo', 'Año']].to_dict('records')
    except (ValueError, TypeError):
         
         print(f"Invalid client_id type for unit lookup: {client_id}")
         return []

def get_unit_details(all_data, vin):
    """Retrieves details for a specific VIN directly from the DataFrame."""
    unidades_df = all_data.get('unidades')
    # Uses internal column name 'Vin'
    if unidades_df is None or unidades_df.empty or vin is None: return None
    unit_data = unidades_df[unidades_df['Vin'].str.lower() == str(vin).lower()]
    # Converts the found row to a dictionary
    return unit_data.iloc[0].to_dict() if not unit_data.empty else None

def get_service_history(all_data, vin):
    """Retrieves the last 3 service visits, including kit and operation details."""
    servicios_df = all_data.get('servicios')
    kits_df = all_data.get('kits')
    operaciones_df = all_data.get('operaciones')

    
    if any(df is None or df.empty for df in [servicios_df, kits_df, operaciones_df]) or vin is None:
        print("Warning: Missing dataframes (servicios, kits, or operaciones) for service history lookup.")
        return []

    # Get the last 3 services for the VIN
    # Uses internal column names 'Vin', 'FechaCierre'
    unit_services = servicios_df[servicios_df['Vin'].str.lower() == str(vin).lower()].sort_values(by='FechaCierre', ascending=False).head(3).copy() # Use .copy() to avoid SettingWithCopyWarning

    if unit_services.empty:
        return []

    # Ensure date columns are datetime objects
    # Uses internal column names 'FechaApertura', 'FechaCierre'
    for col in ['FechaApertura', 'FechaCierre']:
        if col in unit_services.columns and not pd.api.types.is_datetime64_any_dtype(unit_services[col]):
            unit_services[col] = pd.to_datetime(unit_services[col], errors='coerce')

    # Prepare operations data for merging (ensure IdOperacion is numeric for lookup)
    try:
        # Uses internal column names 'IdOperacion', 'DescripcionOperacion'
        operaciones_df['IdOperacion'] = pd.to_numeric(operaciones_df['IdOperacion'], errors='coerce')
        operaciones_df.dropna(subset=['IdOperacion'], inplace=True) # Drop rows where conversion failed
        operaciones_df['IdOperacion'] = operaciones_df['IdOperacion'].astype(int) # Convert to int
        operaciones_map = operaciones_df.set_index('IdOperacion')['DescripcionOperacion'].to_dict()
    except Exception as e:
        
        print(f"Error preparing operations map: {e}")
        operaciones_map = {} # Use empty map if error

    # Prepare kits data for merging, including PrecioKit
    try:
        # Uses internal column names 'CodigoKit', 'IdOperacion', 'IdOperacion2', 'PrecioKit'
        # PrecioKit should be numeric from data_handler cleaning
        kits_map = kits_df.set_index('CodigoKit')[['IdOperacion', 'IdOperacion2', 'PrecioKit']].to_dict('index')
    except Exception as e:
        
        print(f"Error preparing kits map: {e}")
        kits_map = {} # Use empty map if error

    # Function to get operation descriptions and price from kit code
    def get_kit_details(kit_code):
        kit_info = kits_map.get(kit_code)
        op1_desc = "N/A"
        op2_desc = "N/A"
        price = 0.0 # Default price

        if kit_info:
            price = kit_info.get('PrecioKit', 0.0) # Get cleaned price
            op1_id = kit_info.get('IdOperacion')
            op2_id = kit_info.get('IdOperacion2')

            # Safely lookup operation 1 description
            try:
                if pd.notna(op1_id):
                    op1_id_int = int(op1_id) # Convert to int for lookup
                    op1_desc = operaciones_map.get(op1_id_int, f"Op ID {op1_id_int} not found")
            except (ValueError, TypeError):
                op1_desc = f"Invalid Op1 ID ({op1_id})" 

            # Safely lookup operation 2 description
            try:
                if pd.notna(op2_id):
                    op2_id_int = int(op2_id) # Convert to int for lookup
                    op2_desc = operaciones_map.get(op2_id_int, f"Op ID {op2_id_int} not found")
            except (ValueError, TypeError):
                 op2_desc = f"Invalid Op2 ID ({op2_id})" 

        return op1_desc, op2_desc, price

    # Apply the function to get details
    # Uses internal column name 'UltimoServicio'
    kit_details = unit_services['UltimoServicio'].apply(lambda x: get_kit_details(x) if pd.notna(x) else ("N/A", "N/A", 0.0))
    unit_services['Operacion1'] = kit_details.apply(lambda x: x[0])
    unit_services['Operacion2'] = kit_details.apply(lambda x: x[1])
    unit_services['PrecioKit'] = kit_details.apply(lambda x: x[2]) # Add price column

    # Select and return the relevant columns, including PrecioKit
    # Uses internal column names
    result_columns = ['Orden', 'FechaApertura', 'FechaCierre', 'UltimoServicio', 'Operacion1', 'Operacion2', 'PrecioKit']
    # Ensure all columns exist before selecting
    final_columns = [col for col in result_columns if col in unit_services.columns]

    # Returns dictionary records using the internal column names
    return unit_services[final_columns].to_dict('records')


def get_next_service_recommendation_context(all_data, vin):
    """Gathers context data (mileage, last service) and determines the specific next recommended service kit details."""
    servicios_df = all_data.get('servicios')
    unidades_df = all_data.get('unidades') # Kilometraje is cleaned here
    kits_df = all_data.get('kits') # PrecioKit is cleaned here
    operaciones_df = all_data.get('operaciones') # For op descriptions

    
    if any(df is None or df.empty for df in [servicios_df, unidades_df, kits_df, operaciones_df]) or vin is None:
         print("Warning: Missing dataframes (servicios, unidades, kits, or operaciones) for next service recommendation context.")
         return None

    unit_details = get_unit_details(all_data, vin) # Gets cleaned Kilometraje
    if not unit_details: return {'vin': vin, 'message': 'Vehicle details not found.'}

    context = {'vin': vin}
    # Uses internal column names 'Marca', 'Modelo', 'Año', 'Kilometraje'
    context['make'] = unit_details.get('Marca')
    context['model'] = unit_details.get('Modelo')
    context['year'] = unit_details.get('Año')
    current_mileage = unit_details.get('Kilometraje', 0) # Default to 0 if missing
    context['current_mileage'] = current_mileage

    # Uses internal column names 'Vin', 'FechaCierre'
    last_service = servicios_df[servicios_df['Vin'].str.lower() == str(vin).lower()].sort_values(by='FechaCierre', ascending=False).head(1)
    if not last_service.empty and 'FechaCierre' in last_service.columns and pd.api.types.is_datetime64_any_dtype(last_service['FechaCierre']):
         context['last_service_date'] = last_service['FechaCierre'].iloc[0]
    else:
         context['last_service_date'] = None

    # --- Determine Next Service Milestone and Find Specific Kit ---
    recommended_kit_details = None
    if current_mileage > 0:
        # Calculate next 10k milestone
        next_milestone = ((current_mileage // 10000) + 1) * 10000
        # Handle potential edge case for very high mileage if needed, assume max kit is 100k for now
        next_milestone = min(next_milestone, 100000) # Cap at 100k based on observed kit codes

        recommended_kit_code = f"SERV{next_milestone // 1000}" # Generate code like SERV10, SERV20, SERV100
        print(f"DEBUG: Calculated next milestone: {next_milestone}km, Recommended Kit Code: {recommended_kit_code}")

        # Find the specific kit in the kits dataframe
        # Uses internal column name 'CodigoKit'
        recommended_kit_row = kits_df[kits_df['CodigoKit'] == recommended_kit_code]

        if not recommended_kit_row.empty:
            kit_data = recommended_kit_row.iloc[0]

            # Prepare operations map (similar to get_service_history)
            try:
                # Uses internal column names 'IdOperacion', 'DescripcionOperacion'
                operaciones_df['IdOperacion'] = pd.to_numeric(operaciones_df['IdOperacion'], errors='coerce')
                operaciones_df.dropna(subset=['IdOperacion'], inplace=True)
                operaciones_df['IdOperacion'] = operaciones_df['IdOperacion'].astype(int)
                operaciones_map = operaciones_df.set_index('IdOperacion')['DescripcionOperacion'].to_dict()
            except Exception as e:
                
                print(f"Error preparing operations map for recommendation: {e}")
                operaciones_map = {}

            # Get operation descriptions for the recommended kit
            op1_desc = "N/A"
            op2_desc = "N/A"
            # Uses internal column names 'IdOperacion', 'IdOperacion2'
            op1_id = kit_data.get('IdOperacion')
            op2_id = kit_data.get('IdOperacion2')

            try:
                if pd.notna(op1_id):
                    op1_id_int = int(op1_id)
                    op1_desc = operaciones_map.get(op1_id_int, f"Op ID {op1_id_int} not found") 
            except (ValueError, TypeError): op1_desc = f"Invalid Op1 ID ({op1_id})" 

            try:
                if pd.notna(op2_id):
                    op2_id_int = int(op2_id)
                    op2_desc = operaciones_map.get(op2_id_int, f"Op ID {op2_id_int} not found") 
            except (ValueError, TypeError): op2_desc = f"Invalid Op2 ID ({op2_id})" 

            recommended_kit_details = {
                'code': recommended_kit_code,
                'description': kit_data.get('DescripcionKit'), # Internal column name
                'price': kit_data.get('PrecioKit', 0.0), # Use cleaned price
                'op1_desc': op1_desc,
                'op2_desc': op2_desc
            }
            context['recommended_kit'] = recommended_kit_details # Add specific kit details to context
        else:

            print(f"Warning: Recommended kit code {recommended_kit_code} not found in kits data.")
            context['message'] = f"Specific information for the recommended service ({recommended_kit_code}) was not found."

    else: # If mileage is 0 or missing
        context['message'] = "Could not determine current mileage to recommend a service."


    return context

def get_promotions(all_data, vin):
    """Finds currently active promotions for a specific VIN, using cleaned operations data."""
    campanas_df = all_data.get('campanas') # Dates parsed in load_data
    operaciones_df = all_data.get('operaciones') # Duplicates removed in load_data

    
    if any(df is None or df.empty for df in [campanas_df, operaciones_df]) or vin is None:
        print("Warning: Missing dataframes (campanas or operaciones) for promotions lookup.")
        return []

    today = pd.Timestamp.now().normalize()
    vin_lower = str(vin).lower()

    # Dates should be datetime objects from load_data
    # Handle potential NaT in date columns after coercion during loading
    # Uses internal column names 'FechaInicio', 'FechaFin'
    valid_campanas = campanas_df.dropna(subset=['FechaInicio', 'FechaFin'])

    # Filter active campaigns for the VIN
    # Uses internal column name 'Vin'
    # Enable date filtering (using internal column names) if desired
    # active_campanas = valid_campanas[
    #     (valid_campanas['Vin'].str.lower() == vin_lower) &
    #     (valid_campanas['FechaInicio'] <= today) &
    #     (valid_campanas['FechaFin'] >= today)
    # ].copy() # Use .copy()
    active_campanas = valid_campanas[
        valid_campanas['Vin'].str.lower() == vin_lower
    ].copy()

    if active_campanas.empty:
        return [] # No active campaigns for this VIN

    # Merge with cleaned operations data
    # Ensure IdOperacion types match for merging
    try:
        # Attempt conversion just in case, coercing errors
        # Uses internal column name 'IdOperacion'
        active_campanas['IdOperacion'] = pd.to_numeric(active_campanas['IdOperacion'], errors='coerce')
        operaciones_df['IdOperacion'] = pd.to_numeric(operaciones_df['IdOperacion'], errors='coerce')

        # Drop rows where IdOperacion became NaN after coercion, if any
        active_campanas.dropna(subset=['IdOperacion'], inplace=True)
        # No need to drop from operaciones_df as it's the lookup table

        if not active_campanas.empty:
             # Perform the merge only if active_campanas is not empty after potential drops
             active_campanas = pd.merge(
                 active_campanas,
                 # Uses internal column names 'IdOperacion', 'DescripcionOperacion'
                 operaciones_df[['IdOperacion', 'DescripcionOperacion']],
                 on='IdOperacion',
                 how='left'
             )
             # Fill description for campaigns whose operation ID wasn't found in the cleaned operaciones table
             active_campanas['DescripcionOperacion'] = active_campanas['DescripcionOperacion'].fillna('Details not available') 
        else:
             # If all campaigns had invalid IdOperacion, return empty
             return []

    except KeyError as e:
         
         print(f"KeyError during promotion merge preparation: {e}. Check column names.")
         # Add the column if merge failed due to missing key, to prevent downstream errors
         if 'DescripcionOperacion' not in active_campanas.columns:
              active_campanas['DescripcionOperacion'] = 'Lookup Error' 
    except Exception as e:
         
         print(f"Unexpected error during promotion merge: {e}")
         if 'DescripcionOperacion' not in active_campanas.columns:
              active_campanas['DescripcionOperacion'] = 'Lookup Error' 


    # Select final columns, ensuring they exist
    # Uses internal column names 'CodigoCampana', 'DescripcionCampana', 'FechaFin', 'DescripcionOperacion'
    final_columns = ['CodigoCampana', 'DescripcionCampana', 'FechaFin']
    if 'DescripcionOperacion' in active_campanas.columns:
        final_columns.insert(2, 'DescripcionOperacion')

    # Filter columns to only those that actually exist in the dataframe before returning
    existing_final_columns = [col for col in final_columns if col in active_campanas.columns]

    # Returns dictionary records using the internal column names
    return active_campanas[existing_final_columns].to_dict('records')


def handle_appointment_request(all_data, vin, desired_service, fecha=None, hora=None):
    """
    Prepares context and prompt for LLM to handle appointment scheduling (English).
    Adjusts prompt based on whether date/time were provided.
    """
    unit_details = get_unit_details(all_data, vin)
    # Uses internal column names 'Año', 'Marca', 'Modelo', 'Vin'
    context_str = f"Vehicle: {unit_details.get('Año')} {unit_details.get('Marca')} {unit_details.get('Modelo')} (VIN: {vin})" if unit_details else f"VIN: {vin}"

    if fecha or hora:
        # User provided date/time details
        requested_slot = f"for {fecha}" if fecha else ""
        requested_slot += f" at {hora}" if hora else ""
        requested_slot = requested_slot.strip() if requested_slot else "at a specific date/time"

        prompt = f"""
        The user wants to schedule an appointment for their vehicle ({context_str}).
        Requested Service: "{desired_service}"
        Requested Date/Time: "{requested_slot}"

        Confirm receipt of the request for {requested_slot}.
        Indicate that you will check availability and will follow up, or that additional confirmation is needed (simulate this process, you don't have access to a real calendar).
        Keep the response conversational and helpful in English.
        """
    else:
        # User did not provide date/time details, ask for them 
        prompt = f"""
        The user wants to schedule an appointment for their vehicle ({context_str}).
        Requested Service: "{desired_service}"
        Guide the user in English to select a date and time. Ask for their preferred day or time slot (e.g., morning/afternoon).
        Keep the response conversational and helpful.
        """

    # Call LLM with the generated English prompt
    return generate_response_llm(prompt)