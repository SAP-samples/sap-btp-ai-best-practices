import json
import pandas as pd
from datetime import datetime
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
import os
load_dotenv()

# Adjust imports for top-level structure
from config_chatbot import INITIAL_PROMPT
from data_handler_chatbot import load_data
from llm_handler_chatbot import llm, generate_response_llm, parse_intent_llm
from actions_chatbot import (
    find_client, get_vin_from_matricula, get_client_units, get_unit_details,
    get_service_history, get_next_service_recommendation_context,
    get_promotions, handle_appointment_request
)

# Explicitly set template and static folders relative to the script's location (resources/)
app = Flask(__name__, template_folder='templates', static_folder='static')

# Load data at application startup
# Assuming data files are accessible relative to data_handler_chatbot.py
all_data = load_data()

# Conversation history and state (simplified, in memory)
# Consider using Flask sessions for multi-user support in production
chat_history = []
conversation_state = {
    "client_id": None,
    "selected_vin": None,
    "last_intent": None,
}

def process_user_message(user_message, current_chat_history, current_conversation_state):
    """Processes the user message and returns the chatbot response."""
    if not user_message.strip():
        return "Please type a message.", current_chat_history, current_conversation_state 

    # Append user message to history BEFORE parsing intent
    current_chat_history.append({"role": "user", "content": user_message})

    # Parse intent using LLM
    # Pass a copy to avoid modifying the global history directly if parse_intent modifies it
    parsed_result = parse_intent_llm(user_message, list(current_chat_history))
    intent = parsed_result.get('intent', 'unknown') # Default to unknown
    entities = parsed_result.get('entities', {})
    print(f"DEBUG: Intent={intent}, Entities={entities}") # Debug output

    # --- Action Execution based on Intent ---
    bot_response = "Sorry, I didn't understand that. Could you rephrase?" # Default English response

    # --- Handle Client Identification ---
    if intent == "identify_client":
        identifier_type = None
        identifier_value = None
        # Using renamed entity keys from parse_intent_llm
        if 'email' in entities: identifier_type, identifier_value = 'Correo', entities['email'] # Keep internal DB column names
        elif 'phone' in entities: identifier_type, identifier_value = 'NumeroTelefonico', entities['phone']
        elif 'vin' in entities: identifier_type, identifier_value = 'Vin', entities['vin']
        elif 'license_plate' in entities: identifier_type, identifier_value = 'Matricula', entities['license_plate']
        elif 'full_name' in entities: identifier_type, identifier_value = 'NombreCompleto', entities['full_name']

        if identifier_type and identifier_value:
            client_id = find_client(all_data, identifier_type, identifier_value)
            if client_id:
                current_conversation_state["client_id"] = client_id
                current_conversation_state["selected_vin"] = None # Reset selected VIN on new identification

                prompt = f"You have identified the client with ID {client_id}. Greet them warmly (if not already done in the history) and ask how you can help them (e.g., check vehicles, history, promotions, schedule appointment)."
                bot_response = generate_response_llm(prompt)
            else:
                
                bot_response = f"I couldn't find any client with {identifier_type}: {identifier_value}. Could you please verify it or try another piece of information?"
        else:
             
             bot_response = "To help you better, I need to identify you. Could you please provide your email, phone number, VIN, or license plate?"

    # --- Handle Actions Requiring Client ID ---
    elif intent in ["list_units", "select_unit"] and not current_conversation_state["client_id"]:
         
         bot_response = "First, I need to know who you are to show you your vehicles. Could you provide your email, phone number, or VIN?"

    elif intent == "list_units":
        units = get_client_units(all_data, current_conversation_state["client_id"])
        if units:
            units_str = "\n".join([f"- VIN: {u['Vin']} ({u.get('Marca', '')} {u.get('Modelo', '')} {u.get('Año', '')})" for u in units])

            prompt = f"Client {current_conversation_state['client_id']} has these vehicles:\n{units_str}\n\nAsk them in English which vehicle (by VIN or license plate) they want to inquire about."
            bot_response = generate_response_llm(prompt)
        else:
            
            bot_response = "It seems there are no vehicles registered under your name."

    elif intent == "select_unit":
         vin_to_select = entities.get('vin')
         # Use renamed entity key 'license_plate'
         matricula_to_select = entities.get('license_plate') # Check for license_plate too

         if not vin_to_select and matricula_to_select:
             # Try to find VIN from matricula if VIN wasn't provided directly
             print(f"DEBUG: Attempting to find VIN for License Plate {matricula_to_select}")
             found_vin = get_vin_from_matricula(all_data, matricula_to_select) # Function still expects 'matricula' internally
             if found_vin:
                 vin_to_select = found_vin # Use the found VIN
                 print(f"DEBUG: Found VIN {vin_to_select} for License Plate {matricula_to_select}")
             else:
                 
                 bot_response = f"I couldn't find any vehicle with the license plate {matricula_to_select}. Could you please verify it or try with the VIN?"

         # Use the default English response for comparison
         if not vin_to_select and bot_response == "Sorry, I didn't understand that. Could you rephrase?":
             # If still no VIN after checking license plate, ask user to clarify
             
             bot_response = "I didn't understand which vehicle you want to select. Could you please provide the VIN or license plate?"

         elif vin_to_select: # Proceed if we have a VIN (either direct or from license plate)
             client_units = get_client_units(all_data, current_conversation_state["client_id"])
             valid_vins = [u['Vin'].upper() for u in client_units]
             if vin_to_select.upper() in valid_vins:
                 current_conversation_state["selected_vin"] = vin_to_select.upper()
                 unit_details = get_unit_details(all_data, current_conversation_state["selected_vin"])
                 details_str = f"{unit_details.get('Marca')} {unit_details.get('Modelo')} {unit_details.get('Año')}" if unit_details else current_conversation_state["selected_vin"]

                 prompt = f"The user selected the vehicle VIN: {current_conversation_state['selected_vin']} ({details_str}). Confirm the selection in English and ask what they need to know about it (details, history, promotions, next service, schedule appointment)."
                 bot_response = generate_response_llm(prompt)
             else:
                 
                 bot_response = f"The VIN {vin_to_select} doesn't seem to be associated with your account. Please select one from your list."

    # --- Handle Actions Requiring VIN (Selected or Provided) ---
    # Using renamed intents
    elif intent in ["get_details", "get_history", "get_recommendation", "get_promotions", "schedule_appointment"]:
        target_vin = None
        if 'vin' in entities: target_vin = entities['vin']
        # Use renamed entity key 'license_plate'
        elif 'license_plate' in entities:
            # Function still expects 'matricula' internally
            found_vin = get_vin_from_matricula(all_data, entities['license_plate'])
            if found_vin:
                target_vin = found_vin
                print(f"DEBUG: Found VIN {target_vin} for License Plate {entities['license_plate']}")
            else:
                
                bot_response = f"I couldn't find a vehicle with the license plate {entities['license_plate']}."
        elif current_conversation_state.get("selected_vin"): target_vin = current_conversation_state["selected_vin"]

        if not target_vin and bot_response == "Sorry, I didn't understand that. Could you rephrase?": # Only ask if no other error occurred
             
             bot_response = "Please specify which vehicle (VIN or license plate) your query is about, or select one from your list if I've already shown it to you."
        elif target_vin: # Proceed if we have a VIN
             vin_is_valid = True
             # Check if VIN belongs to client only if client is identified
             if current_conversation_state["client_id"]:
                 client_units = get_client_units(all_data, current_conversation_state["client_id"])
                 valid_vins = [u['Vin'].upper() for u in client_units]
                 if target_vin.upper() not in valid_vins:
                     
                     bot_response = f"The VIN {target_vin} is not associated with your account."
                     vin_is_valid = False

             if vin_is_valid:
                # Update selected VIN if a new one is targeted and valid
                if target_vin.upper() != current_conversation_state.get("selected_vin"):
                     current_conversation_state["selected_vin"] = target_vin.upper()
                     print(f"DEBUG: Updated selected VIN to {current_conversation_state['selected_vin']}")

                unit_details = get_unit_details(all_data, target_vin)
                details_str = f"{unit_details.get('Marca')} {unit_details.get('Modelo')} {unit_details.get('Año')}" if unit_details else target_vin

                if intent == "get_details":
                    if unit_details:
                        details_data_str = json.dumps(unit_details, indent=2, default=str)
                        # ted Prompt
                        prompt = f"Provide the user with the details of their vehicle (VIN: {target_vin}) in English, based on:\n{details_data_str}\nFormat the response clearly and friendly."
                        bot_response = generate_response_llm(prompt)
                    else:
                        
                        bot_response = f"Sorry, I couldn't find details for VIN {target_vin}."

                elif intent == "get_history":
                    history = get_service_history(all_data, target_vin)
                    if history:
                         history_formatted = []
                         for h in history:
                             try:
                                 # Keep date format or change as needed
                                 apertura_dt = pd.to_datetime(h.get('FechaApertura')).strftime('%d-%m-%Y') if pd.notna(h.get('FechaApertura')) else 'N/A'
                                 cierre_dt = pd.to_datetime(h.get('FechaCierre')).strftime('%d-%m-%Y') if pd.notna(h.get('FechaCierre')) else 'N/A'
                                 # Get operation details
                                 op1 = h.get('Operacion1', 'N/A')
                                 op2 = h.get('Operacion2', 'N/A')
                                 ops_list = [desc for desc in [op1, op2] if desc and desc != 'N/A' and 'not found' not in desc.lower() and 'invalid' not in desc.lower()] # Adjusted checks
                                 ops_str = f"Operations: {', '.join(ops_list)}" if ops_list else "Operations: N/A"

                                 # Get and format price
                                 price = h.get('PrecioKit', 0.0)
                                 price_str = f"${price:,.2f}" if price > 0 else "Price not available" # Format as currency

                                 # Append formatted string (English)
                                 history_formatted.append(f"- Order: {h.get('Orden', 'N/A')}, Opened: {apertura_dt}, Closed: {cierre_dt}, Service: {h.get('UltimoServicio', 'N/A')} ({ops_str}), Kit Price: {price_str}")
                             except Exception as fmt_e:
                                 print(f"Error formatting history item: {h}. Error: {fmt_e}")
                                 history_formatted.append(f"- Order: {h.get('Orden', 'N/A')}, Error formatting details.")
                         history_str = "\n".join(history_formatted)

                         # ted Prompt
                         prompt = f"""
The user asked for the history for VIN {target_vin} ({details_str}). The last recorded visits, including the main operations and associated kit price, are:
{history_str}

Present this information in English clearly and friendly.
Mention the main operations performed (if available and valid) and the kit price for each visit.
"""
                         bot_response = generate_response_llm(prompt)
                    else:
                        
                        bot_response = f"I couldn't find service history for VIN {target_vin}."

                elif intent == "get_recommendation":
                    context = get_next_service_recommendation_context(all_data, target_vin)
                    if context and 'message' not in context :
                        last_service_date_str = context.get('last_service_date').strftime('%d-%m-%Y') if context.get('last_service_date') else 'N/A'
                        # Basic vehicle context (English)
                        vehicle_context_str = f"Make: {context.get('make')}, Model: {context.get('model')}, Year: {context.get('year')}, Mileage: {context.get('current_mileage')} km, Last Service: {last_service_date_str}" # Keep km or change unit? Assuming km is okay.

                        # Detailed recommended kit info (English)
                        recommended_kit_info_str = "No specific recommended kit found."
                        if context.get('recommended_kit'):
                            kit = context['recommended_kit']
                            kit_desc = kit.get('description', 'Recommended Service')
                            kit_price = kit.get('price', 0.0)
                            # Removed 'pesos mexicanos', assuming generic currency format
                            price_str = f"${kit_price:,.2f} (estimated)" if kit_price > 0 else "Price not available"

                            # Get operation descriptions
                            op1 = kit.get('op1_desc', 'N/A')
                            op2 = kit.get('op2_desc', 'N/A')
                            ops_list = [desc for desc in [op1, op2] if desc and desc != 'N/A' and 'not found' not in desc.lower() and 'invalid' not in desc.lower()] # Adjusted checks
                            ops_str = f"Mainly includes: {', '.join(ops_list)}." if ops_list else ""

                            recommended_kit_info_str = f"Recommended Service: '{kit_desc}' (Code: {kit.get('code', 'N/A')})\nEstimated Price: {price_str}\n{ops_str}"

                        # Construct the final prompt for the LLM (English)
                        prompt = f"""
Vehicle Context:
{vehicle_context_str}

Recommended Service Information:
{recommended_kit_info_str}

Task:
Based on the vehicle context (especially mileage) and the recommended service information, generate a clear and friendly recommendation in English for the user.
- Explain why this service is recommended (based on mileage and typical 10,000 km intervals).
- Mention the name of the recommended service (e.g., '{kit.get('description', 'Recommended Service')}' if available).
- State the main operations included in this specific service.
- Clearly state the estimated price of the service.
"""
                        bot_response = generate_response_llm(prompt)
                    # Handle cases where context generation failed earlier (e.g., missing mileage)
                    elif context and 'message' in context:
                         bot_response = context['message'] # Use the error message from the context function
                    else: # General fallback
                         
                         bot_response = f"Sorry, I couldn't get information to recommend the next service for VIN {target_vin}."

                elif intent == "get_promotions":
                    promos = get_promotions(all_data, target_vin)
                    if promos:
                        promos_formatted = []
                        for p in promos:
                            try:
                                fecha_fin_dt = pd.to_datetime(p['FechaFin']).strftime('%d-%m-%Y') if pd.notna(p['FechaFin']) else 'N/A'

                                promos_formatted.append(f"- {p['DescripcionCampana']} ({p.get('DescripcionOperacion', 'N/A')}) - Valid until: {fecha_fin_dt}")
                            except Exception as fmt_e:
                                 print(f"Error formatting date for promotion: {fmt_e}")
                                 promos_formatted.append(f"- {p['DescripcionCampana']} ({p.get('DescripcionOperacion', 'N/A')}) - End date not available")
                        promos_str = "\n".join(promos_formatted)
                        # ted Prompt
                        prompt = f"The user asked for promotions for VIN {target_vin} ({details_str}). The active promotions are:\n{promos_str}\n\nPresent the list clearly in English."
                        bot_response = generate_response_llm(prompt)
                    else:
                        
                        bot_response = f"I couldn't find any active promotions for VIN {target_vin}."

                elif intent == "schedule_appointment":
                     # Use renamed entity keys
                     service_req = entities.get('service_type') or "service not specified"
                     fecha_req = entities.get('date')
                     hora_req = entities.get('time')
                     # handle_appointment_request now generates English responses
                     bot_response = handle_appointment_request(all_data, target_vin, service_req, fecha=fecha_req, hora=hora_req)

    # --- Handle General Intents ---
    # Using renamed intents
    elif intent == "greet":
         # ted Prompt
         bot_response = generate_response_llm("The user greeted you ('Hi', 'Good morning', etc.). Respond politely in English and ask how you can help them.")
    elif intent == "farewell":
         # ted Prompt
         bot_response = generate_response_llm("The user is saying goodbye ('Bye', 'See you', etc.). Say goodbye politely in English.")
         # In a web context, we don't 'break' the loop, just send the response.
    elif intent == "confirm":
         history_context_str = "\n".join([f"{msg['role']}: {msg['content']}" for msg in current_chat_history])
         # ted Prompt
         prompt = f"Context:\n{history_context_str}\n\nThe user confirmed ('Yes', 'Ok', 'Alright'). Respond appropriately in English based on the previous context. If the context is unclear, simply acknowledge receipt."
         bot_response = generate_response_llm(prompt)
    elif intent == "deny":
         history_context_str = "\n".join([f"{msg['role']}: {msg['content']}" for msg in current_chat_history])
         # ted Prompt
         prompt = f"Context:\n{history_context_str}\n\nThe user denied ('No'). Respond appropriately in English based on the previous context. If the context is unclear, simply acknowledge receipt and ask what they would like to do."
         bot_response = generate_response_llm(prompt)

    # --- Fallback for Unknown Intent ---
    # Check if the default English response is still set
    if bot_response == "Sorry, I didn't understand that. Could you rephrase?":
         history_context_str = "\n".join([f"{msg['role']}: {msg['content']}" for msg in current_chat_history])
         # Check if the intent was truly 'unknown' or if an action failed silently
         if intent == 'unknown':
             # ted Prompt
             prompt = f"Context:\n{history_context_str}\n\nThe intent of the user's last message could not be determined. Apologize in English and ask the user to rephrase their request."
         else: # An intent was recognized, but maybe failed to produce a specific response
             # ted Prompt
             prompt = f"Context:\n{history_context_str}\n\nThere was a problem processing your request regarding '{intent}'. Could you please try again or rephrase your request?"
         bot_response = generate_response_llm(prompt)

    # Append assistant response to history

    bot_response_content = bot_response if bot_response else "Sorry, I encountered a problem processing that."
    current_chat_history.append({"role": "assistant", "content": bot_response_content})
    current_conversation_state["last_intent"] = intent

    return bot_response_content, current_chat_history, current_conversation_state

@app.route("/", methods=["GET"])
def index():
    """Render the chat interface."""
    # Ensure the template is loaded from the correct folder specified in Flask app init
    return render_template("index.html") # Assume index.html is generic or will be updated separately

@app.route("/api/chat", methods=["POST"])
def chat():
    """Process chat messages from the frontend."""
    global chat_history, conversation_state # Use global state (simple approach)

    data = request.get_json()
    user_message = data.get("message", "").strip()

    if not user_message:
        return jsonify({"message": "Empty message."}), 400

    # Process the message using our chatbot logic
    # Pass the current global state and history
    bot_response, updated_history, updated_state = process_user_message(
        user_message, chat_history, conversation_state
    )

    # Update global state (important!)
    chat_history = updated_history
    conversation_state = updated_state

    return jsonify({"message": bot_response})

@app.route("/api/reset", methods=["POST"])
def reset_conversation():
    """Reset the conversation history and state."""
    global chat_history, conversation_state

    # Reset global state
    chat_history = []
    conversation_state = {
        "client_id": None,
        "selected_vin": None,
        "last_intent": None,
    }

    # Initial greeting using LLM 
    initial_greeting = generate_response_llm("Start the conversation by greeting the user in English and asking how you can help them today. Introduce yourself as Apex Assistant. Ask them to identify themselves with their Full Name, email address, or phone number.")
    chat_history.append({"role": "assistant", "content": initial_greeting})

    return jsonify({"message": initial_greeting})

# --- Main Execution ---
if __name__ == "__main__":
    if all_data and llm: # Check if data loaded AND llm initialized
        print("Data and LLM loaded successfully. Starting web server...")
        # Initial greeting for chat history (only if history is empty)
        if not chat_history:
             initial_greeting = generate_response_llm("Start the conversation by greeting the user in English and asking how you can help them today. Introduce yourself as Apex Assistant. Ask them to identify themselves with their Full Name, email address, or phone number.")
             chat_history.append({"role": "assistant", "content": initial_greeting})
        # Run Flask app
        # Use 0.0.0.0 to make it accessible on the network if needed
        # debug=True removed for production readiness (though Gunicorn bypasses this)
        app.run(host='0.0.0.0', port=5001)
    elif not all_data:
        print("\nError loading data. The chatbot cannot start.") 
    else: # llm failed to initialize
        print("\nError initializing LLM. The chatbot cannot start.") 