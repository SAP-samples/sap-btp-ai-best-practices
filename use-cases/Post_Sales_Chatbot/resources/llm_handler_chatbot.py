import json
from gen_ai_hub.proxy.core.proxy_clients import get_proxy_client
from gen_ai_hub.proxy.langchain.openai import ChatOpenAI
# Adjust import for top-level structure
from config_chatbot import INITIAL_PROMPT, LLM_MODEL_NAME

# --- LLM Initialization ---
try:
    # Initialize the proxy client first
    proxy_client = get_proxy_client('gen-ai-hub')
    # Initialize ChatOpenAI using the proxy client and model name from config
    llm = ChatOpenAI(proxy_model_name=LLM_MODEL_NAME, temperature=0.0, proxy_client=proxy_client)
    print("ChatOpenAI LLM initialized successfully via llm_handler.")
except Exception as e:
    print(f"Error initializing LLM in llm_handler: {e}")
    llm = None

# --- LLM Generation Function ---
def generate_response_llm(prompt):
    """Generates a response using the initialized LLM."""
    if llm is None:
        return "Error: LLM not initialized." 
    try:
        # print(f"\n--- Sending Prompt to LLM ---\n{prompt}\n--------------------------") # Keep for debug if needed
        # Add the initial persona prompt to every generation call
        full_prompt = INITIAL_PROMPT + "\n\n" + prompt
        response = llm.invoke(full_prompt)
        content = response.content
        # print(f"--- LLM Response ---\n{content}\n---------------------") # Keep for debug if needed
        return content
    except Exception as e:
        print(f"Error during LLM invocation: {e}")
        return f"Sorry, an error occurred while trying to generate a response. ({e})" 

# --- Intent Parsing Function ---
def parse_intent_llm(user_input, chat_history):
    """Uses LLM to parse intent and entities from user input (English), considering history."""
    history_str = "\n".join([f"{msg['role']}: {msg['content']}" for msg in chat_history])

    intents = [
        "identify_client", "list_units", "select_unit",
        "get_details", "get_history", "get_recommendation",
        "get_promotions", "schedule_appointment", "greet", "farewell",
        "confirm", "deny", "unknown" 
    ]
    entities_description = """
    - email (string)
    - phone (string)
    - vin (string)
    - license_plate (string): License plate number).
    - full_name (string): User's full name .
    - service_type (string): Description of the service requested (e.g., 'oil change', 'inspection').
    - date (string): Requested date for an appointment (e.g., 'tomorrow', 'April 10th', 'next Monday').
    - time (string): Requested time for an appointment (e.g., 'at 10:00', 'in the afternoon').
    """

    prompt = f"""
    Analyze the following conversation and the user's last message to determine the intent and extract relevant entities.
    The conversation is in English.

    Possible Intents: {', '.join(intents)}
    Possible Entities to Extract:
    {entities_description}

    Conversation History:
    {history_str}

    User's Last Message: "{user_input}"

    Respond ONLY with a JSON object containing "intent" and "entities" keys.
    Identification Example: {{"intent": "identify_client", "entities": {{"email": "example@email.com"}}}}
    History Example: {{"intent": "get_history", "entities": {{"vin": "VIN123XYZ"}}}}
    Appointment Example with Date/Time: {{"intent": "schedule_appointment", "entities": {{"service_type": "inspection", "date": "April 10th", "time": "10:00"}}}}
    Appointment Example without Date/Time: {{"intent": "schedule_appointment", "entities": {{"service_type": "oil change"}}}}
    If misunderstood: {{"intent": "unknown", "entities": {{}}}}

    Your JSON response:
    """

    # Intent parsing might not need the full persona, but use the main generator for now
    llm_response_str = generate_response_llm(prompt)

    try:
        # Attempt to find JSON within the response if it's not pure JSON
        json_match = None
        try:
            parsed_json = json.loads(llm_response_str.strip())
            json_match = parsed_json
        except json.JSONDecodeError:
            start_index = llm_response_str.find('{')
            end_index = llm_response_str.rfind('}')
            if start_index != -1 and end_index != -1 and end_index > start_index:
                json_substring = llm_response_str[start_index:end_index+1]
                try:
                    parsed_json = json.loads(json_substring)
                    json_match = parsed_json
                except json.JSONDecodeError:
                    print(f"Could not find valid JSON within LLM response for intent: {llm_response_str}")
            else:
                 print(f"Could not find JSON markers in LLM response for intent: {llm_response_str}")

        if json_match and isinstance(json_match, dict) and "intent" in json_match and "entities" in json_match and isinstance(json_match["entities"], dict):
             if json_match["intent"] not in intents:
                 print(f"Warning: LLM returned unknown intent '{json_match['intent']}'. Defaulting to 'unknown'.")
                 json_match["intent"] = "unknown" # Default to 'unknown'
             # Rename entities for consistency before returning
             renamed_entities = {}
             for key, value in json_match["entities"].items():
                 if key == "telefono":
                     renamed_entities["phone"] = value
                 elif key == "matricula":
                     renamed_entities["license_plate"] = value
                 elif key == "nombre_completo":
                     renamed_entities["full_name"] = value
                 elif key == "tipo_servicio":
                     renamed_entities["service_type"] = value
                 elif key == "fecha":
                     renamed_entities["date"] = value
                 elif key == "hora":
                     renamed_entities["time"] = value
                 else: # Keep other entities like email, vin
                     renamed_entities[key] = value
             json_match["entities"] = renamed_entities
             return json_match
        else:
            print(f"Warning: LLM response for intent parsing was not valid JSON structure or missing keys: {llm_response_str}")
            return {"intent": "unknown", "entities": {}} # Default to 'unknown'

    except Exception as e:
        print(f"Unexpected error parsing LLM intent response: {e}")
        return {"intent": "unknown", "entities": {}} # Default to 'unknown'