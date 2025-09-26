# Zapata Post-Sale Service Chatbot Documentation

## 1. Overview

This document describes the structure and functionality of the Zapata Post-Sale Service Chatbot. The chatbot is designed to interact with users in Spanish, understand their requests regarding vehicle service history, promotions, and appointments, retrieve relevant information from CSV data files, and generate natural language responses using a Large Language Model (LLM).

The chatbot follows a Retrieval-Augmented Generation (RAG) pattern:
1.  User input is parsed to identify intent and entities using an LLM (`llm_handler_chatbot.py`).
2.  Based on the intent, specific functions retrieve relevant data directly from pandas DataFrames loaded from CSV files (`actions_chatbot.py`, `data_handler_chatbot.py`).
3.  This retrieved, structured data is then formatted into a prompt.
4.  The LLM generates a user-facing response in Spanish based on the prompt and retrieved data (`llm_handler_chatbot.py`).
5.  Conversation history is maintained to provide context for intent parsing and response generation (`main_chatbot.py`).

## 2. Project Structure

```
/
├── main_chatbot.py             # Main application entry point & interaction loop
├── config_chatbot.py           # Configuration constants (paths, prompts, model names)
├── data_handler_chatbot.py     # Data loading from CSV files
├── llm_handler_chatbot.py      # LLM initialization, intent parsing, response generation
├── actions_chatbot.py          # Core logic functions (data retrieval, specific actions)
├── tables/                     # Directory containing CSV data files
│   ├── Cliente.csv
│   ├── Unidad.csv
│   ├── Servicios.csv
│   ├── Operacion.csv
│   ├── Campanas.csv
│   ├── Kits.csv
│   └── Materiales.csv
├── .env                        # Environment variables (API keys, etc.) - Not created by bot
└── requirements.txt            # Python package dependencies
└── DOCUMENTATION.md            # This file
```

## 3. File Descriptions

*   **`config_chatbot.py`**:
    *   Stores configuration constants like the path to the data tables (`TABLES_PATH`), the LLM model name (`LLM_MODEL_NAME`), and the initial persona prompt (`INITIAL_PROMPT`).
    *   Centralizes settings for easy modification.

*   **`data_handler_chatbot.py`**:
    *   Responsible for loading data from the CSV files located in the `tables/` directory.
    *   Contains the `load_data()` function which reads CSVs into pandas DataFrames, handling date parsing.

*   **`llm_handler_chatbot.py`**:
    *   Handles all direct interactions with the Large Language Model (LLM).
    *   Initializes the `ChatOpenAI` client using settings from `config_chatbot.py` and environment variables.
    *   Contains `generate_response_llm()` for generating natural language responses based on provided prompts.
    *   Contains `parse_intent_llm()` which uses the LLM to analyze user input and chat history, returning structured intent and entities in JSON format.

*   **`actions_chatbot.py`**:
    *   Contains the core business logic and data retrieval functions that interact with the loaded DataFrames.
    *   Functions like `find_client`, `get_client_units`, `get_service_history`, `get_promotions`, etc., perform direct lookups based on parameters provided by the main loop (derived from parsed intent/entities).
    *   Functions like `handle_appointment_request` or `get_next_service_recommendation_context` prepare context and call `generate_response_llm` from `llm_handler_chatbot.py` for tasks requiring LLM generation based on retrieved data.

*   **`main_chatbot.py`**:
    *   The main entry point for the application (`if __name__ == "__main__":`).
    *   Imports necessary functions from other modules.
    *   Loads data using `data_handler_chatbot.load_data()`.
    *   Contains the `run_chatbot_interaction()` function which manages the main conversation loop:
        *   Takes user input.
        *   Maintains `chat_history`.
        *   Calls `llm_handler_chatbot.parse_intent_llm()` to understand the user.
        *   Calls appropriate functions from `actions_chatbot.py` based on the parsed intent and entities.
        *   Calls `llm_handler_chatbot.generate_response_llm()` to formulate the final response in Spanish.
        *   Prints the chatbot's response.
        *   Manages basic conversation state (`client_id`, `selected_vin`).

*   **`tables/`**:
    *   Contains the source data in CSV format used by the chatbot.

*   **`.env`**:
    *   Stores sensitive information like API keys required by the LLM client. 

*   **`requirements.txt`**:
    *   Lists the necessary Python packages to run the chatbot (e.g., `pandas`, `langchain-openai`, `python-dotenv`, `generative-ai-hub-sdk`). Install using `pip install -r requirements.txt`.

## 4. Key Functions

*   **`data_handler_chatbot.load_data()`**: Loads CSV files into a dictionary of pandas DataFrames. Handles date parsing.
*   **`llm_handler_chatbot.generate_response_llm(prompt)`**: Takes a prompt string, adds the initial persona, sends it to the LLM, and returns the generated text response.
*   **`llm_handler_chatbot.parse_intent_llm(user_input, chat_history)`**: Takes the latest user input and conversation history, prompts the LLM to return a JSON string identifying the intent and extracted entities. Parses the JSON safely.
*   **`actions_chatbot.find_client(all_data, identifier_type, identifier_value)`**: Looks up `NumeroDeCliente` in the data based on provided identifiers (email, phone, VIN, matricula, name).
*   **`actions_chatbot.get_vin_from_matricula(all_data, matricula)`**: Looks up a VIN using a license plate (`Matricula`).
*   **`actions_chatbot.get_client_units(all_data, client_id)`**: Retrieves all vehicles associated with a client ID.
*   **`actions_chatbot.get_unit_details(all_data, vin)`**: Retrieves detailed information for a specific VIN.
*   **`actions_chatbot.get_service_history(all_data, vin)`**: Retrieves the last 3 service visit records (dates only) for a specific VIN.
*   **`actions_chatbot.get_next_service_recommendation_context(all_data, vin)`**: Gathers data (mileage, last service date) needed by the LLM to generate a service recommendation.
*   **`actions_chatbot.get_promotions(all_data, vin)`**: Retrieves currently active promotions for a specific VIN.
*   **`actions_chatbot.handle_appointment_request(all_data, vin, desired_service, fecha=None, hora=None)`**: Prepares a prompt (asking for time slot or confirming received request) and calls the LLM to handle appointment scheduling dialogue.
*   **`main_chatbot.run_chatbot_interaction(all_data)`**: Manages the main interactive loop, state, history, and orchestrates calls to parsing, action, and generation functions.

## 5. Running the Chatbot

1.  Ensure all dependencies are installed: `pip install -r requirements.txt`
2.  Create a `.env` file in the root directory and add your necessary environment variables.
3.  Make sure the `tables/` directory with the CSV files is present in the root directory.
4.  Run the main script from the root directory: `python main_chatbot.py`

The chatbot will start, display a welcome message, and prompt for user input ("Tú: "). Type 'salir' to exit.

## 6. Adding New Intents/Situations

To add functionality for handling a new type of user request (a new "intent" or "situation"), you need to modify the following files:

1.  **`llm_handler_chatbot.py`**:
    *   **Update `intents` list:** Add your new intent name (e.g., `"consultar_estado_reparacion"`) to the `intents` list within the `parse_intent_llm` function. This tells the LLM that this is a possible intent it can identify.
    *   **Update `entities_description`:** If your new intent requires specific information to be extracted from the user's message (like an order number or a specific part), define a new entity in the `entities_description` string within `parse_intent_llm`. For example: `- numero_orden (string): The service order number the user is asking about.`
    *   **Update Examples (Optional but Recommended):** Add an example of the expected JSON output for your new intent in the prompt within `parse_intent_llm` to help guide the LLM.

2.  **`actions_chatbot.py`**:
    *   **Create Action Function:** Define a new Python function that contains the logic for handling your new intent. This function will typically:
        *   Accept `all_data` and any necessary `entities` (extracted by the LLM) as arguments.
        *   Retrieve required data from the `all_data` DataFrames (e.g., look up the status based on `numero_orden`).
        *   Prepare a specific prompt containing the retrieved data (or indicating if data is missing).
        *   Call `generate_response_llm` (imported from `llm_handler_chatbot`) with the prepared prompt to get the final Spanish response for the user.
        *   Return the generated response string.

3.  **`main_chatbot.py`**:
    *   **Add `elif` block:** In the `run_chatbot_interaction` function, find the main `if/elif` block that handles different intents. Add a new `elif intent == "your_new_intent_name":` block.
    *   **Call Action Function:** Inside this new `elif` block, call the corresponding action function you created in `actions_chatbot.py`, passing `all_data` and the relevant `entities` dictionary. Assign the returned response to the `bot_response` variable.
    *   **Handle Dependencies:** Ensure your new action requires the correct state (e.g., does the user need to be identified first? Does a VIN need to be selected?). Add checks for `conversation_state["client_id"]` or `conversation_state["selected_vin"]` if necessary before calling your action function, providing appropriate feedback if the prerequisites aren't met.

By following these steps, you integrate the new capability into the chatbot's understanding (intent parsing), logic execution (action function), and conversation flow (main loop).
