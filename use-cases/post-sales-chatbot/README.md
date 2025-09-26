# Apex Automotive Services Post-Sale Service Chatbot

## 1. Overview

This document describes the structure and functionality of the Apex Automotive Services Post-Sale Service Chatbot. The chatbot is designed to interact with users via a web interface, understand their requests regarding vehicle service history, promotions, and appointments, retrieve relevant information from CSV data files, and generate natural language responses using a Large Language Model (LLM) accessed via SAP AI Core.

The application uses Flask to provide the web UI and Gunicorn as the WSGI server for deployment, suitable for platforms like Cloud Foundry.

The chatbot follows a Retrieval-Augmented Generation (RAG) pattern:
1.  User input from the web interface is sent to the Flask backend.
2.  The input is parsed to identify intent and entities using an LLM (`llm_handler_chatbot.py`).
3.  Based on the intent, specific functions retrieve relevant data directly from pandas DataFrames loaded from CSV files (`actions_chatbot.py`, `data_handler_chatbot.py`).
4.  This retrieved, structured data is then formatted into a prompt.
5.  The LLM generates a user-facing response based on the prompt and retrieved data (`llm_handler_chatbot.py`).
6.  The response is sent back to the web interface for display, with Markdown formatting rendered in the browser.
7.  Conversation history is maintained server-side (in memory) to provide context for intent parsing and response generation (`main_ui.py`).

## 2. Project Structure

```
/Post_Sales_Chatbot/  # Root Folder
├── main_ui.py                # Flask application entry point & web routes
├── main_chatbot.py           # Original terminal-based chatbot (for reference)
├── config_chatbot.py         # Configuration constants (paths, prompts, model names)
├── data_handler_chatbot.py   # Data loading and cleaning from CSV files
├── llm_handler_chatbot.py    # LLM initialization, intent parsing, response generation
├── actions_chatbot.py        # Core logic functions (data retrieval, specific actions)
├── new_tables/               # Directory containing source CSV data files
│   ├── Cliente.csv
│   ├── Unidad.csv
│   ├── Servicios.csv
│   ├── Operacion.csv
│   ├── Campanas.csv
│   ├── Kits.csv
│   └── Materiales.csv
├── templates/                # HTML templates for Flask UI
│   └── index.html
├── .env.copy                 # Environment variables for LOCAL development (API keys, etc.)
├── requirements.txt          # Python package dependencies
├── manifest.yml              # Cloud Foundry deployment manifest
├── DOCUMENTATION.md          # Additional documentation
└── README.md                 # This file (Project Root README)
```
*(Note: Data files are located in `new_tables/` directory)*

## 3. File Descriptions

*   **`main_ui.py`**:
    *   The main Flask application script.
    *   Defines web routes (`/`, `/api/chat`, `/api/reset`).
    *   Handles requests from the frontend, calls the chatbot logic (`process_user_message`), and returns JSON responses.
    *   Manages conversation state and history (in memory for simplicity).
    *   Loads data and initializes the LLM on startup.
    *   Configured to serve static files from the `static/` folder and templates from the `templates/` folder.

*   **`config_chatbot.py`**:
    *   Stores configuration constants like the path to the data tables (`NEW_TABLES_PATH`), the LLM model name (`LLM_MODEL_NAME`), and the initial persona prompt (`INITIAL_PROMPT`).
    *   Centralizes settings for easy modification.

*   **`data_handler_chatbot.py`**:
    *   Responsible for loading and cleaning data from the CSV files located in the `new_tables/` directory.
    *   Contains the `load_data()` function which reads CSVs into pandas DataFrames, handling date parsing and basic data cleaning (duplicates, types).

*   **`llm_handler_chatbot.py`**:
    *   Handles all direct interactions with the Large Language Model (LLM) via the SAP AI Core SDK (`gen-ai-hub`).
    *   Initializes the `ChatOpenAI` client using settings from `config_chatbot.py` and environment variables.
    *   Contains `generate_response_llm()` for generating natural language responses based on provided prompts.
    *   Contains `parse_intent_llm()` which uses the LLM to analyze user input and chat history, returning structured intent and entities in JSON format.

*   **`actions_chatbot.py`**:
    *   Contains the core business logic and data retrieval functions that interact with the loaded DataFrames (using original DB column names).
    *   Functions like `find_client`, `get_client_units`, `get_service_history`, `get_promotions`, etc., perform direct lookups.
    *   Functions like `handle_appointment_request` or `get_next_service_recommendation_context` prepare context and call `generate_response_llm` for tasks requiring LLM generation based on retrieved data.

*   **`main_chatbot.py`**:
    *   The original entry point for the terminal-based version of the application. Kept for reference.

*   **`new_tables/`**:
    *   Contains the source data in CSV format used by the chatbot.

*   **`templates/index.html`**:
    *   The main HTML file for the chat interface.
    *   Includes HTML structure, CSS styling, and JavaScript for handling user input, sending requests, displaying messages, and rendering Markdown.

*   **`.env`**:
    *   Stores sensitive information like API keys required by the LLM client **for local development only**.

*   **`requirements.txt`**:
    *   Lists the necessary Python packages. Install using `pip install -r requirements.txt`. Includes `Flask`, `gunicorn`, `pandas`, `langchain-openai`, `generative-ai-hub-sdk`, etc.

*   **`manifest.yml`**:
    *   Configuration file for deploying the application to Cloud Foundry using `cf push`. Specifies memory, buildpack, start command (using Gunicorn), and service bindings.

*   **`DOCUMENTATION.md`**:
    *   Additional documentation file containing detailed information about the chatbot implementation.

## 4. Key Functions (Core Logic)

*   **`data_handler_chatbot.load_data()`**: Loads and cleans CSV files into a dictionary of pandas DataFrames.
*   **`llm_handler_chatbot.generate_response_llm(prompt)`**: Generates LLM response based on a prompt.
*   **`llm_handler_chatbot.parse_intent_llm(user_input, chat_history)`**: Parses user intent and extracts entities using the LLM.
*   **`actions_chatbot.*`**: Various functions to retrieve specific data from DataFrames (using original column names).
*   **`main_ui.process_user_message(...)`**: Orchestrates the process of parsing intent, calling actions, and generating the final response within the Flask app context.

## 5. Running the Chatbot

### 5.1 Local Development

1.  **Navigate to Project Root:** Open your terminal in the `Post_Sales_Chatbot` directory.
2.  **Create Virtual Environment (Recommended):**
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows use `.venv\Scripts\activate`
    ```
3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Set Environment Variables:** Create a `.env` file in the project root by changing the name of the `.env.copy` into `.env` and add your necessary environment variables (e.g., `AICORE_CLIENT_ID`, `AICORE_CLIENT_SECRET`, etc.).
5.  **Ensure Data Files:** Make sure the `new_tables/` directory exists and contains the required CSV files.
6.  **Run the Flask App:**
    ```bash
    python main_ui.py
    ```
7.  **Access UI:** Open your web browser and go to `http://localhost:5001`.

### 5.2 Cloud Foundry Deployment

1.  **Prerequisites:**
    *   Cloud Foundry CLI (`cf`) installed and logged in to the target space.
2.  **Deploy:** Push the application using the manifest file.
    ```bash
    cf push
    ```
    Cloud Foundry will use the `manifest.yml` file, install dependencies from `requirements.txt`, bind the `chatbot-creds` service (or your equivalent service name), and start the application using Gunicorn.


## 6. Adding New Intents/Situations

To add functionality for handling a new type of user request (a new "intent" or "situation"), you need to modify the following files:

1.  **`llm_handler_chatbot.py`**:
    *   **Update `intents` list:** Add your new intent name (e.g., `"check_repair_status"`) to the `intents` list within the `parse_intent_llm` function. 
    *   **Update `entities_description`:** If your new intent requires specific information (entities), define it here (e.g., `repair_order_id (string)`).
    *   **Update Examples (Recommended):** Add an example of the expected JSON output for your new intent in the prompt within `parse_intent_llm` (e.g., `{{"intent": "check_repair_status", "entities": {{"repair_order_id": "RO12345"}}}}`).

2.  **`actions_chatbot.py`**:
    *   **Create Action Function:** Define a new Python function for the new intent's logic (e.g., `def check_repair_status(all_data, repair_order_id):`). It should accept `all_data` and necessary `entities`, retrieve data, prepare a prompt, call `generate_response_llm`, and return the response string.

3.  **`main_ui.py`**:
    *   **Add `elif` block:** In the `process_user_message` function, add a new `elif intent == "your_new_intent_name":` block.
    *   **Call Action Function:** Inside this block, call the corresponding action function from `actions_chatbot.py`, passing the required entities.
    *   **Handle Dependencies:** Add checks for `current_conversation_state` prerequisites if needed (e.g., checking if the client is identified).
