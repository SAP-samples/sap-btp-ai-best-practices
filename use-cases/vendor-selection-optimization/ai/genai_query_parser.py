import json
import os
from dotenv import load_dotenv
load_dotenv()
from gen_ai_hub.proxy.native.openai import chat as openai_chat # Assuming this is the correct proxy import

# --- Configuration Constants ---
QUERY_PARSING_TIMEOUT_SECONDS = 60.0  # Timeout for the LLM call

# System prompt for query parsing
QUERY_PARSING_SYSTEM_PROMPT = """
You are an AI assistant specialized in parsing procurement queries.
Your task is to extract the 'material' and 'quantity' from the user's natural language query.
The material might be referred to by its name, description, or a code (e.g., "EMN-MOTOR", "ECR-SENSOR", "brake pads").
The quantity should be an integer.

Return the extracted information strictly as a JSON object with two keys: "material" and "quantity".
- If the material can be identified, provide its string value. If not, use null.
- If the quantity can be identified, provide its integer value. If not, use null.

Example User Query 1: "I want to buy 1000 units of EMN-MOTOR units"
Example JSON Output 1: {"material": "EMN-MOTOR", "quantity": 1000}

Example User Query 2: "Need ECR-SENSOR, about 50 pieces"
Example JSON Output 2: {"material": "ECR-SENSOR", "quantity": 50}

Example User Query 3: "Looking for some handles"
Example JSON Output 3: {"material": "handles", "quantity": null}

Example User Query 4: "How much for 200 of EMN-THROTTLE?"
Example JSON Output 4: {"material": "EMN-THROTTLE", "quantity": 200}

Example User Query 5: "Show me vendors for EMN-BRAKES"
Example JSON Output 5: {"material": "EMN-BRAKES", "quantity": null}

Example User Query 6: "I need some parts"
Example JSON Output 6: {"material": "parts", "quantity": null}

Example User Query 7: "What about 500 units?"
Example JSON Output 7: {"material": null, "quantity": 500}

Ensure the output is ONLY the JSON object and nothing else.
"""

def clean_text(text: str) -> str:
    # Find $ sign and change it to \$
    if "$" in text:
        text = text.replace("$", "\$")

    return text

def parse_procurement_query(user_query: str):
    """
    Parses a natural language procurement query to extract material and quantity
    using SAP GenAI Hub (via OpenAI proxy).

    Args:
        user_query (str): The natural language query from the user.

    Returns:
        dict: A dictionary with "material" and "quantity" keys, or None if parsing fails.
              Example: {"material": "EMN-MOTOR", "quantity": 1000}
    """
    print(f"Parsing procurement query: '{user_query}'")

    if not user_query:
        print("Error: User query is empty.")
        return None

    messages = [
        {"role": "system", "content": QUERY_PARSING_SYSTEM_PROMPT},
        {"role": "user", "content": user_query}
    ]

    try:
        print(f"Sending request to LLM for query parsing (timeout: {QUERY_PARSING_TIMEOUT_SECONDS}s).")
        # Assuming 'gpt-4o' or a similar capable model is available and configured
        # The model_name might need to be adjusted based on actual GenAI Hub deployment
        response = openai_chat.completions.create(
            model_name="gpt-4.1", # Or another suitable model configured in GenAI Hub
            messages=messages,
            max_tokens=150,      # Sufficient for a small JSON object
            temperature=0.0,     # For deterministic output
            timeout=QUERY_PARSING_TIMEOUT_SECONDS,
            response_format={"type": "json_object"} # Request JSON output
        )

        raw_response_content = response.choices[0].message.content
        print(f"Raw LLM response: {raw_response_content}")

        # Attempt to parse the JSON response
        parsed_json = json.loads(raw_response_content)

        # Validate the structure
        if not isinstance(parsed_json, dict) or \
           "material" not in parsed_json or \
           "quantity" not in parsed_json:
            print(f"Error: LLM response is not in the expected JSON format. Response: {raw_response_content}")
            return {"material": None, "quantity": None, "error": "Invalid JSON structure from LLM"}

        # Ensure quantity is an int or None
        if parsed_json.get("quantity") is not None:
            try:
                parsed_json["quantity"] = int(parsed_json["quantity"])
            except (ValueError, TypeError):
                print(f"Warning: Could not convert quantity '{parsed_json['quantity']}' to int. Setting to null.")
                parsed_json["quantity"] = None
        
        print(f"Successfully parsed query. Result: {parsed_json}")
        return parsed_json

    except json.JSONDecodeError as json_err:
        print(f"Error decoding JSON response from LLM: {json_err}. Response: {raw_response_content}")
        return {"material": None, "quantity": None, "error": f"JSONDecodeError: {json_err}"}
    except TimeoutError: # Specific timeout error for openai client might be openai.Timeout
        error_msg = f"Error parsing query: LLM request timed out after {QUERY_PARSING_TIMEOUT_SECONDS} seconds."
        print(error_msg)
        return {"material": None, "quantity": None, "error": "LLM request timed out"}
    except Exception as e:
        # Check for more specific timeout messages if not caught by TimeoutError
        if "timeout" in str(e).lower() or "timed out" in str(e).lower():
            error_msg = f"Error parsing query: LLM request likely timed out ({QUERY_PARSING_TIMEOUT_SECONDS}s). Details: {e}"
        else:
            error_msg = f"An unexpected error occurred during query parsing with LLM: {e}"
        print(error_msg)
        return {"material": None, "quantity": None, "error": str(e)}

if __name__ == '__main__':
    # Example Usage (for testing)
    test_queries = [
        "I want to buy 1000 units of EMN-MOTOR units",
        "Need ECR-SENSOR, about 50 pieces",
        "Looking for some handles",
        "How much for 200 of EMN-THROTTLE?",
        "Show me vendors for EMN-BRAKES",
        "I need some parts",
        "What about 500 units?",
        "Get me 75 EMN-HANDLEs",
        "gibberish query"
    ]

    for query in test_queries:
        result = parse_procurement_query(query)
        print(f"Query: '{query}' -> Parsed: {result}\n")

    # Test with an empty query
    # print(f"Query: '' -> Parsed: {parse_procurement_query('')}\n")


# --- Vendor Suggestion Constants ---
VENDOR_SUGGESTION_TIMEOUT_SECONDS = 90.0
VENDOR_SUGGESTION_SYSTEM_PROMPT_TEMPLATE = """
You are an expert procurement advisor.
The user is looking to procure '{material_name}'"""
# Add quantity to prompt only if it's available
VENDOR_SUGGESTION_SYSTEM_PROMPT_QUANTITY_SUFFIX = " for a requested quantity of {quantity} units."
VENDOR_SUGGESTION_SYSTEM_PROMPT_MAIN = """
Based on the provided JSON data representing a list of potential vendors, please recommend the best vendor.
Your recommendation should primarily be based on the 'EffectiveCostPerUnit_USD' (name it Effective Cost per Unit).
Explain your choice by discussing its 'EffectiveCostPerUnit_USD' and briefly mentioning significant contributing factors from its cost breakdown (e.g., 'cost_BasePrice', 'cost_Tariff', 'cost_Holding_LeadTime', name those as Base Price Cost, Tariff Cost, Holding Lead Time Cost, etc.) and other relevant metrics like 'AvgLeadTimeDays_raw' or 'OnTimeRate_raw' (name them Avg Lead Time Days, On Time Rate, etc.).
If other vendors are very close in terms of total effective cost, you may mention them as alternatives, highlighting key differences.
Keep your explanation clear, concise, and well-reasoned. Structure your response in a readable paragraph or a short list of points.

The input data is a list of JSON objects, where each object is a vendor. Example structure for one vendor:
{
    "VendorFullID": "VendorA-101",
    "EffectiveCostPerUnit_USD": 100.50,
    "cost_BasePrice": 80.00,
    "cost_Tariff": 5.50,
    "cost_Holding_LeadTime": 3.00,
    "AvgLeadTimeDays_raw": 10,
    "OnTimeRate_raw": 0.95,
    "InFullRate_raw": 0.92
    // ... other cost and economic metrics might be present ...
}
Focus on the provided data to make your recommendation.
Here is an example of output:
Based on the provided data, the best vendor to procure the 'EMN-MOTOR' from is TechGroup, Inc-33. Here is the reasoning for this recommendation:

**Effective Cost per Unit**: TechGroup, Inc-33 offers the lowest Effective Cost per Unit at $484.5361, which is significantly lower than the other vendors.
**Base Price Cost**: Although TechGroup, Inc-33 has a relatively high Base Price Cost of $363.0, it compensates with zero Tariff Cost, unlike some competitors.
**Tariff Cost**: TechGroup, Inc-33 incurs no Tariff Cost, which is a notable advantage over vendors like EV Parts Inc.-12 and WaveCrest Labs-23, who have substantial tariff costs.
**Holding Lead Time Cost**: TechGroup, Inc-33 has a Holding Lead Time Cost of $10.5618, which is reasonable compared to other vendors, contributing positively to its overall effective cost.
**Avg Lead Time Days**: The Avg Lead Time Days for TechGroup, Inc-33 is 59 days, which is better than EV Parts Inc.-12's 90 days, though slightly longer than WaveCrest Labs-24's 51 days.
**On Time Rate**: TechGroup, Inc-33 has an On Time Rate of 0.7, which is superior to EV Parts Inc.-12 and comparable to other vendors, indicating reliable delivery performance.
**In Full Rate**: The In Full Rate for TechGroup, Inc-33 is 0.8, which is competitive and suggests a good likelihood of receiving complete orders.

While **TechGroup, Inc-33** is the recommended vendor, **WaveCrest Labs-24** could be considered as an alternative due to its relatively low Effective Cost per Unit of $502.3253 and shorter Avg Lead Time Days of 51. However, it has a higher Inefficiency In Full Cost, which affects its overall cost-effectiveness.

In conclusion, **TechGroup, Inc-33** stands out as the most cost-effective choice, balancing pricing and delivery reliability effectively.
"""

def generate_vendor_suggestion(filtered_data_json_str: str, material_name: str, quantity: int = None):
    """
    Generates a textual vendor suggestion based on filtered data using SAP GenAI Hub.

    Args:
        filtered_data_json_str (str): A JSON string representing a list of vendor data.
        material_name (str): The name of the material being procured.
        quantity (int, optional): The quantity being procured. Defaults to None.

    Returns:
        str: A textual suggestion from the LLM, or None if an error occurs.
    """
    print(f"Generating vendor suggestion for material: '{material_name}'" + (f", quantity: {quantity}" if quantity else ""))

    if not filtered_data_json_str:
        print("Error: Filtered data for suggestion is empty.")
        return "No vendor data provided to generate a suggestion."
    
    # Construct the system prompt
    system_prompt = VENDOR_SUGGESTION_SYSTEM_PROMPT_TEMPLATE.format(material_name=material_name)
    if quantity is not None:
        system_prompt += VENDOR_SUGGESTION_SYSTEM_PROMPT_QUANTITY_SUFFIX.format(quantity=quantity)
    system_prompt += VENDOR_SUGGESTION_SYSTEM_PROMPT_MAIN

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Here is the vendor data:\n```json\n{filtered_data_json_str}\n```\nPlease provide your recommendation."}
    ]

    try:
        print(f"Sending request to LLM for vendor suggestion (timeout: {VENDOR_SUGGESTION_TIMEOUT_SECONDS}s).")
        response = openai_chat.completions.create(
            model_name="gpt-4.1", # Or another suitable model
            messages=messages,
            max_tokens=500,      # Allow for a reasonably detailed explanation
            temperature=0.3,     # Slightly more creative/natural language for explanation
            timeout=VENDOR_SUGGESTION_TIMEOUT_SECONDS
        )
        suggestion = response.choices[0].message.content

        suggestion = clean_text(suggestion)  # Clean the suggestion text
        print("Successfully generated vendor suggestion.")
        return suggestion.strip()

    except TimeoutError:
        error_msg = f"Error generating suggestion: LLM request timed out after {VENDOR_SUGGESTION_TIMEOUT_SECONDS} seconds."
        print(error_msg)
        return "Sorry, the request for an AI suggestion timed out."
    except Exception as e:
        if "timeout" in str(e).lower() or "timed out" in str(e).lower():
            error_msg = f"Error generating suggestion: LLM request likely timed out ({VENDOR_SUGGESTION_TIMEOUT_SECONDS}s). Details: {e}"
        else:
            error_msg = f"An unexpected error occurred while generating vendor suggestion: {e}"
        print(error_msg)
        return f"Sorry, an error occurred while generating the AI suggestion: {e}"