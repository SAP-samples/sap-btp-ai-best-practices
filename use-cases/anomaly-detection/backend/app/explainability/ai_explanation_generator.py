"""
AI-powered explanation generator for pharmaceutical anomaly detection.

This module uses SAP GenAI Hub to generate human-readable explanations
for detected anomalies based on order data, model predictions, and 
statistical benchmarks.
"""

import json
import os
import base64
import tempfile
from typing import Dict, Any, Optional, List, Tuple
import pandas as pd
from dotenv import load_dotenv
import numpy as np
load_dotenv()

# Import the GenAI Hub proxy for OpenAI-compatible interface
try:
    from gen_ai_hub.proxy.native.openai import chat as openai_chat
except ImportError:
    print("Warning: gen_ai_hub module not found. AI explanations will be disabled.")
    openai_chat = None

# Configuration
LLM_REQUEST_TIMEOUT_SECONDS = 60.0
MODEL_NAME = "gpt-4.1"  # Or another model available in your GenAI Hub

# Load the prompt from file
PROMPT_FILE_PATH = os.path.join(os.path.dirname(__file__), "prompt.txt")
try:
    with open(PROMPT_FILE_PATH, 'r') as f:
        SYSTEM_PROMPT = f.read()
except FileNotFoundError:
    print(f"Warning: Prompt file not found at {PROMPT_FILE_PATH}")
    SYSTEM_PROMPT = "You are an expert pharmaceutical supply chain analyst. Analyze the order for anomalies."

# Load the binary prompt from file
BINARY_PROMPT_FILE_PATH = os.path.join(os.path.dirname(__file__), "prompt_binary.txt")
try:
    with open(BINARY_PROMPT_FILE_PATH, 'r') as f:
        BINARY_SYSTEM_PROMPT = f.read()
except FileNotFoundError:
    print(f"Warning: Binary prompt file not found at {BINARY_PROMPT_FILE_PATH}")
    BINARY_SYSTEM_PROMPT = "You are an expert pharmaceutical supply chain analyst. Analyze the order and respond with exactly 'True' for anomaly or 'False' for normal."


def encode_image(image_path: str) -> str:
    """
    Encode an image file to base64 string for AI analysis.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Base64 encoded string of the image
    """
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
    except Exception as e:
        print(f"Error encoding image {image_path}: {e}")
        return None


def cleanup_temp_images(image_paths: List[str]):
    """
    Clean up temporary image files and directories.
    
    Args:
        image_paths: List of image file paths to clean up
    """
    if not image_paths:
        return
        
    try:
        import shutil
        # Get the temporary directory from the first image path
        temp_dir = os.path.dirname(image_paths[0])
        if temp_dir and os.path.exists(temp_dir) and "anomaly_plots_" in temp_dir:
            shutil.rmtree(temp_dir)
            print(f"Cleaned up temporary directory: {temp_dir}")
    except Exception as e:
        print(f"Error cleaning up temporary images: {e}")


def _safe_float(value, default: Optional[float] = None) -> Optional[float]:
    """
    Convert a value to float, gracefully handling non-numeric strings such as '#NAME?'.
    """
    if value is None:
        return default
    if isinstance(value, (float, np.floating)) and pd.isna(value):
        return default
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped or stripped.lower() in {"nan", "null"}:
            return default
        value = stripped
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value, default: Optional[int] = 0) -> Optional[int]:
    """Convert a value to int, returning default on failure."""
    if value is None:
        return default
    if isinstance(value, (float, np.floating)) and pd.isna(value):
        return default
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _safe_bool(value, default: bool = False) -> bool:
    """Convert common truthy/falsey encodings to bool."""
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, np.integer)):
        return value != 0
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes", "y"}:
            return True
        if lowered in {"false", "0", "no", "n"}:
            return False
    if isinstance(value, (float, np.floating)):
        if pd.isna(value):
            return default
        return value != 0
    return bool(value)


def prepare_order_data(row: pd.Series, features_df: pd.DataFrame = None) -> Dict[str, Any]:
    """
    Prepare order data in the format expected by the LLM prompt.
    
    Args:
        row: A pandas Series containing the order data
        features_df: Optional DataFrame with historical feature data
        
    Returns:
        Dictionary with formatted order data
    """
    # Extract order identifiers
    order_data = {
        # Order Identifiers & Basic Info
        "order_sales_document_number": str(row.get('Sales Document Number', '')),
        "order_sales_document_item": str(row.get('Sales Document Item', '')),
        "order_customer_po_number": str(row.get('Customer PO number', '')),
        "order_material_number": str(row.get('Material Number', '')),
        "order_material_description": str(row.get('Material Description', '')),
        "order_sold_to_number": str(row.get('Sold To number', '')),
        "order_ship_to_party": str(row.get('Ship-To Party', '')),
        "order_created_date": str(row.get('Sales Document Created Date', ''))[:10] if pd.notna(row.get('Sales Document Created Date')) else '',
        
        # Anomaly Model Output
        "model_used": str(row.get('model_used', 'sklearn')),
        "model_anomaly_score": _safe_float(row.get('anomaly_score'), 0.0),
        "model_predicted_anomaly": _safe_int(row.get('predicted_anomaly'), 0),
        
        # Key Order Features (Actual Values)
        "actual_sales_order_item_qty": _safe_float(row.get('Sales Order item qty'), 0.0),
        "actual_unit_price": _safe_float(row.get('Unit Price'), 0.0),
        "actual_price_z_vs_customer": _safe_float(row.get('price_z_vs_customer')),
        "actual_order_item_value": _safe_float(row.get('Order item value'), 0.0),
        "actual_current_month_total_qty": _safe_float(row.get('current_month_total_qty'), 0.0),
        "actual_month_rolling_z": _safe_float(row.get('month_rolling_z')),
        "actual_order_share_of_month": _safe_float(row.get('order_share_of_month')),
        "actual_qty_trend_slope_lastN": _safe_float(row.get('qty_trend_slope_lastN')),
        "actual_fulfillment_duration_days": _safe_float(row.get('fulfillment_duration_days'), 0.0),
        "actual_qty_z_score": _safe_float(row.get('qty_z_score'), 0.0),
        "actual_qty_deviation_from_mean": _safe_float(row.get('qty_deviation_from_mean'), 0.0),
        
        # Comparative Benchmarks & Historical Context
        "typical_qty_p05": _safe_float(row.get('p05'), 0.0),
        "typical_qty_p95": _safe_float(row.get('p95'), 0.0),
        "historical_median_qty_for_material": _safe_float(row.get('hist_median')),
        "num_historical_orders_for_material": _safe_int(row.get('num_historical_orders')),
        "typical_price_p05": _safe_float(row.get('price_p05'), 0.0),
        "typical_price_p95": _safe_float(row.get('price_p95'), 0.0),
        "expected_order_item_value_model": _safe_float(row.get('expected_order_item_value'), 0.0),
        "month_rolling_mean": _safe_float(row.get('month_rolling_mean')),
        "month_rolling_std": _safe_float(row.get('month_rolling_std')),
        "typical_fulfillment_p05": _safe_float(row.get('fulfillment_p05'), 0.0),
        "typical_fulfillment_p95": _safe_float(row.get('fulfillment_p95'), 0.0),
        "ship_to_percentage_for_sold_to": _safe_float(row.get('ship_to_percentage_for_sold_to')),
        "typical_qty_z_score_range": "-2 to +2",
        
        # Boolean Anomaly Flags
        "flag_is_first_time_cust_material_order": _safe_bool(row.get('is_first_time_cust_material_order', False)),
        "flag_is_rare_material": _safe_bool(row.get('is_rare_material', False)),
        "flag_is_suspected_duplicate_order": _safe_bool(row.get('is_suspected_duplicate_order', False)),
        "flag_is_unusual_unit_price": _safe_bool(row.get('is_unusual_unit_price', False)),
        "flag_is_qty_outside_typical_range": _safe_bool(row.get('is_qty_outside_typical_range', False)),
        "flag_is_unusual_fulfillment_time": _safe_bool(row.get('is_unusual_fulfillment_time', False)),
        "flag_is_order_qty_high_z": _safe_bool(row.get('is_order_qty_high_z', False)),
        "flag_is_value_mismatch_price_qty": _safe_bool(row.get('is_value_mismatch_price_qty', False)),
        "flag_is_unusual_uom": _safe_bool(row.get('is_unusual_uom', False)),
        "flag_is_unusual_ship_to_for_sold_to": _safe_bool(row.get('is_unusual_ship_to_for_sold_to', False)),
        
        # Raw Textual Explanations
        "raw_rule_based_explanation_string": str(row.get('anomaly_explanation', '')) if pd.notna(row.get('anomaly_explanation')) else '',
        "raw_shap_explanation_string": str(row.get('shap_explanation', '')) if pd.notna(row.get('shap_explanation')) else ''
    }
    
    # Calculate historical median if we have features_df
    if features_df is not None and 'Material Number' in features_df.columns:
        material_data = features_df[features_df['Material Number'] == row.get('Material Number')]
        if not material_data.empty and 'Sales Order item qty' in material_data.columns:
            order_data["historical_median_qty_for_material"] = _safe_float(material_data['Sales Order item qty'].median())
            order_data["num_historical_orders_for_material"] = len(material_data)

    return order_data


def generate_ai_explanation(row: pd.Series, features_df: pd.DataFrame = None) -> Optional[str]:
    """
    Generate an AI-powered explanation for the anomaly status of an order.
    
    Args:
        row: A pandas Series containing the order data
        features_df: Optional DataFrame with historical feature data
        
    Returns:
        String containing the AI-generated explanation or None if generation fails
    """
    # DEBUG: Log function entry
    order_id = f"{row.get('Sales Document Number', 'Unknown')}_{row.get('Sales Document Item', 'Unknown')}"
    print("🚀 DEBUG: Starting AI explanation generation")
    print(f"📄 Order ID: {order_id}")
    print(f"🔧 Function: generate_ai_explanation (standard)")
    print(f"📊 Features DataFrame: {'Available' if features_df is not None else 'Not Available'}")
    
    if openai_chat is None:
        print("❌ DEBUG: OpenAI chat module not available")
        return None
    
    try:
        # Prepare the order data
        order_data = prepare_order_data(row, features_df)
        
        # Create the user prompt with the order data
        user_prompt = f"""
Please analyze the following pharmaceutical order data and provide a comprehensive explanation 
of whether this order is anomalous and why.

Order Data:
{json.dumps(order_data, indent=2)}

Please follow the reasoning process and output format specified in your instructions.
"""
        
        # Create messages for the chat completion
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ]
        
        # DEBUG: Print complete information being sent to AI
        print("=" * 100)
        print("🔍 DEBUG: COMPLETE AI REQUEST DETAILS")
        print("=" * 100)
        print(f"📋 Model: {MODEL_NAME}")
        print(f"🔧 Temperature: 0.3")
        print(f"📏 Max Tokens: 1500")
        print(f"⏱️  Timeout: {LLM_REQUEST_TIMEOUT_SECONDS}s")
        print("-" * 100)
        print("📜 SYSTEM PROMPT:")
        print("-" * 50)
        print(SYSTEM_PROMPT)
        print("-" * 100)
        print("👤 USER PROMPT:")
        print("-" * 50)
        print(user_prompt)
        print("-" * 100)
        print("📊 ORDER DATA BEING SENT:")
        print("-" * 50)
        for key, value in order_data.items():
            print(f"  {key}: {value}")
        print("=" * 100)
        
        # Make the API call to GenAI Hub
        response = openai_chat.completions.create(
            model_name=MODEL_NAME,
            messages=messages,
            max_tokens=1500,  # Increased for detailed explanations
            temperature=0.3,  # Low temperature for consistency
            timeout=LLM_REQUEST_TIMEOUT_SECONDS
        )
        
        # Extract and return the response
        explanation = response.choices[0].message.content
        
        # DEBUG: Print AI response details
        print("=" * 100)
        print("✅ DEBUG: AI RESPONSE RECEIVED")
        print("=" * 100)
        print(f"📝 Response Length: {len(explanation)} chars")
        print(f"🎯 Response Preview (first 500 chars): {explanation[:500]}...")
        print("=" * 100)
        
        return explanation.strip()
        
    except TimeoutError:
        print(f"AI explanation generation timed out after {LLM_REQUEST_TIMEOUT_SECONDS} seconds.")
        return None
    except Exception as e:
        if "timeout" in str(e).lower():
            print(f"AI explanation generation likely timed out. Details: {e}")
        else:
            print(f"Error generating AI explanation: {e}")
        return None


def generate_ai_explanation_with_images(row: pd.Series, features_df: pd.DataFrame = None, 
                                       image_paths: List[str] = None) -> Optional[str]:
    """
    Generate an AI-powered explanation with both textual data and visual analysis.
    
    Args:
        row: A pandas Series containing the order data
        features_df: Optional DataFrame with historical feature data
        image_paths: List of paths to plot images for visual analysis
        
    Returns:
        String containing the AI-generated explanation or None if generation fails
    """
    # DEBUG: Log function entry
    order_id = f"{row.get('Sales Document Number', 'Unknown')}_{row.get('Sales Document Item', 'Unknown')}"
    print("🚀 DEBUG: Starting AI explanation generation")
    print(f"📄 Order ID: {order_id}")
    print(f"🔧 Function: generate_ai_explanation_with_images")
    print(f"📊 Features DataFrame: {'Available' if features_df is not None else 'Not Available'}")
    print(f"🖼️  Image Paths: {image_paths if image_paths else 'None'}")
    
    if openai_chat is None:
        print("❌ DEBUG: OpenAI chat module not available")
        return None
    
    try:
        # Prepare the order data
        order_data = prepare_order_data(row, features_df)
        
        # Create the base message content
        message_content = [
            {
                "type": "text", 
                "text": f"""Please analyze the following pharmaceutical order data and provide a comprehensive explanation 
of whether this order is anomalous and why.

Order Data:
{json.dumps(order_data, indent=2)}

"""
            }
        ]
        
        # Add the complete analysis image if available
        if image_paths:
            # Find and use only the complete analysis image
            complete_analysis_path = None
            for path in image_paths:
                if "complete_analysis" in os.path.basename(path):
                    complete_analysis_path = path
                    break
            
            if complete_analysis_path and os.path.exists(complete_analysis_path):
                message_content[0]["text"] += """
Additionally, I am providing you with a comprehensive visual analysis figure that shows four integrated plots for this order:

**Visual Analysis Figure Contains:**
- **Top Left - Order Quantity Distribution**: KDE plot showing historical quantity distribution vs. current order (red line)
- **Top Right - Unit Price Distribution**: Violin plot showing historical price distribution vs. current price (red line)  
- **Bottom Left - Monthly Volume Analysis**: Stacked bar chart showing current order's contribution to monthly volume
- **Bottom Right - Anomaly Contributors**: Horizontal bar chart showing which features contribute most to the anomaly score

**Important**: When referencing visual insights in your analysis, please clearly indicate that the insight comes from the visual analysis figure using phrases like "As shown in the visual analysis figure..." or "The plots confirm that..." This helps distinguish between numerical analysis and visual confirmation.

Please analyze this comprehensive figure alongside the numerical data to provide enhanced insights.
"""
                
                encoded_image = encode_image(complete_analysis_path)
                if encoded_image:
                    message_content.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{encoded_image}",
                            "detail": "high"
                        }
                    })
                        
        message_content[0]["text"] += "\nPlease follow the reasoning process and output format specified in your instructions."
        
        # Create messages for the chat completion
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": message_content}
        ]
        
        # DEBUG: Print complete information being sent to AI (With Images)
        print("=" * 100)
        print("🔍 DEBUG: COMPLETE AI REQUEST DETAILS (WITH VISUAL ANALYSIS)")
        print("=" * 100)
        print(f"📋 Model: {MODEL_NAME}")
        print(f"🔧 Temperature: 0.3")
        print(f"📏 Max Tokens: 2000")
        print(f"⏱️  Timeout: {LLM_REQUEST_TIMEOUT_SECONDS}s")
        print(f"🖼️  Visual Analysis: Enabled")
        print(f"📦 Message Content Parts: {len(message_content)}")
        print("-" * 100)
        print("📜 SYSTEM PROMPT:")
        print("-" * 50)
        print(SYSTEM_PROMPT)
        print("-" * 100)
        for i, part in enumerate(message_content):
            if part["type"] == "text":
                print(f"👤 USER PROMPT (Part {i+1} - Text, {len(part['text'])} chars):")
                print("-" * 50)
                print(part["text"])
                print("-" * 50)
            elif part["type"] == "image_url":
                print(f"🖼️  IMAGE CONTENT (Part {i+1} - Image):")
                print("-" * 50)
                print(f"  Detail Level: {part['image_url']['detail']}")
                print(f"  Format: PNG (base64 encoded)")
                print(f"  Data Length: {len(part['image_url']['url'])} chars")
                print("-" * 50)
        print("📊 ORDER DATA BEING SENT:")
        print("-" * 50)
        for key, value in order_data.items():
            print(f"  {key}: {value}")
        print("=" * 100)
        
        # Make the API call to GenAI Hub
        response = openai_chat.completions.create(
            model_name=MODEL_NAME,
            messages=messages,
            max_tokens=2000,  # Increased for visual analysis
            temperature=0.3,
            timeout=LLM_REQUEST_TIMEOUT_SECONDS
        )
        
        print(f"[DEBUG] Received response from GenAI Hub")
        
        # Extract and return the response
        explanation = response.choices[0].message.content
        
        # DEBUG: Print AI response details (With Images)
        print("=" * 100)
        print("✅ DEBUG: AI RESPONSE RECEIVED (WITH VISUAL ANALYSIS)")
        print("=" * 100)
        print(f"📝 Response Length: {len(explanation)} chars")
        print(f"🎯 Response Preview (first 500 chars): {explanation[:500]}...")
        print("=" * 100)
        
        return explanation.strip()
        
    except TimeoutError:
        print(f"AI explanation generation with images timed out after {LLM_REQUEST_TIMEOUT_SECONDS} seconds.")
        return None
    except Exception as e:
        if "timeout" in str(e).lower():
            print(f"AI explanation generation with images likely timed out. Details: {e}")
        else:
            print(f"Error generating AI explanation with images: {e}")
        return None
    finally:
        # Clean up temporary images
        if image_paths:
            cleanup_temp_images(image_paths)


def generate_ai_explanation_streaming(row: pd.Series, features_df: pd.DataFrame = None):
    """
    Generate an AI-powered explanation with streaming support for real-time display.
    
    Args:
        row: A pandas Series containing the order data
        features_df: Optional DataFrame with historical feature data
        
    Yields:
        String chunks of the AI-generated explanation as they arrive
        
    Returns:
        Generator that yields text chunks, or None if streaming not supported
    """
    # DEBUG: Log function entry
    order_id = f"{row.get('Sales Document Number', 'Unknown')}_{row.get('Sales Document Item', 'Unknown')}"
    print("🚀 DEBUG: Starting AI explanation generation")
    print(f"📄 Order ID: {order_id}")
    print(f"🔧 Function: generate_ai_explanation_streaming")
    print(f"📊 Features DataFrame: {'Available' if features_df is not None else 'Not Available'}")
    
    if openai_chat is None:
        print("❌ DEBUG: OpenAI chat module not available")
        return None
    
    try:
        # Prepare the order data
        order_data = prepare_order_data(row, features_df)
        
        # Create the user prompt with the order data
        user_prompt = f"""
Please analyze the following pharmaceutical order data and provide a comprehensive explanation 
of whether this order is anomalous and why.

Order Data:
{json.dumps(order_data, indent=2)}

Please follow the reasoning process and output format specified in your instructions.
"""
        
        # Create messages for the chat completion
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ]
        
        # DEBUG: Print complete information being sent to AI (Streaming)
        print("=" * 100)
        print("🔍 DEBUG: COMPLETE AI STREAMING REQUEST DETAILS")
        print("=" * 100)
        print(f"📋 Model: {MODEL_NAME}")
        print(f"🔧 Temperature: 0.3")
        print(f"📏 Max Tokens: 1500")
        print(f"⏱️  Timeout: {LLM_REQUEST_TIMEOUT_SECONDS}s")
        print(f"🌊 Streaming: Enabled")
        print("-" * 100)
        print("📜 SYSTEM PROMPT:")
        print("-" * 50)
        print(SYSTEM_PROMPT)
        print("-" * 100)
        print("👤 USER PROMPT:")
        print("-" * 50)
        print(user_prompt)
        print("-" * 100)
        print("📊 ORDER DATA BEING SENT:")
        print("-" * 50)
        for key, value in order_data.items():
            print(f"  {key}: {value}")
        print("=" * 100)
        
        # Try streaming first
        try:
            response = openai_chat.completions.create(
                model_name=MODEL_NAME,
                messages=messages,
                max_tokens=1500,
                temperature=0.3,
                timeout=LLM_REQUEST_TIMEOUT_SECONDS,
                stream=True  # Enable streaming
            )
            
            # Stream the response chunks
            chunk_count = 0
            total_content = ""
            for chunk in response:
                chunk_count += 1
                if hasattr(chunk, 'choices') and chunk.choices:
                    if hasattr(chunk.choices[0], 'delta') and hasattr(chunk.choices[0].delta, 'content'):
                        content = chunk.choices[0].delta.content
                        if content:
                            total_content += content
                            # print(f"[DEBUG] Streaming chunk {chunk_count}: {len(content)} chars")
                            yield escape_markdown_dollars(content)
                    elif hasattr(chunk.choices[0], 'message') and hasattr(chunk.choices[0].message, 'content'):
                        # Fallback for different response format
                        content = chunk.choices[0].message.content
                        if content:
                            total_content += content
                            # print(f"[DEBUG] Streaming chunk {chunk_count} (fallback): {len(content)} chars")
                            yield escape_markdown_dollars(content)
            
            # DEBUG: Print streaming completion summary
            print("=" * 100)
            print("✅ DEBUG: AI STREAMING RESPONSE COMPLETED")
            print("=" * 100)
            print(f"📦 Total Chunks: {chunk_count}")
            print(f"📝 Total Content Length: {len(total_content)} chars")
            print(f"🎯 Content Preview (first 500 chars): {total_content[:500]}...")
            print("=" * 100)
            
            return  # Successful streaming
            
        except Exception as stream_error:
            print(f"Streaming not supported or failed: {stream_error}")
            print("Falling back to regular generation...")
            
            # Fallback to regular generation if streaming fails
            response = openai_chat.completions.create(
                model_name=MODEL_NAME,
                messages=messages,
                max_tokens=1500,
                temperature=0.3,
                timeout=LLM_REQUEST_TIMEOUT_SECONDS
                # No stream parameter - regular generation
            )
            
            # Return the complete response as a single chunk
            explanation = response.choices[0].message.content
            if explanation:
                yield escape_markdown_dollars(explanation.strip())
        
    except TimeoutError:
        print(f"AI explanation generation timed out after {LLM_REQUEST_TIMEOUT_SECONDS} seconds.")
        return None
    except Exception as e:
        if "timeout" in str(e).lower():
            print(f"AI explanation generation likely timed out. Details: {e}")
        else:
            print(f"Error generating AI explanation: {e}")
        return None


def generate_ai_explanation_streaming_with_images(row: pd.Series, features_df: pd.DataFrame = None, 
                                                 image_paths: List[str] = None):
    """
    Generate an AI-powered explanation with streaming support and visual analysis.
    
    Args:
        row: A pandas Series containing the order data
        features_df: Optional DataFrame with historical feature data
        image_paths: List of paths to plot images for visual analysis
        
    Yields:
        String chunks of the AI-generated explanation as they arrive
        
    Returns:
        Generator that yields text chunks, or None if streaming not supported
    """
    if openai_chat is None:
        return None
    
    try:
        # Prepare the order data
        order_data = prepare_order_data(row, features_df)
        
        # Create the base message content
        message_content = [
            {
                "type": "text", 
                "text": f"""Please analyze the following pharmaceutical order data and provide a comprehensive explanation 
of whether this order is anomalous and why.

Order Data:
{json.dumps(order_data, indent=2)}

"""
            }
        ]
        
        # Add the complete analysis image if available
        if image_paths:
            # Find and use only the complete analysis image
            complete_analysis_path = None
            for path in image_paths:
                if "complete_analysis" in os.path.basename(path):
                    complete_analysis_path = path
                    break
            
            if complete_analysis_path and os.path.exists(complete_analysis_path):
                message_content[0]["text"] += """
Additionally, I am providing you with a comprehensive visual analysis figure that shows four integrated plots for this order:

**Visual Analysis Figure Contains:**
- **Top Left - Order Quantity Distribution**: KDE plot showing historical quantity distribution vs. current order (red line)
- **Top Right - Unit Price Distribution**: Violin plot showing historical price distribution vs. current price (red line)  
- **Bottom Left - Monthly Volume Analysis**: Stacked bar chart showing current order's contribution to monthly volume
- **Bottom Right - Anomaly Contributors**: Horizontal bar chart showing which features contribute most to the anomaly score

**Important**: When referencing visual insights in your analysis, please clearly indicate that the insight comes from the visual analysis figure using phrases like "As shown in the visual analysis figure..." or "The plots confirm that..." This helps distinguish between numerical analysis and visual confirmation.

Please analyze this comprehensive figure alongside the numerical data to provide enhanced insights.
"""
                
                encoded_image = encode_image(complete_analysis_path)
                if encoded_image:
                    message_content.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{encoded_image}",
                            "detail": "high"
                        }
                    })
                        
        message_content[0]["text"] += "\nPlease follow the reasoning process and output format specified in your instructions."
        
        # Create messages for the chat completion
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": message_content}
        ]
        
        # DEBUG: Print complete information being sent to AI (Streaming With Images)
        print("=" * 100)
        print("🔍 DEBUG: COMPLETE AI STREAMING REQUEST DETAILS (WITH VISUAL ANALYSIS)")
        print("=" * 100)
        print(f"📋 Model: {MODEL_NAME}")
        print(f"🔧 Temperature: 0.3")
        print(f"📏 Max Tokens: 2000")
        print(f"⏱️  Timeout: {LLM_REQUEST_TIMEOUT_SECONDS}s")
        print(f"🌊 Streaming: Enabled")
        print(f"🖼️  Visual Analysis: Enabled")
        print(f"📦 Message Content Parts: {len(message_content)}")
        print("-" * 100)
        print("📜 SYSTEM PROMPT:")
        print("-" * 50)
        print(SYSTEM_PROMPT)
        print("-" * 100)
        for i, part in enumerate(message_content):
            if part["type"] == "text":
                print(f"👤 USER PROMPT (Part {i+1} - Text, {len(part['text'])} chars):")
                print("-" * 50)
                print(part["text"])
                print("-" * 50)
            elif part["type"] == "image_url":
                print(f"🖼️  IMAGE CONTENT (Part {i+1} - Image):")
                print("-" * 50)
                print(f"  Detail Level: {part['image_url']['detail']}")
                print(f"  Format: PNG (base64 encoded)")
                print(f"  Data Length: {len(part['image_url']['url'])} chars")
                print("-" * 50)
        print("📊 ORDER DATA BEING SENT:")
        print("-" * 50)
        for key, value in order_data.items():
            print(f"  {key}: {value}")
        print("=" * 100)
        
        # Try streaming first
        try:
            response = openai_chat.completions.create(
                model_name=MODEL_NAME,
                messages=messages,
                max_tokens=2000,  # Increased for visual analysis
                temperature=0.3,
                timeout=LLM_REQUEST_TIMEOUT_SECONDS,
                stream=True  # Enable streaming
            )
            
            # Stream the response chunks
            chunk_count = 0
            total_content = ""
            for chunk in response:
                chunk_count += 1
                if hasattr(chunk, 'choices') and chunk.choices:
                    if hasattr(chunk.choices[0], 'delta') and hasattr(chunk.choices[0].delta, 'content'):
                        content = chunk.choices[0].delta.content
                        if content:
                            total_content += content
                            # print(f"[DEBUG] Streaming chunk {chunk_count} (with images): {len(content)} chars")
                            yield escape_markdown_dollars(content)
                    elif hasattr(chunk.choices[0], 'message') and hasattr(chunk.choices[0].message, 'content'):
                        # Fallback for different response format
                        content = chunk.choices[0].message.content
                        if content:
                            total_content += content
                            # print(f"[DEBUG] Streaming chunk {chunk_count} (with images, fallback): {len(content)} chars")
                            yield escape_markdown_dollars(content)
            
            # DEBUG: Print streaming completion summary
            print("=" * 100)
            print("✅ DEBUG: AI STREAMING RESPONSE COMPLETED (WITH VISUAL ANALYSIS)")
            print("=" * 100)
            print(f"📦 Total Chunks: {chunk_count}")
            print(f"📝 Total Content Length: {len(total_content)} chars")
            print(f"🎯 Content Preview (first 500 chars): {total_content[:500]}...")
            print("=" * 100)
            
            return  # Successful streaming
            
        except Exception as stream_error:
            print(f"Streaming with images not supported or failed: {stream_error}")
            print("Falling back to regular generation with images...")
            
            # Fallback to regular generation with images if streaming fails
            response = openai_chat.completions.create(
                model_name=MODEL_NAME,
                messages=messages,
                max_tokens=2000,
                temperature=0.3,
                timeout=LLM_REQUEST_TIMEOUT_SECONDS
                # No stream parameter - regular generation
            )
            
            # Return the complete response as a single chunk
            explanation = response.choices[0].message.content
            if explanation:
                yield escape_markdown_dollars(explanation.strip())
        
    except TimeoutError:
        print(f"AI explanation generation with images timed out after {LLM_REQUEST_TIMEOUT_SECONDS} seconds.")
        return None
    except Exception as e:
        if "timeout" in str(e).lower():
            print(f"AI explanation generation with images likely timed out. Details: {e}")
        else:
            print(f"Error generating AI explanation with images: {e}")
        return None
    finally:
        # Clean up temporary images
        if image_paths:
            cleanup_temp_images(image_paths)


def get_fallback_explanation(row: pd.Series) -> str:
    """
    Provide a fallback explanation when AI generation is unavailable.
    
    Args:
        row: A pandas Series containing the order data
        
    Returns:
        String containing a basic explanation
    """
    is_anomaly = row.get('predicted_anomaly', 0) == 1
    score = row.get('anomaly_score', 0)
    model_type = row.get('model_used', 'sklearn')
    
    if is_anomaly:
        explanation = f"""
**Overall Assessment:** Anomaly Detected

**Model Anomaly Score:** {score:.4f}

**Summary:** This order has been flagged as anomalous by the {model_type} model. 
The AI-powered detailed explanation is currently unavailable. Please review the 
rule-based flags and SHAP contributions below for specific factors that contributed 
to this classification.
"""
    else:
        explanation = f"""
**Overall Assessment:** Normal Order

**Model Anomaly Score:** {score:.4f}

**Summary:** This order appears to be within normal parameters according to the {model_type} model.
The AI-powered detailed explanation is currently unavailable.
"""
    
    return escape_markdown_dollars(explanation)


def escape_markdown_dollars(text: str) -> str:
    """
    Escape dollar signs in text to prevent markdown from interpreting them as LaTeX equations.
    
    Args:
        text: Text that may contain dollar signs
        
    Returns:
        Text with dollar signs escaped for markdown rendering
    """
    if not text:
        return text
    
    # Use HTML entity encoding to prevent LaTeX interpretation
    # This is more robust than backslash escaping for various markdown renderers
    return text.replace("$", "&#36;")


def format_ai_explanation_for_display(explanation: str) -> str:
    """
    Format the AI explanation for better display in Streamlit.
    
    Args:
        explanation: Raw AI explanation text
        
    Returns:
        Formatted explanation with proper styling
    """
    if not explanation:
        return ""
    
    # First escape dollar signs to prevent LaTeX interpretation
    explanation = escape_markdown_dollars(explanation)
    
    # The explanation is already well-formatted from the LLM
    # We can add some Streamlit-specific styling if needed
    
    # Replace section headers with styled versions
    explanation = explanation.replace("**Overall Assessment:**", "### 🎯 Overall Assessment:")
    explanation = explanation.replace("**Model Anomaly Score:**", "**📊 Model Anomaly Score:**")
    explanation = explanation.replace("**Summary of Key Reasons**", "### 📌 Summary of Key Reasons")
    explanation = explanation.replace("**Detailed Explanation:**", "### 📋 Detailed Explanation:")
    explanation = explanation.replace("**Concluding Remarks:**", "### 💡 Concluding Remarks:")
    
    return explanation


# Cache for storing generated explanations to avoid redundant API calls
_explanation_cache = {}


def get_cached_explanation(cache_key: str) -> Optional[str]:
    """Get explanation from cache if available."""
    return _explanation_cache.get(cache_key)


def cache_explanation(cache_key: str, explanation: str):
    """Store explanation in cache."""
    _explanation_cache[cache_key] = explanation


def generate_explanation_with_cache(row: pd.Series, features_df: pd.DataFrame = None) -> str:
    """
    Generate AI explanation with caching to avoid redundant API calls.
    
    Args:
        row: Order data
        features_df: Historical feature data
        
    Returns:
        AI-generated or fallback explanation
    """
    # Create a cache key from order identifiers
    cache_key = f"{row.get('Sales Document Number')}_{row.get('Sales Document Item')}"
    
    # DEBUG: Log cache attempt
    print(f"🔍 DEBUG: Checking cache for key: {cache_key}")
    
    # Check cache first
    cached = get_cached_explanation(cache_key)
    if cached:
        print(f"✅ DEBUG: Cache hit! Using cached explanation")
        return cached
    
    print(f"❌ DEBUG: Cache miss - generating new explanation")
    
    # Generate new explanation
    explanation = generate_ai_explanation(row, features_df)
    
    if explanation is None:
        explanation = get_fallback_explanation(row)
    else:
        explanation = format_ai_explanation_for_display(explanation)
    
    # Cache the result
    cache_explanation(cache_key, explanation)
    
    return explanation


def generate_explanation_with_cache_enhanced(row: pd.Series, features_df: pd.DataFrame = None, 
                                           use_visual_analysis: bool = False) -> str:
    """
    Generate AI explanation with caching and optional visual analysis.
    
    Args:
        row: Order data
        features_df: Historical feature data
        use_visual_analysis: If True, generate and analyze plots for enhanced explanations
        
    Returns:
        AI-generated or fallback explanation
    """
    # Create a cache key from order identifiers (include visual analysis in cache key)
    cache_key = f"{row.get('Sales Document Number')}_{row.get('Sales Document Item')}"
    if use_visual_analysis:
        cache_key += "_visual"
    
    # Check cache first
    cached = get_cached_explanation(cache_key)
    if cached:
        return cached
    
    # Generate plots for visual analysis if requested
    image_paths = None
    if use_visual_analysis:
        try:
            # Import the plotting function from the separate module to avoid Streamlit conflicts
            from visualization.feature_analysis import create_feature_plots
            
            print(f"[DEBUG] Starting plot generation for visual analysis...")
            # Generate plots and save as images
            fig, image_paths = create_feature_plots(row, features_df, save_for_ai_analysis=True)
            print(f"[DEBUG] Generated {len(image_paths) if image_paths else 0} plot images: {image_paths}")
            
            # Close the matplotlib figure to free memory
            import matplotlib.pyplot as plt
            plt.close(fig)
            
        except Exception as e:
            print(f"[ERROR] Error generating plots for visual analysis: {e}")
            import traceback
            print(f"[ERROR] Full traceback: {traceback.format_exc()}")
            image_paths = None
    
    # Generate new explanation
    if use_visual_analysis and image_paths:
        explanation = generate_ai_explanation_with_images(row, features_df, image_paths)
    else:
        explanation = generate_ai_explanation(row, features_df)
    
    if explanation is None:
        explanation = get_fallback_explanation(row)
    else:
        explanation = format_ai_explanation_for_display(explanation)
    
    # Cache the result
    cache_explanation(cache_key, explanation)
    
    return explanation


def generate_explanation_streaming_with_cache(row: pd.Series, features_df: pd.DataFrame = None, 
                                             use_visual_analysis: bool = False):
    """
    Generate AI explanation with streaming support, caching, and optional visual analysis.
    
    Args:
        row: Order data
        features_df: Historical feature data
        use_visual_analysis: If True, generate and analyze plots for enhanced explanations
        
    Yields:
        String chunks of the explanation for real-time display
        
    Returns:
        Generator that yields (chunk, is_final, full_explanation) tuples
        where is_final indicates if this is the last chunk and full_explanation
        is the complete text (for caching when is_final=True)
    """
    # Create a cache key from order identifiers (include visual analysis in cache key)
    cache_key = f"{row.get('Sales Document Number')}_{row.get('Sales Document Item')}"
    if use_visual_analysis:
        cache_key += "_visual"
    
    # Check cache first - if cached, yield complete explanation
    cached = get_cached_explanation(cache_key)
    if cached:
        yield (cached, True, cached)
        return
    
    # Generate plots for visual analysis if requested
    image_paths = None
    if use_visual_analysis:
        try:
            # Import the plotting function from the separate module to avoid Streamlit conflicts
            from visualization.feature_analysis import create_feature_plots
            
            print(f"[DEBUG] Starting plot generation for streaming visual analysis...")
            # Generate plots and save as images
            fig, image_paths = create_feature_plots(row, features_df, save_for_ai_analysis=True)
            print(f"[DEBUG] Generated {len(image_paths) if image_paths else 0} plot images for streaming: {image_paths}")
            
            # Close the matplotlib figure to free memory
            import matplotlib.pyplot as plt
            plt.close(fig)
            
        except Exception as e:
            print(f"[ERROR] Error generating plots for streaming visual analysis: {e}")
            import traceback
            print(f"[ERROR] Full traceback for streaming: {traceback.format_exc()}")
            image_paths = None
    
    # Generate new explanation with streaming
    full_explanation = ""
    if use_visual_analysis and image_paths:
        stream_generator = generate_ai_explanation_streaming_with_images(row, features_df, image_paths)
    else:
        stream_generator = generate_ai_explanation_streaming(row, features_df)
    
    if stream_generator is None:
        # Fallback to non-streaming
        explanation = get_fallback_explanation(row)
        formatted_explanation = format_ai_explanation_for_display(explanation)
        cache_explanation(cache_key, formatted_explanation)
        yield (formatted_explanation, True, formatted_explanation)
        return
    
    try:
        stream_successful = False
        for chunk in stream_generator:
            if chunk:
                stream_successful = True
                full_explanation += chunk
                # Format the partial explanation for display
                formatted_partial = format_ai_explanation_for_display(full_explanation)
                yield (formatted_partial, False, full_explanation)
        
        # Final formatting and caching after successful streaming
        if stream_successful and full_explanation:
            final_explanation = format_ai_explanation_for_display(full_explanation)
            cache_explanation(cache_key, final_explanation)
            yield (final_explanation, True, final_explanation)
        else:
            # Fallback if streaming didn't produce content
            explanation = get_fallback_explanation(row)
            formatted_explanation = format_ai_explanation_for_display(explanation)
            cache_explanation(cache_key, formatted_explanation)
            yield (formatted_explanation, True, formatted_explanation)
            
    except Exception as e:
        print(f"Error in streaming generation: {e}")
        # Fallback to regular explanation
        explanation = get_fallback_explanation(row)
        formatted_explanation = format_ai_explanation_for_display(explanation)
        cache_explanation(cache_key, formatted_explanation)
        yield (formatted_explanation, True, formatted_explanation)


def generate_ai_binary_classification_with_images(row: pd.Series, features_df: pd.DataFrame = None, 
                                                  image_paths: List[str] = None) -> Optional[str]:
    """
    Generate a binary AI classification (True/False) for the anomaly status of an order with visual analysis.
    
    Args:
        row: A pandas Series containing the order data
        features_df: Optional DataFrame with historical feature data
        image_paths: List of paths to plot images for visual analysis
        
    Returns:
        String containing "True" (anomaly) or "False" (normal), or None if generation fails
    """
    # DEBUG: Log function entry
    order_id = f"{row.get('Sales Document Number', 'Unknown')}_{row.get('Sales Document Item', 'Unknown')}"
    print("🚀 DEBUG: Starting binary AI classification")
    print(f"📄 Order ID: {order_id}")
    print(f"🔧 Function: generate_ai_binary_classification_with_images")
    print(f"📊 Features DataFrame: {'Available' if features_df is not None else 'Not Available'}")
    print(f"🖼️  Image Paths: {image_paths if image_paths else 'None'}")
    
    if openai_chat is None:
        print("❌ DEBUG: OpenAI chat module not available")
        return None
    
    try:
        # Prepare the order data
        order_data = prepare_order_data(row, features_df)
        
        # Create the base message content
        message_content = [
            {
                "type": "text", 
                "text": f"""Please analyze the following pharmaceutical order data and provide a binary classification.

Order Data:
{json.dumps(order_data, indent=2)}

"""
            }
        ]
        
        # Add the complete analysis image if available
        if image_paths:
            # Find and use only the complete analysis image
            complete_analysis_path = None
            for path in image_paths:
                if "complete_analysis" in os.path.basename(path):
                    complete_analysis_path = path
                    break
            
            if complete_analysis_path and os.path.exists(complete_analysis_path):
                message_content[0]["text"] += """Additionally, I am providing you with a comprehensive visual analysis figure that shows four integrated plots for this order:

**Visual Analysis Figure Contains:**
- **Top Left - Order Quantity Distribution**: KDE plot showing historical quantity distribution vs. current order (red line)
- **Top Right - Unit Price Distribution**: Violin plot showing historical price distribution vs. current price (red line)  
- **Bottom Left - Monthly Volume Analysis**: Stacked bar chart showing current order's contribution to monthly volume
- **Bottom Right - Anomaly Contributors**: Horizontal bar chart showing which features contribute most to the anomaly score

Please analyze this comprehensive figure alongside the numerical data for your binary classification.

"""
                
                encoded_image = encode_image(complete_analysis_path)
                if encoded_image:
                    message_content.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{encoded_image}",
                            "detail": "high"
                        }
                    })
                        
        message_content[0]["text"] += "\nProvide your binary classification following the instructions."
        
        # Create messages for the chat completion
        messages = [
            {"role": "system", "content": BINARY_SYSTEM_PROMPT},
            {"role": "user", "content": message_content}
        ]
        
        # DEBUG: Print complete information being sent to AI (Binary Classification)
        print("=" * 100)
        print("🎯 DEBUG: COMPLETE BINARY AI CLASSIFICATION REQUEST")
        print("=" * 100)
        print(f"📋 Model: {MODEL_NAME}")
        print(f"🔧 Temperature: 0.3")
        print(f"📏 Max Tokens: 50")
        print(f"⏱️  Timeout: 30s")
        print(f"🖼️  Visual Analysis: {'Enabled' if image_paths else 'Disabled'}")
        print(f"📦 Message Content Parts: {len(message_content)}")
        print("-" * 100)
        print("📜 BINARY SYSTEM PROMPT:")
        print("-" * 50)
        print(BINARY_SYSTEM_PROMPT)
        print("-" * 100)
        for i, part in enumerate(message_content):
            if part["type"] == "text":
                print(f"👤 USER PROMPT (Part {i+1} - Text, {len(part['text'])} chars):")
                print("-" * 50)
                print(part["text"])
                print("-" * 50)
            elif part["type"] == "image_url":
                print(f"🖼️  IMAGE CONTENT (Part {i+1} - Image):")
                print("-" * 50)
                print(f"  Detail Level: {part['image_url']['detail']}")
                print(f"  Format: PNG (base64 encoded)")
                print(f"  Data Length: {len(part['image_url']['url'])} chars")
                print("-" * 50)
        print("📊 ORDER DATA BEING SENT:")
        print("-" * 50)
        for key, value in order_data.items():
            print(f"  {key}: {value}")
        print("=" * 100)
        
        # Make the API call to GenAI Hub with shorter timeout and fewer tokens
        # TODO: CHANGE BACK TO GPT-4.1
        # response = openai_chat.completions.create(
        #     model_name='o3-mini',
        #     messages=messages
        # )
        response = openai_chat.completions.create(
            model_name=MODEL_NAME,
            messages=messages,
            max_tokens=50,  # Short response expected
            temperature=0.3,
            timeout=30.0  # Shorter timeout for binary classification
        )
        
        # Extract and return the response
        raw_response = response.choices[0].message.content.strip()
        
        # DEBUG: Print AI response details (Binary Classification)
        print("=" * 100)
        print("✅ DEBUG: BINARY AI RESPONSE RECEIVED")
        print("=" * 100)
        print(f"📝 Raw Response: '{raw_response}'")
        print(f"📏 Response Length: {len(raw_response)} chars")
        
        # Clean and validate the response
        cleaned_response = raw_response.strip().strip('"').strip("'")
        
        # Validate binary response
        if cleaned_response.lower() == "true":
            processed_result = "True"
        elif cleaned_response.lower() == "false":
            processed_result = "False"
        else:
            # Handle invalid responses
            print(f"⚠️  WARNING: Invalid binary response received: '{cleaned_response}'")
            print("🔄 Attempting to extract binary decision from response...")
            
            # Try to extract True/False from response text
            response_lower = cleaned_response.lower()
            if "true" in response_lower and "false" not in response_lower:
                processed_result = "True"
            elif "false" in response_lower and "true" not in response_lower:
                processed_result = "False"
            else:
                print("❌ Could not determine binary classification from response")
                return None
        
        print(f"🎯 Processed Result: '{processed_result}'")
        print("=" * 100)
        
        return processed_result
        
    except TimeoutError:
        print(f"Binary AI classification timed out after 30 seconds.")
        return None
    except Exception as e:
        if "timeout" in str(e).lower():
            print(f"Binary AI classification likely timed out. Details: {e}")
        else:
            print(f"Error generating binary AI classification: {e}")
        return None
    finally:
        # Clean up temporary images
        if image_paths:
            cleanup_temp_images(image_paths)


def format_binary_result_for_display(binary_result: Optional[str]) -> str:
    """
    Format the binary AI classification result for display in the UI.
    
    Args:
        binary_result: "True", "False", or None from the binary classification
        
    Returns:
        Formatted display text with appropriate styling
    """
    if binary_result is None:
        return "Classification unavailable"
    elif binary_result == "True":
        return "Anomalous Sales Order"
    elif binary_result == "False":
        return "Normal Sales Order"
    else:
        return f"Unexpected result: {binary_result}"
