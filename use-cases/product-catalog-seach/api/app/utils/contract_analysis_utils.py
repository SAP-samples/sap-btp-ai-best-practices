"""Utility functions for contract analysis using LLM."""

import os
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List
import json

logger = logging.getLogger(__name__)

# Get the contracts directory path
CONTRACTS_DIR = Path(__file__).parent.parent.parent / "storage" / "contracts"


def get_contract_file_path(contract_name: str) -> Path:
    """Get the full path to a contract file.

    Args:
        contract_name: Name of the contract (with or without .pdf extension)

    Returns:
        Path: Full path to the contract file

    Raises:
        FileNotFoundError: If the contract file doesn't exist
    """
    # Ensure .pdf extension
    if not contract_name.lower().endswith(".pdf"):
        contract_name += ".pdf"

    # Sanitize filename
    safe_filename = os.path.basename(contract_name)
    file_path = CONTRACTS_DIR / safe_filename

    if not file_path.exists():
        raise FileNotFoundError(f"Contract file not found: {safe_filename}")

    # Security check - ensure file is within contracts directory
    if not str(file_path.resolve()).startswith(str(CONTRACTS_DIR.resolve())):
        raise ValueError(f"Access denied to file: {safe_filename}")

    return file_path


def extract_text_from_pdf(file_path: Path) -> str:
    """Extract text content from a PDF file.

    Uses multiple PDF processing libraries for robust text extraction.
    Falls back to placeholder content if extraction fails.

    Args:
        file_path: Path to the PDF file

    Returns:
        str: Extracted text content
    """
    try:
        # First try with pdfplumber (generally more reliable for text extraction)
        try:
            import pdfplumber

            logger.info(f"Extracting text from PDF using pdfplumber: {file_path.name}")

            text_content = []
            with pdfplumber.open(file_path) as pdf:
                for page_num, page in enumerate(pdf.pages, 1):
                    try:
                        page_text = page.extract_text()
                        if page_text:
                            text_content.append(f"--- Page {page_num} ---\n{page_text}")
                    except Exception as e:
                        logger.warning(
                            f"Failed to extract text from page {page_num}: {e}"
                        )
                        continue

            if text_content:
                extracted_text = "\n\n".join(text_content)
                logger.info(
                    f"Successfully extracted {len(extracted_text)} characters from {file_path.name}"
                )
                return extracted_text
            else:
                raise ValueError("No text content extracted from PDF")

        except ImportError:
            logger.warning("pdfplumber not available, trying PyPDF2")

            try:
                # Fallback to PyPDF2
                import PyPDF2

                logger.info(f"Extracting text from PDF using PyPDF2: {file_path.name}")

                text_content = []
                with open(file_path, "rb") as pdf_file:
                    pdf_reader = PyPDF2.PdfReader(pdf_file)

                    for page_num, page in enumerate(pdf_reader.pages, 1):
                        try:
                            page_text = page.extract_text()
                            if page_text:
                                text_content.append(
                                    f"--- Page {page_num} ---\n{page_text}"
                                )
                        except Exception as e:
                            logger.warning(
                                f"Failed to extract text from page {page_num}: {e}"
                            )
                            continue

                if text_content:
                    extracted_text = "\n\n".join(text_content)
                    logger.info(
                        f"Successfully extracted {len(extracted_text)} characters from {file_path.name} using PyPDF2"
                    )
                    return extracted_text
                else:
                    raise ValueError("No text content extracted from PDF")

            except ImportError:
                logger.error(
                    "Neither pdfplumber nor PyPDF2 are available for PDF text extraction"
                )
                raise ImportError("PDF processing libraries not available")
            except Exception as e:
                logger.error(f"PyPDF2 extraction failed: {e}")
                raise e

    except Exception as e:
        logger.error(f"PDF text extraction failed for {file_path.name}: {e}")
        raise RuntimeError(
            f"Failed to extract text from PDF {file_path.name}: {str(e)}"
        )


def build_analysis_prompt(
    contract_text: str,
    product_name: str,
    product_description: Optional[str],
    current_date: str,
) -> str:
    """Build the prompt for LLM contract analysis.

    Args:
        contract_text: Full text of the contract
        product_name: Name of the product to analyze
        product_description: Optional description of the product
        current_date: Current date for validity check

    Returns:
        str: Formatted prompt for the LLM
    """
    product_info = f"Product: {product_name}"
    if product_description:
        product_info += f"\nDescription: {product_description}"

    prompt = f"""
You are a contract analysis expert. Please analyze the following contract to determine:

1. Whether the contract supports purchasing the specified product
2. Whether the contract is still valid on the current date

CONTRACT TEXT:
{contract_text}

PRODUCT TO ANALYZE:
{product_info}

CURRENT DATE: {current_date}

CRITICAL: Your response must be ONLY a valid JSON object. No explanatory text, no code blocks, no markdown - just pure JSON starting with {{ and ending with }}.

Please provide your analysis in exactly this JSON format:
{{
    "supports_purchase": true,
    "is_valid": true,
    "validity_end_date": "2025-12-31",
    "confidence_score": 0.95,
    "reasoning": "Detailed explanation of your analysis",
    "relevant_clauses": ["Contract clause 1", "Contract clause 2"]
}}

Analysis Guidelines:
- For "supports_purchase": Check if the product falls under the contract's scope of supply (boolean: true or false)
- For "is_valid": Compare contract validity dates with the current date (boolean: true or false) 
- For "validity_end_date": Extract the contract expiration date in YYYY-MM-DD format, or null if not found
- For "confidence_score": Rate your confidence in the analysis (number between 0.0 and 1.0)
- For "reasoning": Provide clear explanation of your decision-making process (string, keep it concise)
- For "relevant_clauses": Extract specific contract text that supports your conclusions (array of strings)

Be thorough and precise in your analysis. If information is unclear or missing, indicate this in your reasoning and adjust the confidence score accordingly.

IMPORTANT REMINDERS:
- Start your response immediately with {{ (opening brace)
- End your response with }} (closing brace)
- Use proper JSON formatting with double quotes for strings
- Do not include any text before or after the JSON
- Ensure all strings are properly escaped (use \\" for quotes within strings)
"""

    return prompt


def parse_llm_response(response_text: str) -> Dict[str, Any]:
    """Parse the LLM response and extract structured analysis results.

    Args:
        response_text: Raw response text from the LLM

    Returns:
        Dict[str, Any]: Parsed analysis results
    """
    import re

    try:
        # Clean the response text
        cleaned_text = response_text.strip()

        # Log the raw response for debugging
        logger.debug(f"Attempting to parse LLM response: {cleaned_text[:200]}...")

        # Try multiple approaches to extract JSON
        json_str = None

        # Approach 1: Look for JSON block between curly braces with better brace matching
        start_idx = cleaned_text.find("{")
        if start_idx != -1:
            # Find the matching closing brace by counting braces
            brace_count = 0
            end_idx = start_idx
            in_string = False
            escape_next = False

            for i, char in enumerate(cleaned_text[start_idx:], start_idx):
                if escape_next:
                    escape_next = False
                    continue

                if char == "\\":
                    escape_next = True
                    continue

                if char == '"' and not escape_next:
                    in_string = not in_string
                    continue

                if not in_string:
                    if char == "{":
                        brace_count += 1
                    elif char == "}":
                        brace_count -= 1
                        if brace_count == 0:
                            end_idx = i + 1
                            break

            if brace_count == 0:  # Found matching closing brace
                json_str = cleaned_text[start_idx:end_idx]

        # Approach 2: If no valid JSON found, try to extract from code blocks
        if not json_str:
            # Look for JSON in code blocks (```json ... ```)
            json_match = re.search(
                r"```(?:json)?\s*(\{.*?\})\s*```", cleaned_text, re.DOTALL
            )
            if json_match:
                json_str = json_match.group(1)

        # Approach 3: If still no JSON, try the entire response if it looks like JSON
        if not json_str and cleaned_text.startswith("{") and cleaned_text.endswith("}"):
            json_str = cleaned_text

        # Approach 4: Look for JSON between triple quotes or similar delimiters
        if not json_str:
            patterns = [
                r"```json\s*(\{.*?\})\s*```",
                r"```\s*(\{.*?\})\s*```",
                r'"json":\s*(\{.*?\})',
                r"JSON:\s*(\{.*?\})",
                r"Response:\s*(\{.*?\})",
            ]
            for pattern in patterns:
                match = re.search(pattern, cleaned_text, re.DOTALL | re.IGNORECASE)
                if match:
                    json_str = match.group(1)
                    break

        # Approach 5: Extract JSON from text that might have prefix/suffix text
        if not json_str:
            # Find the largest JSON-like structure
            matches = re.finditer(
                r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", cleaned_text, re.DOTALL
            )
            largest_match = None
            max_length = 0

            for match in matches:
                if len(match.group(0)) > max_length:
                    max_length = len(match.group(0))
                    largest_match = match

            if largest_match:
                json_str = largest_match.group(0)

        if not json_str:
            raise ValueError("No valid JSON found in response")

        # Clean up common JSON formatting issues
        json_str = json_str.strip()

        # Remove extra whitespace and line breaks within the JSON
        json_str = re.sub(r"\s+", " ", json_str)

        # Fix common trailing comma issues
        json_str = re.sub(r",\s*}", "}", json_str)
        json_str = re.sub(r",\s*]", "]", json_str)

        # Fix unescaped quotes in strings (common LLM error)
        json_str = re.sub(
            r':\s*"([^"]*)"([^",\]\}]*)"([^",\]\}]*)"', r': "\1\2\3"', json_str
        )

        # Fix missing quotes around string values
        json_str = re.sub(
            r':\s*([^"\{\[\]\},\s][^,\}\]]*?)(\s*[,\}\]])', r': "\1"\2', json_str
        )

        logger.debug(f"Cleaned JSON string: {json_str}")

        # Parse the JSON
        result = json.loads(json_str)

        # Validate required fields
        required_fields = [
            "supports_purchase",
            "is_valid",
            "confidence_score",
            "reasoning",
        ]
        for field in required_fields:
            if field not in result:
                raise ValueError(f"Missing required field: {field}")

        # Ensure confidence_score is between 0 and 1
        if not isinstance(result["confidence_score"], (int, float)):
            try:
                result["confidence_score"] = float(result["confidence_score"])
            except (ValueError, TypeError):
                result["confidence_score"] = 0.5

        if not 0 <= result["confidence_score"] <= 1:
            result["confidence_score"] = max(0, min(1, result["confidence_score"]))

        # Ensure boolean fields are actually booleans
        for bool_field in ["supports_purchase", "is_valid"]:
            if not isinstance(result[bool_field], bool):
                if isinstance(result[bool_field], str):
                    result[bool_field] = result[bool_field].lower() in [
                        "true",
                        "1",
                        "yes",
                    ]
                else:
                    result[bool_field] = bool(result[bool_field])

        # Ensure relevant_clauses is a list
        if "relevant_clauses" not in result:
            result["relevant_clauses"] = []
        elif not isinstance(result["relevant_clauses"], list):
            if isinstance(result["relevant_clauses"], str):
                # Try to split by common delimiters if it's a string
                result["relevant_clauses"] = [
                    clause.strip()
                    for clause in result["relevant_clauses"].split(";")
                    if clause.strip()
                ]
            else:
                result["relevant_clauses"] = [str(result["relevant_clauses"])]

        # Ensure reasoning is a string
        if not isinstance(result["reasoning"], str):
            result["reasoning"] = str(result["reasoning"])

        return result

    except (json.JSONDecodeError, ValueError) as e:
        logger.error(f"Failed to parse LLM response: {e}")
        logger.error(f"Raw response (first 1000 chars): {response_text[:1000]}")

        # Try to extract reasoning from the raw text as a fallback
        reasoning_text = "Failed to parse LLM response properly."
        if response_text:
            # Look for any text that might be reasoning
            sentences = re.split(r"[.!?]+", response_text)
            meaningful_sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
            if meaningful_sentences:
                reasoning_text = (
                    meaningful_sentences[0][:200]
                    + "... (parsed from malformed response)"
                )

        # Return a fallback result with low confidence
        return {
            "supports_purchase": False,
            "is_valid": False,
            "validity_end_date": None,
            "confidence_score": 0.1,
            "reasoning": f"{reasoning_text} Error: {str(e)}",
            "relevant_clauses": [],
        }


def get_current_date_string() -> str:
    """Get the current date as a string in YYYY-MM-DD format.

    Returns:
        str: Current date in YYYY-MM-DD format
    """
    return datetime.now().strftime("%Y-%m-%d")


def validate_date_format(date_string: str) -> bool:
    """Validate if a date string is in YYYY-MM-DD format.

    Args:
        date_string: Date string to validate

    Returns:
        bool: True if valid, False otherwise
    """
    try:
        datetime.strptime(date_string, "%Y-%m-%d")
        return True
    except ValueError:
        return False
