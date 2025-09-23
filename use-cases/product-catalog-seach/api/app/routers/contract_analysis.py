"""Contract analysis router for LLM-based contract evaluation."""

import logging
from typing import Dict, Any
from fastapi import APIRouter, HTTPException, Depends

# Gen AI Hub imports
from gen_ai_hub.proxy.native.openai import chat

from ..models.contract_analysis import (
    ContractAnalysisRequest,
    ContractAnalysisResponse,
    ContractAnalysisResult,
    BulkContractAnalysisRequest,
    BulkContractAnalysisResponse,
)
from ..utils.contract_analysis_utils import (
    get_contract_file_path,
    extract_text_from_pdf,
    build_analysis_prompt,
    parse_llm_response,
    get_current_date_string,
    validate_date_format,
)
from ..security import get_api_key

# Setup logging
logger = logging.getLogger(__name__)

router = APIRouter(dependencies=[Depends(get_api_key)])


@router.post("/analyze", response_model=ContractAnalysisResponse)
async def analyze_contract(
    request: ContractAnalysisRequest,
) -> ContractAnalysisResponse:
    """Analyze a contract to determine if it supports purchasing a specific product.

    Uses an LLM to analyze contract text and determine:
    1. Whether the contract supports buying the specified product
    2. Whether the contract is still valid on the current date

    Args:
        request: ContractAnalysisRequest containing contract name, product details, and analysis parameters.

    Returns:
        ContractAnalysisResponse: Analysis results including purchase support, validity status,
        confidence score, reasoning, and relevant contract clauses.

    Raises:
        HTTPException: If contract file is not found, analysis fails, or other errors occur.
    """
    try:
        logger.info(
            f"Analyzing contract '{request.contract_name}' for product '{request.product_name}'"
        )

        # Validate and set current date
        current_date = request.current_date or get_current_date_string()
        if not validate_date_format(current_date):
            raise HTTPException(
                status_code=400,
                detail=f"Invalid date format. Expected YYYY-MM-DD, got: {current_date}",
            )

        # Get contract file path and extract text
        try:
            contract_file_path = get_contract_file_path(request.contract_name)
            contract_text = extract_text_from_pdf(contract_file_path)
        except FileNotFoundError as e:
            logger.error(f"Contract file not found: {e}")
            raise HTTPException(status_code=404, detail=str(e))
        except ValueError as e:
            logger.error(f"Access denied to contract file: {e}")
            raise HTTPException(status_code=403, detail=str(e))

        # Build analysis prompt
        prompt = build_analysis_prompt(
            contract_text=contract_text,
            product_name=request.product_name,
            product_description=request.product_description,
            current_date=current_date,
        )

        # Call LLM for analysis
        messages: list[Dict[str, str]] = [{"role": "user", "content": prompt}]

        response = chat.completions.create(
            messages=messages,
            model="gpt-4.1",
            temperature=request.temperature,
            max_tokens=request.max_tokens,
        )

        llm_response_text: str = response.choices[0].message.content
        usage: Dict[str, int] = {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
        }

        # Parse LLM response
        analysis_data = parse_llm_response(llm_response_text)

        # Create analysis result
        result = ContractAnalysisResult(
            product_name=request.product_name,
            supports_purchase=analysis_data["supports_purchase"],
            is_valid=analysis_data["is_valid"],
            validity_end_date=analysis_data.get("validity_end_date"),
            confidence_score=analysis_data["confidence_score"],
            reasoning=analysis_data["reasoning"],
            relevant_clauses=analysis_data["relevant_clauses"],
        )

        logger.info(
            f"Contract analysis completed successfully for '{request.product_name}'"
        )

        return ContractAnalysisResponse(
            contract_name=request.contract_name,
            analysis_date=current_date,
            result=result,
            model="gpt-4.1",
            success=True,
            usage=usage,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Contract analysis error: {e}")
        return ContractAnalysisResponse(
            contract_name=request.contract_name,
            analysis_date=request.current_date or get_current_date_string(),
            result=None,
            model="gpt-4.1",
            success=False,
            error=str(e),
        )


@router.post("/analyze-bulk", response_model=BulkContractAnalysisResponse)
async def analyze_contract_bulk(
    request: BulkContractAnalysisRequest,
) -> BulkContractAnalysisResponse:
    """Analyze a contract against multiple products in a single request.

    Efficiently analyzes one contract against multiple products by sending all
    product information in a single LLM call.

    Args:
        request: BulkContractAnalysisRequest containing contract name and list of products.

    Returns:
        BulkContractAnalysisResponse: Analysis results for all products.

    Raises:
        HTTPException: If contract file is not found or analysis fails.
    """
    try:
        logger.info(
            f"Bulk analyzing contract '{request.contract_name}' for {len(request.products)} products"
        )

        # Validate and set current date
        current_date = request.current_date or get_current_date_string()
        if not validate_date_format(current_date):
            raise HTTPException(
                status_code=400,
                detail=f"Invalid date format. Expected YYYY-MM-DD, got: {current_date}",
            )

        # Get contract file path and extract text
        try:
            contract_file_path = get_contract_file_path(request.contract_name)
            contract_text = extract_text_from_pdf(contract_file_path)
        except FileNotFoundError as e:
            logger.error(f"Contract file not found: {e}")
            raise HTTPException(status_code=404, detail=str(e))
        except ValueError as e:
            logger.error(f"Access denied to contract file: {e}")
            raise HTTPException(status_code=403, detail=str(e))

        # Build bulk analysis prompt
        products_info = "\n".join(
            [
                f"Product {i+1}: {product.get('name', 'Unknown')}"
                + (
                    f"\nDescription: {product.get('description', '')}"
                    if product.get("description")
                    else ""
                )
                for i, product in enumerate(request.products)
            ]
        )

        prompt = f"""
You are a contract analysis expert. Please analyze the following contract to determine for each product:

1. Whether the contract supports purchasing the product
2. Whether the contract is still valid on the current date

CONTRACT TEXT:
{contract_text}

PRODUCTS TO ANALYZE:
{products_info}

CURRENT DATE: {current_date}

IMPORTANT: You must respond with ONLY valid JSON. Do not include any explanatory text, comments, or additional formatting.

Please provide your analysis in the following JSON format:
{{
    "results": [
        {{
            "product_name": "Product 1 name",
            "supports_purchase": true,
            "is_valid": true,
            "validity_end_date": "2025-12-31",
            "confidence_score": 0.95,
            "reasoning": "Detailed explanation for this product",
            "relevant_clauses": ["Contract clause 1", "Contract clause 2"]
        }},
        {{
            "product_name": "Product 2 name",
            "supports_purchase": false,
            "is_valid": true,
            "validity_end_date": "2025-12-31",
            "confidence_score": 0.85,
            "reasoning": "Detailed explanation for this product",
            "relevant_clauses": ["Contract clause 3"]
        }}
    ]
}}

Analysis Guidelines:
- For "supports_purchase": Check if each product falls under the contract's scope of supply (boolean: true or false)
- For "is_valid": Compare contract validity dates with the current date (boolean: true or false, same for all products)
- For "validity_end_date": Extract the contract expiration date in YYYY-MM-DD format, or null if not found
- For "confidence_score": Rate your confidence in the analysis for each product (number between 0.0 and 1.0)
- For "reasoning": Provide clear explanation specific to each product (string)
- For "relevant_clauses": Extract contract text that supports your conclusions for each product (array of strings)

Be thorough and precise in your analysis. Analyze each product separately.

Remember: Return ONLY the JSON object, no additional text or comments.
"""

        # Call LLM for bulk analysis
        messages: list[Dict[str, str]] = [{"role": "user", "content": prompt}]

        response = chat.completions.create(
            messages=messages,
            model="gpt-4.1",
            temperature=request.temperature,
            max_tokens=request.max_tokens,
        )

        llm_response_text: str = response.choices[0].message.content
        usage: Dict[str, int] = {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
        }

        # Parse bulk LLM response using robust parsing function
        try:
            # Use the same robust parsing function as single analysis
            from app.utils.contract_analysis_utils import parse_llm_response
            import json
            import re

            # For bulk analysis, we need to extract the array of results
            cleaned_text = llm_response_text.strip()
            logger.debug(
                f"Attempting to parse bulk LLM response: {cleaned_text[:200]}..."
            )

            # Try multiple approaches to extract JSON array
            json_str = None

            # Approach 1: Look for JSON array between square brackets
            start_idx = cleaned_text.find("[")
            if start_idx != -1:
                # Find the matching closing bracket by counting brackets
                bracket_count = 0
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
                        if char == "[":
                            bracket_count += 1
                        elif char == "]":
                            bracket_count -= 1
                            if bracket_count == 0:
                                end_idx = i + 1
                                break

                if bracket_count == 0:  # Found matching closing bracket
                    json_str = cleaned_text[start_idx:end_idx]

            # Approach 2: Look for JSON in code blocks
            if not json_str:
                json_match = re.search(
                    r"```(?:json)?\s*(\[.*?\])\s*```", cleaned_text, re.DOTALL
                )
                if json_match:
                    json_str = json_match.group(1)

            # Approach 3: If still no JSON array, try the entire response if it looks like an array
            if (
                not json_str
                and cleaned_text.startswith("[")
                and cleaned_text.endswith("]")
            ):
                json_str = cleaned_text

            # Approach 4: Look for results wrapped in an object
            if not json_str:
                obj_match = re.search(
                    r'\{\s*"results"\s*:\s*(\[.*?\])\s*\}', cleaned_text, re.DOTALL
                )
                if obj_match:
                    json_str = obj_match.group(1)

            # Approach 5: Look for the largest array-like structure
            if not json_str:
                matches = re.finditer(
                    r"\[[^\[\]]*(?:\[[^\[\]]*\][^\[\]]*)*\]", cleaned_text, re.DOTALL
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
                raise ValueError("No valid JSON array found in response")

            # Clean up common JSON formatting issues
            json_str = json_str.strip()
            json_str = re.sub(r"\s+", " ", json_str)
            json_str = re.sub(r",\s*}", "}", json_str)
            json_str = re.sub(r",\s*]", "]", json_str)

            logger.debug(f"Cleaned bulk JSON string: {json_str}")

            # Parse the JSON array
            results_array = json.loads(json_str)

            # Validate that we have an array of analysis results
            if not isinstance(results_array, list):
                raise ValueError("Expected JSON array of analysis results")

            # Process each result in the array
            results = []
            for i, result_data in enumerate(results_array):
                # Ensure each result has the required fields
                if not isinstance(result_data, dict):
                    logger.warning(f"Bulk result {i} is not a dict, skipping")
                    continue

                # Apply the same validation as single analysis
                required_fields = [
                    "supports_purchase",
                    "is_valid",
                    "confidence_score",
                    "reasoning",
                ]
                for field in required_fields:
                    if field not in result_data:
                        result_data[field] = (
                            False
                            if field in ["supports_purchase", "is_valid"]
                            else (
                                0.1 if field == "confidence_score" else "Missing field"
                            )
                        )

                # Ensure data types are correct
                if not isinstance(result_data.get("confidence_score"), (int, float)):
                    try:
                        result_data["confidence_score"] = float(
                            result_data["confidence_score"]
                        )
                    except (ValueError, TypeError):
                        result_data["confidence_score"] = 0.5

                if not 0 <= result_data["confidence_score"] <= 1:
                    result_data["confidence_score"] = max(
                        0, min(1, result_data["confidence_score"])
                    )

                # Ensure boolean fields are actually booleans
                for bool_field in ["supports_purchase", "is_valid"]:
                    if not isinstance(result_data[bool_field], bool):
                        if isinstance(result_data[bool_field], str):
                            result_data[bool_field] = result_data[
                                bool_field
                            ].lower() in ["true", "1", "yes"]
                        else:
                            result_data[bool_field] = bool(result_data[bool_field])

                # Ensure relevant_clauses is a list
                if "relevant_clauses" not in result_data:
                    result_data["relevant_clauses"] = []
                elif not isinstance(result_data["relevant_clauses"], list):
                    if isinstance(result_data["relevant_clauses"], str):
                        result_data["relevant_clauses"] = [
                            clause.strip()
                            for clause in result_data["relevant_clauses"].split(";")
                            if clause.strip()
                        ]
                    else:
                        result_data["relevant_clauses"] = [
                            str(result_data["relevant_clauses"])
                        ]

                # Ensure reasoning is a string
                if not isinstance(result_data["reasoning"], str):
                    result_data["reasoning"] = str(result_data["reasoning"])

                # Create the ContractAnalysisResult
                product_name = (
                    request.products[i].get("name", "Unknown Product")
                    if i < len(request.products)
                    else f"Product {i+1}"
                )
                result = ContractAnalysisResult(
                    product_name=product_name,
                    supports_purchase=result_data["supports_purchase"],
                    is_valid=result_data["is_valid"],
                    validity_end_date=result_data.get("validity_end_date"),
                    confidence_score=result_data["confidence_score"],
                    reasoning=result_data["reasoning"],
                    relevant_clauses=result_data["relevant_clauses"],
                )
                results.append(result)

        except (json.JSONDecodeError, ValueError, IndexError) as e:
            logger.error(f"Failed to parse bulk LLM response: {e}")
            logger.debug(f"Raw response (first 500 chars): {llm_response_text[:500]}")
            # Create fallback results
            results = []
            for product in request.products:
                result = ContractAnalysisResult(
                    product_name=product.get("name", "Unknown Product"),
                    supports_purchase=False,
                    is_valid=False,
                    validity_end_date=None,
                    confidence_score=0.1,
                    reasoning=f"Failed to parse LLM response. Error: {str(e)}",
                    relevant_clauses=[],
                )
                results.append(result)

        logger.info(f"Bulk contract analysis completed for {len(results)} products")

        return BulkContractAnalysisResponse(
            contract_name=request.contract_name,
            analysis_date=current_date,
            results=results,
            model="gpt-4.1",
            success=True,
            usage=usage,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Bulk contract analysis error: {e}")
        return BulkContractAnalysisResponse(
            contract_name=request.contract_name,
            analysis_date=request.current_date or get_current_date_string(),
            results=[],
            model="gpt-4.1",
            success=False,
            error=str(e),
        )


@router.get("/contracts")
async def list_contracts():
    """List available contract files for analysis.

    Returns:
        Dict: List of available contract files.
    """
    try:
        from ..utils.contract_analysis_utils import CONTRACTS_DIR

        if not CONTRACTS_DIR.exists():
            return {"contracts": [], "message": "Contracts directory not found"}

        contract_files = []
        for file_path in CONTRACTS_DIR.glob("*.pdf"):
            contract_files.append(
                {
                    "name": file_path.stem,  # filename without extension
                    "filename": file_path.name,  # full filename with extension
                    "size": file_path.stat().st_size,
                }
            )

        return {
            "contracts": contract_files,
            "count": len(contract_files),
        }

    except Exception as e:
        logger.error(f"Error listing contracts: {e}")
        raise HTTPException(status_code=500, detail="Failed to list contracts")
