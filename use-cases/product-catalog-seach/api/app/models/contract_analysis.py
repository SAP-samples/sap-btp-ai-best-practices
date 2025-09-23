"""Contract analysis models for LLM-based contract evaluation."""

from pydantic import BaseModel
from typing import Optional, Dict, Any, List
from datetime import datetime


class ContractAnalysisRequest(BaseModel):
    """Request model for contract analysis endpoints.

    Attributes:
        contract_name: Name of the contract file to analyze (without .pdf extension).
        product_name: Name of the product to check against the contract.
        product_description: Optional description of the product for better analysis.
        current_date: Current date for validity check. If not provided, uses today's date.
        temperature: Controls response randomness (0.0-1.0). Defaults to 0.1 for more deterministic analysis.
        max_tokens: Maximum number of tokens in the response. Defaults to 2000.
    """

    contract_name: str
    product_name: str
    product_description: Optional[str] = None
    current_date: Optional[str] = None  # Format: YYYY-MM-DD
    temperature: Optional[float] = 0.1
    max_tokens: Optional[int] = 2000


class ContractAnalysisResult(BaseModel):
    """Result of contract analysis for a specific product.

    Attributes:
        product_name: Name of the analyzed product.
        supports_purchase: Whether the contract supports buying this product.
        is_valid: Whether the contract is still valid on the current date.
        validity_end_date: End date of contract validity if found.
        confidence_score: Confidence level of the analysis (0.0-1.0).
        reasoning: Explanation of the analysis decision.
        relevant_clauses: List of relevant contract clauses that influenced the decision.
    """

    product_name: str
    supports_purchase: bool
    is_valid: bool
    validity_end_date: Optional[str] = None
    confidence_score: float
    reasoning: str
    relevant_clauses: List[str] = []


class ContractAnalysisResponse(BaseModel):
    """Response model for contract analysis endpoints.

    Attributes:
        contract_name: Name of the analyzed contract.
        analysis_date: Date when the analysis was performed.
        result: Analysis result for the requested product.
        model: Identifier of the LLM model used for analysis.
        success: Whether the analysis completed successfully.
        usage: Token usage statistics from the LLM call.
        error: Error message if analysis failed.
    """

    contract_name: str
    analysis_date: str
    result: Optional[ContractAnalysisResult] = None
    model: str
    success: bool
    usage: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class BulkContractAnalysisRequest(BaseModel):
    """Request model for analyzing multiple products against a contract.

    Attributes:
        contract_name: Name of the contract file to analyze.
        products: List of products to analyze against the contract.
        current_date: Current date for validity check.
        temperature: Controls response randomness.
        max_tokens: Maximum number of tokens in the response.
    """

    contract_name: str
    products: List[Dict[str, str]]  # [{"name": "Product A", "description": "..."}]
    current_date: Optional[str] = None
    temperature: Optional[float] = 0.1
    max_tokens: Optional[int] = 3000


class BulkContractAnalysisResponse(BaseModel):
    """Response model for bulk contract analysis.

    Attributes:
        contract_name: Name of the analyzed contract.
        analysis_date: Date when the analysis was performed.
        results: List of analysis results for each product.
        model: Identifier of the LLM model used.
        success: Whether the analysis completed successfully.
        usage: Token usage statistics.
        error: Error message if analysis failed.
    """

    contract_name: str
    analysis_date: str
    results: List[ContractAnalysisResult] = []
    model: str
    success: bool
    usage: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
