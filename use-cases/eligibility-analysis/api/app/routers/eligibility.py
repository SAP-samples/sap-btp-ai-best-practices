"""
Eligibility API Router

Endpoints for invoice eligibility analysis, including:
- Upload and analyze offer files
- Download generated summary files
- Query seller rejection statistics
- View current configuration
- Pattern analysis and trend detection
- Bulk historical import
"""

import io
import logging
from datetime import date, datetime
from typing import List, Optional

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile, status
from fastapi.responses import Response, StreamingResponse

from ..config.eligibility_config import EligibilitySettings, get_output_directory
from ..models.eligibility import (
    AnalysisResponse,
    ConfigResponse,
    CustomerLogEntry,
    CustomerLogSummary,
    SellerHistoryResponse,
)
from ..models.patterns import (
    BulkImportResult,
    DebtorRuleProfile,
    PatternFilters,
    PatternSummary,
    TrendDataPoint,
)
from ..security import get_api_key
from ..services.eligibility.customer_log import CustomerLogService
from ..services.eligibility.engine import EligibilityEngine
from ..services.eligibility.excel_generator import ExcelGenerator
from ..services.eligibility.parser import parse_offer_file
from ..services.eligibility.pattern_analyzer import PatternAnalyzer

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/eligibility",
    tags=["eligibility"],
    dependencies=[Depends(get_api_key)],
)

# Service instances (lazy initialization)
_customer_log_service: Optional[CustomerLogService] = None
_excel_generator: Optional[ExcelGenerator] = None
_pattern_analyzer: Optional[PatternAnalyzer] = None


def get_customer_log_service() -> CustomerLogService:
    """Get or create the customer log service instance."""
    global _customer_log_service
    if _customer_log_service is None:
        _customer_log_service = CustomerLogService()
    return _customer_log_service


def get_excel_generator() -> ExcelGenerator:
    """Get or create the Excel generator instance."""
    global _excel_generator
    if _excel_generator is None:
        _excel_generator = ExcelGenerator()
    return _excel_generator


def get_pattern_analyzer() -> PatternAnalyzer:
    """Get or create the pattern analyzer instance."""
    global _pattern_analyzer
    if _pattern_analyzer is None:
        _pattern_analyzer = PatternAnalyzer()
    return _pattern_analyzer


def _build_settings(
    purchase_date: Optional[date],
    nddt: Optional[int],
    teih: Optional[int],
    isspur: Optional[int],
    eligible_currencies: Optional[str],
) -> tuple[date, EligibilitySettings]:
    """Build purchase date and EligibilitySettings from query parameters."""
    if purchase_date is None:
        purchase_date = date.today()
    currencies = None
    if eligible_currencies:
        currencies = [c.strip().upper() for c in eligible_currencies.split(",")]
    settings = EligibilitySettings(
        nddt=nddt,
        teih=teih,
        isspur=isspur,
        eligible_currencies=currencies,
    )
    return purchase_date, settings


@router.post("/analyze", response_model=AnalysisResponse)
async def analyze_offer_file(
    file: UploadFile = File(..., description="Excel offer file to analyze"),
    purchase_date: Optional[date] = Query(
        None,
        description="Purchase date for calculations (defaults to today)",
    ),
    nddt: Optional[int] = Query(
        None,
        ge=0,
        description="Minimum days to due date (R1). Default: 6",
    ),
    teih: Optional[int] = Query(
        None,
        ge=1,
        description="Maximum tenor days (R16). Default: 15",
    ),
    isspur: Optional[int] = Query(
        None,
        ge=0,
        description="Minimum days since issuance (R17). Default: 0",
    ),
    eligible_currencies: Optional[str] = Query(
        None,
        description="Comma-separated allowed currencies (R11). Default: EUR,USD",
    ),
) -> AnalysisResponse:
    """
    Upload an offer Excel file and analyze invoice eligibility.

    The analysis applies all eligibility rules to each invoice and returns:
    - Summary counts of funded vs non-funded invoices
    - Detailed lists of both categories
    - Generated Excel file for download

    All threshold parameters are optional and will use defaults if not specified.
    """
    logger.info(f"Analyzing offer file: {file.filename}")

    purchase_date, settings = _build_settings(
        purchase_date, nddt, teih, isspur, eligible_currencies
    )

    try:
        # Read file content
        content = await file.read()

        # Parse Excel file
        invoices = parse_offer_file(content, file.filename or "offer.xlsx")

        if not invoices:
            return AnalysisResponse(
                success=True,
                total_invoices=0,
                funded_count=0,
                non_funded_count=0,
                funded_invoices=[],
                non_funded_invoices=[],
                settings_used=settings.to_dict(),
            )

        # Run eligibility analysis
        engine = EligibilityEngine(settings=settings)
        results, funded, non_funded = engine.analyze_batch(
            invoices=invoices,
            purchase_date=purchase_date,
        )

        # Log results to database
        customer_log = get_customer_log_service()
        customer_log.log_batch(invoices, results, purchase_date=purchase_date)

        # Generate output Excel
        excel_gen = get_excel_generator()
        filename = excel_gen.generate_filename()
        excel_gen.generate_summary(funded, non_funded, filename)

        logger.info(
            f"Analysis complete: {len(funded)} funded, {len(non_funded)} non-funded"
        )

        return AnalysisResponse(
            success=True,
            total_invoices=len(invoices),
            funded_count=len(funded),
            non_funded_count=len(non_funded),
            funded_invoices=funded,
            non_funded_invoices=non_funded,
            output_file=filename,
            settings_used=settings.to_dict(),
        )

    except ValueError as e:
        logger.error(f"Validation error: {e}")
        return AnalysisResponse(
            success=False,
            total_invoices=0,
            funded_count=0,
            non_funded_count=0,
            error=str(e),
        )
    except Exception as e:
        logger.error(f"Analysis error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Analysis failed: {str(e)}",
        )


@router.get("/download/{filename}")
async def download_file(filename: str) -> Response:
    """
    Download a generated summary Excel file.

    The filename is returned in the analyze response's output_file field.
    """
    excel_gen = get_excel_generator()
    filepath = excel_gen.get_generated_file_path(filename)

    if filepath is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"File not found: {filename}",
        )

    # Read file content
    with open(filepath, "rb") as f:
        content = f.read()

    return Response(
        content=content,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
        },
    )


@router.get("/seller/{seller_id}/summary", response_model=CustomerLogSummary)
async def get_seller_summary(seller_id: str) -> CustomerLogSummary:
    """
    Get non-eligibility statistics for a specific seller.

    Returns historical analysis including:
    - Total invoices processed
    - Eligibility rate
    - Breakdown of non-eligibility by rule (e.g., "70% of non-eligible invoices are R1")
    """
    customer_log = get_customer_log_service()
    summary = customer_log.get_seller_summary(seller_id)
    return summary


@router.get("/seller/{seller_id}/history", response_model=SellerHistoryResponse)
async def get_seller_history(
    seller_id: str,
    limit: int = Query(100, ge=1, le=1000, description="Maximum entries to return"),
    offset: int = Query(0, ge=0, description="Number of entries to skip"),
) -> SellerHistoryResponse:
    """
    Get paginated history of processed invoices for a seller.
    """
    customer_log = get_customer_log_service()
    history = customer_log.get_seller_history(seller_id, limit=limit, offset=offset)
    return SellerHistoryResponse(records=history, total=len(history))


@router.get("/config", response_model=ConfigResponse)
async def get_config() -> ConfigResponse:
    """
    Get current default configuration values.

    These are the default values used when query parameters are not specified.
    """
    settings = EligibilitySettings()
    return ConfigResponse(
        nddt=settings.nddt,
        teih=settings.teih,
        isspur=settings.isspur,
        eligible_currencies=settings.eligible_currencies,
    )


@router.post("/customer-logs/reset")
async def reset_customer_logs() -> dict:
    """Delete all customer log entries."""
    customer_log = get_customer_log_service()
    deleted = customer_log.reset_logs()
    return {"success": True, "deleted": deleted}


# ------------------------------------------------------------------
# Pattern Analysis Endpoints
# ------------------------------------------------------------------


@router.get("/patterns", response_model=PatternSummary)
async def get_patterns(
    seller_id: Optional[str] = Query(
        None, description="Filter patterns to a specific seller"
    ),
    debtor_id: Optional[str] = Query(
        None, description="Filter patterns to a specific debtor"
    ),
    programa: Optional[str] = Query(
        None, description="Filter patterns to a specific program"
    ),
    insurer_id: Optional[str] = Query(
        None, description="Filter patterns to a specific insurer"
    ),
    lookback_days: int = Query(
        90, ge=1, le=365, description="Number of days of history to analyze"
    ),
    min_invoices: int = Query(
        3, ge=1, le=100, description="Minimum invoices for a pattern to be flagged"
    ),
) -> PatternSummary:
    """
    Detect recurring non-eligibility patterns in historical eligibility data.

    Returns alerts for five pattern types:
    - Chronic Rule Failures: debtor consistently fails a specific rule
    - Trending Increases: non-eligibility rate worsening over time
    - Repeat Offenders: debtor non-eligible in most batches
    - Rule Concentration: single rule dominates all non-eligibility
    - Amount at Risk: high financial impact from non-eligible invoices
    """
    analyzer = get_pattern_analyzer()
    filters = PatternFilters(
        seller_id=seller_id,
        debtor_id=debtor_id,
        programa=programa,
        insurer_id=insurer_id,
        lookback_days=lookback_days,
        min_invoices=min_invoices,
    )
    return analyzer.analyze_all(filters=filters)


@router.get("/patterns/filters")
async def get_pattern_filter_options(
    lookback_days: int = Query(90, ge=1, le=365),
) -> dict:
    """Get distinct values for each filter dimension within the lookback window."""
    analyzer = get_pattern_analyzer()
    return analyzer.get_filter_options(lookback_days=lookback_days)


@router.get("/patterns/debtor-profiles", response_model=List[DebtorRuleProfile])
async def get_debtor_profiles(
    seller_id: Optional[str] = Query(
        None, description="Filter to a specific seller"
    ),
    debtor_id: Optional[str] = Query(
        None, description="Filter to a specific debtor"
    ),
    programa: Optional[str] = Query(
        None, description="Filter to a specific program"
    ),
    insurer_id: Optional[str] = Query(
        None, description="Filter to a specific insurer"
    ),
    lookback_days: int = Query(
        90, ge=1, le=365, description="Number of days of history to analyze"
    ),
) -> List[DebtorRuleProfile]:
    """
    Get per-debtor non-eligibility profiles with rule breakdowns.

    Returns one profile per (seller, debtor) pair, sorted by non-eligibility
    rate descending. Each profile includes the dominant failure rule,
    amounts at risk, and batch-level statistics.
    """
    analyzer = get_pattern_analyzer()
    filters = PatternFilters(
        seller_id=seller_id,
        debtor_id=debtor_id,
        programa=programa,
        insurer_id=insurer_id,
        lookback_days=lookback_days,
    )
    return analyzer.get_debtor_profiles(filters=filters)


@router.get("/patterns/trend", response_model=List[TrendDataPoint])
async def get_rejection_trend(
    seller_id: Optional[str] = Query(
        None, description="Filter to a specific seller"
    ),
    debtor_id: Optional[str] = Query(
        None, description="Filter to a specific debtor"
    ),
    programa: Optional[str] = Query(
        None, description="Filter to a specific program"
    ),
    insurer_id: Optional[str] = Query(
        None, description="Filter to a specific insurer"
    ),
    granularity: str = Query(
        "week",
        description="Time bucket granularity: day, week, month, or quarter",
        pattern="^(day|week|month|quarter)$",
    ),
    lookback_days: int = Query(
        90, ge=1, le=365, description="Number of days of history to analyze"
    ),
) -> List[TrendDataPoint]:
    """
    Get non-eligibility rate time-series for charting.

    Returns chronologically sorted data points at the specified granularity,
    each containing total invoices, non-eligible count, non-eligibility rate,
    and per-rule breakdown.
    """
    analyzer = get_pattern_analyzer()
    filters = PatternFilters(
        seller_id=seller_id,
        debtor_id=debtor_id,
        programa=programa,
        insurer_id=insurer_id,
        lookback_days=lookback_days,
    )
    return analyzer.get_rejection_trend(
        granularity=granularity,
        filters=filters,
    )


@router.get("/patterns/report")
async def get_pattern_report(
    seller_id: Optional[str] = Query(None, description="Filter to a specific seller"),
    debtor_id: Optional[str] = Query(None, description="Filter to a specific debtor"),
    programa: Optional[str] = Query(None, description="Filter to a specific program"),
    insurer_id: Optional[str] = Query(None, description="Filter to a specific insurer"),
    lookback_days: int = Query(90, ge=1, le=365),
    min_invoices: int = Query(3, ge=1, le=100),
    format: str = Query("pdf", pattern="^(pdf)$"),
) -> StreamingResponse:
    """Generate and download a pattern analysis PDF report."""
    filters = PatternFilters(
        seller_id=seller_id,
        debtor_id=debtor_id,
        programa=programa,
        insurer_id=insurer_id,
        lookback_days=lookback_days,
        min_invoices=min_invoices,
    )
    from ..services.eligibility.pattern_report_builder import generate_pattern_report

    pdf_bytes = generate_pattern_report(filters)
    return StreamingResponse(
        io.BytesIO(pdf_bytes),
        media_type="application/pdf",
        headers={"Content-Disposition": "attachment; filename=pattern_analysis_report.pdf"},
    )


# ------------------------------------------------------------------
# Bulk Historical Import
# ------------------------------------------------------------------


@router.post("/import-history", response_model=BulkImportResult)
async def import_history(
    files: List[UploadFile] = File(
        ..., description="One or more historical offer Excel files"
    ),
    purchase_date: Optional[date] = Query(
        None,
        description="Purchase date to use for all files (defaults to today)",
    ),
    nddt: Optional[int] = Query(None, ge=0, description="Minimum days to due date (R1)"),
    teih: Optional[int] = Query(None, ge=1, description="Maximum tenor days (R16)"),
    isspur: Optional[int] = Query(None, ge=0, description="Minimum days since issuance (R17)"),
    eligible_currencies: Optional[str] = Query(
        None, description="Comma-separated allowed currencies (R11)"
    ),
) -> BulkImportResult:
    """
    Import multiple historical offer files to build the analysis database.

    Each file is processed through the eligibility engine with the same rules,
    and all results are logged to the customer log database. This bootstraps
    the historical dataset needed for meaningful pattern analysis.
    """
    purchase_date, settings = _build_settings(
        purchase_date, nddt, teih, isspur, eligible_currencies
    )

    engine = EligibilityEngine(settings=settings)
    customer_log = get_customer_log_service()

    total_invoices = 0
    total_eligible = 0
    total_rejected = 0
    files_processed = 0
    errors: List[str] = []

    for file in files:
        try:
            content = await file.read()
            invoices = parse_offer_file(content, file.filename or "offer.xlsx")

            if not invoices:
                errors.append(f"{file.filename}: no invoices found")
                continue

            results, funded, non_funded = engine.analyze_batch(
                invoices=invoices,
                purchase_date=purchase_date,
            )

            customer_log.log_batch(invoices, results, purchase_date=purchase_date)

            total_invoices += len(invoices)
            total_eligible += len(funded)
            total_rejected += len(non_funded)
            files_processed += 1

            logger.info(
                f"Imported {file.filename}: {len(funded)} funded, "
                f"{len(non_funded)} non-funded"
            )
        except Exception as e:
            logger.error(f"Failed to import {file.filename}: {e}")
            errors.append(f"{file.filename}: {str(e)}")

    return BulkImportResult(
        success=len(errors) == 0,
        files_processed=files_processed,
        total_invoices=total_invoices,
        total_eligible=total_eligible,
        total_rejected=total_rejected,
        errors=errors,
    )
