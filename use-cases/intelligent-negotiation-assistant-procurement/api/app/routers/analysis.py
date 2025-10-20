from __future__ import annotations

from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from ..security import get_api_key
from ..services.analysis_service import AnalysisService

router = APIRouter(dependencies=[Depends(get_api_key)], tags=["analysis"])
service = AnalysisService()


class AnalyzeRequest(BaseModel):
    supplier_id: Optional[str] = None
    kg_path: Optional[str] = None
    model: Optional[str] = None
    options: Optional[Dict[str, Any]] = None


@router.post("/v1/analyze/cost")
async def analyze_cost(req: AnalyzeRequest) -> Dict[str, Any]:
    try:
        return service.run_cost(supplier_id=req.supplier_id, kg_path=req.kg_path, model=req.model)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/v1/analyze/risk")
async def analyze_risk(req: AnalyzeRequest) -> Dict[str, Any]:
    try:
        return service.run_risk(supplier_id=req.supplier_id, kg_path=req.kg_path, model=req.model)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class PartsAnalyzeRequest(AnalyzeRequest):
    core_part_categories: Optional[list[str]] = None


@router.post("/v1/analyze/parts")
async def analyze_parts(req: PartsAnalyzeRequest) -> Dict[str, Any]:
    try:
        return service.run_parts(
            supplier_id=req.supplier_id,
            kg_path=req.kg_path,
            model=req.model,
            core_part_categories=req.core_part_categories,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/v1/analyze/homepage")
async def analyze_homepage(req: AnalyzeRequest) -> Dict[str, Any]:
    try:
        return service.run_homepage(supplier_id=req.supplier_id, kg_path=req.kg_path, model=req.model)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class TqdcsAnalyzeRequest(AnalyzeRequest):
    prior_analyses: Optional[Dict[str, Any]] = None
    weights: Optional[Dict[str, float]] = None


@router.post("/v1/analyze/tqdcs")
async def analyze_tqdcs(req: TqdcsAnalyzeRequest) -> Dict[str, Any]:
    try:
        return service.run_tqdcs(
            supplier_id=req.supplier_id,
            kg_path=req.kg_path,
            model=req.model,
            prior_analyses=req.prior_analyses,
            weights=req.weights,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class CompareRequest(BaseModel):
    supplier1_name: str
    supplier2_name: str
    supplier1_analyses: Optional[Dict[str, Any]] = None
    supplier2_analyses: Optional[Dict[str, Any]] = None
    tqdcs_weights: Optional[Dict[str, float]] = None
    generate_metrics: bool = True
    generate_strengths_weaknesses: bool = True
    generate_recommendation_and_split: bool = True
    model: Optional[str] = None


@router.post("/v1/analyze/compare")
async def analyze_compare(req: CompareRequest) -> Dict[str, Any]:
    try:
        return service.run_compare(
            supplier1_name=req.supplier1_name,
            supplier2_name=req.supplier2_name,
            supplier1_analyses=req.supplier1_analyses,
            supplier2_analyses=req.supplier2_analyses,
            tqdcs_weights=req.tqdcs_weights,
            generate_metrics=req.generate_metrics,
            generate_strengths_weaknesses=req.generate_strengths_weaknesses,
            generate_recommendation_and_split=req.generate_recommendation_and_split,
            model=req.model,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class SupplierFullRequest(BaseModel):
    supplier_id: Optional[str] = None
    kg_path: Optional[str] = None
    model: Optional[str] = None
    core_part_categories: Optional[list[str]] = None
    force_refresh: bool = False


@router.post("/v1/analyze/supplier_full")
async def analyze_supplier_full(req: SupplierFullRequest) -> Dict[str, Any]:
    try:
        return service.run_supplier_full(
            supplier_id=req.supplier_id,
            kg_path=req.kg_path,
            model=req.model,
            core_part_categories=req.core_part_categories,
            force_refresh=req.force_refresh,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class CompleteRequest(BaseModel):
    supplier1_id: str
    supplier2_id: str
    model: Optional[str] = None
    comparator_model: Optional[str] = None
    core_part_categories: Optional[list[str]] = None
    tqdcs_weights: Optional[Dict[str, float]] = None
    force_refresh: bool = False


@router.post("/v1/analyze/complete")
async def analyze_complete(req: CompleteRequest) -> Dict[str, Any]:
    try:
        return service.run_complete(
            supplier1_id=req.supplier1_id,
            supplier2_id=req.supplier2_id,
            model=req.model,
            comparator_model=req.comparator_model,
            core_part_categories=req.core_part_categories,
            tqdcs_weights=req.tqdcs_weights,
            force_refresh=req.force_refresh,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class EnsureRequest(BaseModel):
    supplier1_id: str
    supplier2_id: str
    model: Optional[str] = None
    comparator_model: Optional[str] = None
    core_part_categories: Optional[list[str]] = None
    tqdcs_weights: Optional[Dict[str, float]] = None
    force_refresh: bool = False
    generate_metrics: bool = True
    generate_strengths_weaknesses: bool = True
    generate_recommendation_and_split: bool = True


@router.post("/v1/analyze/ensure")
async def analyze_ensure(req: EnsureRequest) -> Dict[str, Any]:
    try:
        return service.ensure_complete_or_compare(
            supplier1_id=req.supplier1_id,
            supplier2_id=req.supplier2_id,
            model=req.model,
            comparator_model=req.comparator_model,
            core_part_categories=req.core_part_categories,
            tqdcs_weights=req.tqdcs_weights,
            force_refresh=req.force_refresh,
            generate_metrics=req.generate_metrics,
            generate_strengths_weaknesses=req.generate_strengths_weaknesses,
            generate_recommendation_and_split=req.generate_recommendation_and_split,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class CacheStatusRequest(BaseModel):
    supplier1_id: str
    supplier2_id: str
    core_part_categories: Optional[list[str]] = None


@router.post("/v1/analyze/cache_status")
async def analyze_cache_status(req: CacheStatusRequest) -> Dict[str, Any]:
    try:
        return service.get_cache_status(
            supplier1_id=req.supplier1_id,
            supplier2_id=req.supplier2_id,
            core_part_categories=req.core_part_categories,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
