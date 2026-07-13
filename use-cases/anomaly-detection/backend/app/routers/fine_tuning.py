from fastapi import APIRouter, UploadFile, File, HTTPException
from typing import Optional, List, Dict, Any
import shutil
import os
import json
import pandas as pd
from pathlib import Path
from services.data_loader import load_dataset, find_best_results_directory

router = APIRouter()

UPLOAD_DIR = Path("datasets/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

@router.get("/statistics")
def get_dataset_statistics():
    try:
        loaded_data = load_dataset()
        df = loaded_data.features
        
        if df is None or df.empty:
            return {
                "total_records": 0,
                "anomaly_rate": None,
                "unique_customers": 0,
                "unique_materials": 0
            }

        total_records = len(df)
        unique_customers = df["Sold To number"].nunique() if "Sold To number" in df.columns else 0
        unique_materials = df["Material Number"].nunique() if "Material Number" in df.columns else 0
        
        anomaly_rate = _get_current_anomaly_rate(df)
        
        return {
            "total_records": total_records,
            "anomaly_rate": anomaly_rate,
            "unique_customers": unique_customers,
            "unique_materials": unique_materials
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def _get_current_anomaly_rate(df: Optional[pd.DataFrame] = None) -> float | None:
    # Priority 1: Calculate from loaded dataframe's ai_anomaly_result column
    if df is not None and "ai_anomaly_result" in df.columns:
        # TRUE=anomaly, FALSE=non anomaly
        return df["ai_anomaly_result"].sum() / len(df)

    # Priority 2: Look for results in the latest results directory
    latest_results_dir = find_best_results_directory()
    if not latest_results_dir:
        return None

    metadata_file = Path(latest_results_dir) / "models" / "stratified_models_metadata.json"
    if metadata_file.exists():
        try:
            with open(metadata_file, "r") as handle:
                metadata = json.load(handle)
                return metadata.get("anomaly_rate")
        except Exception:
            pass

    results_path = Path(latest_results_dir) / "anomaly_detection_results.csv"
    if results_path.exists():
        try:
            results_df = pd.read_csv(results_path)
            return results_df["predicted_anomaly"].sum() / len(results_df)
        except Exception:
            return None
    return None

@router.get("/features")
def get_features():
    feature_categories = {
        "Quantity-Based Features": [
            {"id": "qty_z_score", "name": "Qty Z Score", "description": "Z-score deviation from historical mean - captures statistical outliers"},
            {"id": "qty_deviation_from_mean", "name": "Qty Deviation From Mean", "description": "Raw deviation from mean - absolute measure of unusual quantities"},
            {"id": "Sales Order item qty", "name": "Sales Order Item Qty", "description": "Raw quantity - base measure for volume anomalies"},
            {"id": "current_month_total_qty", "name": "Current Month Total Qty", "description": "Monthly accumulation - detects volume breaches"},
        ],
        "Pricing Features": [
            {"id": "Unit Price", "name": "Unit Price", "description": "Raw unit price - base measure for pricing anomalies"},
            {"id": "expected_order_item_value", "name": "Expected Order Item Value", "description": "Expected value calculation - helps detect value mismatches"},
        ],
        "Temporal Features": [
            {"id": "fulfillment_duration_days", "name": "Fulfillment Duration Days", "description": "Delivery time - unusual processing times"},
        ],
        "Boolean Anomaly Flags": [
            {"id": "is_first_time_cust_material_order", "name": "Is First Time Cust Material Order", "description": "First-time orders - new customer-product combinations"},
            {"id": "is_rare_material", "name": "Is Rare Material", "description": "Rare drug indicators - unusual product requests"},
            {"id": "is_qty_outside_typical_range", "name": "Is Qty Outside Typical Range", "description": "Quantity outliers - statistical anomalies"},
            {"id": "is_suspected_duplicate_order", "name": "Is Suspected Duplicate Order", "description": "Potential duplicates - operational anomalies"},
            {"id": "is_monthly_qty_outside_typical_range", "name": "Is Monthly Qty Outside Typical Range", "description": "Monthly volume breaches - accumulation anomalies"},
            {"id": "is_unusual_unit_price", "name": "Is Unusual Unit Price", "description": "Price outliers - pricing anomalies"},
            {"id": "is_value_mismatch_price_qty", "name": "Is Value Mismatch Price Qty", "description": "Value calculation errors - data quality issues"},
            {"id": "is_unusual_fulfillment_time", "name": "Is Unusual Fulfillment Time", "description": "Delivery time outliers - operational anomalies"},
        ],
    }
    return feature_categories

@router.post("/upload")
async def upload_training_data(file: UploadFile = File(...)):
    try:
        file_path = UPLOAD_DIR / file.filename
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        return {"filename": file.filename, "status": "uploaded"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/train")
def start_training(
    contamination: float = 0.05,
    model_type: str = "isolation_forest"
):
    # TODO: Trigger training pipeline
    return {"status": "started", "message": "Training started (mock)"}
