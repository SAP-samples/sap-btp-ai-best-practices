from fastapi import APIRouter, HTTPException, Body
from typing import Dict, Any
import pandas as pd
from services.model_service import model_service
from services import ai_classification
from services.data_loader import load_dataset

router = APIRouter()

@router.post("/predict")
def predict_anomaly(sample: Dict[str, Any] = Body(...)):
    if not model_service.loaded:
        if not model_service.load_models():
             raise HTTPException(status_code=500, detail="Failed to load models")
    
    # Convert dict to DataFrame row
    sample_df = pd.DataFrame([sample])
    
    # This assumes we have feature columns in sample or can derive them
    # For now, this is a placeholder for real-time inference
    return {"message": "Not implemented for raw JSON input yet"}

@router.post("/explain-binary")
def explain_binary(doc_number: str, doc_item: str):
    # Quick AI Classification
    from services import order_selection
    
    try:
        loaded_data = load_dataset()
        results_df = loaded_data.results
        features_df = loaded_data.features

        target_doc = order_selection.normalize_identifier(doc_number)
        target_item = order_selection.normalize_identifier(doc_item)
        
        normalized_docs = results_df["Sales Document Number"].map(order_selection.normalize_identifier)
        normalized_items = results_df["Sales Document Item"].map(order_selection.normalize_identifier)
        
        mask = (normalized_docs == target_doc) & (normalized_items == target_item)
        
        if not mask.any():
            raise HTTPException(status_code=404, detail="Order not found")
            
        row = results_df[mask].iloc[0]
        
        result, _ = ai_classification.run_binary_classification(row, features_df)
        return {"classification": result}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

