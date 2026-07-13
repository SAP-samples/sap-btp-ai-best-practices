from fastapi import APIRouter, HTTPException, Query
from typing import Optional, Dict, Any
import pandas as pd
import numpy as np
from services.data_loader import load_dataset
from services import order_selection, model_runtime, shap_explanations, ai_classification

router = APIRouter()

@router.get("/random-anomalous")
def get_random_anomalous_order():
    """Return a random anomalous order (doc_number/doc_item) for quick demos."""
    try:
        loaded_data = load_dataset()
        results_df = loaded_data.results

        key = order_selection.select_random_anomalous_order(results_df)
        if not key:
            raise HTTPException(status_code=404, detail="No anomalous orders available")

        return {
            "doc_number": key.document_number,
            "doc_item": key.document_item,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{doc_number}/{doc_item}")
def get_order_details(doc_number: str, doc_item: str):
    try:
        loaded_data = load_dataset()
        results_df = loaded_data.results
        features_df = loaded_data.features

        # Normalize identifiers
        target_doc = order_selection.normalize_identifier(doc_number)
        target_item = order_selection.normalize_identifier(doc_item)
        
        # Filter
        normalized_docs = results_df["Sales Document Number"].map(order_selection.normalize_identifier)
        normalized_items = results_df["Sales Document Item"].map(order_selection.normalize_identifier)
        
        mask = (normalized_docs == target_doc) & (normalized_items == target_item)
        
        if not mask.any():
            raise HTTPException(status_code=404, detail="Order not found")
            
        row = results_df[mask].iloc[0]
        
        # Convert row to dict and handle NaNs
        row_dict = row.replace({np.nan: None}).to_dict()
        
        # Compute SHAP if needed
        shap_explanation = row.get("shap_explanation")
        if not shap_explanation or shap_explanation == "N/A":
             # Try on-demand
             shap_explanation = model_runtime.compute_shap(row, features_df)
        
        shap_data = []
        if shap_explanation:
             shap_df = shap_explanations.build_shap_dataframe(shap_explanation, features_df, row)
             if shap_df is not None:
                 shap_data = shap_df.to_dict(orient="records")

        return {
            "order": row_dict,
            "shap_explanation": shap_data,
            "raw_shap_text": shap_explanation
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/{doc_number}/{doc_item}/explain")
def generate_ai_explanation(doc_number: str, doc_item: str):
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
        
        explanation = ai_classification.generate_full_explanation(row, features_df)
        
        return {"explanation": explanation}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

