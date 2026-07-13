from fastapi import APIRouter, HTTPException, Query
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
from services.data_loader import load_dataset
from datetime import datetime

router = APIRouter()

@router.get("/summary")
def get_summary(
    year: Optional[int] = Query(None, description="Year to filter by"),
    month: Optional[int] = Query(None, description="Month to filter by")
):
    try:
        loaded_data = load_dataset()
        df = loaded_data.results

        if "Sales Document Created Date" not in df.columns:
            raise HTTPException(status_code=500, detail="Date column missing in dataset")

        # Ensure date format
        df["date_obj"] = pd.to_datetime(df["Sales Document Created Date"], errors='coerce')
        df = df.dropna(subset=["date_obj"])
        
        # Filter by year and month if provided
        if year is not None:
            df = df[df["date_obj"].dt.year == year]
        if month is not None:
            df = df[df["date_obj"].dt.month == month]
        
        if df.empty:
            return {
                "summary": [],
                "calendar_data": {},
                "metrics": {
                    "total_orders": 0,
                    "anomaly_count": 0,
                    "anomaly_rate": 0,
                    "total_value": 0,
                    "anomaly_value": 0
                }
            }

        # Daily summary for calendar
        daily_summary = df.groupby(df["date_obj"].dt.date).agg(
            total_orders=("Sales Document Number", "count"),
            anomaly_count=("predicted_anomaly", "sum")
        ).reset_index()
        
        daily_summary["anomaly_rate"] = daily_summary["anomaly_count"] / daily_summary["total_orders"]
        daily_summary["date"] = daily_summary["date_obj"].astype(str)
        
        calendar_data = daily_summary.set_index("date").to_dict(orient="index")

        return {
            "calendar_data": calendar_data,
            "years": sorted(df["date_obj"].dt.year.unique().tolist(), reverse=True),
            "months": sorted(df["date_obj"].dt.month.unique().tolist())
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/daily")
def get_daily_details(
    year: int, 
    month: int, 
    day: int
):
    try:
        loaded_data = load_dataset()
        df = loaded_data.results
        
        df["date_obj"] = pd.to_datetime(df["Sales Document Created Date"], errors='coerce')
        target_date = datetime(year, month, day).date()
        
        daily_data = df[df["date_obj"].dt.date == target_date].copy()
        
        # Sort by anomaly score ascending (lowest score = most anomalous for sklearn)
        daily_data = daily_data.sort_values("anomaly_score", ascending=True)
        
        if daily_data.empty:
             return {
                "metrics": {
                    "total_orders": 0,
                    "anomaly_count": 0,
                    "anomaly_rate": 0,
                    "total_value": 0,
                    "anomaly_value": 0
                },
                "orders": [],
                "charts": {}
            }

        # Metrics
        total_orders = len(daily_data)
        anomalies = daily_data[daily_data["predicted_anomaly"] == 1]
        anomaly_count = len(anomalies)
        anomaly_rate = anomaly_count / total_orders if total_orders > 0 else 0
        total_value = daily_data["Order item value"].sum()
        anomaly_total_value = anomalies["Order item value"].sum()
        
        # Orders List (simplified for table)
        orders = daily_data[[
            "Sales Document Number", 
            "Sales Document Item", 
            "Customer PO number",
            "Material Description",
            "Sales Order item qty",
            "Unit Price",
            "anomaly_score",
            "predicted_anomaly"
        ]].copy()
        
        orders = orders.fillna("")
        orders_list = orders.to_dict(orient="records")
        
        return {
            "metrics": {
                "total_orders": total_orders,
                "anomaly_count": anomaly_count,
                "anomaly_rate": anomaly_rate,
                "total_value": float(total_value),
                "anomaly_value": float(anomaly_total_value)
            },
            "orders": orders_list
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
