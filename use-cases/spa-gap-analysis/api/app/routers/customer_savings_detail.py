"""
Customer Savings Detail Endpoint
Returns material-level breakdown of savings for a customer
"""
from fastapi import APIRouter, HTTPException, Depends
import pandas as pd
import logging
from typing import List, Dict

from app.security import require_api_key
from app.services.data_loader import load_from_parquet

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/spa", tags=["SPA Customer Savings Detail"])


@router.get(
    "/customer/{customer_id}/savings-detail",
    dependencies=[Depends(require_api_key)],
    summary="Get Material-Level Savings Detail",
    description="Returns detailed breakdown of savings by material for a customer"
)
async def get_customer_savings_detail(customer_id: str) -> Dict:
    """
    Get material-level savings detail for customer

    Returns:
        {
            "customer_id": str,
            "materials": [
                {
                    "material": str,
                    "material_desc": str,
                    "sales_deal": str,
                    "base_cost": float,
                    "spa_price": float,
                    "savings_amount": float,
                    "savings_percent": float,
                    "category": str
                }
            ],
            "summary": {
                "total_materials": int,
                "total_base_cost": float,
                "total_spa_cost": float,
                "total_savings": float,
                "avg_savings_percent": float
            }
        }
    """
    logger.info(f"Getting savings detail for customer {customer_id}")

    try:
        # Load customer_savings summary
        customer_savings = load_from_parquet('customer_savings.parquet')
        cust_data = customer_savings[customer_savings['customer_id'] == customer_id]

        if cust_data.empty:
            raise HTTPException(
                status_code=404,
                detail=f"No savings data found for customer {customer_id}"
            )

        # Load transactions for this customer
        transactions = load_from_parquet('s712_transactions.parquet')
        cust_trans = transactions[transactions['customer_id'] == customer_id].copy()

        if cust_trans.empty:
            raise HTTPException(
                status_code=404,
                detail=f"No transaction data found for customer {customer_id}"
            )

        # Load material savings (SPA vs base pricing)
        material_savings = load_from_parquet('material_savings.parquet')

        # Load SAP master for descriptions
        sap_master = load_from_parquet('sap_master_enhanced.parquet')

        # Convert material to string for join
        cust_trans['material'] = cust_trans['material'].astype(str)
        material_savings['material'] = material_savings['material'].astype(str)
        sap_master['material'] = sap_master['material'].astype(str)

        # Join transactions with material savings
        materials_with_savings = cust_trans.merge(
            material_savings[['material', 'sales_deal', 'base_cost', 'spa_price', 'savings', 'savings_percent']],
            on='material',
            how='inner'
        )

        # Join with SAP master for descriptions and categories
        materials_with_savings = materials_with_savings.merge(
            sap_master[['material', 'material_desc', 'product_hier_01_desc']],
            on='material',
            how='left'
        )

        # Group by material to get unique list
        material_details = materials_with_savings.groupby('material').agg({
            'material_desc': 'first',
            'sales_deal': 'first',
            'base_cost': 'first',
            'spa_price': 'first',
            'savings': 'first',
            'savings_percent': 'first',
            'product_hier_01_desc': 'first'
        }).reset_index()

        # Sort by savings amount descending
        material_details = material_details.sort_values('savings', ascending=False)

        # Convert to list of dicts
        materials_list = []
        for _, row in material_details.iterrows():
            materials_list.append({
                'material': row['material'],
                'material_desc': row['material_desc'] if pd.notna(row['material_desc']) else 'Unknown',
                'sales_deal': str(int(row['sales_deal'])) if pd.notna(row['sales_deal']) else 'N/A',
                'base_cost': float(row['base_cost']),
                'spa_price': float(row['spa_price']),
                'savings_amount': float(row['savings']),
                'savings_percent': float(row['savings_percent']),
                'category': row['product_hier_01_desc'] if pd.notna(row['product_hier_01_desc']) else 'Uncategorized'
            })

        # Calculate summary
        summary = {
            'total_materials': len(materials_list),
            'total_base_cost': float(material_details['base_cost'].sum()),
            'total_spa_cost': float(material_details['spa_price'].sum()),
            'total_savings': float(material_details['savings'].sum()),
            'avg_savings_percent': float(material_details['savings_percent'].mean())
        }

        logger.info(f"Found {len(materials_list)} materials with savings for customer {customer_id}")

        return {
            'customer_id': customer_id,
            'materials': materials_list,
            'summary': summary
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting savings detail for customer {customer_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )
