"""Procurement tools for the enterprise workflow agent example.

These tools simulate interactions with SAP-style enterprise systems (material master,
warehouse management, controlling, vendor master). They load mock data from CSV files
to mimic real API responses with realistic field structures and error handling.

In production, each function would call an OData service, S/4HANA API, or BTP destination
instead of reading from local CSV files.
"""

from datetime import datetime
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Load mock data from CSV files
# ---------------------------------------------------------------------------

_DATA_DIR = Path(__file__).parent / "data"

products_df = pd.read_csv(_DATA_DIR / "products.csv")
inventory_df = pd.read_csv(_DATA_DIR / "inventory.csv")
budgets_df = pd.read_csv(_DATA_DIR / "budgets.csv")
suppliers_df = pd.read_csv(_DATA_DIR / "suppliers.csv")


# ---------------------------------------------------------------------------
# Tool functions
# ---------------------------------------------------------------------------


def lookup_product(product_name: str) -> str:
    """Search the product catalog by name.

    Performs a case-insensitive substring match against product names.
    Use this to find a product's ID, price, category, and supplier
    before proceeding with inventory or budget checks.

    Args:
        product_name: Full or partial product name to search for.

    Returns:
        Product details if found, or a list of all available products if not.
    """
    matches = products_df[
        products_df["name"].str.lower().str.contains(product_name.lower())
    ]

    if matches.empty:
        available = ", ".join(products_df["name"].tolist())
        return (
            f"No product found matching '{product_name}'. "
            f"Available products: {available}"
        )

    results = []
    for _, row in matches.iterrows():
        results.append(
            f"Product ID: {row['product_id']}\n"
            f"  Name: {row['name']}\n"
            f"  Category: {row['category']}\n"
            f"  Unit Price: {row['unit_price']:.2f} {row['currency']}\n"
            f"  Supplier ID: {row['supplier_id']}"
        )
    return "\n\n".join(results)


def check_inventory(product_id: str, plant: str) -> str:
    """Check stock availability for a product at a specific plant.

    Returns available units (in stock minus reserved) at the requested plant,
    plus stock levels at all other plants for context.

    Args:
        product_id: The product identifier (e.g. 'PRD-001').
        plant: The plant location to check (e.g. 'Berlin', 'Munich', 'Dublin').

    Returns:
        Stock levels at the requested plant and other locations.
    """
    product_inv = inventory_df[inventory_df["product_id"] == product_id]

    if product_inv.empty:
        return f"No inventory records found for product '{product_id}'."

    plant_inv = product_inv[product_inv["plant"].str.lower() == plant.lower()]

    lines = []

    if plant_inv.empty:
        available_plants = ", ".join(product_inv["plant"].tolist())
        lines.append(
            f"Plant '{plant}' not found for product {product_id}. "
            f"Available plants: {available_plants}"
        )
    else:
        row = plant_inv.iloc[0]
        available = row["units_in_stock"] - row["reserved_units"]
        lines.append(
            f"Stock at {row['plant']}:\n"
            f"  Units in stock: {row['units_in_stock']}\n"
            f"  Reserved: {row['reserved_units']}\n"
            f"  Available: {available}"
        )

    # Show other plants for context
    other_plants = product_inv[product_inv["plant"].str.lower() != plant.lower()]
    if not other_plants.empty:
        lines.append("\nStock at other plants:")
        for _, row in other_plants.iterrows():
            available = row["units_in_stock"] - row["reserved_units"]
            lines.append(f"  {row['plant']}: {available} available")

    return "\n".join(lines)


def validate_budget(department: str, amount: float) -> str:
    """Check if a department's budget can cover a requested purchase amount.

    Compares the requested amount against the department's remaining budget
    and returns an APPROVED or REJECTED status with details.

    Args:
        department: Department name (e.g. 'Engineering', 'Marketing').
        amount: Total purchase amount in EUR to validate.

    Returns:
        Budget validation result with remaining budget details.
    """
    dept_budget = budgets_df[
        budgets_df["department"].str.lower() == department.lower()
    ]

    if dept_budget.empty:
        available_depts = ", ".join(budgets_df["department"].tolist())
        return (
            f"Department '{department}' not found. "
            f"Available departments: {available_depts}"
        )

    row = dept_budget.iloc[0]
    remaining = float(row["remaining"])

    if amount <= remaining:
        return (
            f"APPROVED: {row['department']} department budget check passed.\n"
            f"  Requested amount: {amount:.2f} EUR\n"
            f"  Remaining budget: {remaining:.2f} EUR\n"
            f"  Budget after purchase: {remaining - amount:.2f} EUR"
        )
    else:
        return (
            f"REJECTED: {row['department']} department budget exceeded.\n"
            f"  Requested amount: {amount:.2f} EUR\n"
            f"  Remaining budget: {remaining:.2f} EUR\n"
            f"  Over budget by: {amount - remaining:.2f} EUR"
        )


def get_supplier_info(supplier_id: str) -> str:
    """Retrieve supplier details including lead time and order constraints.

    Use this to check delivery timelines and minimum order quantities
    before finalizing a purchase order.

    Args:
        supplier_id: The supplier identifier (e.g. 'SUP-001').

    Returns:
        Supplier name, lead time, minimum order quantity, and reliability rating.
    """
    supplier = suppliers_df[suppliers_df["supplier_id"] == supplier_id]

    if supplier.empty:
        available = ", ".join(suppliers_df["supplier_id"].tolist())
        return (
            f"Supplier '{supplier_id}' not found. "
            f"Available suppliers: {available}"
        )

    row = supplier.iloc[0]
    return (
        f"Supplier: {row['name']} ({row['supplier_id']})\n"
        f"  Lead time: {row['lead_time_days']} days\n"
        f"  Minimum order quantity: {row['min_order_qty']} units\n"
        f"  Reliability rating: {row['reliability_rating']}/5.0"
    )


def search_alternative_products(category: str, max_unit_price: float) -> str:
    """Search for products in a category within a price limit.

    Use this when the originally requested product is too expensive,
    out of stock, or unavailable, to find alternatives the user might consider.

    Args:
        category: Product category to search in (e.g. 'IT Equipment', 'Office Furniture').
        max_unit_price: Maximum acceptable unit price in EUR.

    Returns:
        List of matching products, or a message if none found.
    """
    matches = products_df[
        (products_df["category"].str.lower() == category.lower())
        & (products_df["unit_price"] <= max_unit_price)
    ]

    if matches.empty:
        categories = ", ".join(products_df["category"].unique().tolist())
        return (
            f"No products found in category '{category}' "
            f"with unit price <= {max_unit_price:.2f} EUR. "
            f"Available categories: {categories}"
        )

    results = []
    for _, row in matches.iterrows():
        results.append(
            f"- {row['name']} ({row['product_id']}): "
            f"{row['unit_price']:.2f} {row['currency']}"
        )
    return (
        f"Products in '{category}' under {max_unit_price:.2f} EUR:\n"
        + "\n".join(results)
    )


def create_purchase_order_draft(
    product_id: str, quantity: int, plant: str, department: str
) -> str:
    """Create a draft purchase order for review.

    Validates that the product, plant, and department exist, then generates
    a formatted purchase order draft with a PO number and cost breakdown.
    This does NOT submit the order -- it creates a draft for human review.

    Args:
        product_id: The product to order (e.g. 'PRD-001').
        quantity: Number of units to order.
        plant: Delivery plant location.
        department: Requesting department for cost allocation.

    Returns:
        Formatted purchase order draft or an error message.
    """
    product = products_df[products_df["product_id"] == product_id]
    if product.empty:
        return f"Cannot create PO: product '{product_id}' not found."

    plant_exists = inventory_df[
        inventory_df["plant"].str.lower() == plant.lower()
    ]
    if plant_exists.empty:
        return f"Cannot create PO: plant '{plant}' not found."

    dept_exists = budgets_df[
        budgets_df["department"].str.lower() == department.lower()
    ]
    if dept_exists.empty:
        return f"Cannot create PO: department '{department}' not found."

    row = product.iloc[0]
    total = row["unit_price"] * quantity
    po_number = f"PO-{datetime.now().strftime('%Y%m%d%H%M%S')}"

    return (
        f"===== PURCHASE ORDER DRAFT =====\n"
        f"PO Number: {po_number}\n"
        f"Date: {datetime.now().strftime('%Y-%m-%d')}\n"
        f"Status: DRAFT (pending approval)\n"
        f"\n"
        f"Requesting Department: {department}\n"
        f"Delivery Plant: {plant}\n"
        f"\n"
        f"Line Items:\n"
        f"  1. {row['name']} ({product_id})\n"
        f"     Quantity: {quantity}\n"
        f"     Unit Price: {row['unit_price']:.2f} {row['currency']}\n"
        f"     Line Total: {total:.2f} {row['currency']}\n"
        f"\n"
        f"Order Total: {total:.2f} {row['currency']}\n"
        f"Supplier: {row['supplier_id']}\n"
        f"================================"
    )
