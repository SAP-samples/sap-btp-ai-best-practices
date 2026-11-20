"""Read access to (mock) ECC master data: customers, partners, sales areas, materials, units.

Every function returns plain dicts keyed by SAP field names (``KUNNR``, ``NAME1``...), so the
callers (agent tools, validator, UI value help) are independent of where the data comes from.

Mock implementation: SQL against the ``*_POAGENT`` HANA tables.
Production swap: re-implement the same functions on top of SAP Integration Suite iFlows
(``BAPI_CUSTOMER_GETDETAIL2`` / ``RFC_READ_TABLE``) or master data replicated into HANA.

All functions are synchronous (blocking DB I/O); async callers use ``asyncio.to_thread``.

ponytail: full-table loads per call are fine for mock-sized data; add a TTL cache or
server-side candidate retrieval when the master data grows.
"""

from __future__ import annotations

from typing import Any

from ..db.sql import fetch_all
from ..orders.matching import alpha_in


def _rows(sql: str, **params: Any) -> list[dict[str, Any]]:
    """Run a query and return rows keyed by upper-case SAP field names (see ``db.sql.fetch_all``)."""
    return fetch_all(sql, **params)


# ---------------------------------------------------------------------------
# Customers
# ---------------------------------------------------------------------------
def load_customers() -> list[dict[str, Any]]:
    """Load all customers with their contact e-mails and sales areas.

    Returns:
        KNA1 rows (deleted customers excluded) with extra keys ``EMAILS`` (list[str]) and
        ``SALES_AREAS`` (list of {VKORG, VTWEG, SPART}).
    """
    customers = {row["KUNNR"]: {**row, "EMAILS": [], "SALES_AREAS": []}
                 for row in _rows('SELECT * FROM "KNA1_POAGENT" WHERE COALESCE("LOEVM", \'\') = \'\'')}
    for row in _rows('SELECT "KUNNR", "SMTP_ADDR" FROM "KNVK_POAGENT" WHERE "SMTP_ADDR" IS NOT NULL'):
        if row["KUNNR"] in customers:
            customers[row["KUNNR"]]["EMAILS"].append(row["SMTP_ADDR"])
    for row in _rows('SELECT "KUNNR", "VKORG", "VTWEG", "SPART" FROM "KNVV_POAGENT"'):
        if row["KUNNR"] in customers:
            customers[row["KUNNR"]]["SALES_AREAS"].append({k: row[k] for k in ("VKORG", "VTWEG", "SPART")})
    return list(customers.values())


def get_customer(kunnr: str) -> dict[str, Any] | None:
    """Return one KNA1 row (deleted customers included, so callers can report them).

    Args:
        kunnr: Customer number (any format).

    Returns:
        The row or None.
    """
    rows = _rows('SELECT * FROM "KNA1_POAGENT" WHERE "KUNNR" = :k', k=alpha_in(kunnr, 10))
    return rows[0] if rows else None


def get_sales_areas(kunnr: str) -> list[dict[str, Any]]:
    """Return the KNVV sales-area rows of a customer.

    Args:
        kunnr: Customer number.

    Returns:
        KNVV rows (currency, payment terms, incoterms, delivering plant, blocks).
    """
    return _rows('SELECT * FROM "KNVV_POAGENT" WHERE "KUNNR" = :k ORDER BY "VKORG"', k=alpha_in(kunnr, 10))


def get_partners(kunnr: str, vkorg: str | None = None) -> list[dict[str, Any]]:
    """Return partner functions of a sold-to with the partner's name and address.

    Args:
        kunnr: Sold-to customer number.
        vkorg: Optional sales organisation filter.

    Returns:
        Rows with PARVW, PARZA, KUNN2, DEFPA, VKORG/VTWEG/SPART and the partner's
        NAME1, NAME2, STRAS, PSTLZ, ORT01, LAND1, AUFSD, LOEVM.
    """
    sql = f"""
        SELECT p."VKORG", p."VTWEG", p."SPART", p."PARVW", p."PARZA", p."KUNN2", p."DEFPA",
               k."NAME1", k."NAME2", k."STRAS", k."PSTLZ", k."ORT01", k."LAND1", k."AUFSD", k."LOEVM"
          FROM "KNVP_POAGENT" p LEFT JOIN "KNA1_POAGENT" k ON k."KUNNR" = p."KUNN2"
         WHERE p."KUNNR" = :k {'AND p."VKORG" = :vkorg' if vkorg else ''}
         ORDER BY p."VKORG", p."PARVW", p."PARZA"
    """
    return _rows(sql, k=alpha_in(kunnr, 10), **({"vkorg": vkorg} if vkorg else {}))


def get_contacts(kunnr: str) -> list[dict[str, Any]]:
    """Return KNVK contact persons of a customer.

    Args:
        kunnr: Customer number.

    Returns:
        KNVK rows.
    """
    return _rows('SELECT * FROM "KNVK_POAGENT" WHERE "KUNNR" = :k', k=alpha_in(kunnr, 10))


def list_sales_orgs() -> list[str]:
    """Return the distinct sales organisations present in customer sales areas."""
    return [row["VKORG"] for row in _rows('SELECT DISTINCT "VKORG" FROM "KNVV_POAGENT" ORDER BY "VKORG"')]


# ---------------------------------------------------------------------------
# Materials
# ---------------------------------------------------------------------------
def load_materials(include_deleted: bool = False) -> list[dict[str, Any]]:
    """Load materials with English description, sales data and alternative units.

    Args:
        include_deleted: Also return materials flagged for deletion (LVORM = 'X').

    Returns:
        MARA rows with extra keys ``MAKTX`` (English text), ``SALES`` (list of MVKE rows)
        and ``UNITS`` (list of MARM rows).
    """
    where = "" if include_deleted else 'WHERE COALESCE(m."LVORM", \'\') = \'\''
    materials = {
        row["MATNR"]: {**row, "SALES": [], "UNITS": []}
        for row in _rows(
            f'SELECT m.*, t."MAKTX" FROM "MARA_POAGENT" m '
            f'LEFT JOIN "MAKT_POAGENT" t ON t."MATNR" = m."MATNR" AND t."SPRAS" = \'E\' {where}'
        )
    }
    for row in _rows('SELECT * FROM "MVKE_POAGENT"'):
        if row["MATNR"] in materials:
            materials[row["MATNR"]]["SALES"].append(row)
    for row in _rows('SELECT * FROM "MARM_POAGENT"'):
        if row["MATNR"] in materials:
            materials[row["MATNR"]]["UNITS"].append(row)
    return list(materials.values())


def get_material(matnr: str) -> dict[str, Any] | None:
    """Return one material (deleted ones included) with MAKTX, SALES and UNITS.

    Args:
        matnr: Material number (any format).

    Returns:
        The material dict or None.
    """
    wanted = alpha_in(matnr, 18)
    for material in load_materials(include_deleted=True):
        if material["MATNR"] == wanted:
            return material
    return None


def get_customer_material_codes(kunnr: str, vkorg: str | None = None) -> list[dict[str, Any]]:
    """Return customer-material info records (KNMT) of a customer.

    Args:
        kunnr: Customer number.
        vkorg: Optional sales organisation filter.

    Returns:
        KNMT rows (KDMAT = customer's own material code).
    """
    sql = 'SELECT * FROM "KNMT_POAGENT" WHERE "KUNNR" = :k' + (' AND "VKORG" = :vkorg' if vkorg else "")
    return _rows(sql, k=alpha_in(kunnr, 10), **({"vkorg": vkorg} if vkorg else {}))


def get_customer_prices(kunnr: str, vkorg: str | None = None) -> dict[str, dict[str, Any]]:
    """Return the customer-specific material prices (condition table A305 with KONP rate).

    Args:
        kunnr: Sold-to customer number.
        vkorg: Optional sales organisation filter.

    Returns:
        {MATNR: row with KBETR (price), KONWA (currency), KPEIN (per quantity), KMEIN (unit)}.
    """
    sql = 'SELECT * FROM "A305_POAGENT" WHERE "KUNNR" = :k' + (' AND "VKORG" = :vkorg' if vkorg else "")
    return {row["MATNR"]: row for row in _rows(sql, k=alpha_in(kunnr, 10), **({"vkorg": vkorg} if vkorg else {}))}


def load_unit_aliases() -> dict[str, str]:
    """Return the free-text unit alias map {alias (lower case): SAP unit}."""
    return {row["ALIAS"]: row["MSEHI"] for row in _rows('SELECT "ALIAS", "MSEHI" FROM "ZUOM_ALIAS_POAGENT"')}
