#!/usr/bin/env python3
import argparse
import csv
import hashlib
import os
from datetime import datetime, timedelta
from typing import Dict, Callable, Optional, List


SALTED_NAMESPACE = "ov-open-source-salt-v1"


def stable_int(value: str, modulo: int, offset: int = 0) -> int:
    h = hashlib.sha256(
        (SALTED_NAMESPACE + "::" + str(value)).encode("utf-8")
    ).hexdigest()
    return (int(h[:12], 16) + offset) % modulo


def strip_bom(s: str) -> str:
    return s.lstrip("\ufeff")


def map_plant(plant: str) -> str:
    idx = stable_int(plant, 9000, offset=137)
    return f"P{idx:04d}"


def map_material(material: str) -> str:
    idx = stable_int(material, 900000, offset=731)
    return f"M{idx:06d}"


def map_product_name(material: str) -> str:
    return f"Product {map_material(material)}"


def parse_date_keep_format(raw: str) -> Optional[Callable[[int], str]]:
    # Returns a function that shifts the date by N days and renders in same format
    raw = raw.strip()
    if not raw:
        return None
    patterns: List[str] = [
        "%m/%d/%Y",  # 4/12/2024
        "%d-%b-%y",  # 20-Jun-23
        "%d-%b-%Y",  # 20-Jun-2023
        "%Y-%m-%d",  # 2024-04-12
    ]
    for fmt in patterns:
        try:
            dt = datetime.strptime(raw, fmt)
        except ValueError:
            continue

        def _render(shift_days: int, _fmt=fmt, _dt=dt) -> str:
            return (_dt + timedelta(days=shift_days)).strftime(_fmt)

        return _render
    # If unknown format, return identity
    return lambda shift_days: raw


def parse_date_with_fmt(raw: str) -> Optional[tuple[datetime, str]]:
    raw = raw.strip()
    if not raw:
        return None
    patterns: List[str] = [
        "%m/%d/%Y",
        "%d-%b-%y",
        "%d-%b-%Y",
        "%Y-%m-%d",
    ]
    for fmt in patterns:
        try:
            dt = datetime.strptime(raw, fmt)
            return dt, fmt
        except ValueError:
            continue
    return None


def per_material_scale(material: str) -> float:
    # Deterministic factor in [0.75, 1.25]
    base = stable_int(material, 5000, offset=3131) / 10000.0  # [0, 0.4999]
    return 0.75 + base  # [0.75, 1.2499]


def try_float(x: str) -> Optional[float]:
    x = x.strip()
    if not x:
        return None
    try:
        return float(x)
    except ValueError:
        return None


def fmt_number_like(original: str, value: float) -> str:
    # Preserve decimals count if present, else no decimals
    original = original.strip()
    if "." in original:
        decimals = len(original.split(".")[-1])
        return f"{value:.{decimals}f}"
    else:
        return str(int(round(value)))


def anonymize_po_csv(input_path: str, output_path: str) -> None:
    with open(input_path, newline="") as f_in:
        reader = csv.DictReader(f_in)
        fieldnames = [strip_bom(h) for h in (reader.fieldnames or [])]
        # Expected headers (order preserved):
        # Plant,Material Number,MAKTX,Delivery Date (Requested),Purchase Order Quantity (Requested),Purchase Order Quantity (Delivered),Planned Input (lbs),Actual Input (lbs)
        raw_rows = list(reader)

    # Normalize potential BOM-prefixed keys in rows
    rows = []
    for r in raw_rows:
        nr = {}
        for k, v in r.items():
            nr[strip_bom(k)] = v
        rows.append(nr)

    out_rows = []
    for row in rows:
        material_raw = row.get("Material Number", "").strip()
        plant_raw = row.get("Plant", "").strip()
        factor = per_material_scale(material_raw)

        # Identifiers
        row["Plant"] = map_plant(plant_raw) if plant_raw else plant_raw
        row["Material Number"] = (
            map_material(material_raw) if material_raw else material_raw
        )
        row["MAKTX"] = (
            map_product_name(material_raw)
            if row.get("MAKTX", "").strip()
            else row.get("MAKTX", "")
        )

        # Date shift: deterministic per material (+/- 1024 days window)
        date_parsed = parse_date_with_fmt(row.get("Delivery Date (Requested)", ""))
        if date_parsed:
            original_dt, fmt = date_parsed
            # Deterministic shift of 0-45 days into the past
            shift_days = -stable_int(material_raw, 46, offset=97)
            shifted = original_dt + timedelta(days=shift_days)
            today = datetime.today().replace(hour=0, minute=0, second=0, microsecond=0)
            if shifted > today:
                shifted = today
            row["Delivery Date (Requested)"] = shifted.strftime(fmt)

        # Numeric scaling (preserve blanks)
        for k in [
            "Purchase Order Quantity (Requested)",
            "Purchase Order Quantity (Delivered)",
            "Planned Input (lbs)",
            "Actual Input (lbs)",
        ]:
            v = row.get(k, "")
            num = try_float(v)
            if num is None:
                continue
            scaled = num * factor
            row[k] = fmt_number_like(v, scaled)

        out_rows.append(row)

    with open(output_path, "w", newline="") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)


def anonymize_scrapping_csv(input_path: str, output_path: str) -> None:
    with open(input_path, newline="") as f_in:
        reader = csv.DictReader(f_in)
        fieldnames = [strip_bom(h) for h in (reader.fieldnames or [])]
        raw_rows = list(reader)

    rows = []
    for r in raw_rows:
        nr = {}
        for k, v in r.items():
            nr[strip_bom(k)] = v
        rows.append(nr)

    out_rows = []
    for row in rows:
        material_raw = row.get("Material Number", "").strip()
        plant_raw = row.get("Plant", "").strip()
        factor = per_material_scale(material_raw)

        row["Plant"] = map_plant(plant_raw) if plant_raw else plant_raw
        row["Material Number"] = (
            map_material(material_raw) if material_raw else material_raw
        )
        row["MAKTX"] = (
            map_product_name(material_raw)
            if row.get("MAKTX", "").strip()
            else row.get("MAKTX", "")
        )

        date_parsed = parse_date_with_fmt(row.get("Date of Manufacture", ""))
        if date_parsed:
            original_dt, fmt = date_parsed
            shift_days = -stable_int(material_raw, 46, offset=197)
            shifted = original_dt + timedelta(days=shift_days)
            today = datetime.today().replace(hour=0, minute=0, second=0, microsecond=0)
            if shifted > today:
                shifted = today
            row["Date of Manufacture"] = shifted.strftime(fmt)

        qty = try_float(row.get("Quantity", ""))
        if qty is not None:
            row["Quantity"] = fmt_number_like(row.get("Quantity", ""), qty * factor)

        out_rows.append(row)

    with open(output_path, "w", newline="") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)


def guess_and_anonymize(input_path: str, output_path: Optional[str]) -> str:
    # Choose strategy by headers
    with open(input_path, newline="") as f_in:
        reader = csv.reader(f_in)
        headers = next(reader)

    if output_path is None:
        base, ext = os.path.splitext(input_path)
        output_path = f"{base}.anonymized{ext or '.csv'}"

    header_set = set([strip_bom(h).strip() for h in headers])
    if {"Plant", "Material Number", "MAKTX", "Delivery Date (Requested)"}.issubset(
        header_set
    ):
        anonymize_po_csv(input_path, output_path)
    elif {
        "Plant",
        "Date of Manufacture",
        "MAKTX",
        "MEINS",
        "Material Number",
        "Quantity",
    }.issubset(header_set):
        anonymize_scrapping_csv(input_path, output_path)
    else:
        raise ValueError(f"Unrecognized CSV schema for {input_path}: {headers}")

    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Deterministically anonymize CSV seed data."
    )
    parser.add_argument(
        "inputs",
        nargs="*",
        help="CSV file paths to anonymize. If none provided, defaults to known files.",
    )
    parser.add_argument(
        "--inplace", action="store_true", help="Overwrite input files in place."
    )
    parser.add_argument(
        "--suffix",
        default=".anonymized",
        help="Suffix to append before extension when not using --inplace.",
    )
    args = parser.parse_args()

    default_files = [
        os.path.join("api", "app", "mock", "data", "po_data_seed.csv"),
        os.path.join("api", "app", "mock", "data", "scrapping_data_seed.csv"),
    ]
    inputs = args.inputs or default_files

    outputs = []
    for p in inputs:
        if not os.path.isabs(p):
            p_abs = os.path.abspath(p)
        else:
            p_abs = p

        if not os.path.exists(p_abs):
            raise FileNotFoundError(p_abs)

        if args.inplace:
            out_path = p_abs
        else:
            base, ext = os.path.splitext(p_abs)
            out_path = f"{base}{args.suffix}{ext or '.csv'}"

        produced = guess_and_anonymize(p_abs, out_path if not args.inplace else None)
        outputs.append(produced)

    for o in outputs:
        print(o)


if __name__ == "__main__":
    main()
