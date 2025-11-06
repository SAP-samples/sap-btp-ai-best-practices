from typing import Any, Dict, List, Optional


def markdown_table_from_rows(
    headers: List[str], rows: List[List[Any]], max_rows: int = 30
) -> str:
    display_rows = rows[:max_rows]
    lines = ["| " + " | ".join(str(h) for h in headers) + " |"]
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for r in display_rows:
        lines.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(lines)


def summarize_rows_for_llm(raw_rows: List[List[Any]]) -> Dict[str, Any]:
    def _is_empty_value(value: Any) -> bool:
        if value is None:
            return True
        if isinstance(value, str) and value.strip() == "":
            return True
        return False

    if not raw_rows:
        return {"headers": [], "rows": []}

    max_scan = min(200, len(raw_rows))

    def _row_non_empty_count(row: List[Any]) -> int:
        return sum(1 for v in row if not _is_empty_value(v))

    header_idx = 0
    header_non_empty = _row_non_empty_count(raw_rows[0])
    for i in range(1, max_scan):
        cnt = _row_non_empty_count(raw_rows[i])
        if cnt > header_non_empty:
            header_non_empty = cnt
            header_idx = i
        # If equal counts, keep the earlier header_idx (prefer earliest max)

    window_end = min(len(raw_rows), header_idx + 500)
    col_count = max((len(r) for r in raw_rows), default=0)

    counts_by_col: List[int] = [0] * col_count
    for i in range(header_idx, window_end):
        row = raw_rows[i]
        for j in range(col_count):
            val = row[j] if j < len(row) else None
            if not _is_empty_value(val):
                counts_by_col[j] += 1

    used_cols: List[int] = []
    header_row = raw_rows[header_idx]
    for j in range(col_count):
        in_header = (j < len(header_row)) and (not _is_empty_value(header_row[j]))
        if in_header or counts_by_col[j] >= 2:
            used_cols.append(j)

    if len(used_cols) <= 1:
        ranked = sorted(range(col_count), key=lambda x: counts_by_col[x], reverse=True)
        used_cols = [j for j in ranked[: min(10, col_count)] if counts_by_col[j] > 0]

    if not used_cols:
        return {"headers": [], "rows": []}

    headers: List[str] = []
    for pos, j in enumerate(used_cols):
        cell = header_row[j] if j < len(header_row) else None
        headers.append(str(cell) if not _is_empty_value(cell) else f"col_{pos}")

    data_rows: List[List[Any]] = []
    data_row_numbers: List[int] = []  # Excel 1-based row numbers for each data row
    last_idx = -1
    for i in range(header_idx + 1, window_end):
        row = raw_rows[i]
        vals = [(row[j] if j < len(row) else None) for j in used_cols]
        data_rows.append(vals)
        data_row_numbers.append(i + 1)  # worksheet rows are 1-based
        if any(not _is_empty_value(v) for v in vals):
            last_idx = i - (header_idx + 1)

    if last_idx >= 0:
        data_rows = data_rows[: last_idx + 1]
        data_row_numbers = data_row_numbers[: last_idx + 1]
    else:
        data_rows = []
        data_row_numbers = []

    col_letters = [excel_col_letter(i) for i in range(len(headers))]
    # Preserve actual worksheet row numbers for the included data rows
    row_numbers = data_row_numbers

    return {
        "headers": headers,
        "rows": data_rows,
        "rowNumbers": row_numbers,
        "colLetters": col_letters,
    }


def summarize_sheet_for_llm(sheet) -> Dict[str, Any]:
    raw_rows: List[List[Any]] = [list(r) for r in sheet.iter_rows(values_only=True)]
    return summarize_rows_for_llm(raw_rows)


def is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and value is not None


def percentile(sorted_values: List[float], p: float) -> float:
    if not sorted_values:
        return 0.0
    k = (len(sorted_values) - 1) * p
    f = int(k)
    c = min(f + 1, len(sorted_values) - 1)
    if f == c:
        return float(sorted_values[f])
    d0 = sorted_values[f] * (c - k)
    d1 = sorted_values[c] * (k - f)
    return float(d0 + d1)


def excel_col_letter(zero_based_index: int) -> str:
    n = int(zero_based_index)
    s = ""
    while n >= 0:
        s = chr((n % 26) + 65) + s
        n = (n // 26) - 1
    return s


def build_column_profiles(headers: List[str], rows: List[List[Any]]) -> Dict[str, Any]:
    import math

    profiles: Dict[str, Any] = {"columns": []}
    for col_idx, header in enumerate(headers):
        h_lower = (header or "").strip().lower()

        def _is_identifier_header(name: str) -> bool:
            tokens = [
                "id",
                "po",
                "plant",
                "material",
                "uom",
                "co-product",
                "co product",
                "by-product",
                "by product",
                "code",
                "number",
            ]
            if name.startswith("col_"):
                return True
            return any(t in name for t in tokens)

        if _is_identifier_header(h_lower):
            profiles["columns"].append(
                {"name": header, "index": col_idx, "count": 0, "candidates": []}
            )
            continue

        values: List[float] = []
        value_rows: List[int] = []
        for r_idx, row in enumerate(rows):
            if col_idx < len(row) and is_number(row[col_idx]):
                values.append(float(row[col_idx]))
                value_rows.append(r_idx)

        if len(values) < 8:
            profiles["columns"].append(
                {"name": header, "index": col_idx, "count": 0, "candidates": []}
            )
            continue

        sorted_vals = sorted(values)
        n = len(values)
        mean = sum(values) / n
        var = sum((v - mean) ** 2 for v in values) / n
        std = math.sqrt(var)
        q1 = percentile(sorted_vals, 0.25)
        q3 = percentile(sorted_vals, 0.75)
        iqr = q3 - q1
        lower_fence = q1 - 1.5 * iqr
        upper_fence = q3 + 1.5 * iqr

        candidates: List[Dict[str, Any]] = []
        for v, r_idx in zip(values, value_rows):
            z = (v - mean) / std if std > 0 else 0.0
            if (v < lower_fence or v > upper_fence) and abs(z) >= 3.5:
                candidates.append(
                    {
                        "rowIndex": r_idx,
                        "value": v,
                        "z": z,
                        "rule": "iqr_and_3.5sigma",
                    }
                )

        profiles["columns"].append(
            {
                "name": header,
                "index": col_idx,
                "count": n,
                "mean": mean,
                "std": std,
                "q1": q1,
                "q3": q3,
                "lower_fence": lower_fence,
                "upper_fence": upper_fence,
                "candidates": candidates,
            }
        )

    return profiles


def build_row_candidates(headers: List[str], rows: List[List[Any]]) -> Dict[str, Any]:
    candidates: List[Dict[str, Any]] = []
    header_lowers = [(h or "").strip().lower() for h in headers]

    def _is_measure_col(name: str) -> bool:
        keys = ["quantity", "consumed", "produced", "amount", "expected", "milk"]
        return any(k in name for k in keys)

    for r_idx, row in enumerate(rows):
        numeric_positions: List[int] = []
        for c_idx, val in enumerate(row):
            if is_number(val):
                numeric_positions.append(c_idx)
        if len(numeric_positions) == 1:
            c_idx = numeric_positions[0]
            name = header_lowers[c_idx] if c_idx < len(header_lowers) else ""
            if _is_measure_col(name):
                candidates.append(
                    {
                        "rowIndex": r_idx,
                        "columnIndex": c_idx,
                        "column": (
                            headers[c_idx] if c_idx < len(headers) else f"col_{c_idx}"
                        ),
                        "value": row[c_idx],
                        "rule": "singleton_numeric_in_measure_column",
                    }
                )

    return {"rows": candidates}
