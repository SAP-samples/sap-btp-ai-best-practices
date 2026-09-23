"""Run live SAP RPT-1.5/1.6 API contract and quality validation.

Examples:
    python live_test_rpt_client_api.py --dataset-root /path/to/datasets
    python live_test_rpt_client_api.py --dataset-root /path/to/datasets \
        --models sap-rpt-1.6 sap-rpt-1.6-large --output /tmp/rpt-report.json

The command reads AI Core credentials from the environment or ``.env``. It
never prints credentials, prediction rows, or source records. Contract failures
return a non-zero exit code; probabilistic quality metrics are report-only.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import sys
import tempfile
from collections import defaultdict
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

SKILL_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SKILL_ROOT / "assets"))

from rpt_client_api import (  # noqa: E402 - reusable asset is added to sys.path above
    RPTAPIError,
    RPTClientAPI,
    flatten_predictions,
    map_explanation_rows,
)

SUPPORTED_MODELS = (
    "sap-rpt-1.5",
    "sap-rpt-1.5-large",
    "sap-rpt-1.6",
    "sap-rpt-1.6-large",
)
PLACEHOLDER = "[PREDICT]"


class ProgressBar:
    """Render a small dependency-free progress bar for the live request matrix.

    Args:
        total: Number of live validation cases expected.
    """

    def __init__(self, total: int) -> None:
        """Initialize an empty bar for the positive number of expected cases."""
        self.total = total
        self.current = 0

    def advance(self, label: str) -> None:
        """Advance by one case and print the current label.

        Args:
            label: Short description of the completed case.
        """
        self.current += 1
        width = 24
        filled = round(width * self.current / self.total)
        bar = "#" * filled + "-" * (width - filled)
        print(f"[{bar}] {self.current}/{self.total} {label}", flush=True)


def _json_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """Convert a DataFrame to JSON-safe records while preserving null values.

    Args:
        frame: Source table in the desired API row order.

    Returns:
        Record dictionaries containing Python scalars and ``None`` for nulls.
    """
    clean = frame.astype(object).where(pd.notna(frame), None)
    return clean.to_dict(orient="records")


def _data_schema(frame: pd.DataFrame) -> dict[str, dict[str, str]]:
    """Build the minimal RPT schema from stable pandas source dtypes.

    Args:
        frame: Original table before prediction placeholders are inserted.

    Returns:
        Mapping from column name to RPT ``numeric``, ``date``, or ``string`` dtype.
    """
    schema: dict[str, dict[str, str]] = {}
    for column in frame.columns:
        if pd.api.types.is_numeric_dtype(frame[column]):
            dtype = "numeric"
        elif pd.api.types.is_datetime64_any_dtype(frame[column]):
            dtype = "date"
        else:
            dtype = "string"
        schema[column] = {"dtype": dtype}
    return schema


def _split_frame(
    frame: pd.DataFrame, *, query_count: int, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create deterministic disjoint context and query tables.

    Args:
        frame: Complete source table.
        query_count: Number of held-out query rows.
        seed: Shuffle seed used for reproducibility.

    Returns:
        Context and query DataFrames in their eventual API order.
    """
    if query_count <= 0 or query_count >= len(frame):
        raise ValueError("query_count must be positive and smaller than the dataset")
    shuffled = frame.sample(frac=1, random_state=seed).reset_index(drop=True)
    return shuffled.iloc[:-query_count].copy(), shuffled.iloc[-query_count:].copy()


def _prediction_config(
    targets: Sequence[tuple[str, str, int | None]],
    *,
    context_mode: str | None = None,
) -> dict[str, Any]:
    """Create a multi-target configuration used across the live cases.

    Args:
        targets: ``(column, task_type, top_k)`` tuples.
        context_mode: Optional RPT-1.6 context mode.

    Returns:
        Prediction configuration with explainability enabled.
    """
    target_columns = []
    for name, task_type, top_k in targets:
        target: dict[str, Any] = {
            "name": name,
            "prediction_placeholder": PLACEHOLDER,
            "task_type": task_type,
        }
        if top_k is not None:
            target["top_k"] = top_k
        target_columns.append(target)
    config: dict[str, Any] = {
        "target_columns": target_columns,
        "explanations": {
            "top_column_scores": 4,
            "top_relevant_context_rows": 3,
        },
    }
    if context_mode:
        config["context_mode"] = context_mode
    return config


def _build_payload(
    context: pd.DataFrame,
    queries: pd.DataFrame,
    targets: Sequence[tuple[str, str, int | None]],
    *,
    layout: str = "rows",
    context_mode: str | None = None,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """Build a complete row- or column-oriented request and held-out truth.

    Args:
        context: In-context examples with known target values.
        queries: Held-out rows whose target values will be replaced.
        targets: ``(column, task_type, top_k)`` tuples.
        layout: ``rows`` or ``columns`` JSON representation.
        context_mode: Optional RPT-1.6 context mode.

    Returns:
        Request payload and truth values keyed by ``row_id``.
    """
    query_marked = queries.copy()
    truth = {
        str(row["row_id"]): {name: row[name] for name, _, _ in targets}
        for _, row in queries.iterrows()
    }
    for name, _, _ in targets:
        query_marked[name] = PLACEHOLDER
    combined = pd.concat([context, query_marked], ignore_index=True)
    payload: dict[str, Any] = {
        "index_column": "row_id",
        "prediction_config": _prediction_config(targets, context_mode=context_mode),
        "parse_data_types": True,
        "data_schema": _data_schema(pd.concat([context, queries], ignore_index=True)),
    }
    records = _json_records(combined)
    if layout == "rows":
        payload["rows"] = records
    elif layout == "columns":
        payload["columns"] = {
            column: [record[column] for record in records]
            for column in combined.columns
        }
    else:
        raise ValueError("layout must be 'rows' or 'columns'")
    return payload, truth


def _normalize_label(value: Any) -> str:
    """Normalize numeric and string class labels for metric comparison."""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _prediction_groups(
    response: Mapping[str, Any], target: str
) -> dict[str, list[dict[str, Any]]]:
    """Group flattened candidates for one target by response index.

    Args:
        response: Raw successful RPT response.
        target: Target column whose candidates should be grouped.

    Returns:
        Candidate records keyed by string form of ``row_id``.
    """
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in flatten_predictions(response, index_column="row_id"):
        if item["target"] == target:
            grouped[str(item["index"])].append(item)
    return dict(grouped)


def _interval_bounds(value: Any) -> tuple[float, float] | None:
    """Read confidence interval bounds from either documented JSON representation."""
    if (
        isinstance(value, Sequence)
        and not isinstance(value, (str, bytes))
        and len(value) == 2
    ):
        return float(value[0]), float(value[1])
    if isinstance(value, Mapping):
        lower = value.get("lower", value.get("lower_bound"))
        upper = value.get("upper", value.get("upper_bound"))
        if lower is not None and upper is not None:
            return float(lower), float(upper)
    return None


def regression_metrics(
    response: Mapping[str, Any], truth: Mapping[str, Mapping[str, Any]], target: str
) -> dict[str, float | int | None]:
    """Compute report-only regression and confidence-interval metrics.

    Args:
        response: Raw successful RPT response.
        truth: Held-out values keyed by response index.
        target: Regression target column.

    Returns:
        Count, MAE, RMSE, R-squared, interval coverage, and mean interval width.
    """
    grouped = _prediction_groups(response, target)
    if set(grouped) != set(truth):
        raise AssertionError(
            f"Regression indices differ: got {len(grouped)}, expected {len(truth)}"
        )
    actual = [float(truth[index][target]) for index in truth]
    predicted = [float(grouped[index][0]["prediction"]) for index in truth]
    errors = [estimate - observed for estimate, observed in zip(predicted, actual)]
    mae = sum(abs(error) for error in errors) / len(errors)
    rmse = math.sqrt(sum(error * error for error in errors) / len(errors))
    mean_actual = sum(actual) / len(actual)
    denominator = sum((value - mean_actual) ** 2 for value in actual)
    r_squared = (
        None
        if denominator == 0
        else 1 - sum(error * error for error in errors) / denominator
    )

    intervals = [
        _interval_bounds(grouped[index][0].get("confidence_interval"))
        for index in truth
    ]
    available = [
        (bounds, observed)
        for bounds, observed in zip(intervals, actual)
        if bounds is not None
    ]
    if len(available) != len(actual):
        raise AssertionError(
            f"Expected {len(actual)} regression confidence intervals, received {len(available)}"
        )
    coverage = None
    mean_width = None
    if available:
        coverage = sum(
            lower <= observed <= upper for (lower, upper), observed in available
        ) / len(available)
        mean_width = sum(upper - lower for (lower, upper), _ in available) / len(
            available
        )
    return {
        "query_rows": len(actual),
        "mae": mae,
        "rmse": rmse,
        "r_squared": r_squared,
        "interval_rows": len(available),
        "interval_coverage": coverage,
        "mean_interval_width": mean_width,
    }


def classification_metrics(
    response: Mapping[str, Any], truth: Mapping[str, Mapping[str, Any]], target: str
) -> dict[str, float | int]:
    """Compute report-only accuracy, macro-F1, and top-k accuracy.

    Args:
        response: Raw successful RPT response.
        truth: Held-out values keyed by response index.
        target: Classification target column.

    Returns:
        Query count and classification quality metrics.
    """
    grouped = _prediction_groups(response, target)
    if set(grouped) != set(truth):
        raise AssertionError(
            f"Classification indices differ: got {len(grouped)}, expected {len(truth)}"
        )
    actual = [_normalize_label(truth[index][target]) for index in truth]
    top_one = [_normalize_label(grouped[index][0]["prediction"]) for index in truth]
    top_k = [
        {_normalize_label(candidate["prediction"]) for candidate in grouped[index]}
        for index in truth
    ]
    accuracy = sum(
        observed == estimate for observed, estimate in zip(actual, top_one)
    ) / len(actual)
    top_k_accuracy = sum(
        observed in candidates for observed, candidates in zip(actual, top_k)
    ) / len(actual)
    labels = sorted(set(actual) | set(top_one))
    f1_values = []
    for label in labels:
        true_positive = sum(a == label and p == label for a, p in zip(actual, top_one))
        false_positive = sum(a != label and p == label for a, p in zip(actual, top_one))
        false_negative = sum(a == label and p != label for a, p in zip(actual, top_one))
        denominator = 2 * true_positive + false_positive + false_negative
        f1_values.append(0.0 if denominator == 0 else 2 * true_positive / denominator)
    return {
        "query_rows": len(actual),
        "accuracy": accuracy,
        "macro_f1": sum(f1_values) / len(f1_values),
        "top_k_accuracy": top_k_accuracy,
    }


def _assert_explanations(
    response: Mapping[str, Any],
    source_data: Sequence[Mapping[str, Any]] | Mapping[str, Sequence[Any]],
    query_count: int,
) -> None:
    """Assert that requested explanation arrays map back to every query row."""
    mapped = map_explanation_rows(response, source_data)
    if len(mapped) != query_count:
        raise AssertionError(
            f"Expected {query_count} explanations, received {len(mapped)}"
        )
    if any(not item["column_scores"] for item in mapped):
        raise AssertionError("Every query row must contain top_column_scores")
    if any(not item["relevant_context_rows"] for item in mapped):
        raise AssertionError("Every query row must contain top_relevant_context_rows")


def _single_target_case(
    client: RPTClientAPI,
    frame: pd.DataFrame,
    *,
    target: str,
    task_type: str,
    query_count: int,
    seed: int,
    use_async: bool,
) -> dict[str, Any]:
    """Run one explained regression or top-k classification live case."""
    context, queries = _split_frame(frame, query_count=query_count, seed=seed)
    top_k = 2 if task_type == "classification" else None
    payload, truth = _build_payload(context, queries, [(target, task_type, top_k)])
    response = (
        asyncio.run(client.apredict_json(payload))
        if use_async
        else client.predict_json(payload)
    )
    _assert_explanations(response, payload["rows"], query_count)
    metrics = (
        regression_metrics(response, truth, target)
        if task_type == "regression"
        else classification_metrics(response, truth, target)
    )
    if task_type == "classification" and any(
        len(candidates) < 2
        for candidates in _prediction_groups(response, target).values()
    ):
        raise AssertionError("top_k=2 did not return two candidates for every query")
    return {
        "response_id": response.get("id"),
        "status": response.get("status"),
        "transport": "async" if use_async else "sync",
        "metrics": metrics,
    }


def _multi_target_case(client: RPTClientAPI, frame: pd.DataFrame) -> dict[str, Any]:
    """Run gzip column-oriented joint regression and classification validation."""
    prepared = frame.copy()
    prepared["MEDV_BIN"] = (prepared["MEDV"] >= prepared["MEDV"].median()).astype(int)
    context, queries = _split_frame(prepared, query_count=16, seed=43)
    targets = [("MEDV", "regression", None), ("MEDV_BIN", "classification", 2)]
    payload, truth = _build_payload(
        context, queries, targets, layout="columns", context_mode="default"
    )
    response = client.predict_json(payload, compress=True)
    _assert_explanations(response, payload["columns"], len(queries))
    return {
        "response_id": response.get("id"),
        "status": response.get("status"),
        "layout": "columns",
        "compression": "gzip",
        "regression_metrics": regression_metrics(response, truth, "MEDV"),
        "classification_metrics": classification_metrics(response, truth, "MEDV_BIN"),
    }


def _parquet_case(client: RPTClientAPI, frame: pd.DataFrame) -> dict[str, Any]:
    """Run explained multi-target Parquet validation with safe placeholders."""
    prepared = frame.copy()
    prepared["MEDV_BIN"] = (prepared["MEDV"] >= prepared["MEDV"].median()).astype(int)
    context, queries = _split_frame(prepared, query_count=16, seed=44)
    targets = [("MEDV", "regression", None), ("MEDV_BIN", "classification", 2)]
    payload, truth = _build_payload(context, queries, targets, context_mode="default")
    parquet_frame = pd.DataFrame(payload.pop("rows"))
    # Parquet columns have one physical type, so numeric target values become strings
    # before string prediction placeholders are inserted.
    for target, _, _ in targets:
        parquet_frame[target] = parquet_frame[target].astype(str)
    with tempfile.TemporaryDirectory(prefix="rpt-live-") as temporary_directory:
        parquet_path = Path(temporary_directory) / "multi_target.parquet"
        parquet_frame.to_parquet(parquet_path, index=False)
        response = client.predict_parquet(
            parquet_path,
            payload["prediction_config"],
            index_column="row_id",
            parse_data_types=True,
        )
    _assert_explanations(response, _json_records(parquet_frame), len(queries))
    return {
        "response_id": response.get("id"),
        "status": response.get("status"),
        "format": "parquet",
        "regression_metrics": regression_metrics(response, truth, "MEDV"),
        "classification_metrics": classification_metrics(response, truth, "MEDV_BIN"),
    }


def _deep_context_case(client: RPTClientAPI, frame: pd.DataFrame) -> dict[str, Any]:
    """Run RPT-1.6-large deep-context regression above 8,000 context rows."""
    prepared = frame.drop(columns=["casual", "registered"]).copy()
    context, queries = _split_frame(prepared, query_count=16, seed=45)
    context = context.iloc[:8_001].copy()
    payload, truth = _build_payload(
        context,
        queries,
        [("cnt", "regression", None)],
        context_mode="deep",
    )
    response = client.predict_json(payload, compress=True)
    _assert_explanations(response, payload["rows"], len(queries))
    metadata = response.get("metadata") or {}
    if metadata.get("context_mode") != "deep":
        raise AssertionError(f"Expected deep context metadata, received {metadata!r}")
    return {
        "response_id": response.get("id"),
        "status": response.get("status"),
        "context_rows": len(context),
        "compression": "gzip",
        "metrics": regression_metrics(response, truth, "cnt"),
    }


def _load_datasets(
    dataset_root: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load and minimally prepare the three deterministic validation datasets."""
    boston = pd.read_csv(dataset_root / "boston" / "boston.csv")
    titanic = pd.read_csv(dataset_root / "titanic" / "train.csv").drop(
        columns=["Name", "Ticket", "Cabin"]
    )
    bike = pd.read_csv(dataset_root / "bike_sharing" / "hour.csv")
    boston.insert(0, "row_id", [f"b-{index:04d}" for index in range(len(boston))])
    titanic.insert(0, "row_id", [f"t-{index:04d}" for index in range(len(titanic))])
    bike.insert(0, "row_id", [f"h-{index:05d}" for index in range(len(bike))])
    return boston, titanic, bike


def run_live_matrix(
    dataset_root: Path, models: Sequence[str], timeout_s: int
) -> dict[str, Any]:
    """Execute the requested live contract and report-only quality matrix.

    Args:
        dataset_root: Directory containing ``boston``, ``titanic``, and
            ``bike_sharing`` subdirectories.
        models: Model names to validate.
        timeout_s: Per-request client timeout.

    Returns:
        JSON-serializable validation report.
    """
    boston, titanic, bike = _load_datasets(dataset_root)
    special_cases = int("sap-rpt-1.6" in models) * 2 + int(
        "sap-rpt-1.6-large" in models
    )
    progress = ProgressBar(len(models) * 2 + special_cases)
    results: list[dict[str, Any]] = []
    clients: dict[str, RPTClientAPI] = {}

    for model in models:
        client = RPTClientAPI(model_name=model, model_version="1", timeout_s=timeout_s)
        clients[model] = client
        regression = _single_target_case(
            client,
            boston,
            target="MEDV",
            task_type="regression",
            query_count=32,
            seed=41,
            use_async=False,
        )
        results.append({"model": model, "case": "boston_regression", **regression})
        progress.advance(f"{model} regression")

        classification = _single_target_case(
            client,
            titanic,
            target="Survived",
            task_type="classification",
            query_count=32,
            seed=42,
            use_async=True,
        )
        results.append(
            {"model": model, "case": "titanic_classification", **classification}
        )
        progress.advance(f"{model} classification")

    if "sap-rpt-1.6" in clients:
        multi_target = _multi_target_case(clients["sap-rpt-1.6"], boston)
        results.append(
            {"model": "sap-rpt-1.6", "case": "multi_target_json", **multi_target}
        )
        progress.advance("sap-rpt-1.6 multi-target gzip columns")

        parquet = _parquet_case(clients["sap-rpt-1.6"], boston)
        results.append(
            {"model": "sap-rpt-1.6", "case": "multi_target_parquet", **parquet}
        )
        progress.advance("sap-rpt-1.6 multi-target parquet")

    if "sap-rpt-1.6-large" in clients:
        deep = _deep_context_case(clients["sap-rpt-1.6-large"], bike)
        results.append({"model": "sap-rpt-1.6-large", "case": "deep_context", **deep})
        progress.advance("sap-rpt-1.6-large deep context")

    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "quality_gate": "report-only",
        "dataset_root": str(dataset_root.resolve()),
        "models": list(models),
        "results": results,
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line options for the live validation matrix."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path.cwd() / "datasets",
        help="Directory containing the Boston, Titanic, and bike-sharing datasets.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=SUPPORTED_MODELS,
        default=list(SUPPORTED_MODELS),
        help="RPT deployments to validate; defaults to all supported models.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=600,
        help="Per-request timeout in seconds; deep mode can take several minutes.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional JSON report path. No source or prediction rows are written.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run live validation and return a shell-friendly status code."""
    args = _parse_args(argv)
    try:
        report = run_live_matrix(args.dataset_root, args.models, args.timeout)
    except (RPTAPIError, AssertionError, FileNotFoundError, ValueError) as exc:
        print(f"Validation failed: {exc}", file=sys.stderr)
        return 1

    rendered = json.dumps(report, indent=2, default=str)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
        print(f"Report written to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
