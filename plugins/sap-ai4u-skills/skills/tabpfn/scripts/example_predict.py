"""Runnable TabPFN 3.5-plus demo: one classification and one regression call.

Uses a tiny synthetic in-context dataset (no external files, no pandas) so it
runs anywhere the AICORE_* environment is configured. Proves the deployment is
reachable and shows the response shape for both tasks.

Run:
    # from a directory containing a .env with the AICORE_* service key:
    python example_predict.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Import the client sitting next to this script.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from tabpfn_client_api import AICoreError, TabPFNClient  # noqa: E402


def _synthetic() -> tuple[list[list[float]], list[str], list[float], list[list[float]]]:
    """Build a small deterministic dataset: 25 context rows, 5 query rows.

    Feature 0 is numeric; feature 1 is a small integer category (0/1/2).
    Classification label: "high" when feature 0 >= 15, else "low".
    Regression target: y = 2*feature0 + feature1 (a clean linear rule).
    """
    x = [[float(i), float(i % 3)] for i in range(30)]
    y_class = ["high" if i >= 15 else "low" for i in range(30)]
    y_reg = [2.0 * i + (i % 3) for i in range(30)]
    x_train, x_test = x[:25], x[25:]
    return x_train, y_class[:25], y_reg[:25], x_test


def main() -> int:
    """Run a classification and a regression prediction and print the results."""
    x_train, y_class, y_reg, x_test = _synthetic()
    try:
        client = TabPFNClient()
    except AICoreError as exc:
        print(f"Setup failed: {exc}")
        return 1
    print(f"Resolved deployment: {client.deployment_url}\n")

    # feature 1 is categorical -> declare it, a free accuracy lever.
    cfg = {"categorical_features_indices": [1]}

    clf = client.predict_raw(x_train, y_class, x_test, "classification", tabpfn_config=cfg)
    classes = clf["metadata"]["classes"]
    labels = [classes[max(range(len(p)), key=p.__getitem__)] for p in clf["prediction"]]
    print("Classification")
    print(f"  classes      : {classes}")
    print(f"  probabilities: {[[round(v, 3) for v in row] for row in clf['prediction']]}")
    print(f"  labels       : {labels}")
    print(f"  billed cells : {clf['usage'].get('num_cells')}\n")

    reg = client.predict_raw(x_train, y_reg, x_test, "regression", tabpfn_config=cfg)
    print("Regression (true rule y = 2*f0 + f1)")
    print(f"  x_test       : {x_test}")
    print(f"  predictions  : {[round(v, 2) for v in reg['prediction']]}")
    print(f"  billed cells : {reg['usage'].get('num_cells')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
