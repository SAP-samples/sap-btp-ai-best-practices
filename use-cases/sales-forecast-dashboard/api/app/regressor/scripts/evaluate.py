#!/usr/bin/env python
"""
Evaluation Script.

Evaluate model predictions against ground truth.

Usage:
    python -m forecasting.regressor.scripts.evaluate \
        --predictions-bm output/predictions_bm.csv \
        --predictions-web output/predictions_web.csv \
        --output evaluation/

Arguments:
    --predictions-bm: Path to B&M predictions CSV
    --predictions-web: Path to WEB predictions CSV
    --output: Output directory for evaluation summary (default: output)

Metrics computed:
    - MAE: Mean Absolute Error
    - WMAPE: Weighted Mean Absolute Percentage Error
    - Bias: Mean Error (signed)
    - R2: Coefficient of Determination
"""

import argparse
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate model predictions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--predictions-bm",
        help="Path to B&M predictions CSV",
    )
    parser.add_argument(
        "--predictions-web",
        help="Path to WEB predictions CSV",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="output",
        help="Output directory for evaluation summary (default: output)",
    )

    args = parser.parse_args()

    if not args.predictions_bm and not args.predictions_web:
        print("Error: At least one of --predictions-bm or --predictions-web required")
        parser.print_help()
        sys.exit(1)

    # Import and run
    from app.regressor.pipelines import evaluate

    result = evaluate(
        predictions_bm_path=args.predictions_bm,
        predictions_web_path=args.predictions_web,
        output_dir=args.output,
    )

    # Print and save summary
    result.print_summary()
    result.save()


if __name__ == "__main__":
    main()
