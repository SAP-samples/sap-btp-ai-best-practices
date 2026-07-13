#!/usr/bin/env python
"""
Generate baseline forecast data for inference.

This script generates "business as usual" baseline data for future horizons
using a seasonal naive approach. It creates both Model A (explainability) and
Model B (prediction) compatible outputs.

Usage:
    python -m forecasting.regressor.scripts.generate_baseline \
        --model-b final_data/model_b.csv \
        --output output/baselines \
        --horizons 1 52

    # With specific origin date override
    python -m forecasting.regressor.scripts.generate_baseline \
        --model-b final_data/model_b.csv \
        --output output/baselines \
        --origin-date 2024-12-02

    # B&M channel only
    python -m forecasting.regressor.scripts.generate_baseline \
        --model-b final_data/model_b.csv \
        --output output/baselines \
        --channels B&M

Output:
    - baseline_model_a.csv: Actionable features for SHAP analysis
    - baseline_model_b.csv: Full features for prediction

Author: EPIC 4 Feature Engineering
Status: Step 3 - Baseline Generator
"""

import argparse
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Generate baseline forecast data for inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Generate baselines for horizons 1-52 weeks
    python -m forecasting.regressor.scripts.generate_baseline \\
        --model-b final_data/model_b.csv \\
        --output output/baselines

    # Generate with specific origin date
    python -m forecasting.regressor.scripts.generate_baseline \\
        --model-b final_data/model_b.csv \\
        --output output/baselines \\
        --origin-date 2024-12-02 \\
        --horizons 1 26
        """
    )

    parser.add_argument(
        "--model-b",
        required=True,
        type=Path,
        help="Path to model_b.csv (required)"
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Output directory for baseline files"
    )
    parser.add_argument(
        "--horizons",
        nargs=2,
        type=int,
        default=[1, 52],
        metavar=("START", "END"),
        help="Horizon range (start end), default: 1 52"
    )
    parser.add_argument(
        "--origin-date",
        type=str,
        default=None,
        help="Override origin date for all stores (YYYY-MM-DD). "
             "If not specified, uses last available date per store."
    )
    parser.add_argument(
        "--channels",
        nargs="+",
        default=["B&M", "WEB"],
        help="Channels to generate (default: B&M WEB)"
    )

    args = parser.parse_args()

    # Validate model_b path
    if not args.model_b.exists():
        print(f"Error: Model B file not found: {args.model_b}")
        sys.exit(1)

    # Import here to avoid slow import on --help
    from app.regressor.baseline_generator import BaselineGenerator, BaselineConfig

    # Create config
    config = BaselineConfig(
        model_b_path=args.model_b,
        output_dir=args.output,
        horizons=range(args.horizons[0], args.horizons[1] + 1),
        channels=args.channels,
        origin_date=args.origin_date,
    )

    print("=" * 60)
    print("Baseline Generator")
    print("=" * 60)
    print(f"Input:       {args.model_b}")
    print(f"Output:      {args.output}")
    print(f"Horizons:    {args.horizons[0]} to {args.horizons[1]}")
    print(f"Channels:    {', '.join(args.channels)}")
    print(f"Origin Date: {args.origin_date or 'Last available per store'}")
    print("=" * 60)
    print()

    # Generate baselines
    generator = BaselineGenerator(config)
    model_a_df, model_b_df = generator.generate()

    # Save outputs
    model_a_path, model_b_path = generator.save(model_a_df, model_b_df)

    print()
    print("=" * 60)
    print("Generation Complete")
    print("=" * 60)
    print(f"Model A saved: {model_a_path}")
    print(f"  Rows: {len(model_a_df):,}")
    print(f"  Columns: {len(model_a_df.columns)}")
    print()
    print(f"Model B saved: {model_b_path}")
    print(f"  Rows: {len(model_b_df):,}")
    print(f"  Columns: {len(model_b_df.columns)}")
    print("=" * 60)

    # Print sample of generated dates
    print()
    print("Sample target weeks (first store):")
    sample = model_b_df.head(min(5, len(model_b_df)))[
        ["profit_center_nbr", "channel", "origin_week_date", "horizon", "target_week_date"]
    ]
    print(sample.to_string(index=False))


if __name__ == "__main__":
    main()
