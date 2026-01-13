#!/usr/bin/env python
"""
Unified Pipeline Entry Point.

Single entry point for all forecasting regressor operations:
- generate: Generate canonical training data
- train: Train models on feature data
- infer: Run inference with saved models
- evaluate: Evaluate model predictions

Usage:
    # Generate training data
    python -m forecasting.regressor.scripts.run_pipeline generate \
        --horizons 1 52 \
        --output data/canonical_training.csv

    # Train models
    python -m forecasting.regressor.scripts.run_pipeline train \
        --model-b data/model_b.csv \
        --model-a data/model_a.csv \
        --output output/

    # Run inference
    python -m forecasting.regressor.scripts.run_pipeline infer \
        --model-b data/new_features.csv \
        --checkpoints output/checkpoints \
        --output predictions/

    # Evaluate
    python -m forecasting.regressor.scripts.run_pipeline evaluate \
        --predictions-bm predictions/predictions_bm.csv \
        --predictions-web predictions/predictions_web.csv
"""

import argparse
import sys
from pathlib import Path

from app.regressor.features.model_views import (
    MODEL_A_FEATURES,
    MODEL_A_BM_ONLY_FEATURES,
)


def add_generate_parser(subparsers):
    """Add generate subcommand parser."""
    parser = subparsers.add_parser(
        "generate",
        help="Generate canonical training data",
        description="Generate the canonical training table with features for model training.",
    )
    parser.add_argument(
        "--horizons",
        nargs=2,
        type=int,
        default=[1, 52],
        metavar=("START", "END"),
        help="Horizon range (default: 1 52)",
    )
    parser.add_argument(
        "--output",
        "-o",
        required=True,
        help="Output CSV path",
    )
    parser.add_argument(
        "--include-crm",
        action="store_true",
        help="Include CRM demographic features",
    )
    parser.add_argument(
        "--model",
        choices=["A", "B", "both"],
        default="both",
        help="Generate Model A, B, or both feature sets (default: both)",
    )
    return parser


def add_train_parser(subparsers):
    """Add train subcommand parser."""
    parser = subparsers.add_parser(
        "train",
        help="Train forecasting models",
        description="Train B&M and/or WEB channel models with optional explainability.",
    )
    parser.add_argument(
        "--model-b",
        required=True,
        help="Path to Model B CSV (full features)",
    )
    parser.add_argument(
        "--model-a",
        help="Path to Model A CSV (business features) for explainability",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="output",
        help="Output directory (default: output)",
    )
    parser.add_argument(
        "--channels",
        nargs="+",
        choices=["bm", "web", "B&M", "WEB"],
        default=["bm", "web"],
        help="Channels to train (default: bm web)",
    )
    parser.add_argument(
        "--correct-bm",
        action="store_true",
        help="Apply bias correction to B&M predictions",
    )
    parser.add_argument(
        "--correct-web",
        action="store_true",
        help="Apply bias correction to all WEB predictions",
    )
    parser.add_argument(
        "--correct-web-sales",
        action="store_true",
        help="Apply bias correction to WEB Sales only",
    )
    parser.add_argument(
        "--correct-web-aov",
        action="store_true",
        help="Apply bias correction to WEB AOV only",
    )
    parser.add_argument(
        "--no-surrogate",
        action="store_true",
        help="Skip surrogate model training",
    )
    return parser


def add_infer_parser(subparsers):
    """Add infer subcommand parser."""
    parser = subparsers.add_parser(
        "infer",
        help="Run inference with saved models",
        description="Load trained models and generate predictions on new data.",
    )
    parser.add_argument(
        "--model-b",
        required=True,
        help="Path to Model B feature CSV",
    )
    parser.add_argument(
        "--model-a",
        help="Path to Model A feature CSV for explainability",
    )
    parser.add_argument(
        "--checkpoints",
        default="output/checkpoints",
        help="Directory with saved models (default: output/checkpoints)",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="output_infer",
        help="Output directory (default: output_infer)",
    )
    parser.add_argument(
        "--channels",
        nargs="+",
        choices=["bm", "web", "B&M", "WEB"],
        default=["bm", "web"],
        help="Channels to process (default: bm web)",
    )
    parser.add_argument(
        "--correct-bm",
        action="store_true",
        help="Apply bias correction to B&M predictions",
    )
    parser.add_argument(
        "--correct-web",
        action="store_true",
        help="Apply bias correction to all WEB predictions",
    )
    parser.add_argument(
        "--correct-web-sales",
        action="store_true",
        help="Apply bias correction to WEB Sales only",
    )
    parser.add_argument(
        "--correct-web-aov",
        action="store_true",
        help="Apply bias correction to WEB AOV only",
    )
    parser.add_argument(
        "--no-explainability",
        action="store_true",
        help="Skip explainability analysis",
    )
    return parser


def add_evaluate_parser(subparsers):
    """Add evaluate subcommand parser."""
    parser = subparsers.add_parser(
        "evaluate",
        help="Evaluate model predictions",
        description="Compute metrics comparing predictions to ground truth.",
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
    return parser


def normalize_channels(channels):
    """Normalize channel names to standard format."""
    mapping = {"bm": "B&M", "web": "WEB", "B&M": "B&M", "WEB": "WEB"}
    return [mapping.get(c.lower() if hasattr(c, 'lower') else c, c) for c in channels]


def _expected_model_a_features(channels_present):
    """Return the ordered feature list expected for the detected channels."""
    channels = {str(ch).upper() for ch in channels_present if ch is not None}
    if not channels:
        # Default to full set when channel info is missing.
        return list(MODEL_A_FEATURES)
    if channels <= {"WEB"}:
        return [f for f in MODEL_A_FEATURES if f not in MODEL_A_BM_ONLY_FEATURES]
    return list(MODEL_A_FEATURES)


def ensure_model_a_columns(df):
    """
    Validate that the generated Model A dataframe includes all actionable levers.

    Returns a dataframe with feature columns reordered to match MODEL_A_FEATURES.
    """
    if "channel" not in df.columns:
        raise ValueError("Model A dataframe is missing 'channel'; cannot validate features.")

    expected = _expected_model_a_features(df["channel"].unique())
    missing = [col for col in expected if col not in df.columns]
    if missing:
        raise ValueError(
            "Model A output is missing expected actionable features: "
            f"{missing}. Ensure feature engineering ran successfully."
        )

    non_feature_cols = [col for col in df.columns if col not in expected]
    ordered_cols = non_feature_cols + expected
    return df.loc[:, ordered_cols]


def cmd_generate(args):
    """Execute generate command."""
    from app.regressor.etl import build_canonical_training_table

    horizons = range(args.horizons[0], args.horizons[1] + 1)

    print(f"Generating canonical training table...")
    print(f"  Horizons: {args.horizons[0]} to {args.horizons[1]}")
    print(f"  Model variant: {args.model}")
    print(f"  Include CRM: {args.include_crm}")
    print(f"  Output: {args.output}")

    # Determine model variant
    if args.model == "both":
        # Generate both Model A and Model B features
        print("\nGenerating Model B features...")
        df_b = build_canonical_training_table(
            horizons=horizons,
            include_features=True,
            model_variant='B',
            include_crm=args.include_crm,
        )

        # Save Model B
        output_path = Path(args.output)
        if output_path.suffix == '.csv':
            model_b_path = output_path.with_stem(output_path.stem + '_model_b')
            model_a_path = output_path.with_stem(output_path.stem + '_model_a')
            # Ensure parent directory exists for file paths
            model_b_path.parent.mkdir(parents=True, exist_ok=True)
        else:
            model_b_path = output_path / 'model_b.csv'
            model_a_path = output_path / 'model_a.csv'
            output_path.mkdir(parents=True, exist_ok=True)

        df_b.to_csv(model_b_path, index=False)
        print(f"  Saved Model B: {model_b_path} ({len(df_b)} rows)")

        print("\nGenerating Model A features...")
        df_a = build_canonical_training_table(
            horizons=horizons,
            include_features=True,
            model_variant='A',
            include_crm=args.include_crm,
        )
        df_a = ensure_model_a_columns(df_a)
        df_a.to_csv(model_a_path, index=False)
        print(f"  Saved Model A: {model_a_path} ({len(df_a)} rows)")
    else:
        # Generate single model variant
        df = build_canonical_training_table(
            horizons=horizons,
            include_features=True,
            model_variant=args.model,
            include_crm=args.include_crm,
        )

        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"\nSaved: {output_path} ({len(df)} rows)")

    print("\nGeneration complete!")


def cmd_train(args):
    """Execute train command."""
    from app.regressor.pipelines import train
    from app.regressor.configs import BiasCorrection

    channels = normalize_channels(args.channels)

    print(f"Training models...")
    print(f"  Model B: {args.model_b}")
    print(f"  Model A: {args.model_a or 'None (no explainability)'}")
    print(f"  Channels: {channels}")
    print(f"  Output: {args.output}")

    result = train(
        model_b_path=args.model_b,
        model_a_path=args.model_a,
        output_dir=args.output,
        channels=channels,
        correct_bm=args.correct_bm,
        correct_web=args.correct_web,
        correct_web_sales=args.correct_web_sales,
        correct_web_aov=args.correct_web_aov,
        train_surrogate=not args.no_surrogate,
    )

    print("\nTraining complete!")
    if result.bm_result:
        print(f"  B&M: RMSE Sales={result.bm_result.rmse_sales:.4f}, "
              f"AOV={result.bm_result.rmse_aov:.4f}, "
              f"Conv={result.bm_result.rmse_conv:.4f}")
    if result.web_result:
        print(f"  WEB: RMSE Sales={result.web_result.rmse_sales:.4f}, "
              f"AOV={result.web_result.rmse_aov:.4f}")


def cmd_infer(args):
    """Execute infer command."""
    from app.regressor.pipelines import infer

    channels = normalize_channels(args.channels)

    print(f"Running inference...")
    print(f"  Model B: {args.model_b}")
    print(f"  Model A: {args.model_a or 'None'}")
    print(f"  Checkpoints: {args.checkpoints}")
    print(f"  Output: {args.output}")

    result = infer(
        model_b_path=args.model_b,
        model_a_path=args.model_a,
        checkpoint_dir=args.checkpoints,
        output_dir=args.output,
        channels=channels,
        correct_bm=args.correct_bm,
        correct_web=args.correct_web,
        correct_web_sales=args.correct_web_sales,
        correct_web_aov=args.correct_web_aov,
        run_explainability=not args.no_explainability,
    )

    print("\nInference complete!")
    if result.bm_predictions is not None:
        print(f"  B&M predictions: {len(result.bm_predictions)} rows")
    if result.web_predictions is not None:
        print(f"  WEB predictions: {len(result.web_predictions)} rows")


def cmd_evaluate(args):
    """Execute evaluate command."""
    from app.regressor.pipelines import evaluate

    if not args.predictions_bm and not args.predictions_web:
        print("Error: At least one of --predictions-bm or --predictions-web required")
        sys.exit(1)

    print(f"Evaluating predictions...")

    result = evaluate(
        predictions_bm_path=args.predictions_bm,
        predictions_web_path=args.predictions_web,
        output_dir=args.output,
    )

    result.print_summary()
    result.save()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        prog="forecasting.regressor",
        description="Unified pipeline for sales forecasting regressor",
    )

    subparsers = parser.add_subparsers(
        dest="command",
        title="commands",
        description="Available commands",
    )

    add_generate_parser(subparsers)
    add_train_parser(subparsers)
    add_infer_parser(subparsers)
    add_evaluate_parser(subparsers)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    # Dispatch to appropriate command handler
    commands = {
        "generate": cmd_generate,
        "train": cmd_train,
        "infer": cmd_infer,
        "evaluate": cmd_evaluate,
    }

    handler = commands.get(args.command)
    if handler:
        handler(args)
    else:
        print(f"Unknown command: {args.command}")
        sys.exit(1)


if __name__ == "__main__":
    main()
