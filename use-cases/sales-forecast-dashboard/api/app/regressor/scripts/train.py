#!/usr/bin/env python
"""
Training Script.

Train forecasting models on prepared feature data.

Usage:
    python -m forecasting.regressor.scripts.train \
        --model-b data/model_b.csv \
        --model-a data/model_a.csv \
        --output output/ \
        --channels bm web \
        --correct-bm

Arguments:
    --model-b (required): Path to Model B CSV (full autoregressive features)
    --model-a (optional): Path to Model A CSV (business features) for explainability
    --output: Output directory (default: output)
    --channels: Channels to train (default: bm web)
    --correct-bm: Apply bias correction to B&M predictions
    --correct-web: Apply bias correction to all WEB predictions
"""

import argparse
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Train forecasting models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
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

    args = parser.parse_args()

    # Normalize channel names
    channel_map = {"bm": "B&M", "web": "WEB", "B&M": "B&M", "WEB": "WEB"}
    channels = [channel_map.get(c.lower() if hasattr(c, 'lower') else c, c) for c in args.channels]

    # Import and run
    from app.regressor.pipelines import TrainingPipeline
    from app.regressor.configs import TrainingConfig, BiasCorrection

    config = TrainingConfig(
        output_dir=Path(args.output),
        channels=channels,
        train_surrogate=not args.no_surrogate,
        bias_correction=BiasCorrection(
            correct_bm=args.correct_bm,
            correct_web=args.correct_web,
            correct_web_sales=args.correct_web_sales,
            correct_web_aov=args.correct_web_aov,
        ),
    )

    # Load data
    import pandas as pd
    model_b_data = pd.read_csv(args.model_b)
    model_a_data = pd.read_csv(args.model_a) if args.model_a else None

    # Run pipeline
    pipeline = TrainingPipeline(config)
    result = pipeline.run(model_b_data, model_a_data, channels)

    # Print summary
    print("\n" + "=" * 60)
    print("TRAINING SUMMARY")
    print("=" * 60)

    if result.bm_result:
        print(f"\nB&M Channel:")
        print(f"  Train samples: {result.bm_result.train_samples:,}")
        print(f"  Test samples: {result.bm_result.test_samples:,}")
        print(f"  RMSE Sales: {result.bm_result.rmse_sales:.4f}")
        print(f"  RMSE AOV: {result.bm_result.rmse_aov:.4f}")
        print(f"  RMSE Conversion: {result.bm_result.rmse_conv:.4f}")
        if result.bm_result.surrogate_r2 > 0:
            print(f"  Surrogate R2: {result.bm_result.surrogate_r2:.4f}")

    if result.web_result:
        print(f"\nWEB Channel:")
        print(f"  Train samples: {result.web_result.train_samples:,}")
        print(f"  Test samples: {result.web_result.test_samples:,}")
        print(f"  RMSE Sales: {result.web_result.rmse_sales:.4f}")
        print(f"  RMSE AOV: {result.web_result.rmse_aov:.4f}")
        if result.web_result.surrogate_r2 > 0:
            print(f"  Surrogate R2: {result.web_result.surrogate_r2:.4f}")

    print(f"\nOutputs saved to: {result.output_dir}")
    print(f"Checkpoints saved to: {result.checkpoint_dir}")


if __name__ == "__main__":
    main()
