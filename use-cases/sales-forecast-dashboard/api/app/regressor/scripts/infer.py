#!/usr/bin/env python
"""
Inference Script.

Run inference with saved models on new data.

Usage:
    python -m forecasting.regressor.scripts.infer \
        --model-b data/new_features.csv \
        --checkpoints output/checkpoints \
        --output predictions/

Arguments:
    --model-b (required): Path to Model B feature CSV
    --model-a (optional): Path to Model A feature CSV for explainability
    --checkpoints: Directory with saved models (default: output/checkpoints)
    --output: Output directory (default: output_infer)
    --correct-bm: Apply bias correction to B&M predictions
    --correct-web: Apply bias correction to all WEB predictions
"""

import argparse
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Run inference with saved models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
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

    args = parser.parse_args()

    # Normalize channel names
    channel_map = {"bm": "B&M", "web": "WEB", "B&M": "B&M", "WEB": "WEB"}
    channels = [channel_map.get(c.lower() if hasattr(c, 'lower') else c, c) for c in args.channels]

    # Import and run
    from app.regressor.pipelines import InferencePipeline
    from app.regressor.configs import InferenceConfig, BiasCorrection

    config = InferenceConfig(
        checkpoint_dir=Path(args.checkpoints),
        output_dir=Path(args.output),
        channels=channels,
        run_explainability=not args.no_explainability,
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
    pipeline = InferencePipeline(config)
    result = pipeline.run(model_b_data, model_a_data, channels)

    # Print summary
    print("\n" + "=" * 60)
    print("INFERENCE SUMMARY")
    print("=" * 60)

    if result.bm_predictions is not None:
        print(f"\nB&M Channel: {len(result.bm_predictions):,} predictions")
        print(f"  Output: {args.output}/predictions_bm.csv")

    if result.web_predictions is not None:
        print(f"\nWEB Channel: {len(result.web_predictions):,} predictions")
        print(f"  Output: {args.output}/predictions_web.csv")


if __name__ == "__main__":
    main()
