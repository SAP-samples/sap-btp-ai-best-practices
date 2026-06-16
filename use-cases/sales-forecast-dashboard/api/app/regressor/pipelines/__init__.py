"""
Pipelines Module.

Provides orchestration pipelines for training, inference, and evaluation:
- TrainingPipeline: End-to-end model training with SHAP explainability
- InferencePipeline: Production scoring with saved models
- EvaluationPipeline: Model evaluation with comprehensive metrics

Usage:
    from app.regressor.pipelines import (
        TrainingPipeline,
        InferencePipeline,
        EvaluationPipeline,
        train,
        infer,
        evaluate,
    )

    # Train models
    result = train(
        model_b_path="data/model_b.csv",
        model_a_path="data/model_a.csv",
        output_dir="output",
    )

    # Run inference
    predictions = infer(
        model_b_path="data/new_features.csv",
        checkpoint_dir="output/checkpoints",
    )

    # Evaluate
    metrics = evaluate(
        predictions_bm_path="output/predictions_bm.csv",
        predictions_web_path="output/predictions_web.csv",
    )
    metrics.print_summary()
"""

from .training import (
    TrainingPipeline,
    TrainingResult,
    ChannelTrainingResult,
    train,
)

from .inference import (
    InferencePipeline,
    InferenceResult,
    infer,
)

from .evaluation import (
    EvaluationPipeline,
    EvaluationResult,
    MetricResult,
    evaluate,
    evaluate_bm,
    evaluate_web,
    compute_metrics,
)


__all__ = [
    # Training
    "TrainingPipeline",
    "TrainingResult",
    "ChannelTrainingResult",
    "train",
    # Inference
    "InferencePipeline",
    "InferenceResult",
    "infer",
    # Evaluation
    "EvaluationPipeline",
    "EvaluationResult",
    "MetricResult",
    "evaluate",
    "evaluate_bm",
    "evaluate_web",
    "compute_metrics",
]
