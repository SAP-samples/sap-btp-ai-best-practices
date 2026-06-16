"""
Regressor API routes.

Provides endpoints for forecasting model operations:
- Generate: Create canonical training data
- Train: Train forecasting models
- Infer: Run inference with trained models
- Evaluate: Evaluate model predictions
"""

import logging
import traceback

from fastapi import APIRouter, Depends, HTTPException

from ..security import get_api_key
from ..models.regressor import (
    GenerateRequest,
    GenerateResponse,
    TrainRequest,
    TrainResponse,
    InferRequest,
    InferResponse,
    EvaluateRequest,
    EvaluateResponse,
)
from ..services.regressor_service import (
    generate_training_data,
    train_models,
    run_inference,
    evaluate_predictions,
)

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/regressor",
    tags=["regressor"],
    dependencies=[Depends(get_api_key)],
)


@router.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest):
    """
    Generate canonical training data for forecasting models.

    Creates the training dataset with features for Model A (explainability)
    and/or Model B (full autoregressive features). The generated data is
    saved to the specified output path.

    Parameters in request body:
    - horizons_start: Start of forecast horizon range (1-52)
    - horizons_end: End of forecast horizon range (1-52)
    - model_variant: "A", "B", or "both"
    - include_crm: Include CRM demographic features
    - output_path: Path to save generated CSV file(s)
    """
    try:
        # Validate horizons
        if request.horizons_start > request.horizons_end:
            raise HTTPException(
                status_code=400,
                detail="horizons_start must be <= horizons_end",
            )

        result = generate_training_data(
            horizons_start=request.horizons_start,
            horizons_end=request.horizons_end,
            model_variant=request.model_variant,
            include_crm=request.include_crm,
            output_path=request.output_path,
        )

        return GenerateResponse(
            success=True,
            rows_generated=result["rows"],
            columns=result["columns"],
            output_files=result["output_files"],
            message=f"Successfully generated {result['rows']} rows with {result['columns']} columns",
        )

    except HTTPException:
        raise
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        raise HTTPException(
            status_code=404,
            detail=f"File not found: {str(e)}",
        )
    except Exception as e:
        logger.error(f"Generate failed: {e}\n{traceback.format_exc()}")
        raise HTTPException(
            status_code=500,
            detail=f"Generation failed: {str(e)}",
        )


@router.post("/train", response_model=TrainResponse)
async def train(request: TrainRequest):
    """
    Train B&M and/or WEB forecasting models.

    Trains CatBoost models for sales, AOV, and (for B&M) conversion
    prediction. Optionally trains surrogate models for SHAP-based
    explainability analysis.

    Parameters in request body:
    - model_b_path: Path to Model B CSV (required)
    - model_a_path: Path to Model A CSV (optional, for explainability)
    - output_dir: Output directory for checkpoints
    - channels: List of channels to train ["B&M", "WEB"]
    - bias_correction: Bias correction configuration
    - train_surrogate: Whether to train surrogate models
    """
    try:
        result = train_models(
            model_b_path=request.model_b_path,
            model_a_path=request.model_a_path,
            output_dir=request.output_dir,
            channels=request.channels,
            bias_correction=request.bias_correction,
            train_surrogate=request.train_surrogate,
        )

        channels_trained = []
        if result["bm_result"]:
            channels_trained.append("B&M")
        if result["web_result"]:
            channels_trained.append("WEB")

        return TrainResponse(
            success=True,
            bm_result=result["bm_result"],
            web_result=result["web_result"],
            checkpoint_dir=result["checkpoint_dir"],
            output_files=result["output_files"],
            message=f"Successfully trained models for channels: {', '.join(channels_trained)}",
        )

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        raise HTTPException(
            status_code=404,
            detail=f"Input file not found: {str(e)}",
        )
    except Exception as e:
        logger.error(f"Training failed: {e}\n{traceback.format_exc()}")
        raise HTTPException(
            status_code=500,
            detail=f"Training failed: {str(e)}",
        )


@router.post("/infer", response_model=InferResponse)
async def infer(request: InferRequest):
    """
    Run inference with trained models on new data.

    Loads trained model checkpoints and generates predictions for
    the provided feature data. Supports bias correction and
    optional SHAP-based explainability analysis.

    Parameters in request body:
    - model_b_path: Path to Model B CSV with features (required)
    - model_a_path: Path to Model A CSV (optional, for explainability)
    - checkpoint_dir: Path to model checkpoints
    - output_dir: Output directory for predictions
    - channels: List of channels to process ["B&M", "WEB"]
    - bias_correction: Bias correction configuration
    - run_explainability: Whether to run SHAP analysis
    """
    try:
        result = run_inference(
            model_b_path=request.model_b_path,
            model_a_path=request.model_a_path,
            checkpoint_dir=request.checkpoint_dir,
            output_dir=request.output_dir,
            channels=request.channels,
            bias_correction=request.bias_correction,
            run_explainability=request.run_explainability,
        )

        total_predictions = (result["bm_predictions_count"] or 0) + (
            result["web_predictions_count"] or 0
        )

        return InferResponse(
            success=True,
            bm_predictions_count=result["bm_predictions_count"],
            web_predictions_count=result["web_predictions_count"],
            output_files=result["output_files"],
            message=f"Successfully generated {total_predictions} predictions",
        )

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        raise HTTPException(
            status_code=404,
            detail=f"File not found: {str(e)}",
        )
    except Exception as e:
        logger.error(f"Inference failed: {e}\n{traceback.format_exc()}")
        raise HTTPException(
            status_code=500,
            detail=f"Inference failed: {str(e)}",
        )


@router.post("/evaluate", response_model=EvaluateResponse)
async def evaluate(request: EvaluateRequest):
    """
    Evaluate model predictions against ground truth.

    Computes evaluation metrics (MAE, WMAPE, Bias, R2) for
    predictions from both B&M and WEB channels.

    Parameters in request body:
    - predictions_bm_path: Path to B&M predictions CSV (optional)
    - predictions_web_path: Path to WEB predictions CSV (optional)
    - output_dir: Output directory for evaluation results

    At least one predictions file must be provided.
    """
    try:
        # Validate that at least one predictions file is provided
        if not request.predictions_bm_path and not request.predictions_web_path:
            raise HTTPException(
                status_code=400,
                detail="At least one predictions file (predictions_bm_path or predictions_web_path) is required",
            )

        result = evaluate_predictions(
            predictions_bm_path=request.predictions_bm_path,
            predictions_web_path=request.predictions_web_path,
            output_dir=request.output_dir,
        )

        return EvaluateResponse(
            success=True,
            metrics=result["metrics"],
            output_file=result["output_file"],
            message=f"Successfully evaluated {len(result['metrics'])} targets",
        )

    except HTTPException:
        raise
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        raise HTTPException(
            status_code=404,
            detail=f"Predictions file not found: {str(e)}",
        )
    except Exception as e:
        logger.error(f"Evaluation failed: {e}\n{traceback.format_exc()}")
        raise HTTPException(
            status_code=500,
            detail=f"Evaluation failed: {str(e)}",
        )
