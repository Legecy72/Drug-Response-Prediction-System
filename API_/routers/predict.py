"""
Prediction router for FastAPI.

This router handles:
- Single predictions
- Batch predictions
- Model comparisons
"""

from fastapi import APIRouter, HTTPException, status
from typing import List

from api.schemas.request import (
    PredictionRequest,
    BatchPredictionRequest,
    ModelComparisonRequest
)
from api.schemas.response import (
    PredictionResponse,
    BatchPredictionResponse,
    ModelComparisonResponse,
    ErrorResponse
)
from api.services.ml_service import get_ml_service

router = APIRouter(
    prefix="/predict",
    tags=["prediction"],
    responses={
        400: {"model": ErrorResponse},
        500: {"model": ErrorResponse}
    }
)

ml_service = get_ml_service()


@router.post(
    "/single",
    response_model=PredictionResponse,
    summary="Make a single prediction",
    description="Predict LN_IC50 for a single cell line-drug combination."
)
async def predict_single(request: PredictionRequest):
    """
    Make a single prediction.
    
    - **cell_line_name**: Cell line name
    - **tcga_desc**: TCGA description  
    - **site**: Site
    - **histology**: Histology
    - **gdsc_tissue_descriptor_1**: GDSC tissue descriptor 1
    - **gdsc_tissue_descriptor_2**: GDSC tissue descriptor 2
    - **cancer_type_matching_tcga_label**: Cancer type matching TCGA label
    - **drug_name**: Drug name
    - **target**: Drug target
    - **target_pathway**: Target pathway
    - **auc**: AUC value (0-1)
    - **z_score**: Z-score value
    - **model_name**: Optional model name (uses default if not specified)
    
    Returns predicted LN_IC50 value.
    """
    try:
        result = ml_service.make_prediction(request)
        
        if not result["success"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=result.get("detail", "Prediction failed"),
                headers={"X-Error": result.get("error", "Unknown error")}
            )
        
        return PredictionResponse(
            success=True,
            prediction=result["prediction"]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )


@router.post(
    "/batch",
    response_model=BatchPredictionResponse,
    summary="Make batch predictions",
    description="Predict LN_IC50 for multiple cell line-drug combinations."
)
async def predict_batch(request: BatchPredictionRequest):
    """
    Make batch predictions.
    
    - **samples**: List of prediction requests
    - **model_name**: Optional model name (uses default if not specified)
    
    Returns predictions for all samples.
    """
    try:
        result = ml_service.make_batch_prediction(request)
        
        if not result["success"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=result.get("detail", "Batch prediction failed"),
                headers={"X-Error": result.get("error", "Unknown error")}
            )
        
        return BatchPredictionResponse(
            success=True,
            predictions=result["predictions"],
            sample_count=result["sample_count"],
            model_used=result["model_used"]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )


@router.post(
    "/compare",
    response_model=ModelComparisonResponse,
    summary="Compare model predictions",
    description="Compare predictions from multiple models for the same input."
)
async def compare_models(request: ModelComparisonRequest):
    """
    Compare predictions from multiple models.
    
    - **sample**: Prediction request sample
    - **model_names**: Optional list of model names to compare (uses all available if not specified)
    
    Returns predictions from all specified models.
    """
    try:
        result = ml_service.compare_models(request)
        
        if not result["success"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=result.get("detail", "Model comparison failed"),
                headers={"X-Error": result.get("error", "Unknown error")}
            )
        
        return ModelComparisonResponse(
            success=True,
            comparisons=result["comparisons"],
            sample_info=result["sample_info"]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )