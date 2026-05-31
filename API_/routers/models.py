"""
Models router for FastAPI.

This router handles:
- Model information
- Model management
"""

from fastapi import APIRouter, HTTPException, status
from typing import List

from api.schemas.response import ErrorResponse
from api.services.ml_service import get_ml_service

router = APIRouter(
    prefix="/models",
    tags=["models"],
    responses={
        500: {"model": ErrorResponse}
    }
)

ml_service = get_ml_service()


@router.get(
    "",
    summary="Get available models",
    description="Get list of all available ML models."
)
async def get_models():
    """
    Get list of all available models.
    
    Returns information about all trained models.
    """
    try:
        result = ml_service.get_model_info()
        
        if not result["success"]:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=result.get("detail", "Failed to get model information")
            )
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get model information: {str(e)}"
        )


@router.get(
    "/{model_name}",
    summary="Get model details",
    description="Get detailed information about a specific model."
)
async def get_model_details(model_name: str):
    """
    Get detailed information about a specific model.
    
    - **model_name**: Name of the model
    
    Returns detailed model information.
    """
    try:
        model_manager = ml_service.model_manager
        
        # Check if model exists
        available_models = model_manager.get_available_models()
        if model_name not in available_models:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Model '{model_name}' not found. Available models: {available_models}"
            )
        
        # Get model info
        model_info = model_manager.get_model_info(model_name)
        
        # Try to load model to get more details
        try:
            model = model_manager.load_model(model_name)
            model_info["model_type"] = type(model).__name__
            
            # Add model-specific information if available
            if hasattr(model, "n_features_in_"):
                model_info["n_features"] = model.n_features_in_
            if hasattr(model, "get_params"):
                model_info["parameters"] = model.get_params()
                
        except Exception as e:
            model_info["load_error"] = str(e)
        
        return {
            "success": True,
            "model": model_info
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get model details: {str(e)}"
        )


@router.post(
    "/{model_name}/load",
    summary="Load a model",
    description="Explicitly load a model into memory."
)
async def load_model(model_name: str):
    """
    Load a model into memory.
    
    - **model_name**: Name of the model to load
    
    Returns confirmation of model loading.
    """
    try:
        model_manager = ml_service.model_manager
        
        # Check if model exists
        available_models = model_manager.get_available_models()
        if model_name not in available_models:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Model '{model_name}' not found. Available models: {available_models}"
            )
        
        # Load model
        model = model_manager.load_model(model_name)
        
        return {
            "success": True,
            "message": f"Model '{model_name}' loaded successfully",
            "model_type": type(model).__name__,
            "loaded": True
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to load model: {str(e)}"
        )


@router.post(
    "/load-all",
    summary="Load all models",
    description="Load all available models into memory."
)
async def load_all_models():
    """
    Load all available models into memory.
    
    Returns confirmation of model loading.
    """
    try:
        model_manager = ml_service.model_manager
        
        # Load all models
        model_manager.load_all_models()
        
        loaded_count = len(model_manager.models)
        total_count = len(model_manager.get_available_models())
        
        return {
            "success": True,
            "message": f"Loaded {loaded_count} of {total_count} models",
            "loaded_count": loaded_count,
            "total_count": total_count
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to load models: {str(e)}"
        )


@router.get(
    "/default",
    summary="Get default model",
    description="Get information about the default model."
)
async def get_default_model():
    """
    Get information about the default model.
    
    Returns default model information.
    """
    try:
        predictor = ml_service.predictor
        default_model = predictor.default_model
        
        return {
            "success": True,
            "default_model": default_model,
            "available_models": predictor.get_available_models()
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get default model: {str(e)}"
        )