"""
Health router for FastAPI.

This router handles:
- Health checks
- Service status
- Version information
"""

from fastapi import APIRouter, HTTPException, status
from datetime import datetime

from api.schemas.response import HealthResponse, ErrorResponse
from api.services.ml_service import get_ml_service

router = APIRouter(
    prefix="/health",
    tags=["health"],
    responses={
        500: {"model": ErrorResponse}
    }
)

ml_service = get_ml_service()


@router.get(
    "",
    response_model=HealthResponse,
    summary="Health check",
    description="Check if the service is healthy and get version information."
)
async def health_check():
    """
    Health check endpoint.
    
    Returns service health status and version information.
    """
    try:
        health_info = ml_service.get_health_info()
        
        return HealthResponse(
            status=health_info["status"],
            version=health_info["version"],
            model_count=health_info["model_count"],
            uptime=health_info["uptime"]
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Health check failed: {str(e)}"
        )


@router.get(
    "/ready",
    summary="Readiness check",
    description="Check if the service is ready to handle requests."
)
async def readiness_check():
    """
    Readiness check endpoint.
    
    Returns whether the service is ready to handle requests.
    """
    try:
        health_info = ml_service.get_health_info()
        
        # Check critical components
        model_count = health_info["model_count"]
        drug_count = health_info["drug_count"]
        
        is_ready = model_count > 0  # At least one model should be available
        
        return {
            "ready": is_ready,
            "timestamp": datetime.now().isoformat(),
            "model_count": model_count,
            "drug_count": drug_count,
            "checks": {
                "models_available": model_count > 0,
                "service_initialized": True
            }
        }
        
    except Exception as e:
        return {
            "ready": False,
            "timestamp": datetime.now().isoformat(),
            "error": str(e),
            "checks": {
                "models_available": False,
                "service_initialized": False
            }
        }


@router.get(
    "/version",
    summary="Version information",
    description="Get service version information."
)
async def version_info():
    """
    Version information endpoint.
    
    Returns service version and build information.
    """
    try:
        import sys
        import platform
        
        # Get Python version
        python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        
        # Get package versions (if available)
        packages = {}
        try:
            import pandas as pd
            packages["pandas"] = pd.__version__
        except:
            pass
            
        try:
            import sklearn
            packages["scikit-learn"] = sklearn.__version__
        except:
            pass
            
        try:
            import numpy as np
            packages["numpy"] = np.__version__
        except:
            pass
        
        return {
            "service": "Drug Response Prediction API",
            "version": "1.0.0",
            "build_date": "2026-04-29",
            "python_version": python_version,
            "platform": platform.platform(),
            "packages": packages
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get version information: {str(e)}"
        )


@router.get(
    "/stats",
    summary="Service statistics",
    description="Get service usage statistics."
)
async def service_stats():
    """
    Service statistics endpoint.
    
    Returns service usage statistics.
    """
    try:
        health_info = ml_service.get_health_info()
        
        # In a production system, you would track actual usage statistics
        # For now, return basic information
        return {
            "uptime_seconds": health_info["uptime"],
            "models_available": health_info["model_count"],
            "models_loaded": health_info["models_loaded"],
            "drugs_in_database": health_info["drug_count"],
            "service_status": health_info["status"]
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get service statistics: {str(e)}"
        )