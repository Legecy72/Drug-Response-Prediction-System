"""
Recommendation router for FastAPI.

This router handles:
- Drug recommendations for cell lines
- Drug information retrieval
"""

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field
from typing import List, Optional

from api.schemas.request import RecommendationRequest
from api.schemas.response import (
    RecommendationResponse,
    ErrorResponse
)
from api.services.ml_service import get_ml_service


class BatchRecommendRequest(BaseModel):
    """Request schema for batch drug recommendations."""
    cell_line_name: str = Field(..., description="Cell line name")
    tcga_desc: str = Field(..., description="TCGA description")
    site: str = Field(..., description="Site")
    histology: str = Field(..., description="Histology")
    gdsc_tissue_descriptor_1: str = Field(..., description="GDSC tissue descriptor 1")
    gdsc_tissue_descriptor_2: str = Field(..., description="GDSC tissue descriptor 2")
    cancer_type_matching_tcga_label: str = Field(..., description="Cancer type matching TCGA label")
    auc: float = Field(0.5, description="AUC value", ge=0.0, le=1.0)
    z_score: float = Field(0.0, description="Z-score value")
    drug_names: List[str] = Field(..., description="List of drug names to evaluate")
    model_name: Optional[str] = Field(None, description="Model name to use")

router = APIRouter(
    prefix="/recommend",
    tags=["recommendation"],
    responses={
        400: {"model": ErrorResponse},
        500: {"model": ErrorResponse}
    }
)

ml_service = get_ml_service()


@router.post(
    "",
    response_model=RecommendationResponse,
    summary="Get drug recommendations",
    description="Get ranked drug recommendations for a cell line."
)
async def recommend_drugs(request: RecommendationRequest):
    """
    Get drug recommendations for a cell line.
    
    - **cell_line_name**: Cell line name
    - **tcga_desc**: TCGA description  
    - **site**: Site
    - **histology**: Histology
    - **gdsc_tissue_descriptor_1**: GDSC tissue descriptor 1
    - **gdsc_tissue_descriptor_2**: GDSC tissue descriptor 2
    - **cancer_type_matching_tcga_label**: Cancer type matching TCGA label
    - **auc**: AUC value (0-1, default: 0.5)
    - **z_score**: Z-score value (default: 0.0)
    - **max_drugs**: Maximum number of drugs to return (default: 10)
    - **model_name**: Optional model name (uses default if not specified)
    - **include_score**: Whether to include recommendation score (default: True)
    
    Returns ranked list of drug recommendations.
    """
    try:
        result = ml_service.generate_recommendations(request)
        
        if not result["success"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=result.get("detail", "Recommendation generation failed"),
                headers={"X-Error": result.get("error", "Unknown error")}
            )
        
        # Handle case where no drugs matched AUC reference data
        if not result.get("recommendations"):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No drugs matched the given cancer context in AUC reference data. "
                        "Try different cancer context fields (TCGA_DESC, SITE, etc.).",
                headers={"X-Warning": "No AUC reference matches"}
            )
        
        return RecommendationResponse(
            success=True,
            recommendations=result["recommendations"],
            cell_line_name=result["cell_line_name"],
            total_drugs_evaluated=result["total_drugs_evaluated"],
            total_matches=result.get("total_matches", len(result["recommendations"]))
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )


@router.get(
    "/drugs",
    summary="Get available drugs",
    description="Get list of all available drugs in the database."
)
async def get_available_drugs():
    """
    Get list of all available drugs.
    
    Returns list of drug names and counts.
    """
    try:
        recommender = ml_service.recommender
        drug_names = recommender.get_drug_names()
        drug_count = recommender.get_drug_count()
        
        return {
            "success": True,
            "drug_count": drug_count,
            "drugs": drug_names[:100],  # Limit to 100 for response size
            "total_available": drug_count
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get drug list: {str(e)}"
        )


@router.get(
    "/drugs/{drug_name}",
    summary="Get drug information",
    description="Get detailed information about a specific drug."
)
async def get_drug_info(drug_name: str):
    """
    Get information about a specific drug.
    
    - **drug_name**: Name of the drug
    
    Returns drug information including target and pathway.
    """
    try:
        recommender = ml_service.recommender
        drug_info = recommender.get_drug_info(drug_name)
        
        if drug_info is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Drug '{drug_name}' not found"
            )
        
        return {
            "success": True,
            "drug": drug_info
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get drug information: {str(e)}"
        )


@router.post(
    "/batch",
    summary="Get recommendations for specific drugs",
    description="Get recommendations for a cell line with specific drugs only."
)
async def recommend_specific_drugs(request: BatchRecommendRequest):
    """
    Get recommendations for specific drugs only.
    
    - **cell_line_name**: Cell line name
    - **drug_names**: List of drug names to evaluate
    
    Returns recommendations for the specified drugs only.
    """
    try:
        recommender = ml_service.recommender
        
        # Build cell line data dict
        cell_line_data = {
            "CELL_LINE_NAME": request.cell_line_name,
            "TCGA_DESC": request.tcga_desc,
            "SITE": request.site,
            "HISTOLOGY": request.histology,
            "GDSC_TISSUE_DESCRIPTOR_1": request.gdsc_tissue_descriptor_1,
            "GDSC_TISSUE_DESCRIPTOR_2": request.gdsc_tissue_descriptor_2,
            "CANCER_TYPE_MATCHING_TCGA_LABEL": request.cancer_type_matching_tcga_label,
            "AUC": request.auc,
            "Z_SCORE": request.z_score
        }
        
        # Generate recommendations for specific drugs
        recommendations = recommender.recommend_batch(
            cell_line_data,
            request.drug_names,
            model_name=request.model_name
        )
        
        return {
            "success": True,
            "recommendations": recommendations,
            "cell_line_name": request.cell_line_name,
            "drugs_evaluated": len(recommendations),
            "total_drugs_requested": len(request.drug_names)
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate recommendations: {str(e)}"
        )