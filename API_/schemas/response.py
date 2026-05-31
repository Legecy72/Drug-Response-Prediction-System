"""
Response schemas for API.
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime


class PredictionResult(BaseModel):
    """Single prediction result."""
    
    predicted_ic50: float = Field(..., description="Predicted LN_IC50 value")
    model_used: str = Field(..., description="Model used for prediction")
    sample_index: Optional[int] = Field(None, description="Index of sample in batch")
    
    class Config:
        schema_extra = {
            "example": {
                "predicted_ic50": -1.46,
                "model_used": "CatBoost",
                "sample_index": 0
            }
        }


class PredictionResponse(BaseModel):
    """Response schema for single prediction."""
    
    success: bool = Field(..., description="Whether prediction was successful")
    prediction: PredictionResult = Field(..., description="Prediction result")
    timestamp: datetime = Field(default_factory=datetime.now, description="Timestamp of prediction")
    
    class Config:
        schema_extra = {
            "example": {
                "success": True,
                "prediction": {
                    "predicted_ic50": -1.46,
                    "model_used": "CatBoost"
                },
                "timestamp": "2023-10-01T12:00:00Z"
            }
        }


class BatchPredictionResponse(BaseModel):
    """Response schema for batch predictions."""
    
    success: bool = Field(..., description="Whether predictions were successful")
    predictions: List[PredictionResult] = Field(..., description="List of prediction results")
    sample_count: int = Field(..., description="Number of samples processed")
    model_used: str = Field(..., description="Model used for predictions")
    timestamp: datetime = Field(default_factory=datetime.now, description="Timestamp of predictions")
    
    class Config:
        schema_extra = {
            "example": {
                "success": True,
                "predictions": [
                    {"predicted_ic50": -1.46, "model_used": "CatBoost", "sample_index": 0},
                    {"predicted_ic50": -0.89, "model_used": "CatBoost", "sample_index": 1}
                ],
                "sample_count": 2,
                "model_used": "CatBoost",
                "timestamp": "2023-10-01T12:00:00Z"
            }
        }


class DrugRecommendation(BaseModel):
    """Single drug recommendation."""
    
    drug_id: int = Field(..., description="Drug ID")
    drug_name: str = Field(..., description="Drug name")
    target: str = Field(..., description="Drug target")
    target_pathway: str = Field(..., description="Target pathway")
    predicted_ln_ic50: float = Field(..., description="Predicted LN_IC50 value (log-scale, model output)")
    predicted_ic50: float = Field(..., description="Predicted IC50 value (exp of LN_IC50, actual concentration)")
    response_category: str = Field(..., description="Response category: Very Sensitive, Sensitive, Moderate, or Resistant")
    ic50_score: float = Field(..., description="IC50 score = 1/(1+Predicted_IC50), higher is better")
    ic50_score_percentage: float = Field(..., description="IC50 score as percentage (IC50_Score * 100)")
    used_auc: float = Field(..., description="Mean AUC from matching reference rows, used as input for prediction")
    score: Optional[float] = Field(None, description="Recommendation score (legacy, equals ic50_score)")
    rank: int = Field(..., description="Rank (1 = best)")
    model_used: str = Field(..., description="Model used for prediction")
    
    class Config:
        schema_extra = {
            "example": {
                "drug_id": 1003,
                "drug_name": "Camptothecin",
                "target": "TOP1",
                "target_pathway": "DNA replication",
                "predicted_ln_ic50": -1.46,
                "predicted_ic50": 0.2317,
                "response_category": "Sensitive",
                "ic50_score": 0.8117,
                "ic50_score_percentage": 81.17,
                "used_auc": 0.85,
                "score": 0.8117,
                "rank": 1,
                "model_used": "CatBoost"
            }
        }


class RecommendationResponse(BaseModel):
    """Response schema for drug recommendations."""
    
    success: bool = Field(..., description="Whether recommendation was successful")
    recommendations: List[DrugRecommendation] = Field(..., description="List of drug recommendations")
    cell_line_name: str = Field(..., description="Cell line name")
    total_drugs_evaluated: int = Field(..., description="Total number of drugs evaluated (with AUC matches)")
    total_matches: Optional[int] = Field(None, description="Total number of drugs that matched AUC reference data")
    timestamp: datetime = Field(default_factory=datetime.now, description="Timestamp of recommendation")
    
    class Config:
        schema_extra = {
            "example": {
                "success": True,
                "recommendations": [
                    {
                        "drug_id": 1003,
                        "drug_name": "Camptothecin",
                        "target": "TOP1",
                        "target_pathway": "DNA replication",
                        "predicted_ln_ic50": -1.46,
                        "predicted_ic50": 0.2317,
                        "response_category": "Sensitive",
                        "ic50_score": 0.8117,
                        "ic50_score_percentage": 81.17,
                        "used_auc": 0.85,
                        "score": 0.8117,
                        "rank": 1,
                        "model_used": "CatBoost"
                    },
                    {
                        "drug_id": 1,
                        "drug_name": "Erlotinib",
                        "target": "EGFR",
                        "target_pathway": "EGFR signaling",
                        "predicted_ln_ic50": -0.89,
                        "predicted_ic50": 0.4107,
                        "response_category": "Sensitive",
                        "ic50_score": 0.7093,
                        "ic50_score_percentage": 70.93,
                        "used_auc": 0.72,
                        "score": 0.7093,
                        "rank": 2,
                        "model_used": "CatBoost"
                    }
                ],
                "cell_line_name": "A172",
                "total_drugs_evaluated": 5,
                "total_matches": 5,
                "timestamp": "2023-10-01T12:00:00Z"
            }
        }


class ModelInfo(BaseModel):
    """Information about a model."""
    
    model_name: str = Field(..., description="Model name")
    filename: str = Field(..., description="Model filename")
    loaded: bool = Field(..., description="Whether model is loaded in memory")
    model_type: Optional[str] = Field(None, description="Type of model")
    
    class Config:
        schema_extra = {
            "example": {
                "model_name": "CatBoost",
                "filename": "CatBoost_regression_model.pkl",
                "loaded": True,
                "model_type": "CatBoostRegressor"
            }
        }


class ModelComparisonResult(BaseModel):
    """Single model comparison result."""
    
    model_name: str = Field(..., description="Model name")
    predicted_ic50: Optional[float] = Field(None, description="Predicted LN_IC50 value")
    error: Optional[str] = Field(None, description="Error message if prediction failed")
    
    class Config:
        schema_extra = {
            "example": {
                "model_name": "CatBoost",
                "predicted_ic50": -1.46
            }
        }


class ModelComparisonResponse(BaseModel):
    """Response schema for model comparison."""
    
    success: bool = Field(..., description="Whether comparison was successful")
    comparisons: List[ModelComparisonResult] = Field(..., description="List of model comparisons")
    sample_info: Dict[str, Any] = Field(..., description="Sample information")
    timestamp: datetime = Field(default_factory=datetime.now, description="Timestamp of comparison")
    
    class Config:
        schema_extra = {
            "example": {
                "success": True,
                "comparisons": [
                    {"model_name": "CatBoost", "predicted_ic50": -1.46},
                    {"model_name": "XGBoost", "predicted_ic50": -1.38},
                    {"model_name": "LightGBM", "predicted_ic50": -1.42}
                ],
                "sample_info": {
                    "cell_line_name": "A172",
                    "drug_name": "Camptothecin"
                },
                "timestamp": "2023-10-01T12:00:00Z"
            }
        }


class ErrorResponse(BaseModel):
    """Error response schema."""
    
    success: bool = Field(False, description="Always false for error responses")
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Detailed error information")
    timestamp: datetime = Field(default_factory=datetime.now, description="Timestamp of error")
    
    class Config:
        schema_extra = {
            "example": {
                "success": False,
                "error": "Invalid input data",
                "detail": "Missing required field: cell_line_name",
                "timestamp": "2023-10-01T12:00:00Z"
            }
        }


class HealthResponse(BaseModel):
    """Health check response."""
    
    status: str = Field(..., description="Service status")
    version: str = Field(..., description="API version")
    model_count: int = Field(..., description="Number of available models")
    uptime: Optional[float] = Field(None, description="Service uptime in seconds")
    timestamp: datetime = Field(default_factory=datetime.now, description="Timestamp of health check")
    
    class Config:
        schema_extra = {
            "example": {
                "status": "healthy",
                "version": "1.0.0",
                "model_count": 7,
                "uptime": 3600.5,
                "timestamp": "2023-10-01T12:00:00Z"
            }
        }