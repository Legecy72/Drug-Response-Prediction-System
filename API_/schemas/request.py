"""
Request schemas for API validation.
"""

from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any


class PredictionRequest(BaseModel):
    """Request schema for single prediction."""
    
    # Cell line features
    cell_line_name: str = Field(..., description="Cell line name")
    tcga_desc: str = Field(..., description="TCGA description")
    site: str = Field(..., description="Site")
    histology: str = Field(..., description="Histology")
    gdsc_tissue_descriptor_1: str = Field(..., description="GDSC tissue descriptor 1")
    gdsc_tissue_descriptor_2: str = Field(..., description="GDSC tissue descriptor 2")
    cancer_type_matching_tcga_label: str = Field(..., description="Cancer type matching TCGA label")
    
    # Drug features
    drug_name: str = Field(..., description="Drug name")
    target: str = Field(..., description="Drug target")
    target_pathway: str = Field(..., description="Target pathway")
    
    # Numeric features
    auc: float = Field(0.5, description="AUC value", ge=0.0, le=1.0)
    z_score: float = Field(0.0, description="Z-score value")
    
    # Optional model selection
    model_name: Optional[str] = Field(None, description="Model name to use for prediction")
    
    class Config:
        schema_extra = {
            "example": {
                "cell_line_name": "A172",
                "tcga_desc": "GBM",
                "site": "nervous_system",
                "histology": "glioma",
                "gdsc_tissue_descriptor_1": "nervous_system",
                "gdsc_tissue_descriptor_2": "glioma",
                "cancer_type_matching_tcga_label": "GBM",
                "drug_name": "Camptothecin",
                "target": "TOP1",
                "target_pathway": "DNA replication",
                "auc": 0.93,
                "z_score": 0.43,
                "model_name": "CatBoost"
            }
        }


class BatchPredictionRequest(BaseModel):
    """Request schema for batch predictions."""
    
    samples: List[PredictionRequest] = Field(..., description="List of prediction samples")
    model_name: Optional[str] = Field(None, description="Model name to use for prediction")
    
    class Config:
        schema_extra = {
            "example": {
                "samples": [
                    {
                        "cell_line_name": "A172",
                        "tcga_desc": "GBM",
                        "site": "nervous_system",
                        "histology": "glioma",
                        "gdsc_tissue_descriptor_1": "nervous_system",
                        "gdsc_tissue_descriptor_2": "glioma",
                        "cancer_type_matching_tcga_label": "GBM",
                        "drug_name": "Camptothecin",
                        "target": "TOP1",
                        "target_pathway": "DNA replication",
                        "auc": 0.93,
                        "z_score": 0.43
                    },
                    {
                        "cell_line_name": "A172",
                        "tcga_desc": "GBM",
                        "site": "nervous_system",
                        "histology": "glioma",
                        "gdsc_tissue_descriptor_1": "nervous_system",
                        "gdsc_tissue_descriptor_2": "glioma",
                        "cancer_type_matching_tcga_label": "GBM",
                        "drug_name": "Erlotinib",
                        "target": "EGFR",
                        "target_pathway": "EGFR signaling",
                        "auc": 0.87,
                        "z_score": -0.12
                    }
                ],
                "model_name": "CatBoost"
            }
        }


class RecommendationRequest(BaseModel):
    """Request schema for drug recommendations."""
    
    # Cell line features (drug features will be filled from drug database)
    cell_line_name: str = Field(..., description="Cell line name")
    tcga_desc: str = Field(..., description="TCGA description")
    site: str = Field(..., description="Site")
    histology: str = Field(..., description="Histology")
    gdsc_tissue_descriptor_1: str = Field(..., description="GDSC tissue descriptor 1")
    gdsc_tissue_descriptor_2: str = Field(..., description="GDSC tissue descriptor 2")
    cancer_type_matching_tcga_label: str = Field(..., description="Cancer type matching TCGA label")
    
    # Numeric features (default values)
    auc: float = Field(0.5, description="AUC value", ge=0.0, le=1.0)
    z_score: float = Field(0.0, description="Z-score value")
    
    # Recommendation parameters
    max_drugs: Optional[int] = Field(10, description="Maximum number of drugs to return", ge=1, le=100)
    model_name: Optional[str] = Field(None, description="Model name to use for prediction")
    include_score: bool = Field(True, description="Whether to include recommendation score")
    
    class Config:
        schema_extra = {
            "example": {
                "cell_line_name": "A172",
                "tcga_desc": "GBM",
                "site": "nervous_system",
                "histology": "glioma",
                "gdsc_tissue_descriptor_1": "nervous_system",
                "gdsc_tissue_descriptor_2": "glioma",
                "cancer_type_matching_tcga_label": "GBM",
                "auc": 0.5,
                "z_score": 0.0,
                "max_drugs": 10,
                "model_name": "CatBoost",
                "include_score": True
            }
        }


class ModelComparisonRequest(BaseModel):
    """Request schema for model comparison."""
    
    sample: PredictionRequest = Field(..., description="Sample for prediction")
    model_names: Optional[List[str]] = Field(None, description="List of model names to compare")
    
    class Config:
        schema_extra = {
            "example": {
                "sample": {
                    "cell_line_name": "A172",
                    "tcga_desc": "GBM",
                    "site": "nervous_system",
                    "histology": "glioma",
                    "gdsc_tissue_descriptor_1": "nervous_system",
                    "gdsc_tissue_descriptor_2": "glioma",
                    "cancer_type_matching_tcga_label": "GBM",
                    "drug_name": "Camptothecin",
                    "target": "TOP1",
                    "target_pathway": "DNA replication",
                    "auc": 0.93,
                    "z_score": 0.43
                },
                "model_names": ["CatBoost", "XGBoost", "LightGBM"]
            }
        }