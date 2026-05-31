"""
ML service for the API.

This service provides the business logic for:
- Making predictions
- Generating recommendations
- Managing models
"""

import sys
import os
import logging
from typing import Dict, List, Any, Optional
import warnings

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from core.predictor import DrugResponsePredictor
from core.recommender import DrugRecommender
from core.model_manager import ModelManager

from api.schemas.request import (
    PredictionRequest, 
    BatchPredictionRequest,
    RecommendationRequest,
    ModelComparisonRequest
)
from api.schemas.response import (
    PredictionResponse,
    BatchPredictionResponse,
    RecommendationResponse,
    ModelComparisonResponse,
    ErrorResponse
)

logger = logging.getLogger(__name__)


class MLService:
    """
    Machine Learning service for the API.
    
    This class orchestrates the ML operations and handles
    conversion between API schemas and core module interfaces.
    """
    
    def __init__(self):
        """Initialize the ML service."""
        logger.info("Initializing ML service...")
        
        self.predictor = DrugResponsePredictor()
        self.recommender = DrugRecommender(predictor=self.predictor)
        self.model_manager = self.predictor.model_manager
        
        # Track service start time for uptime calculation
        import time
        self.start_time = time.time()
        
        logger.info(
            f"ML service initialized: "
            f"preprocessor_fitted={self.predictor.preprocessor.is_fitted}, "
            f"models={self.model_manager.get_available_models()}, "
            f"drug_count={self.recommender.get_drug_count()}"
        )
    
    def get_health_info(self) -> Dict[str, Any]:
        """
        Get health information about the service.
        
        Returns:
            Dictionary with health information
        """
        import time
        
        return {
            "status": "healthy",
            "version": "1.0.0",
            "model_count": len(self.model_manager.get_available_models()),
            "models_loaded": len(self.model_manager.models),
            "drug_count": self.recommender.get_drug_count(),
            "preprocessor_fitted": self.predictor.preprocessor.is_fitted,
            "uptime": time.time() - self.start_time
        }
    
    def make_prediction(self, request: PredictionRequest) -> Dict[str, Any]:
        """
        Make a single prediction.
        
        Args:
            request: Prediction request
            
        Returns:
            Dictionary with prediction results
        """
        try:
            # Convert request to dictionary for predictor
            input_data = self._request_to_dict(request)
            logger.info(
                f"[API] Prediction request: cell_line={input_data.get('CELL_LINE_NAME')}, "
                f"drug={input_data.get('DRUG_NAME')}, target={input_data.get('TARGET')}, "
                f"tcga={input_data.get('TCGA_DESC')}, "
                f"auc={input_data.get('AUC')}, z_score={input_data.get('Z_SCORE')}, "
                f"model={request.model_name}"
            )
            
            # Make prediction
            result = self.predictor.predict(
                input_data,
                model_name=request.model_name
            )
            
            # Prepare response
            response = {
                "success": True,
                "prediction": {
                    "predicted_ic50": result["predictions"][0],
                    "model_used": result["model_used"]
                }
            }
            
            logger.info(
                f"[API] Prediction result: ic50={result['predictions'][0]:.4f}, "
                f"model={result['model_used']}"
            )
            return response
            
        except ValueError as e:
            logger.error(f"Prediction validation error: {e}")
            return {
                "success": False,
                "error": str(e),
                "detail": "Input validation failed. Please check that all required fields are provided with valid values."
            }
        except RuntimeError as e:
            logger.error(f"Prediction runtime error: {e}")
            return {
                "success": False,
                "error": str(e),
                "detail": "Model prediction failed. The model may not be loaded."
            }
        except Exception as e:
            logger.error(f"Prediction unexpected error: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "detail": "Prediction failed due to an unexpected error."
            }
    
    def make_batch_prediction(self, request: BatchPredictionRequest) -> Dict[str, Any]:
        """
        Make batch predictions.
        
        Args:
            request: Batch prediction request
            
        Returns:
            Dictionary with batch prediction results
        """
        try:
            # Convert samples to list of dictionaries
            samples = [self._request_to_dict(sample) for sample in request.samples]
            logger.info(f"Batch prediction request: {len(samples)} samples, model={request.model_name}")
            
            # Make predictions
            result = self.predictor.predict(
                samples,
                model_name=request.model_name
            )
            
            # Prepare response with individual results
            predictions = []
            for i, pred_value in enumerate(result["predictions"]):
                predictions.append({
                    "predicted_ic50": pred_value,
                    "model_used": result["model_used"],
                    "sample_index": i
                })
            
            response = {
                "success": True,
                "predictions": predictions,
                "sample_count": result["sample_count"],
                "model_used": result["model_used"]
            }
            
            return response
            
        except ValueError as e:
            logger.error(f"Batch prediction validation error: {e}")
            return {
                "success": False,
                "error": str(e),
                "detail": "Input validation failed for one or more samples."
            }
        except Exception as e:
            logger.error(f"Batch prediction error: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "detail": "Batch prediction failed."
            }
    
    def generate_recommendations(self, request: RecommendationRequest) -> Dict[str, Any]:
        """
        Generate drug recommendations.
        
        Enhanced workflow uses AUC reference data filtering, Used_AUC calculation,
        response category classification, IC50 scoring, deduplication, and ranking.
        
        Args:
            request: Recommendation request
            
        Returns:
            Dictionary with recommendation results including new fields:
            predicted_ln_ic50, predicted_ic50, response_category,
            ic50_score, ic50_score_percentage, used_auc
        """
        try:
            # Convert request to dictionary for recommender
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
            
            logger.info(
                f"Recommendation request: cell_line={request.cell_line_name}, "
                f"max_drugs={request.max_drugs}, model={request.model_name}"
            )
            
            # Generate recommendations (enhanced with AUC reference, scoring, ranking)
            recommendations = self.recommender.recommend(
                cell_line_data,
                max_drugs=request.max_drugs,
                model_name=request.model_name,
                include_score=request.include_score
            )
            
            total_matches = len(recommendations)
            
            # Prepare response
            response = {
                "success": True,
                "recommendations": recommendations,
                "cell_line_name": request.cell_line_name,
                "total_drugs_evaluated": total_matches,
                "total_matches": total_matches
            }
            
            logger.info(
                f"Generated {total_matches} recommendations "
                f"(drugs with AUC reference matches)"
            )
            return response
            
        except ValueError as e:
            logger.error(f"Recommendation validation error: {e}")
            return {
                "success": False,
                "error": str(e),
                "detail": "Input validation failed. Please check that all required fields are provided."
            }
        except Exception as e:
            logger.error(f"Recommendation error: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "detail": "Recommendation generation failed."
            }
    
    def compare_models(self, request: ModelComparisonRequest) -> Dict[str, Any]:
        """
        Compare predictions from multiple models.
        
        Args:
            request: Model comparison request
            
        Returns:
            Dictionary with comparison results
        """
        try:
            # Convert sample to dictionary
            sample_data = self._request_to_dict(request.sample)
            
            # Get model names to compare
            model_names = request.model_names
            if not model_names:
                model_names = self.model_manager.get_available_models()
            
            logger.info(f"Model comparison request: models={model_names}")
            
            # Make comparisons
            comparisons = []
            for model_name in model_names:
                try:
                    result = self.predictor.predict(
                        sample_data,
                        model_name=model_name
                    )
                    comparisons.append({
                        "model_name": model_name,
                        "predicted_ic50": result["predictions"][0]
                    })
                except Exception as e:
                    logger.warning(f"Model {model_name} prediction failed: {e}")
                    comparisons.append({
                        "model_name": model_name,
                        "error": str(e)
                    })
            
            # Prepare response
            response = {
                "success": True,
                "comparisons": comparisons,
                "sample_info": {
                    "cell_line_name": request.sample.cell_line_name,
                    "drug_name": request.sample.drug_name
                }
            }
            
            return response
            
        except ValueError as e:
            logger.error(f"Model comparison validation error: {e}")
            return {
                "success": False,
                "error": str(e),
                "detail": "Input validation failed."
            }
        except Exception as e:
            logger.error(f"Model comparison error: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "detail": "Model comparison failed."
            }
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about all available models.
        
        Returns:
            Dictionary with model information
        """
        try:
            models_info = self.model_manager.get_all_model_info()
            
            # Convert to list for easier consumption
            models_list = []
            for model_name, info in models_info.items():
                models_list.append({
                    "model_name": model_name,
                    "filename": info.get("filename", ""),
                    "loaded": info.get("loaded", False),
                    "model_type": info.get("model_type", "Unknown")
                })
            
            return {
                "success": True,
                "models": models_list,
                "total_models": len(models_list)
            }
            
        except Exception as e:
            logger.error(f"Failed to get model info: {e}")
            return {
                "success": False,
                "error": str(e),
                "detail": "Failed to get model information"
            }
    
    def _request_to_dict(self, request: PredictionRequest) -> Dict[str, Any]:
        """
        Convert PredictionRequest to dictionary for core modules.
        
        Maps API snake_case field names to UPPER_CASE core module field names.
        This mapping is critical - the core preprocessing module expects exact
        column names matching the training data.
        
        Args:
            request: Prediction request
            
        Returns:
            Dictionary with field names matching core module expectations
        """
        # Map from API field names to core module field names
        field_mapping = {
            "cell_line_name": "CELL_LINE_NAME",
            "tcga_desc": "TCGA_DESC",
            "site": "SITE",
            "histology": "HISTOLOGY",
            "gdsc_tissue_descriptor_1": "GDSC_TISSUE_DESCRIPTOR_1",
            "gdsc_tissue_descriptor_2": "GDSC_TISSUE_DESCRIPTOR_2",
            "cancer_type_matching_tcga_label": "CANCER_TYPE_MATCHING_TCGA_LABEL",
            "drug_name": "DRUG_NAME",
            "target": "TARGET",
            "target_pathway": "TARGET_PATHWAY",
            "auc": "AUC",
            "z_score": "Z_SCORE"
        }
        
        result = {}
        for api_field, core_field in field_mapping.items():
            value = getattr(request, api_field)
            result[core_field] = value
        
        logger.debug(
            f"[API] Field mapping: "
            f"CELL_LINE={result.get('CELL_LINE_NAME')}, "
            f"DRUG={result.get('DRUG_NAME')}, "
            f"TARGET={result.get('TARGET')}, "
            f"TARGET_PATHWAY={result.get('TARGET_PATHWAY')}, "
            f"TCGA={result.get('TCGA_DESC')}, "
            f"SITE={result.get('SITE')}, "
            f"HISTOLOGY={result.get('HISTOLOGY')}, "
            f"AUC={result.get('AUC')}, "
            f"Z_SCORE={result.get('Z_SCORE')}"
        )
        return result


# Singleton instance
_ml_service_instance = None

def get_ml_service() -> MLService:
    """
    Get or create the ML service singleton.
    
    Returns:
        MLService instance
    """
    global _ml_service_instance
    if _ml_service_instance is None:
        _ml_service_instance = MLService()
    return _ml_service_instance
