"""
Prediction pipeline for drug response.

This module provides the main prediction interface that:
1. Takes raw input data
2. Applies preprocessing (using saved preprocessor ONLY — never fits at runtime)
3. Makes predictions using selected model
4. Returns structured results

CRITICAL: The preprocessor must be loaded from disk (core/saved_preprocessor/).
It must NEVER be fitted during API runtime. If no saved preprocessor is found,
the predictor will raise an error rather than silently creating an unfitted one.
"""

import pandas as pd
import numpy as np
import os
import logging
from typing import Dict, List, Any, Optional, Union
import warnings

from .preprocessing import DrugResponsePreprocessor, validate_input_data
from .model_manager import ModelManager, get_default_model_manager

logger = logging.getLogger(__name__)


class DrugResponsePredictor:
    """
    Main prediction pipeline for drug response.
    
    This class orchestrates the complete prediction workflow:
    1. Input validation
    2. Preprocessing (using saved preprocessor from disk)
    3. Model prediction
    4. Result formatting
    
    Attributes:
        preprocessor (DrugResponsePreprocessor): Preprocessor instance (loaded from disk)
        model_manager (ModelManager): Model manager instance
        default_model (str): Default model name for predictions
    """
    
    def __init__(self,
                 preprocessor: Optional[DrugResponsePreprocessor] = None,
                 model_manager: Optional[ModelManager] = None,
                 default_model: str = "CatBoost"):
        """
        Initialize the predictor.
        
        Args:
            preprocessor: Preprocessor instance (loads saved one from disk if None)
            model_manager: Model manager instance (uses default if None)
            default_model: Default model name for predictions
            
        Raises:
            RuntimeError: If no saved preprocessor is found on disk and none is provided
        """
        self.model_manager = model_manager or get_default_model_manager()
        
        # Load preprocessor: either from argument or from disk
        if preprocessor is not None:
            self.preprocessor = preprocessor
            logger.info("Using provided preprocessor instance")
        else:
            self.preprocessor = self._load_saved_preprocessor()
        
        self.default_model = default_model
        
        # Check if default model is available
        available_models = self.model_manager.get_available_models()
        if default_model not in available_models and available_models:
            # Use first available model as default
            self.default_model = available_models[0]
            warnings.warn(f"Default model '{default_model}' not found. Using '{self.default_model}' instead.")
        
        logger.info(
            f"DrugResponsePredictor initialized: "
            f"preprocessor_fitted={self.preprocessor.is_fitted}, "
            f"default_model={self.default_model}, "
            f"available_models={available_models}"
        )
    
    def _load_saved_preprocessor(self) -> DrugResponsePreprocessor:
        """
        Load the saved preprocessor from disk.
        
        This is the ONLY way the preprocessor should be obtained for prediction.
        It must NEVER be fitted at API runtime — only loaded from the artifacts
        saved during the training/notebook phase.
        
        Returns:
            DrugResponsePreprocessor instance (fitted, loaded from disk)
            
        Raises:
            RuntimeError: If saved preprocessor cannot be found or loaded
        """
        preprocessor_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "saved_preprocessor")
        scaler_path = os.path.join(preprocessor_dir, "scaler.pkl")
        
        if not os.path.exists(scaler_path):
            logger.error(
                f"Saved preprocessor not found at {preprocessor_dir}. "
                f"The preprocessor must be fitted during training and saved to disk "
                f"before the API can make predictions. "
                f"Run the training notebook to generate the saved preprocessor artifacts."
            )
            raise RuntimeError(
                f"Saved preprocessor not found at {preprocessor_dir}. "
                f"The prediction pipeline requires a pre-fitted preprocessor saved during training. "
                f"Cannot proceed without it — DO NOT fit a new preprocessor at runtime."
            )
        
        try:
            preprocessor = DrugResponsePreprocessor.load(preprocessor_dir)
            if not preprocessor.is_fitted:
                logger.error(
                    f"Loaded preprocessor from {preprocessor_dir} but is_fitted=False. "
                    f"The saved artifacts may be corrupted."
                )
                raise RuntimeError(
                    f"Loaded preprocessor from {preprocessor_dir} but it reports is_fitted=False. "
                    f"The saved preprocessor artifacts may be corrupted. Re-run training to regenerate."
                )
            logger.info(
                f"Successfully loaded saved preprocessor from {preprocessor_dir} "
                f"(is_fitted={preprocessor.is_fitted})"
            )
            return preprocessor
        except RuntimeError:
            raise
        except Exception as e:
            logger.error(
                f"Failed to load saved preprocessor from {preprocessor_dir}: {e}. "
                f"Cannot create predictions without a fitted preprocessor."
            )
            raise RuntimeError(
                f"Failed to load saved preprocessor from {preprocessor_dir}: {e}. "
                f"The prediction pipeline requires a pre-fitted preprocessor. "
                f"Do NOT create a new unfitted preprocessor."
            )
    
    def predict(self,
                input_data: Union[pd.DataFrame, Dict[str, Any], List[Dict[str, Any]]],
                model_name: Optional[str] = None,
                return_features: bool = False) -> Dict[str, Any]:
        """
        Make predictions for input data.
        
        Args:
            input_data: Input data in various formats:
                - DataFrame with required columns
                - Dictionary with column-value pairs (single sample)
                - List of dictionaries (multiple samples)
            model_name: Name of model to use (uses default if None)
            return_features: Whether to return preprocessed features
            
        Returns:
            Dictionary with prediction results
        """
        # Convert input to DataFrame if needed
        df = self._prepare_input_data(input_data)
        
        # Log input features summary (INFO) and per-sample details (DEBUG)
        logger.info(
            f"[PREDICT] Input features: shape={df.shape}, columns={list(df.columns)}, "
            f"sample_count={len(df)}"
        )
        for idx, row in df.iterrows():
            logger.debug(
                f"[PREDICT] Sample {idx}: CELL_LINE={row.get('CELL_LINE_NAME', 'N/A')}, "
                f"DRUG={row.get('DRUG_NAME', 'N/A')}, TCGA={row.get('TCGA_DESC', 'N/A')}, "
                f"TARGET={row.get('TARGET', 'N/A')}, AUC={row.get('AUC', 'N/A')}, "
                f"Z_SCORE={row.get('Z_SCORE', 'N/A')}"
            )
        
        # Validate input data
        try:
            validate_input_data(df)
        except ValueError as e:
            logger.error(f"[PREDICT] Input validation failed: {e}")
            required_fields = [
                "CELL_LINE_NAME", "TCGA_DESC", "DRUG_NAME", "TARGET", "TARGET_PATHWAY",
                "SITE", "HISTOLOGY", "GDSC_TISSUE_DESCRIPTOR_1", "GDSC_TISSUE_DESCRIPTOR_2",
                "CANCER_TYPE_MATCHING_TCGA_LABEL", "AUC", "Z_SCORE"
            ]
            raise ValueError(
                f"Input validation failed: {e}. "
                f"Ensure all required fields are provided: {required_fields}"
            )
        
        # Check preprocessor is fitted
        if not self.preprocessor.is_fitted:
            logger.error(
                "[PREDICT] Preprocessor is not fitted. "
                "This should never happen if the preprocessor was loaded from disk correctly."
            )
            raise ValueError(
                "Preprocessor is not fitted. The prediction pipeline requires a fitted preprocessor "
                "loaded from disk (core/saved_preprocessor/). "
                "Do NOT fit a new preprocessor at runtime."
            )
        
        # Preprocess data
        try:
            features = self.preprocessor.transform(df)
            logger.debug(
                f"[PREDICT] Transformed features: shape={features.shape}, "
                f"dtype={features.dtype}, "
                f"min={features.min():.4f}, max={features.max():.4f}, mean={features.mean():.4f}"
            )
        except ValueError as e:
            logger.error(f"[PREDICT] Preprocessing transform failed: {e}")
            raise ValueError(f"Preprocessing failed: {e}")
        
        # Get model name
        model_to_use = model_name or self.default_model
        
        # Make predictions
        try:
            predictions = self.model_manager.predict(model_to_use, features)
            logger.debug(
                f"[PREDICT] Model prediction: model={model_to_use}, "
                f"input_shape={features.shape}, "
                f"predictions={predictions.tolist()}, "
                f"pred_count={len(predictions)}"
            )
        except Exception as e:
            logger.error(f"[PREDICT] Model prediction failed for '{model_to_use}': {e}")
            raise RuntimeError(f"Model prediction failed for '{model_to_use}': {e}")
        
        # Prepare results
        results = {
            'model_used': model_to_use,
            'predictions': predictions.tolist(),
            'sample_count': len(predictions)
        }
        
        # Add preprocessed features if requested
        if return_features:
            results['preprocessed_features'] = features.tolist()
            results['feature_names'] = self.preprocessor.get_feature_names()
        
        # Add input data reference if single sample
        if isinstance(input_data, dict):
            results['input_data'] = input_data
        elif isinstance(input_data, list) and len(input_data) == 1:
            results['input_data'] = input_data[0]
        
        return results
    
    def predict_batch(self, 
                      df: pd.DataFrame,
                      model_name: Optional[str] = None) -> pd.DataFrame:
        """
        Make batch predictions and return DataFrame with results.
        
        Args:
            df: DataFrame with input data
            model_name: Name of model to use (uses default if None)
            
        Returns:
            DataFrame with original data plus prediction column
        """
        # Make predictions
        results = self.predict(df, model_name=model_name, return_features=False)
        
        # Add predictions to DataFrame
        df_result = df.copy()
        df_result['predicted_LN_IC50'] = results['predictions']
        df_result['prediction_model'] = results['model_used']
        
        return df_result
    
    def compare_models(self, 
                       input_data: Union[pd.DataFrame, Dict[str, Any], List[Dict[str, Any]]],
                       model_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Compare predictions from multiple models.
        
        Args:
            input_data: Input data
            model_names: List of model names to compare (uses all available if None)
            
        Returns:
            Dictionary with predictions from all models
        """
        # Convert input to DataFrame if needed
        df = self._prepare_input_data(input_data)
        
        # Validate input data
        validate_input_data(df)
        
        # Check preprocessor is fitted
        if not self.preprocessor.is_fitted:
            raise ValueError("Preprocessor is not fitted. Cannot make predictions.")
        
        # Preprocess data
        features = self.preprocessor.transform(df)
        
        # Get model names
        if model_names is None:
            model_names = self.model_manager.get_available_models()
        
        # Get predictions from all models
        all_predictions = self.model_manager.predict_all(features)
        
        # Filter to requested models
        results = {}
        for model_name in model_names:
            if model_name in all_predictions and all_predictions[model_name] is not None:
                results[model_name] = all_predictions[model_name].tolist()
            else:
                results[model_name] = None
        
        # Prepare comparison results
        comparison = {
            'input_sample_count': len(features),
            'models_compared': model_names,
            'predictions': results,
            'feature_names': self.preprocessor.get_feature_names()
        }
        
        # Calculate statistics if multiple samples
        if len(features) > 1:
            stats = {}
            for model_name, preds in results.items():
                if preds is not None:
                    stats[model_name] = {
                        'mean': float(np.mean(preds)),
                        'std': float(np.std(preds)),
                        'min': float(np.min(preds)),
                        'max': float(np.max(preds))
                    }
            comparison['statistics'] = stats
        
        return comparison
    
    def get_model_info(self, model_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get information about a model.
        
        Args:
            model_name: Name of model (uses default if None)
            
        Returns:
            Dictionary with model information
        """
        model_to_use = model_name or self.default_model
        return self.model_manager.get_model_info(model_to_use)
    
    def get_available_models(self) -> List[str]:
        """
        Get list of available model names.
        
        Returns:
            List of model names
        """
        return self.model_manager.get_available_models()
    
    def _prepare_input_data(self, input_data: Any) -> pd.DataFrame:
        """
        Convert various input formats to DataFrame.
        
        Args:
            input_data: Input data in various formats
            
        Returns:
            DataFrame with input data
        """
        if isinstance(input_data, pd.DataFrame):
            return input_data.copy()
        
        elif isinstance(input_data, dict):
            # Single sample as dictionary
            return pd.DataFrame([input_data])
        
        elif isinstance(input_data, list) and all(isinstance(item, dict) for item in input_data):
            # Multiple samples as list of dictionaries
            return pd.DataFrame(input_data)
        
        else:
            raise ValueError(
                "Input data must be a DataFrame, dictionary, or list of dictionaries. "
                f"Got {type(input_data)}"
            )
    
    def fit_preprocessor(self, training_data: pd.DataFrame) -> None:
        """
        Fit the preprocessor on training data.
        
        Args:
            training_data: DataFrame with training data
        """
        validate_input_data(training_data)
        self.preprocessor.fit(training_data)
        logger.info("Preprocessor fitted on training data")


def create_sample_prediction() -> None:
    """Create and test a sample prediction."""
    print("Testing DrugResponsePredictor...")
    
    # Create sample data
    sample_data = {
        "CELL_LINE_NAME": "A172",
        "TCGA_DESC": "GBM",
        "DRUG_NAME": "Camptothecin",
        "TARGET": "TOP1",
        "TARGET_PATHWAY": "DNA replication",
        "SITE": "central_nervous_system",
        "HISTOLOGY": "glioma",
        "GDSC_TISSUE_DESCRIPTOR_1": "nervous_system",
        "GDSC_TISSUE_DESCRIPTOR_2": "glioma",
        "CANCER_TYPE_MATCHING_TCGA_LABEL": "GBM",
        "AUC": 0.93,
        "Z_SCORE": 0.43
    }
    
    try:
        # Initialize predictor
        predictor = DrugResponsePredictor()
        
        # Get available models
        available_models = predictor.get_available_models()
        print(f"Available models: {available_models}")
        
        if not available_models:
            print("No models available. Skipping prediction test.")
            return
        
        # Test single prediction
        print(f"\nTesting prediction with sample data...")
        result = predictor.predict(sample_data)
        
        print(f"Model used: {result['model_used']}")
        print(f"Prediction: {result['predictions'][0]}")
        print(f"Sample count: {result['sample_count']}")
        
        # Test batch prediction with multiple samples
        print(f"\nTesting batch prediction...")
        batch_data = [
            sample_data,
            {
                "CELL_LINE_NAME": "A172",
                "TCGA_DESC": "GBM",
                "DRUG_NAME": "Erlotinib",
                "TARGET": "EGFR",
                "TARGET_PATHWAY": "EGFR signaling",
                "SITE": "central_nervous_system",
                "HISTOLOGY": "glioma",
                "GDSC_TISSUE_DESCRIPTOR_1": "nervous_system",
                "GDSC_TISSUE_DESCRIPTOR_2": "glioma",
                "CANCER_TYPE_MATCHING_TCGA_LABEL": "GBM",
                "AUC": 0.87,
                "Z_SCORE": -0.12
            }
        ]
        
        batch_result = predictor.predict(batch_data)
        print(f"Batch predictions: {batch_result['predictions']}")
        
        # Test model comparison
        print(f"\nTesting model comparison...")
        if len(available_models) > 1:
            comparison = predictor.compare_models(sample_data, available_models[:2])
            print(f"Models compared: {comparison['models_compared']}")
            for model_name, preds in comparison['predictions'].items():
                if preds is not None:
                    print(f"  {model_name}: {preds[0]}")
        
        print("\nPrediction test completed successfully!")
        
    except Exception as e:
        print(f"Error during prediction test: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    create_sample_prediction()
