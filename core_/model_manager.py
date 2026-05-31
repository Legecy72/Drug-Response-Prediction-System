"""
Model manager for drug response prediction.

This module provides functionality to:
1. Auto-detect trained models in the /model directory
2. Load models dynamically
3. Provide a unified prediction interface
4. Manage model caching and lifecycle
"""

import os
import joblib
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Union
import warnings
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class ModelManager:
    """
    Manages ML models for drug response prediction.
    
    This class handles:
    - Discovering available models in the model directory
    - Loading models into memory (lazy loading)
    - Providing prediction interface
    - Managing model metadata and versions
    
    Attributes:
        model_dir (str): Directory containing model files
        models (Dict[str, Any]): Loaded models keyed by model name
        model_metadata (Dict[str, Dict]): Metadata for each model
        available_models (List[str]): List of available model names
    """
    
    def __init__(self, model_dir: str = "model"):
        """
        Initialize the model manager.
        
        Args:
            model_dir: Directory containing model files (default: "model")
        """
        self.model_dir = model_dir
        self.models: Dict[str, Any] = {}
        self.model_metadata: Dict[str, Dict] = {}
        self.available_models: List[str] = []
        
        # Discover available models
        self._discover_models()
    
    def _discover_models(self) -> None:
        """Discover available model files in the model directory."""
        if not os.path.exists(self.model_dir):
            warnings.warn(f"Model directory '{self.model_dir}' does not exist")
            return
            
        # Look for .pkl files
        model_files = []
        for file in os.listdir(self.model_dir):
            if file.endswith('.pkl'):
                model_files.append(file)
                
        # Extract model names from filenames
        for model_file in model_files:
            # Remove extension and common suffixes
            model_name = model_file.replace('.pkl', '')
            model_name = model_name.replace('_regression_model', '')
            model_name = model_name.replace('_model', '')
            
            # Store metadata
            self.model_metadata[model_name] = {
                'filename': model_file,
                'path': os.path.join(self.model_dir, model_file),
                'loaded': False
            }
            
        self.available_models = list(self.model_metadata.keys())
        
        if not self.available_models:
            warnings.warn(f"No model files found in '{self.model_dir}'")
        else:
            print(f"Discovered {len(self.available_models)} models: {', '.join(self.available_models)}")
    
    def load_model(self, model_name: str) -> Any:
        """
        Load a specific model into memory.
        
        Args:
            model_name: Name of the model to load
            
        Returns:
            Loaded model object
            
        Raises:
            ValueError: If model_name is not available
            IOError: If model file cannot be loaded
        """
        if model_name not in self.model_metadata:
            raise ValueError(f"Model '{model_name}' not found. Available models: {self.available_models}")
            
        # Check if already loaded
        if model_name in self.models:
            return self.models[model_name]
            
        # Load model from disk
        model_path = self.model_metadata[model_name]['path']
        try:
            print(f"Loading model '{model_name}' from {model_path}")
            model = joblib.load(model_path)
            self.models[model_name] = model
            self.model_metadata[model_name]['loaded'] = True
            return model
        except Exception as e:
            raise IOError(f"Failed to load model '{model_name}' from {model_path}: {e}")
    
    def load_all_models(self) -> None:
        """Load all available models into memory."""
        for model_name in self.available_models:
            try:
                self.load_model(model_name)
            except Exception as e:
                warnings.warn(f"Failed to load model '{model_name}': {e}")
    
    def predict(self, model_name: str, features: np.ndarray) -> np.ndarray:
        """
        Make predictions using a specific model.
        
        Args:
            model_name: Name of the model to use
            features: Preprocessed features array (n_samples, n_features)
            
        Returns:
            Predictions array (n_samples,)
            
        Raises:
            ValueError: If model is not loaded or features are invalid
        """
        # Load model if not already loaded
        if model_name not in self.models:
            self.load_model(model_name)
            
        model = self.models[model_name]
        
        # Validate features
        if not isinstance(features, np.ndarray):
            features = np.array(features)
            
        if features.ndim == 1:
            features = features.reshape(1, -1)
        
        logger.debug(
            f"[MODEL] Predicting with '{model_name}': "
            f"input_shape={features.shape}, input_dtype={features.dtype}"
        )
            
        # Make prediction
        try:
            predictions = model.predict(features)
            logger.debug(
                f"[MODEL] Prediction output: model={model_name}, "
                f"output_shape={predictions.shape}, "
                f"values={predictions.tolist() if len(predictions) <= 10 else predictions[:10].tolist() + ['...']}"
            )
            return predictions
        except Exception as e:
            logger.error(f"[MODEL] Prediction failed for model '{model_name}': {e}")
            raise ValueError(f"Prediction failed for model '{model_name}': {e}")
    
    def predict_all(self, features: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Make predictions using all loaded models.
        
        Args:
            features: Preprocessed features array (n_samples, n_features)
            
        Returns:
            Dictionary mapping model names to predictions
        """
        results = {}
        
        for model_name in self.available_models:
            try:
                # Try to load model if not loaded
                if model_name not in self.models:
                    self.load_model(model_name)
                    
                predictions = self.predict(model_name, features)
                results[model_name] = predictions
            except Exception as e:
                warnings.warn(f"Failed to get predictions from model '{model_name}': {e}")
                results[model_name] = None
                
        return results
    
    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """
        Get information about a specific model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Dictionary with model information
        """
        if model_name not in self.model_metadata:
            raise ValueError(f"Model '{model_name}' not found")
            
        info = self.model_metadata[model_name].copy()
        
        # Add additional info if model is loaded
        if model_name in self.models:
            model = self.models[model_name]
            info['model_type'] = type(model).__name__
            
            # Try to get feature importance if available
            try:
                if hasattr(model, 'feature_importances_'):
                    info['has_feature_importance'] = True
                else:
                    info['has_feature_importance'] = False
            except:
                info['has_feature_importance'] = False
                
        return info
    
    def get_all_model_info(self) -> Dict[str, Dict]:
        """
        Get information about all available models.
        
        Returns:
            Dictionary mapping model names to their info
        """
        return {name: self.get_model_info(name) for name in self.available_models}
    
    def unload_model(self, model_name: str) -> None:
        """
        Unload a model from memory.
        
        Args:
            model_name: Name of the model to unload
        """
        if model_name in self.models:
            del self.models[model_name]
            self.model_metadata[model_name]['loaded'] = False
    
    def unload_all_models(self) -> None:
        """Unload all models from memory."""
        self.models.clear()
        for model_name in self.model_metadata:
            self.model_metadata[model_name]['loaded'] = False
    
    def get_available_models(self) -> List[str]:
        """
        Get list of available model names.
        
        Returns:
            List of model names
        """
        return self.available_models.copy()
    
    def is_model_loaded(self, model_name: str) -> bool:
        """
        Check if a model is loaded in memory.
        
        Args:
            model_name: Name of the model
            
        Returns:
            True if model is loaded, False otherwise
        """
        return model_name in self.models


class ModelRegistry:
    """
    Singleton registry for model managers.
    
    This provides a global access point for model managers to avoid
    redundant loading across the application.
    """
    _instance = None
    _managers: Dict[str, ModelManager] = {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelRegistry, cls).__new__(cls)
        return cls._instance
    
    @classmethod
    def get_manager(cls, model_dir: str = "model") -> ModelManager:
        """
        Get or create a model manager for the specified directory.
        
        Args:
            model_dir: Directory containing model files
            
        Returns:
            ModelManager instance
        """
        if model_dir not in cls._managers:
            cls._managers[model_dir] = ModelManager(model_dir)
            
        return cls._managers[model_dir]
    
    @classmethod
    def clear_registry(cls) -> None:
        """Clear all model managers from the registry."""
        cls._managers.clear()


def get_default_model_manager() -> ModelManager:
    """
    Convenience function to get the default model manager.
    
    Returns:
        ModelManager instance for the default model directory
    """
    return ModelRegistry.get_manager()


if __name__ == "__main__":
    # Test the model manager
    print("Testing ModelManager...")
    
    # Initialize model manager
    manager = ModelManager()
    
    # List available models
    available = manager.get_available_models()
    print(f"Available models: {available}")
    
    if available:
        # Get model info
        for model_name in available[:3]:  # Test first 3 models
            try:
                info = manager.get_model_info(model_name)
                print(f"\nModel: {model_name}")
                print(f"  File: {info['filename']}")
                print(f"  Loaded: {info['loaded']}")
            except Exception as e:
                print(f"Error getting info for {model_name}: {e}")
        
        # Test loading a model
        test_model = available[0]
        print(f"\nTesting loading model: {test_model}")
        
        try:
            model = manager.load_model(test_model)
            print(f"Successfully loaded {test_model}")
            print(f"Model type: {type(model).__name__}")
            
            # Test prediction with dummy data
            # Note: Actual prediction requires properly preprocessed features
            dummy_features = np.random.randn(1, 9)  # 9 features as per preprocessing
            print(f"\nTesting prediction with dummy features shape: {dummy_features.shape}")
            
            try:
                prediction = manager.predict(test_model, dummy_features)
                print(f"Prediction shape: {prediction.shape}")
                print(f"Sample prediction: {prediction[0]}")
            except Exception as e:
                print(f"Prediction test failed (expected for dummy data): {e}")
                
        except Exception as e:
            print(f"Failed to load model {test_model}: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("No models found to test. Ensure model files exist in the 'model' directory.")
        
    print("\nModel manager test completed.")