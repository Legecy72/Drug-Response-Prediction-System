"""
Tests for the model manager module.

Validates:
1. Model discovery from /model directory
2. Model loading and caching
3. Prediction interface
4. Model registry singleton
"""

import sys
import os
import pytest
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.model_manager import ModelManager, ModelRegistry, get_default_model_manager


class TestModelDiscovery:
    """Test model file discovery."""
    
    def test_discovers_pkl_files(self):
        """Should discover .pkl model files in the model directory."""
        manager = ModelManager(model_dir="model")
        assert len(manager.available_models) > 0
    
    def test_model_names_extracted(self):
        """Model names should be extracted from filenames correctly."""
        manager = ModelManager(model_dir="model")
        # Should have names like CatBoost, XGBoost, etc. (without _regression_model suffix)
        for name in manager.available_models:
            assert "_regression_model" not in name
            assert ".pkl" not in name
    
    def test_expected_models_found(self):
        """Should find all 7 expected models."""
        manager = ModelManager(model_dir="model")
        expected = ["CatBoost", "ElasticNet", "GradientBoosting", "Lasso", "LightGBM", "Ridge", "XGBoost"]
        for model_name in expected:
            assert model_name in manager.available_models, f"Model '{model_name}' not found"
    
    def test_nonexistent_directory(self):
        """Should handle nonexistent model directory gracefully."""
        manager = ModelManager(model_dir="/nonexistent/path")
        assert len(manager.available_models) == 0
    
    def test_model_metadata_has_path(self):
        """Each model metadata should have a file path."""
        manager = ModelManager(model_dir="model")
        for name in manager.available_models:
            assert "path" in manager.model_metadata[name]
            assert os.path.exists(manager.model_metadata[name]["path"])


class TestModelLoading:
    """Test model loading functionality."""
    
    def test_load_single_model(self):
        """Should load a single model successfully."""
        manager = ModelManager(model_dir="model")
        if not manager.available_models:
            pytest.skip("No models available")
        
        model_name = manager.available_models[0]
        model = manager.load_model(model_name)
        assert model is not None
        assert manager.model_metadata[model_name]["loaded"] is True
    
    def test_load_model_caching(self):
        """Loading same model twice should return same object."""
        manager = ModelManager(model_dir="model")
        if not manager.available_models:
            pytest.skip("No models available")
        
        model_name = manager.available_models[0]
        model1 = manager.load_model(model_name)
        model2 = manager.load_model(model_name)
        assert model1 is model2
    
    def test_load_nonexistent_model(self):
        """Should raise ValueError for nonexistent model."""
        manager = ModelManager(model_dir="model")
        with pytest.raises(ValueError, match="not found"):
            manager.load_model("NonExistentModel")
    
    def test_load_all_models(self):
        """Should load all available models."""
        manager = ModelManager(model_dir="model")
        if not manager.available_models:
            pytest.skip("No models available")
        
        manager.load_all_models()
        loaded_count = sum(1 for m in manager.model_metadata.values() if m.get("loaded"))
        assert loaded_count == len(manager.available_models)
    
    def test_unload_model(self):
        """Should unload a model from memory."""
        manager = ModelManager(model_dir="model")
        if not manager.available_models:
            pytest.skip("No models available")
        
        model_name = manager.available_models[0]
        manager.load_model(model_name)
        assert manager.is_model_loaded(model_name)
        
        manager.unload_model(model_name)
        assert not manager.is_model_loaded(model_name)


class TestModelPrediction:
    """Test model prediction interface."""
    
    def test_predict_with_loaded_model(self):
        """Should make predictions with a loaded model."""
        manager = ModelManager(model_dir="model")
        if not manager.available_models:
            pytest.skip("No models available")
        
        model_name = manager.available_models[0]
        # Use dummy features (9 features as per preprocessing)
        features = np.random.randn(1, 9)
        
        predictions = manager.predict(model_name, features)
        assert predictions is not None
        assert len(predictions) == 1
    
    def test_predict_auto_loads_model(self):
        """Predict should auto-load model if not already loaded."""
        manager = ModelManager(model_dir="model")
        if not manager.available_models:
            pytest.skip("No models available")
        
        model_name = manager.available_models[0]
        assert not manager.is_model_loaded(model_name)
        
        features = np.random.randn(1, 9)
        predictions = manager.predict(model_name, features)
        assert manager.is_model_loaded(model_name)
    
    def test_predict_batch(self):
        """Should make predictions for multiple samples."""
        manager = ModelManager(model_dir="model")
        if not manager.available_models:
            pytest.skip("No models available")
        
        model_name = manager.available_models[0]
        features = np.random.randn(5, 9)
        
        predictions = manager.predict(model_name, features)
        assert len(predictions) == 5
    
    def test_predict_1d_features(self):
        """Should handle 1D feature array (single sample)."""
        manager = ModelManager(model_dir="model")
        if not manager.available_models:
            pytest.skip("No models available")
        
        model_name = manager.available_models[0]
        features = np.random.randn(9)  # 1D
        
        predictions = manager.predict(model_name, features)
        assert len(predictions) == 1


class TestModelRegistry:
    """Test model registry singleton."""
    
    def test_registry_singleton(self):
        """Registry should be a singleton."""
        r1 = ModelRegistry()
        r2 = ModelRegistry()
        assert r1 is r2
    
    def test_get_manager_returns_same_instance(self):
        """Getting manager for same directory should return same instance."""
        manager1 = ModelRegistry.get_manager("model")
        manager2 = ModelRegistry.get_manager("model")
        assert manager1 is manager2


class TestModelInfo:
    """Test model information retrieval."""
    
    def test_get_model_info(self):
        """Should return model info dictionary."""
        manager = ModelManager(model_dir="model")
        if not manager.available_models:
            pytest.skip("No models available")
        
        model_name = manager.available_models[0]
        info = manager.get_model_info(model_name)
        assert "filename" in info
        assert "loaded" in info
        assert "path" in info
    
    def test_get_all_model_info(self):
        """Should return info for all models."""
        manager = ModelManager(model_dir="model")
        all_info = manager.get_all_model_info()
        assert len(all_info) == len(manager.available_models)
    
    def test_get_model_info_loaded_model(self):
        """Loaded model info should include model type."""
        manager = ModelManager(model_dir="model")
        if not manager.available_models:
            pytest.skip("No models available")
        
        model_name = manager.available_models[0]
        manager.load_model(model_name)
        info = manager.get_model_info(model_name)
        assert "model_type" in info
