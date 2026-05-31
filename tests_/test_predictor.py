"""
Tests for the predictor module.

Validates:
1. Single prediction workflow
2. Batch prediction workflow
3. Model comparison
4. Input format handling
5. End-to-end prediction pipeline
"""

import sys
import os
import pytest
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.preprocessing import DrugResponsePreprocessor
from core.model_manager import ModelManager
from core.predictor import DrugResponsePredictor


@pytest.fixture
def predictor(sample_data):
    """Create a predictor with fitted preprocessor."""
    preprocessor = DrugResponsePreprocessor()
    preprocessor.fit(sample_data)
    model_manager = ModelManager(model_dir="model")
    return DrugResponsePredictor(
        preprocessor=preprocessor,
        model_manager=model_manager
    )


class TestPredictorInit:
    """Test predictor initialization."""
    
    def test_default_initialization(self):
        """Should initialize with default components."""
        predictor = DrugResponsePredictor()
        assert predictor.preprocessor is not None
        assert predictor.model_manager is not None
        assert predictor.default_model is not None
    
    def test_custom_initialization(self, sample_data):
        """Should accept custom preprocessor and model manager."""
        preprocessor = DrugResponsePreprocessor()
        preprocessor.fit(sample_data)
        model_manager = ModelManager(model_dir="model")
        
        predictor = DrugResponsePredictor(
            preprocessor=preprocessor,
            model_manager=model_manager,
            default_model="CatBoost"
        )
        assert predictor.preprocessor is preprocessor
        assert predictor.model_manager is model_manager
    
    def test_default_model_fallback(self, sample_data):
        """Should fall back to first available if default not found."""
        preprocessor = DrugResponsePreprocessor()
        preprocessor.fit(sample_data)
        model_manager = ModelManager(model_dir="model")
        
        predictor = DrugResponsePredictor(
            preprocessor=preprocessor,
            model_manager=model_manager,
            default_model="NonExistentModel"
        )
        # Should have fallen back to first available
        assert predictor.default_model in model_manager.available_models


class TestSinglePrediction:
    """Test single sample prediction."""
    
    def test_predict_dict_input(self, predictor, single_sample):
        """Should predict from dictionary input."""
        result = predictor.predict(single_sample)
        assert "predictions" in result
        assert "model_used" in result
        assert len(result["predictions"]) == 1
        assert isinstance(result["predictions"][0], float)
    
    def test_predict_returns_model_name(self, predictor, single_sample):
        """Result should include which model was used."""
        result = predictor.predict(single_sample)
        assert result["model_used"] is not None
        assert isinstance(result["model_used"], str)
    
    def test_predict_with_specific_model(self, predictor, single_sample):
        """Should use specified model."""
        available = predictor.get_available_models()
        if len(available) < 2:
            pytest.skip("Need at least 2 models")
        
        result = predictor.predict(single_sample, model_name=available[1])
        assert result["model_used"] == available[1]
    
    def test_predict_ic50_reasonable_range(self, predictor, single_sample):
        """Predicted IC50 should be in a reasonable range."""
        result = predictor.predict(single_sample)
        ic50 = result["predictions"][0]
        # IC50 values in the dataset typically range from -5 to 10
        assert -20 < ic50 < 20, f"IC50 {ic50} seems unreasonable"


class TestBatchPrediction:
    """Test batch prediction."""
    
    def test_predict_list_input(self, predictor, batch_samples):
        """Should predict from list of dictionaries."""
        result = predictor.predict(batch_samples)
        assert len(result["predictions"]) == len(batch_samples)
    
    def test_predict_dataframe_input(self, predictor, sample_data):
        """Should predict from DataFrame input."""
        result = predictor.predict(sample_data)
        assert len(result["predictions"]) == len(sample_data)
    
    def test_batch_predictions_count(self, predictor, batch_samples):
        """Sample count should match input."""
        result = predictor.predict(batch_samples)
        assert result["sample_count"] == len(batch_samples)


class TestModelComparison:
    """Test model comparison functionality."""
    
    def test_compare_all_models(self, predictor, single_sample):
        """Should compare all available models."""
        result = predictor.compare_models(single_sample)
        assert "predictions" in result
        assert "models_compared" in result
        assert len(result["models_compared"]) > 0
    
    def test_compare_specific_models(self, predictor, single_sample):
        """Should compare only specified models."""
        available = predictor.get_available_models()
        if len(available) < 2:
            pytest.skip("Need at least 2 models")
        
        models_to_compare = available[:2]
        result = predictor.compare_models(single_sample, model_names=models_to_compare)
        assert set(result["models_compared"]) == set(models_to_compare)
    
    def test_comparison_has_predictions(self, predictor, single_sample):
        """Each model in comparison should have a prediction."""
        result = predictor.compare_models(single_sample)
        for model_name, preds in result["predictions"].items():
            if preds is not None:
                assert len(preds) > 0


class TestInputPreparation:
    """Test input data preparation."""
    
    def test_dict_to_dataframe(self, predictor, single_sample):
        """Should convert dict input to DataFrame."""
        df = predictor._prepare_input_data(single_sample)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1
    
    def test_list_to_dataframe(self, predictor, batch_samples):
        """Should convert list input to DataFrame."""
        df = predictor._prepare_input_data(batch_samples)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == len(batch_samples)
    
    def test_dataframe_passthrough(self, predictor, sample_data):
        """DataFrame input should be copied, not modified."""
        df = predictor._prepare_input_data(sample_data)
        assert isinstance(df, pd.DataFrame)
        # Should be a copy
        assert df is not sample_data
    
    def test_invalid_input_raises(self, predictor):
        """Should raise ValueError for invalid input types."""
        with pytest.raises(ValueError, match="Input data must be"):
            predictor._prepare_input_data("invalid_string")


class TestEndToEnd:
    """End-to-end integration tests."""
    
    def test_full_prediction_pipeline(self, sample_data, single_sample):
        """Full pipeline: fit preprocessor → predict → get result."""
        preprocessor = DrugResponsePreprocessor()
        preprocessor.fit(sample_data)
        model_manager = ModelManager(model_dir="model")
        predictor = DrugResponsePredictor(
            preprocessor=preprocessor,
            model_manager=model_manager
        )
        
        result = predictor.predict(single_sample)
        assert result["predictions"][0] is not None
        assert isinstance(result["predictions"][0], float)
    
    def test_predictions_are_deterministic(self, predictor, single_sample):
        """Same input should produce same prediction."""
        result1 = predictor.predict(single_sample)
        result2 = predictor.predict(single_sample)
        assert result1["predictions"][0] == pytest.approx(result2["predictions"][0])
    
    def test_different_inputs_different_predictions(self, predictor, single_sample):
        """Different inputs should generally produce different predictions."""
        modified_sample = single_sample.copy()
        modified_sample["AUC"] = 0.1
        modified_sample["Z_SCORE"] = -2.0
        
        result1 = predictor.predict(single_sample)
        result2 = predictor.predict(modified_sample)
        # Different inputs should generally give different results
        # (not guaranteed but extremely likely with different AUC/Z_SCORE)
        assert result1["predictions"][0] != result2["predictions"][0]
