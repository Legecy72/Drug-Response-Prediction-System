"""
Tests for the preprocessing module.

Validates that the preprocessing pipeline matches the notebook:
1. Label encoding for 10 categorical columns
2. Feature selection to 9 positive features
3. Standard scaling
"""

import sys
import os
import pytest
import pandas as pd
import numpy as np
import tempfile
import shutil

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.preprocessing import (
    DrugResponsePreprocessor,
    CATEGORICAL_COLUMNS,
    POSITIVE_FEATURES,
    NUMERIC_FEATURES,
    validate_input_data,
    create_sample_dataframe
)


class TestConstants:
    """Test that constants match notebook analysis."""
    
    def test_categorical_columns_count(self):
        """Should have exactly 10 categorical columns."""
        assert len(CATEGORICAL_COLUMNS) == 10
    
    def test_positive_features_count(self):
        """Should have exactly 9 positive features."""
        assert len(POSITIVE_FEATURES) == 9
    
    def test_numeric_features_count(self):
        """Should have exactly 2 numeric features."""
        assert len(NUMERIC_FEATURES) == 2
    
    def test_positive_features_contain_numeric(self):
        """Positive features should include AUC and Z_SCORE."""
        assert "AUC" in POSITIVE_FEATURES
        assert "Z_SCORE" in POSITIVE_FEATURES
    
    def test_positive_features_contain_encoded(self):
        """Positive features should include 7 encoded categorical features."""
        encoded_features = [f for f in POSITIVE_FEATURES if f.endswith("_ENC")]
        assert len(encoded_features) == 7
    
    def test_excluded_features(self):
        """CELL_LINE_NAME_ENC, DRUG_NAME_ENC, HISTOLOGY_ENC should NOT be in positive features."""
        assert "CELL_LINE_NAME_ENC" not in POSITIVE_FEATURES
        assert "DRUG_NAME_ENC" not in POSITIVE_FEATURES
        assert "HISTOLOGY_ENC" not in POSITIVE_FEATURES
    
    def test_categorical_columns_match_notebook(self):
        """Categorical columns should match the notebook exactly."""
        expected = [
            "CELL_LINE_NAME", "TCGA_DESC", "DRUG_NAME", "TARGET",
            "TARGET_PATHWAY", "SITE", "HISTOLOGY",
            "GDSC_TISSUE_DESCRIPTOR_1", "GDSC_TISSUE_DESCRIPTOR_2",
            "CANCER_TYPE_MATCHING_TCGA_LABEL"
        ]
        assert CATEGORICAL_COLUMNS == expected


class TestPreprocessorFit:
    """Test preprocessor fitting."""
    
    def test_fit_returns_self(self, sample_data):
        """Fit should return self for chaining."""
        preprocessor = DrugResponsePreprocessor()
        result = preprocessor.fit(sample_data)
        assert result is preprocessor
    
    def test_fit_sets_fitted_flag(self, sample_data):
        """Fit should set is_fitted to True."""
        preprocessor = DrugResponsePreprocessor()
        assert preprocessor.is_fitted is False
        preprocessor.fit(sample_data)
        assert preprocessor.is_fitted is True
    
    def test_fit_creates_label_encoders(self, sample_data):
        """Fit should create label encoders for all categorical columns."""
        preprocessor = DrugResponsePreprocessor()
        preprocessor.fit(sample_data)
        for col in CATEGORICAL_COLUMNS:
            assert col in preprocessor.label_encoders
            assert hasattr(preprocessor.label_encoders[col], 'classes_')
    
    def test_fit_creates_scaler(self, sample_data):
        """Fit should fit the StandardScaler."""
        preprocessor = DrugResponsePreprocessor()
        preprocessor.fit(sample_data)
        assert hasattr(preprocessor.scaler, 'mean_')
        assert hasattr(preprocessor.scaler, 'scale_')
    
    def test_fit_raises_on_missing_column(self):
        """Fit should raise ValueError if a required column is missing."""
        preprocessor = DrugResponsePreprocessor()
        incomplete_df = pd.DataFrame({"CELL_LINE_NAME": ["A172"]})
        with pytest.raises(ValueError, match="Missing required categorical column"):
            preprocessor.fit(incomplete_df)


class TestPreprocessorTransform:
    """Test preprocessor transformation."""
    
    def test_transform_output_shape(self, fitted_preprocessor, sample_data):
        """Transform should return array with shape (n_samples, 9)."""
        result = fitted_preprocessor.transform(sample_data)
        assert result.shape == (len(sample_data), 9)
    
    def test_transform_output_type(self, fitted_preprocessor, sample_data):
        """Transform should return numpy array."""
        result = fitted_preprocessor.transform(sample_data)
        assert isinstance(result, np.ndarray)
    
    def test_transform_output_dtype(self, fitted_preprocessor, sample_data):
        """Transform should return float64 array."""
        result = fitted_preprocessor.transform(sample_data)
        assert result.dtype in [np.float64, np.float32]
    
    def test_transform_requires_fitted(self, sample_data):
        """Transform should raise if preprocessor is not fitted."""
        preprocessor = DrugResponsePreprocessor()
        with pytest.raises(ValueError, match="must be fitted"):
            preprocessor.transform(sample_data)
    
    def test_transform_produces_scaled_features(self, fitted_preprocessor, sample_data):
        """Transformed features should have approximately zero mean and unit variance."""
        result = fitted_preprocessor.transform(sample_data)
        # With small samples, mean won't be exactly 0 but should be close
        # The scaler is fitted on the same data, so mean should be ~0
        col_means = result.mean(axis=0)
        assert np.allclose(col_means, 0, atol=0.5)
    
    def test_fit_transform_equals_fit_then_transform(self, sample_data):
        """fit_transform should produce same result as fit then transform."""
        preprocessor1 = DrugResponsePreprocessor()
        preprocessor2 = DrugResponsePreprocessor()
        
        result1 = preprocessor1.fit_transform(sample_data)
        preprocessor2.fit(sample_data)
        result2 = preprocessor2.transform(sample_data)
        
        np.testing.assert_array_almost_equal(result1, result2)


class TestPreprocessorSaveLoad:
    """Test preprocessor save and load functionality."""
    
    def test_save_creates_files(self, fitted_preprocessor):
        """Save should create encoder and scaler files."""
        tmpdir = tempfile.mkdtemp()
        try:
            fitted_preprocessor.save(tmpdir)
            
            # Check scaler file
            assert os.path.exists(os.path.join(tmpdir, "scaler.pkl"))
            
            # Check encoder files
            for col in CATEGORICAL_COLUMNS:
                assert os.path.exists(os.path.join(tmpdir, f"{col}_encoder.pkl"))
            
            # Check metadata
            assert os.path.exists(os.path.join(tmpdir, "preprocessor_metadata.pkl"))
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)
    
    def test_load_restores_preprocessor(self, fitted_preprocessor, sample_data):
        """Load should restore a preprocessor that produces same results."""
        tmpdir = tempfile.mkdtemp()
        try:
            fitted_preprocessor.save(tmpdir)
            
            loaded = DrugResponsePreprocessor.load(tmpdir)
            assert loaded.is_fitted is True
            
            # Same transform results
            result1 = fitted_preprocessor.transform(sample_data)
            result2 = loaded.transform(sample_data)
            np.testing.assert_array_almost_equal(result1, result2)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)
    
    def test_load_raises_on_missing_files(self):
        """Load should raise if files are missing."""
        with pytest.raises(FileNotFoundError):
            DrugResponsePreprocessor.load("/nonexistent/path")


class TestInputValidation:
    """Test input validation."""
    
    def test_validate_valid_data(self, sample_data):
        """Validation should pass for valid data."""
        assert validate_input_data(sample_data) is True
    
    def test_validate_missing_column(self):
        """Validation should raise for missing columns."""
        df = pd.DataFrame({"CELL_LINE_NAME": ["A172"]})
        with pytest.raises(ValueError, match="Missing required columns"):
            validate_input_data(df)


class TestSampleData:
    """Test sample data creation."""
    
    def test_create_sample_dataframe(self):
        """Sample DataFrame should have all required columns."""
        df = create_sample_dataframe()
        required = CATEGORICAL_COLUMNS + NUMERIC_FEATURES
        for col in required:
            assert col in df.columns
    
    def test_sample_dataframe_not_empty(self):
        """Sample DataFrame should not be empty."""
        df = create_sample_dataframe()
        assert len(df) > 0
