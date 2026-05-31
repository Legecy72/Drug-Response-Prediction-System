"""
Shared test fixtures and configuration.
"""

import sys
import os
import pytest
import pandas as pd
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture
def sample_data():
    """Create sample data matching GDSC dataset structure."""
    return pd.DataFrame({
        "CELL_LINE_NAME": ["A172", "A172", "A172", "U87MG", "U87MG"],
        "TCGA_DESC": ["GBM", "GBM", "GBM", "GBM", "GBM"],
        "DRUG_NAME": ["Camptothecin", "Erlotinib", "Rapamycin", "Camptothecin", "Erlotinib"],
        "TARGET": ["TOP1", "EGFR", "MTORC1", "TOP1", "EGFR"],
        "TARGET_PATHWAY": ["DNA replication", "EGFR signaling", "PI3K/MTOR signaling", "DNA replication", "EGFR signaling"],
        "SITE": ["nervous_system", "nervous_system", "nervous_system", "nervous_system", "nervous_system"],
        "HISTOLOGY": ["glioma", "glioma", "glioma", "glioma", "glioma"],
        "GDSC_TISSUE_DESCRIPTOR_1": ["nervous_system", "nervous_system", "nervous_system", "nervous_system", "nervous_system"],
        "GDSC_TISSUE_DESCRIPTOR_2": ["glioma", "glioma", "glioma", "glioma", "glioma"],
        "CANCER_TYPE_MATCHING_TCGA_LABEL": ["GBM", "GBM", "GBM", "GBM", "GBM"],
        "AUC": [0.93, 0.87, 0.91, 0.85, 0.79],
        "Z_SCORE": [0.43, -0.12, 0.21, -0.05, -0.34],
        "LN_IC50": [-1.46, -0.89, -1.12, -0.76, -0.45]
    })


@pytest.fixture
def single_sample():
    """Create a single sample dictionary for prediction."""
    return {
        "CELL_LINE_NAME": "A172",
        "TCGA_DESC": "GBM",
        "DRUG_NAME": "Camptothecin",
        "TARGET": "TOP1",
        "TARGET_PATHWAY": "DNA replication",
        "SITE": "nervous_system",
        "HISTOLOGY": "glioma",
        "GDSC_TISSUE_DESCRIPTOR_1": "nervous_system",
        "GDSC_TISSUE_DESCRIPTOR_2": "glioma",
        "CANCER_TYPE_MATCHING_TCGA_LABEL": "GBM",
        "AUC": 0.93,
        "Z_SCORE": 0.43
    }


@pytest.fixture
def cell_line_data():
    """Create cell line data for recommendation (without drug-specific fields)."""
    return {
        "CELL_LINE_NAME": "A172",
        "TCGA_DESC": "GBM",
        "SITE": "nervous_system",
        "HISTOLOGY": "glioma",
        "GDSC_TISSUE_DESCRIPTOR_1": "nervous_system",
        "GDSC_TISSUE_DESCRIPTOR_2": "glioma",
        "CANCER_TYPE_MATCHING_TCGA_LABEL": "GBM",
        "AUC": 0.5,
        "Z_SCORE": 0.0
    }


@pytest.fixture
def batch_samples():
    """Create multiple samples for batch prediction."""
    return [
        {
            "CELL_LINE_NAME": "A172",
            "TCGA_DESC": "GBM",
            "DRUG_NAME": "Camptothecin",
            "TARGET": "TOP1",
            "TARGET_PATHWAY": "DNA replication",
            "SITE": "nervous_system",
            "HISTOLOGY": "glioma",
            "GDSC_TISSUE_DESCRIPTOR_1": "nervous_system",
            "GDSC_TISSUE_DESCRIPTOR_2": "glioma",
            "CANCER_TYPE_MATCHING_TCGA_LABEL": "GBM",
            "AUC": 0.93,
            "Z_SCORE": 0.43
        },
        {
            "CELL_LINE_NAME": "A172",
            "TCGA_DESC": "GBM",
            "DRUG_NAME": "Erlotinib",
            "TARGET": "EGFR",
            "TARGET_PATHWAY": "EGFR signaling",
            "SITE": "nervous_system",
            "HISTOLOGY": "glioma",
            "GDSC_TISSUE_DESCRIPTOR_1": "nervous_system",
            "GDSC_TISSUE_DESCRIPTOR_2": "glioma",
            "CANCER_TYPE_MATCHING_TCGA_LABEL": "GBM",
            "AUC": 0.87,
            "Z_SCORE": -0.12
        }
    ]


@pytest.fixture
def fitted_preprocessor(sample_data):
    """Create a fitted preprocessor."""
    from core.preprocessing import DrugResponsePreprocessor
    preprocessor = DrugResponsePreprocessor()
    preprocessor.fit(sample_data)
    return preprocessor
