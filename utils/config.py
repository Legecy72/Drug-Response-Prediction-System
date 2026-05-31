"""
Configuration module for the Drug Response Prediction System.

This module provides centralized configuration for:
- API settings
- Model paths
- Data paths
- Logging configuration
"""

import os
from pathlib import Path

# Load environment variables from .env file (if available)
try:
    from dotenv import load_dotenv
    # Use explicit path to ensure .env is found regardless of CWD
    _env_path = Path(__file__).parent.parent / ".env"
    _loaded = load_dotenv(_env_path, override=True)
    if _loaded:
        import logging
        logging.getLogger(__name__).debug(f"Loaded .env from {_env_path}")
    else:
        import logging
        logging.getLogger(__name__).warning(f"No .env file found at {_env_path}")
except ImportError:
    import logging
    logging.getLogger(__name__).warning("python-dotenv not installed; skipping .env load")

# Base directory
BASE_DIR = Path(__file__).parent.parent

# Data paths
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "model"
WEB_DIR = BASE_DIR / "web"

# Data files
GDSC_DATASET_PATH = DATA_DIR / "GDSC_DATASET.csv"
GDSC2_DATASET_PATH = DATA_DIR / "GDSC2-dataset.csv"
COMPOUNDS_PATH = DATA_DIR / "Compounds-annotation.csv"
CELL_LINES_PATH = DATA_DIR / "Cell_Lines_Details.xlsx"

# API settings
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))
API_RELOAD = os.getenv("API_RELOAD", "True").lower() == "true"
API_LOG_LEVEL = os.getenv("API_LOG_LEVEL", "info")

# Model settings
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "CatBoost")
MODEL_FILE_PATTERN = "*_regression_model.pkl"

# Preprocessing settings
PREPROCESSOR_SAVE_DIR = BASE_DIR / "core" / "saved_preprocessor"

# Feature configuration
CATEGORICAL_COLUMNS = [
    "CELL_LINE_NAME",
    "TCGA_DESC",
    "DRUG_NAME",
    "TARGET",
    "TARGET_PATHWAY",
    "SITE",
    "HISTOLOGY",
    "GDSC_TISSUE_DESCRIPTOR_1",
    "GDSC_TISSUE_DESCRIPTOR_2",
    "CANCER_TYPE_MATCHING_TCGA_LABEL"
]

POSITIVE_FEATURES = [
    "AUC",
    "Z_SCORE",
    "TARGET_ENC",
    "TARGET_PATHWAY_ENC",
    "TCGA_DESC_ENC",
    "GDSC_TISSUE_DESCRIPTOR_2_ENC",
    "CANCER_TYPE_MATCHING_TCGA_LABEL_ENC",
    "SITE_ENC",
    "GDSC_TISSUE_DESCRIPTOR_1_ENC"
]

NUMERIC_FEATURES = ["AUC", "Z_SCORE"]

# Recommendation settings
DEFAULT_MAX_DRUGS = 10
SCORE_FORMULA = "1 / (1 + abs(IC50))"

# Gemini AI Chatbot settings
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

# CORS settings
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*").split(",")

# Logging
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")


def get_config() -> dict:
    """Get all configuration as a dictionary."""
    return {
        "base_dir": str(BASE_DIR),
        "data_dir": str(DATA_DIR),
        "model_dir": str(MODEL_DIR),
        "web_dir": str(WEB_DIR),
        "api_host": API_HOST,
        "api_port": API_PORT,
        "default_model": DEFAULT_MODEL,
        "categorical_columns": CATEGORICAL_COLUMNS,
        "positive_features": POSITIVE_FEATURES,
        "numeric_features": NUMERIC_FEATURES,
        "gemini_api_key_configured": bool(GEMINI_API_KEY and GEMINI_API_KEY != "your_gemini_api_key_here"),
        "gemini_model": GEMINI_MODEL,
    }


if __name__ == "__main__":
    import json
    print(json.dumps(get_config(), indent=2))