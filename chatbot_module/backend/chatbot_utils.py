"""
Chatbot utility functions for the Drug Response Prediction API.

This module provides:
- Environment variable loading (dotenv)
- Fallback response definitions when Gemini API is unavailable
- Topic matching for fallback responses

Extracted from the original chatbot_service.py for clean modularity.
"""

import os
import logging
from typing import Dict

logger = logging.getLogger(__name__)

# ============================================================================
# ENVIRONMENT LOADING — Load .env before any Gemini initialization
# ============================================================================

try:
    from dotenv import load_dotenv
    from pathlib import Path
    _env_path = Path(__file__).parent.parent.parent / ".env"
    _loaded = load_dotenv(_env_path, override=True)
    if _loaded:
        logger.debug(f"Chatbot utils: Loaded .env from {_env_path}")
    else:
        logger.warning(f"Chatbot utils: No .env file found at {_env_path}")
except ImportError:
    logger.warning("Chatbot utils: python-dotenv not installed; skipping .env load")


# ============================================================================
# FALLBACK RESPONSES — Used when Gemini API is unavailable
# ============================================================================

FALLBACK_RESPONSES = {
    "ic50": (
        "IC50 (Half Maximal Inhibitory Concentration) measures the concentration of a drug "
        "needed to inhibit 50% of cell growth. Lower IC50 values indicate higher drug effectiveness. "
        "In our system, we predict LN_IC50 (natural log of IC50):\n"
        "- IC50 < -2: Very sensitive (drug highly effective)\n"
        "- IC50 between -2 and 0: Sensitive (drug likely effective)\n"
        "- IC50 between 0 and 2: Moderate effectiveness\n"
        "- IC50 > 2: Resistant (drug likely not effective)\n\n"
        "⚠️ Note: The Gemini AI service is currently unavailable. This is a pre-defined response. "
        "For more detailed explanations, please ensure the GEMINI_API_KEY is configured."
    ),
    "recommendation": (
        "The recommendation engine predicts IC50 for all available drugs for a given cell line, "
        "then ranks them using the formula: score = 1/(1+IC50). Lower IC50 → higher score → "
        "better recommendation. The top-ranked drugs are those predicted to be most effective.\n\n"
        "⚠️ Note: The Gemini AI service is currently unavailable. This is a pre-defined response."
    ),
    "preprocessing": (
        "The preprocessing pipeline has 3 stages:\n"
        "1. Label Encoding: 10 categorical columns are encoded using saved LabelEncoder artifacts. "
        "Unseen labels are mapped to -1.\n"
        "2. Feature Selection: 9 features are selected (AUC, Z_SCORE, and 7 encoded columns). "
        "CELL_LINE_NAME and DRUG_NAME are NOT in the final features.\n"
        "3. Standard Scaling: Features are normalized to zero mean and unit variance.\n\n"
        "⚠️ Note: The Gemini AI service is currently unavailable. This is a pre-defined response."
    ),
    "models": (
        "The system uses 7 regression models: CatBoost (default), XGBoost, LightGBM, "
        "GradientBoosting, ElasticNet, Ridge, and Lasso. All models predict LN_IC50 values. "
        "CatBoost is the default as it typically performs best on this dataset.\n\n"
        "⚠️ Note: The Gemini AI service is currently unavailable. This is a pre-defined response."
    ),
    "general": (
        "I'm an AI assistant for the Drug Response Prediction Platform. I can help explain "
        "IC50 values, drug recommendations, preprocessing, and ML models. However, the Gemini "
        "AI service is currently unavailable, so my responses are limited to pre-defined explanations.\n\n"
        "To enable full AI-powered responses, please configure the GEMINI_API_KEY in your .env file.\n\n"
        "⚠️ This tool is for educational and research purposes only. It does not provide medical advice."
    ),
}


def match_fallback_topic(message: str) -> str:
    """
    Match a user message to the closest fallback response topic.
    
    Args:
        message: User message
        
    Returns:
        Key for the fallback response dictionary
    """
    message_lower = message.lower()
    
    # Check for topic keywords
    if any(kw in message_lower for kw in ['ic50', 'ic 50', 'half maximal', 'inhibitory concentration', 'sensitivity', 'ln_ic50']):
        return "ic50"
    if any(kw in message_lower for kw in ['recommend', 'ranking', 'score', 'drug order', 'best drug', 'top drug']):
        return "recommendation"
    if any(kw in message_lower for kw in ['preprocess', 'encoding', 'label encod', 'scaling', 'feature selection', 'pipeline']):
        return "preprocessing"
    if any(kw in message_lower for kw in ['model', 'catboost', 'xgboost', 'lightgbm', 'gradient', 'ridge', 'lasso', 'elasticnet']):
        return "models"
    return "general"