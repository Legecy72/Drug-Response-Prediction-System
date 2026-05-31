from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd

from core.config import Settings, get_settings
from core.schemas import CANCER_CONTEXT_FIELDS, INPUT_FIELDS
from core.engines import (
    ClassificationModelEngine,
    FinalRecommendationSystem,
    IC50RegressionEngine,
)


REGRESSION_MODELS = {
    "catboost": "catboost_regression_model.pkl",
    "elasticnet": "elasticnet_regression_model.pkl",
    "gradientboosting": "gradientboosting_regression_model.pkl",
    "lasso": "lasso_regression_model.pkl",
    "lightgbm": "lightgbm_regression_model.pkl",
    "ridge": "ridge_regression_model.pkl",
    "xgboost": "xgboost_regression_model.pkl",
}

CLASSIFICATION_MODELS = {
    "catboost": "catboost_classification_model.pkl",
    "decision_tree": "decision_tree_classification_model.pkl",
    "gradient_boosting": "gradient_boosting_classification_model.pkl",
    "knn": "knn_classification_model.pkl",
    "lightgbm": "lightgbm_classification_model.pkl",
    "logistic_regression": "logistic_regression_classification_model.pkl",
    "naive_bayes": "naive_bayes_classification_model.pkl",
    "random_forest": "random_forest_classification_model.pkl",
    "xgboost": "xgboost_classification_model.pkl",
}


def _resolve_model_path(directory: Path, registry: dict[str, str], model_name: str) -> Path:
    normalized = model_name.strip().lower()
    if normalized not in registry:
        allowed = ", ".join(sorted(registry))
        raise ValueError(f"Unknown model '{model_name}'. Allowed values: {allowed}")

    path = directory / registry[normalized]
    if not path.exists():
        raise FileNotFoundError(f"Model artifact not found: {path}")
    return path


class DrugResponseService:
    def __init__(self, settings: Settings):
        self.settings = settings

        regression_model_path = _resolve_model_path(
            settings.regression_dir,
            REGRESSION_MODELS,
            settings.regression_model_name,
        )
        classification_model_path = _resolve_model_path(
            settings.classification_dir,
            CLASSIFICATION_MODELS,
            settings.classification_model_name,
        )

        self.regression_engine = IC50RegressionEngine(
            model_path=regression_model_path,
            features_path=settings.regression_dir / "regression_features.pkl",
            encoders_path=settings.regression_dir / "regression_label_encoders.pkl",
            auc_reference_path=settings.regression_dir / "drug_auc_reference.csv",
        )
        self.classification_engine = ClassificationModelEngine(
            model_path=classification_model_path,
            features_path=settings.classification_dir / "classification_features.pkl",
            encoder_path=settings.classification_dir / "classification_encoder.pkl",
            scaler_path=settings.classification_dir / "classification_scaler.pkl",
            num_imputer_path=settings.classification_dir / "classification_num_imputer.pkl",
            cat_imputer_path=settings.classification_dir / "classification_cat_imputer.pkl",
            fallback_label_encoders_path=settings.regression_dir / "regression_label_encoders.pkl",
        )
        self.recommender = FinalRecommendationSystem(
            regression_engine=self.regression_engine,
            classification_engine=self.classification_engine,
        )

    @property
    def auc_reference(self) -> pd.DataFrame:
        return self.regression_engine.auc_reference

    def status(self) -> dict[str, Any]:
        return {
            "regression_model": self.settings.regression_model_name,
            "classification_model": self.settings.classification_model_name,
            "regression_features": self.regression_engine.features,
            "classification_features": self.classification_engine.features,
            "auc_reference_rows": int(len(self.auc_reference)),
            "available_drugs": int(self.auc_reference["DRUG_NAME"].nunique()),
        }

    def options(self, filters: dict[str, str] | None = None) -> dict[str, list[str]]:
        df = self.auc_reference
        filters = filters or {}

        for column, value in filters.items():
            if column in INPUT_FIELDS and value:
                df = df[df[column].astype(str) == str(value)]

        options: dict[str, list[str]] = {}
        for column in INPUT_FIELDS:
            values = (
                df[column]
                .fillna("")
                .astype(str)
                .drop_duplicates()
                .sort_values(kind="stable")
                .tolist()
            )
            options[column] = values
        return options

    def sample_input(self) -> dict[str, str]:
        row = self.auc_reference.iloc[0]
        return {field: str(row.get(field, "")) for field in INPUT_FIELDS}

    def context_options(self, filters: dict[str, str] | None = None) -> dict[str, list[str]]:
        options = self.options(filters)
        return {field: options[field] for field in CANCER_CONTEXT_FIELDS}

    def predict(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self.recommender.predict_for_drug(payload)

    def recommend(self, payload: dict[str, Any], top_n: int | None = None) -> list[dict[str, Any]]:
        return self.recommender.recommend(payload, top_n=top_n)


@lru_cache
def get_drug_response_service() -> DrugResponseService:
    return DrugResponseService(get_settings())

