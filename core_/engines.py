from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd


def _load_pickle(path: Path) -> Any:
    artifact = joblib.load(path)
    if artifact.__class__.__name__ == "SimpleImputer" and not hasattr(artifact, "_fill_dtype"):
        artifact._fill_dtype = getattr(artifact, "_fit_dtype", None)
    return artifact


class IC50RegressionEngine:
    def __init__(
        self,
        model_path: Path,
        features_path: Path,
        encoders_path: Path,
        auc_reference_path: Path,
    ):
        self.model = _load_pickle(model_path)
        self.features = _load_pickle(features_path)
        self.encoders = _load_pickle(encoders_path)
        self.auc_reference = pd.read_csv(auc_reference_path).fillna("")
        self.global_auc_mean = float(pd.to_numeric(self.auc_reference["AUC"], errors="coerce").mean())

    def _encode_value(self, column: str, value: Any) -> int:
        encoder = self.encoders[column]
        value = "" if value is None else str(value)

        if value not in encoder.classes_:
            raise ValueError(
                f"Unknown value for {column}: {value}. Use /api/options to choose supported values."
            )

        return int(encoder.transform([value])[0])

    def _matching_auc_rows(self, user_input: dict[str, Any]) -> pd.DataFrame:
        matched = self.auc_reference
        for column in [
            "DRUG_NAME",
            "TARGET",
            "TARGET_PATHWAY",
            "TCGA_DESC",
            "GDSC_TISSUE_DESCRIPTOR_2",
            "CANCER_TYPE_MATCHING_TCGA_LABEL",
            "SITE",
            "GDSC_TISSUE_DESCRIPTOR_1",
        ]:
            matched = matched[matched[column].astype(str) == str(user_input.get(column, ""))]
        return matched

    def _filter_auc_rows(self, filters: dict[str, Any]) -> pd.DataFrame:
        matched = self.auc_reference
        for column, value in filters.items():
            matched = matched[matched[column].astype(str) == str(value)]
        return matched

    def get_auc(self, user_input: dict[str, Any]) -> float:
        matched = self._matching_auc_rows(user_input)
        if not matched.empty:
            return float(pd.to_numeric(matched["AUC"], errors="coerce").mean())

        fallback_filters = [
            {
                "DRUG_NAME": user_input["DRUG_NAME"],
                "TCGA_DESC": user_input["TCGA_DESC"],
                "SITE": user_input["SITE"],
                "GDSC_TISSUE_DESCRIPTOR_1": user_input["GDSC_TISSUE_DESCRIPTOR_1"],
            },
            {
                "DRUG_NAME": user_input["DRUG_NAME"],
                "TCGA_DESC": user_input["TCGA_DESC"],
            },
            {
                "DRUG_NAME": user_input["DRUG_NAME"],
                "GDSC_TISSUE_DESCRIPTOR_2": user_input["GDSC_TISSUE_DESCRIPTOR_2"],
            },
            {
                "DRUG_NAME": user_input["DRUG_NAME"],
                "SITE": user_input["SITE"],
            },
            {
                "DRUG_NAME": user_input["DRUG_NAME"],
                "TARGET_PATHWAY": user_input["TARGET_PATHWAY"],
            },
            {
                "DRUG_NAME": user_input["DRUG_NAME"],
                "TARGET": user_input["TARGET"],
            },
            {
                "DRUG_NAME": user_input["DRUG_NAME"],
            },
        ]

        for filters in fallback_filters:
            fallback_match = self._filter_auc_rows(filters)
            if not fallback_match.empty:
                return float(pd.to_numeric(fallback_match["AUC"], errors="coerce").mean())

        if np.isnan(self.global_auc_mean):
            raise ValueError("Unable to derive an AUC value for this prediction request.")

        return self.global_auc_mean

    @staticmethod
    def classify_ln_ic50(predicted_ln_ic50: float) -> str:
        if predicted_ln_ic50 < -2:
            return "Very Sensitive"
        if predicted_ln_ic50 < 0:
            return "Sensitive"
        if predicted_ln_ic50 < 2:
            return "Moderate"
        return "Resistant"

    def predict_ic50(self, user_input: dict[str, Any]) -> dict[str, float | str]:
        auc_value = self.get_auc(user_input)

        row = {
            "AUC": auc_value,
            "DRUG_NAME_ENC": self._encode_value("DRUG_NAME", user_input["DRUG_NAME"]),
            "TARGET_ENC": self._encode_value("TARGET", user_input["TARGET"]),
            "TARGET_PATHWAY_ENC": self._encode_value("TARGET_PATHWAY", user_input["TARGET_PATHWAY"]),
            "TCGA_DESC_ENC": self._encode_value("TCGA_DESC", user_input["TCGA_DESC"]),
            "GDSC_TISSUE_DESCRIPTOR_2_ENC": self._encode_value(
                "GDSC_TISSUE_DESCRIPTOR_2", user_input["GDSC_TISSUE_DESCRIPTOR_2"]
            ),
            "CANCER_TYPE_MATCHING_TCGA_LABEL_ENC": self._encode_value(
                "CANCER_TYPE_MATCHING_TCGA_LABEL",
                user_input["CANCER_TYPE_MATCHING_TCGA_LABEL"],
            ),
            "SITE_ENC": self._encode_value("SITE", user_input["SITE"]),
            "GDSC_TISSUE_DESCRIPTOR_1_ENC": self._encode_value(
                "GDSC_TISSUE_DESCRIPTOR_1", user_input["GDSC_TISSUE_DESCRIPTOR_1"]
            ),
        }

        x = pd.DataFrame([row])[self.features]
        predicted_ln_ic50 = float(self.model.predict(x)[0])
        predicted_ic50 = float(np.exp(predicted_ln_ic50))

        return {
            "drug_name": user_input["DRUG_NAME"],
            "predicted_LN_IC50": predicted_ln_ic50,
            "predicted_IC50": predicted_ic50,
            "used_AUC": auc_value,
        }


class ClassificationModelEngine:
    def __init__(
        self,
        model_path: Path,
        features_path: Path,
        encoder_path: Path,
        scaler_path: Path | None = None,
        num_imputer_path: Path | None = None,
        cat_imputer_path: Path | None = None,
        fallback_label_encoders_path: Path | None = None,
    ):
        self.model = _load_pickle(model_path)
        self.features = _load_pickle(features_path)
        self.encoder = _load_pickle(encoder_path)
        self.fallback_label_encoders = (
            _load_pickle(fallback_label_encoders_path)
            if fallback_label_encoders_path and fallback_label_encoders_path.exists()
            else None
        )
        self.scaler = _load_pickle(scaler_path) if scaler_path and scaler_path.exists() else None
        self.num_imputer = _load_pickle(num_imputer_path) if num_imputer_path and num_imputer_path.exists() else None
        self.cat_imputer = _load_pickle(cat_imputer_path) if cat_imputer_path and cat_imputer_path.exists() else None

    @property
    def categorical_features(self) -> list[str]:
        if self.fallback_label_encoders:
            return [col for col in self.features if col in self.fallback_label_encoders]
        if hasattr(self.encoder, "feature_names_in_"):
            return [col for col in self.encoder.feature_names_in_ if col in self.features]
        return [col for col in self.features if col != "AUC"]

    def _apply_encoder(self, x: pd.DataFrame) -> pd.DataFrame:
        categorical = self.categorical_features
        if not categorical:
            return x

        x = x.copy()
        x[categorical] = x[categorical].astype(str).fillna("")

        if self.fallback_label_encoders:
            for column in categorical:
                encoder = self.fallback_label_encoders[column]
                values = []
                for value in x[column].astype(str):
                    if value not in encoder.classes_:
                        values.append(-1)
                    else:
                        values.append(int(encoder.transform([value])[0]))
                x[column] = values
            return x

        if getattr(self.encoder, "n_features_in_", None) == 0:
            return x

        x[categorical] = self.encoder.transform(x[categorical])
        return x

    def predict_sensitivity(self, model_input: dict[str, Any]) -> dict[str, Any]:
        x = pd.DataFrame([model_input])[self.features]
        x = self._apply_encoder(x)

        if self.num_imputer:
            x = pd.DataFrame(self.num_imputer.transform(x), columns=self.features)

        if self.cat_imputer and len(x.select_dtypes(include="object").columns) > 0:
            cat_cols = list(x.select_dtypes(include="object").columns)
            x[cat_cols] = self.cat_imputer.transform(x[cat_cols])

        if self.scaler:
            x = pd.DataFrame(self.scaler.transform(x), columns=self.features)

        pred = int(self.model.predict(x)[0])
        probability = None
        if hasattr(self.model, "predict_proba"):
            probability = float(self.model.predict_proba(x)[0][1])

        return {
            "drug_name": model_input.get("DRUG_NAME"),
            "prediction_label": pred,
            "prediction_text": "Sensitive" if pred == 1 else "Resistant",
            "sensitivity_probability": probability,
        }


@dataclass
class FinalRecommendationSystem:
    regression_engine: IC50RegressionEngine
    classification_engine: ClassificationModelEngine

    def _context_matches(self, user_input: dict[str, Any]) -> pd.DataFrame:
        auc_ref = self.regression_engine.auc_reference.copy()
        for column in [
            "TARGET",
            "TARGET_PATHWAY",
            "TCGA_DESC",
            "GDSC_TISSUE_DESCRIPTOR_2",
            "CANCER_TYPE_MATCHING_TCGA_LABEL",
            "SITE",
            "GDSC_TISSUE_DESCRIPTOR_1",
        ]:
            auc_ref = auc_ref[auc_ref[column].astype(str) == str(user_input.get(column, ""))]
        return auc_ref

    def _classify_input(self, user_input: dict[str, Any], auc_value: float) -> dict[str, Any]:
        return {
            "AUC": auc_value,
            "DRUG_NAME": user_input["DRUG_NAME"],
            "TARGET": user_input["TARGET"],
            "TARGET_PATHWAY": user_input["TARGET_PATHWAY"],
            "TCGA_DESC": user_input["TCGA_DESC"],
            "GDSC_TISSUE_DESCRIPTOR_2": user_input["GDSC_TISSUE_DESCRIPTOR_2"],
            "CANCER_TYPE_MATCHING_TCGA_LABEL": user_input["CANCER_TYPE_MATCHING_TCGA_LABEL"],
            "SITE": user_input["SITE"],
            "GDSC_TISSUE_DESCRIPTOR_1": user_input["GDSC_TISSUE_DESCRIPTOR_1"],
        }

    def predict_for_drug(self, user_input: dict[str, Any]) -> dict[str, Any]:
        reg_result = self.regression_engine.predict_ic50(user_input)
        cls_result = self.classification_engine.predict_sensitivity(
            self._classify_input(user_input, float(reg_result["used_AUC"]))
        )
        derived_sensitivity = self.regression_engine.classify_ln_ic50(float(reg_result["predicted_LN_IC50"]))

        return {
            "DRUG_NAME": user_input["DRUG_NAME"],
            "Predicted_LN_IC50": reg_result["predicted_LN_IC50"],
            "Predicted_IC50": reg_result["predicted_IC50"],
            "Sensitivity": derived_sensitivity,
            "Sensitivity_Probability": cls_result["sensitivity_probability"],
            "Used_AUC": reg_result["used_AUC"],
        }

    def recommend(self, user_input: dict[str, Any], top_n: int | None = None) -> list[dict[str, Any]]:
        matched = self._context_matches(user_input)
        recommendations: list[dict[str, Any]] = []

        for drug_name in matched["DRUG_NAME"].drop_duplicates():
            temp_input = dict(user_input)
            temp_input["DRUG_NAME"] = drug_name
            try:
                recommendations.append(self.predict_for_drug(temp_input))
            except ValueError:
                continue

        if not recommendations:
            return []

        rec_df = pd.DataFrame(recommendations)
        rec_df["Sensitivity_Rank"] = rec_df["Sensitivity"].map(
            {"Very Sensitive": 0, "Sensitive": 1, "Moderate": 2, "Resistant": 3}
        ).fillna(4)
        rec_df = rec_df.sort_values(
            by=["Sensitivity_Rank", "Predicted_IC50", "Sensitivity_Probability"],
            ascending=[True, True, False],
        ).drop_duplicates("DRUG_NAME")
        rec_df = rec_df.drop(columns=["Sensitivity_Rank"])

        if top_n:
            rec_df = rec_df.head(top_n)

        return rec_df.to_dict(orient="records")

