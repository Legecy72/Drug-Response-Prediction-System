from typing import Any

from pydantic import BaseModel, Field, field_validator


INPUT_FIELDS = [
    "DRUG_NAME",
    "TARGET",
    "TARGET_PATHWAY",
    "TCGA_DESC",
    "GDSC_TISSUE_DESCRIPTOR_2",
    "CANCER_TYPE_MATCHING_TCGA_LABEL",
    "SITE",
    "GDSC_TISSUE_DESCRIPTOR_1",
]

CANCER_CONTEXT_FIELDS = [field for field in INPUT_FIELDS if field != "DRUG_NAME"]


class CancerContext(BaseModel):
    TARGET: str
    TARGET_PATHWAY: str
    TCGA_DESC: str
    GDSC_TISSUE_DESCRIPTOR_2: str
    CANCER_TYPE_MATCHING_TCGA_LABEL: str = ""
    SITE: str
    GDSC_TISSUE_DESCRIPTOR_1: str

    @field_validator("*", mode="before")
    @classmethod
    def normalize_values(cls, value: Any) -> str:
        if value is None:
            return ""
        return str(value).strip()


class SinglePredictionRequest(CancerContext):
    DRUG_NAME: str


class RecommendationRequest(CancerContext):
    top_n: int | None = Field(
        default=None,
        ge=1,
        le=1000,
        description="Leave empty to return all matched drugs.",
    )


class PredictionResult(BaseModel):
    DRUG_NAME: str
    Predicted_LN_IC50: float
    Predicted_IC50: float
    Sensitivity: str
    Sensitivity_Probability: float | None = None
    Used_AUC: float


class RecommendationResponse(BaseModel):
    count: int
    recommendations: list[PredictionResult]


class OptionResponse(BaseModel):
    fields: dict[str, list[str]]
    field_order: list[str]


class ModelStatus(BaseModel):
    regression_model: str
    classification_model: str
    regression_features: list[str]
    classification_features: list[str]
    auc_reference_rows: int
    available_drugs: int

