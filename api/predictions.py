from fastapi import APIRouter, Depends, HTTPException

from core.schemas import (
    PredictionResult,
    RecommendationRequest,
    RecommendationResponse,
    SinglePredictionRequest,
)
from core.model_service import DrugResponseService, get_drug_response_service

router = APIRouter(prefix="/api", tags=["predictions"])


@router.post("/predict", response_model=PredictionResult)
def predict(
    request: SinglePredictionRequest,
    service: DrugResponseService = Depends(get_drug_response_service),
):
    try:
        return service.predict(request.model_dump())
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.post("/recommend", response_model=RecommendationResponse)
def recommend(
    request: RecommendationRequest,
    service: DrugResponseService = Depends(get_drug_response_service),
):
    payload = request.model_dump(exclude={"top_n"})
    try:
        recommendations = service.recommend(payload, top_n=request.top_n)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    if not recommendations:
        raise HTTPException(
            status_code=404,
            detail="No drugs matched this cancer context in drug_auc_reference.csv.",
        )

    return {"count": len(recommendations), "recommendations": recommendations}

