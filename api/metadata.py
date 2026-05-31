from fastapi import APIRouter, Depends, Query

from core.schemas import CANCER_CONTEXT_FIELDS, INPUT_FIELDS, ModelStatus, OptionResponse
from core.model_service import DrugResponseService, get_drug_response_service

router = APIRouter(prefix="/api", tags=["metadata"])


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/status", response_model=ModelStatus)
def status(service: DrugResponseService = Depends(get_drug_response_service)):
    return service.status()


@router.get("/options", response_model=OptionResponse)
def options(
    drug_name: str | None = Query(default=None, alias="DRUG_NAME"),
    target: str | None = Query(default=None, alias="TARGET"),
    target_pathway: str | None = Query(default=None, alias="TARGET_PATHWAY"),
    tcga_desc: str | None = Query(default=None, alias="TCGA_DESC"),
    tissue_2: str | None = Query(default=None, alias="GDSC_TISSUE_DESCRIPTOR_2"),
    tcga_label: str | None = Query(default=None, alias="CANCER_TYPE_MATCHING_TCGA_LABEL"),
    site: str | None = Query(default=None, alias="SITE"),
    tissue_1: str | None = Query(default=None, alias="GDSC_TISSUE_DESCRIPTOR_1"),
    service: DrugResponseService = Depends(get_drug_response_service),
):
    filters = {
        "DRUG_NAME": drug_name,
        "TARGET": target,
        "TARGET_PATHWAY": target_pathway,
        "TCGA_DESC": tcga_desc,
        "GDSC_TISSUE_DESCRIPTOR_2": tissue_2,
        "CANCER_TYPE_MATCHING_TCGA_LABEL": tcga_label,
        "SITE": site,
        "GDSC_TISSUE_DESCRIPTOR_1": tissue_1,
    }
    filters = {key: value for key, value in filters.items() if value is not None}
    return {"fields": service.options(filters), "field_order": INPUT_FIELDS}


@router.get("/context-options", response_model=OptionResponse)
def context_options(service: DrugResponseService = Depends(get_drug_response_service)):
    return {"fields": service.context_options(), "field_order": CANCER_CONTEXT_FIELDS}


@router.get("/sample-input")
def sample_input(service: DrugResponseService = Depends(get_drug_response_service)):
    return service.sample_input()

