"""GET /api/pipeline-steps?extension=png — 전·후처리 UI 옵션·프리셋."""
from fastapi import APIRouter, Query

from app.services.pipeline_catalog import (
    POSTPROCESS_CATALOG,
    PREPROCESS_CATALOG,
    PRESET_POSTPROCESS_ESSENTIAL,
    PRESET_SCANNED_DOC,
    file_kind_from_extension,
    filter_steps,
)

router = APIRouter()


@router.get("/pipeline-steps")
def get_pipeline_steps(extension: str | None = Query(default=None)):
    kind = file_kind_from_extension(extension or "png")
    return {
        "file_kind": kind,
        "preprocess": [s.model_dump() for s in filter_steps(PREPROCESS_CATALOG, kind)],
        "postprocess": [s.model_dump() for s in filter_steps(POSTPROCESS_CATALOG, kind)],
        "presets": {
            "scanned_doc": PRESET_SCANNED_DOC,
            "postprocess_essential": PRESET_POSTPROCESS_ESSENTIAL,
            "none": [],
        },
    }
