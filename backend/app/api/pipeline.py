from fastapi import APIRouter, Query

from app.services.pipeline_catalog import (
    POSTPROCESS_CATALOG,
    PREPROCESS_CATALOG,
    PRESET_FULL_TUTORIAL,
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
            "tutorial_full": PRESET_FULL_TUTORIAL,
            "none": [],
        },
    }
