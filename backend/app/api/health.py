from fastapi import APIRouter

from app.utils.gpu_config import gpu_runtime_status

router = APIRouter()


@router.get("/health")
def health_check():
    return {
        "status": "ok",
        "service": "ocr-parser-poc",
        "gpu": gpu_runtime_status(),
    }
