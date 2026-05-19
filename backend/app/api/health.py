"""GET /api/health — Docker healthcheck·프론트 API 온라인 확인용."""
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
