"""GET /api/health — Docker healthcheck·프론트 API 온라인 확인용."""
from fastapi import APIRouter

from app.utils.gpu_config import gpu_runtime_status

router = APIRouter()


@router.get("/health")
def health_check():
    from app.ocr.engines.vlm_registry import _worker_url

    worker = _worker_url()
    return {
        "status": "ok",
        "service": "ocr-parser-poc",
        "gpu": gpu_runtime_status(),
        "vlm_mode": "remote" if worker else "local",
        "vlm_worker_url": worker or None,
    }
