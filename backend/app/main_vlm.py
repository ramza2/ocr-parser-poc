"""
VLM 전용 워커 FastAPI 진입점 (GPU PC).

/api/vlm/* — VLM 추론
/api/health — 헬스체크
"""
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import vlm
from app.utils.gpu_config import gpu_runtime_status

app = FastAPI(title="VLM Worker API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health_check():
    return {
        "status": "ok",
        "service": "vlm-worker",
        "gpu": gpu_runtime_status(),
    }


app.include_router(vlm.router, prefix="/api", tags=["vlm"])
