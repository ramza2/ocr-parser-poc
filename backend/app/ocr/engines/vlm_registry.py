"""
VLM 엔진 레지스트리.

- VLM_WORKER_URL 미설정: 로컬 GPU/CPU 엔진 (Qwen, GOT-OCR 등)
- VLM_WORKER_URL 설정: GPU PC VLM 워커 API 프록시

각 엔진 import 실패는 무시하여 의존성이 없는 환경에서도 서버 기동 가능.
"""
from __future__ import annotations

import logging
import os
from threading import Lock

from app.ocr.engines.vlm_base import VlmEngine

logger = logging.getLogger(__name__)

VLM_ENGINES: dict[str, VlmEngine] = {}
_init_lock = Lock()
_initialized = False


def _worker_url() -> str:
    return os.environ.get("VLM_WORKER_URL", "").strip().rstrip("/")


def _register_local_engines() -> None:
    try:
        from app.ocr.engines.vlm_qwen import QwenVlmEngine
        VLM_ENGINES["qwen_vl"] = QwenVlmEngine()
    except Exception as exc:
        logger.warning("Qwen2.5-VL 엔진 등록 실패: %s", exc)

    try:
        from app.ocr.engines.vlm_got import GotOcrEngine
        VLM_ENGINES["got_ocr"] = GotOcrEngine()
    except Exception as exc:
        logger.warning("GOT-OCR2.0 엔진 등록 실패: %s", exc)


def _register_remote_engines(worker_url: str) -> None:
    from app.ocr.engines.vlm_remote import fetch_remote_engines

    VLM_ENGINES.update(fetch_remote_engines(worker_url))


def ensure_vlm_registry(refresh: bool = False) -> dict[str, VlmEngine]:
    """레지스트리 초기화 (lazy). refresh=True 시 원격 목록 재조회."""
    global _initialized

    with _init_lock:
        if refresh:
            VLM_ENGINES.clear()
            _initialized = False

        if _initialized:
            return VLM_ENGINES

        worker = _worker_url()
        if worker:
            try:
                _register_remote_engines(worker)
            except Exception as exc:
                logger.error("VLM worker 연결 실패 (%s): %s", worker, exc)
        else:
            _register_local_engines()

        _initialized = True
        return VLM_ENGINES


def get_vlm_engine(engine_id: str) -> VlmEngine | None:
    return ensure_vlm_registry().get(engine_id)
