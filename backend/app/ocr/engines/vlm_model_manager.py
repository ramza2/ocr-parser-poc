"""
VLM 모델 싱글턴 매니저.

GPU 메모리 제약 때문에 한 번에 하나의 VLM 모델만 로드한다.
다른 모델로 전환 시 현재 모델을 해제하고 새 모델을 로드.
"""
from __future__ import annotations

import gc
import logging
from threading import Lock

logger = logging.getLogger(__name__)

_lock = Lock()
_current_engine_id: str | None = None


def _get_registry() -> dict:
    from app.ocr.engines.vlm_registry import ensure_vlm_registry
    return ensure_vlm_registry()


def get_current_model_id() -> str | None:
    return _current_engine_id


def switch_model(engine_id: str) -> None:
    """engine_id 모델로 전환. 이미 로드된 모델이면 아무것도 안 함."""
    global _current_engine_id

    with _lock:
        registry = _get_registry()
        if engine_id not in registry:
            raise ValueError(f"알 수 없는 VLM 엔진: {engine_id}")

        if _current_engine_id == engine_id:
            engine = registry[engine_id]
            if engine.is_loaded():
                return

        if _current_engine_id and _current_engine_id in registry:
            old = registry[_current_engine_id]
            if old.is_loaded():
                logger.info("VLM 모델 해제: %s", _current_engine_id)
                old.unload()
                _free_gpu_memory()

        target = registry[engine_id]
        logger.info("VLM 모델 로드: %s", engine_id)
        target.load()
        _current_engine_id = engine_id
        logger.info("VLM 모델 로드 완료: %s", engine_id)


def unload_current() -> None:
    """현재 로드된 모델을 해제."""
    global _current_engine_id
    with _lock:
        if _current_engine_id:
            registry = _get_registry()
            if _current_engine_id in registry:
                registry[_current_engine_id].unload()
            _current_engine_id = None
            _free_gpu_memory()


def _free_gpu_memory() -> None:
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
