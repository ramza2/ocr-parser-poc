"""
VLM 엔진 레지스트리.

VLM 엔진들은 기존 OCR 엔진 레지스트리(registry.py)와 별도로 관리.
각 엔진의 import 실패는 무시하여 의존성이 없는 환경에서도 서버 기동 가능.
"""
from __future__ import annotations

import logging

from app.ocr.engines.vlm_base import VlmEngine

logger = logging.getLogger(__name__)

VLM_ENGINES: dict[str, VlmEngine] = {}

# Qwen2.5-VL
try:
    from app.ocr.engines.vlm_qwen import QwenVlmEngine
    VLM_ENGINES["qwen_vl"] = QwenVlmEngine()
except Exception as exc:
    logger.warning("Qwen2.5-VL 엔진 등록 실패: %s", exc)

# GOT-OCR2.0
try:
    from app.ocr.engines.vlm_got import GotOcrEngine
    VLM_ENGINES["got_ocr"] = GotOcrEngine()
except Exception as exc:
    logger.warning("GOT-OCR2.0 엔진 등록 실패: %s", exc)

# Florence-2
try:
    from app.ocr.engines.vlm_florence import FlorenceVlmEngine
    VLM_ENGINES["florence"] = FlorenceVlmEngine()
except Exception as exc:
    logger.warning("Florence-2 엔진 등록 실패: %s", exc)


def get_vlm_engine(engine_id: str) -> VlmEngine | None:
    return VLM_ENGINES.get(engine_id)
