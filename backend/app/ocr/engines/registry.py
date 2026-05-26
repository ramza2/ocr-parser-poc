"""
OCR 엔진 레지스트리 (파서와 별개).

파서(parser_id)는 UI·권한·PDF 래핑 단위,
엔진(engine_id: tesseract | easyocr | paddleocr | aihub_swin)은 실제 추론 구현.
"""
from __future__ import annotations

import logging

from app.ocr.engines.base import OcrEngine
from app.ocr.engines.easyocr_engine import EasyOcrEngine
from app.ocr.engines.paddle_engine import PaddleOcrEngine
from app.ocr.engines.tesseract_engine import TesseractEngine

_logger = logging.getLogger(__name__)

ENGINES: dict[str, OcrEngine] = {
    "tesseract": TesseractEngine(),
    "easyocr": EasyOcrEngine(),
    "paddleocr": PaddleOcrEngine(),
}

try:
    from app.ocr.engines.aihub import AihubSwinEngine
    ENGINES["aihub_swin"] = AihubSwinEngine()
except Exception as _exc:
    _logger.warning("AI Hub 엔진 등록 실패 (의존성 누락 시 무시): %s", _exc)


def get_engine(engine_id: str) -> OcrEngine | None:
    return ENGINES.get(engine_id)
