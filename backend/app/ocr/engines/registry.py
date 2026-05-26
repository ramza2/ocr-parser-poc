"""
OCR 엔진 레지스트리 (파서와 별개).

파서(parser_id)는 UI·권한·PDF 래핑 단위,
엔진(engine_id: tesseract | easyocr | paddleocr | aihub_swin)은 실제 추론 구현.
"""
from __future__ import annotations

from app.ocr.engines.base import OcrEngine
from app.ocr.engines.easyocr_engine import EasyOcrEngine
from app.ocr.engines.paddle_engine import PaddleOcrEngine
from app.ocr.engines.tesseract_engine import TesseractEngine
from app.ocr.engines.aihub import AihubSwinEngine

ENGINES: dict[str, OcrEngine] = {
    "tesseract": TesseractEngine(),
    "easyocr": EasyOcrEngine(),
    "paddleocr": PaddleOcrEngine(),
    "aihub_swin": AihubSwinEngine(),
}


def get_engine(engine_id: str) -> OcrEngine | None:
    return ENGINES.get(engine_id)
