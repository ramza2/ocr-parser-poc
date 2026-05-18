from __future__ import annotations

from app.ocr.engines.base import OcrEngine
from app.ocr.engines.easyocr_engine import EasyOcrEngine
from app.ocr.engines.paddle_engine import PaddleOcrEngine
from app.ocr.engines.tesseract_engine import TesseractEngine

ENGINES: dict[str, OcrEngine] = {
    "tesseract": TesseractEngine(),
    "easyocr": EasyOcrEngine(),
    "paddleocr": PaddleOcrEngine(),
}


def get_engine(engine_id: str) -> OcrEngine | None:
    return ENGINES.get(engine_id)
