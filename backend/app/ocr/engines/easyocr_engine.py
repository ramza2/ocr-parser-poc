from __future__ import annotations

from app.ocr.engines.base import OcrEngine
from app.utils.gpu_config import easyocr_use_gpu
from app.utils.serialize_utils import make_json_safe

_reader = None


def _get_reader():
    global _reader
    if _reader is None:
        import easyocr

        try:
            _reader = easyocr.Reader(["ko", "en"], gpu=easyocr_use_gpu(), verbose=False)
        except Exception as exc:
            detail = str(exc)
            if "shm.dll" in detail or "torch" in detail.lower():
                raise ImportError(
                    "PyTorch/EasyOCR 로드 실패. Visual C++ Redistributable 설치 후 "
                    "터미널을 재시작하거나 .venv를 다시 활성화해 주세요."
                ) from exc
            raise
    return _reader


class EasyOcrEngine(OcrEngine):
    engine_id = "easyocr"
    name = "EasyOCR"

    def recognize(self, image_path: str, options: dict | None = None) -> tuple[str, list[dict]]:
        opts = options or {}
        min_conf = float(opts.get("min_confidence", 0.25))

        reader = _get_reader()
        raw = reader.readtext(image_path)
        lines: list[str] = []
        blocks: list[dict] = []
        for bbox, text, conf in raw:
            if conf < min_conf:
                continue
            lines.append(text)
            blocks.append(
                {
                    "text": text,
                    "confidence": round(float(conf), 4),
                    "bbox": make_json_safe(bbox),
                }
            )

        return "\n".join(lines).strip(), blocks
