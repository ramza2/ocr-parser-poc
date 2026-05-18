from __future__ import annotations

from app.ocr.engines.base import OcrEngine

_reader = None


class EasyOcrEngine(OcrEngine):
    engine_id = "easyocr"
    name = "EasyOCR"

    def recognize(self, image_path: str, options: dict | None = None) -> tuple[str, list[dict]]:
        global _reader
        import easyocr

        opts = options or {}
        min_conf = float(opts.get("min_confidence", 0.25))

        if _reader is None:
            _reader = easyocr.Reader(["ko", "en"], gpu=False, verbose=False)

        raw = _reader.readtext(image_path)
        lines: list[str] = []
        blocks: list[dict] = []
        for bbox, text, conf in raw:
            if conf < min_conf:
                continue
            lines.append(text)
            blocks.append(
                {"text": text, "confidence": round(float(conf), 4), "bbox": bbox}
            )

        return "\n".join(lines).strip(), blocks
