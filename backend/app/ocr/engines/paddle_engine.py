from __future__ import annotations

from app.ocr.engines.base import OcrEngine

_ocr = None


class PaddleOcrEngine(OcrEngine):
    engine_id = "paddleocr"
    name = "PaddleOCR"

    def recognize(self, image_path: str, options: dict | None = None) -> tuple[str, list[dict]]:
        global _ocr
        try:
            from paddleocr import PaddleOCR
        except ImportError as exc:
            raise ImportError(
                "paddleocr 패키지가 설치되지 않았습니다. pip install paddlepaddle paddleocr"
            ) from exc

        if _ocr is None:
            _ocr = PaddleOCR(
                use_angle_cls=True,
                lang="korean",
                show_log=False,
                use_gpu=False,
            )

        result = _ocr.ocr(image_path, cls=True)
        lines: list[str] = []
        blocks: list[dict] = []

        if not result or result[0] is None:
            return "", blocks

        for line in result[0]:
            bbox, (text, conf) = line
            lines.append(text)
            blocks.append(
                {
                    "text": text,
                    "confidence": round(float(conf), 4),
                    "bbox": bbox,
                }
            )

        return "\n".join(lines).strip(), blocks
