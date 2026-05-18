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
            import sys

            ver = f"{sys.version_info.major}.{sys.version_info.minor}"
            raise ImportError(
                f"paddleocr 미설치 또는 미지원 Python({ver}). "
                "Python 3.10~3.12 가상환경에서 "
                "pip install -r requirements-paddle.txt 를 시도하세요. "
                "Python 3.14는 PaddlePaddle wheel이 없습니다."
            ) from exc

        if _ocr is None:
            try:
                _ocr = PaddleOCR(
                    use_angle_cls=True,
                    lang="korean",
                    use_gpu=False,
                )
            except TypeError:
                _ocr = PaddleOCR(use_angle_cls=True, lang="korean")

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
