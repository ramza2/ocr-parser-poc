from __future__ import annotations

import os

# Paddle 3.x Windows oneDNN 오류 완화 시도 (2.7 설치가 더 안정적)
os.environ.setdefault("FLAGS_use_mkldnn", "0")
os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

from app.ocr.engines.base import OcrEngine
from app.utils.serialize_utils import make_json_safe

_ocr = None


def _paddle_major_version() -> int:
    import paddleocr

    try:
        return int(str(paddleocr.__version__).split(".")[0])
    except (ValueError, AttributeError):
        return 2


def _create_paddle_ocr_v2():
    from paddleocr import PaddleOCR

    return PaddleOCR(use_angle_cls=True, lang="korean", show_log=False)


def _create_paddle_ocr_v3():
    from paddleocr import PaddleOCR

    return PaddleOCR(
        lang="korean",
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
    )


def _parse_v2_result(result) -> tuple[list[str], list[dict]]:
    lines: list[str] = []
    blocks: list[dict] = []
    if not result or result[0] is None:
        return lines, blocks
    for line in result[0]:
        bbox, (text, conf) = line
        lines.append(text)
        blocks.append(
            {
                "text": text,
                "confidence": round(float(conf), 4),
                "bbox": make_json_safe(bbox),
            }
        )
    return lines, blocks


def _parse_v3_result(raw) -> tuple[list[str], list[dict]]:
    lines: list[str] = []
    blocks: list[dict] = []
    if raw is None:
        return lines, blocks
    for item in raw:
        data = getattr(item, "json", item)
        if isinstance(data, dict) and "rec_texts" in data:
            for i, text in enumerate(data["rec_texts"]):
                if text:
                    lines.append(str(text))
                    scores = data.get("rec_scores") or []
                    conf = scores[i] if i < len(scores) else None
                    blocks.append({"text": str(text), "confidence": make_json_safe(conf)})
    return lines, blocks


class PaddleOcrEngine(OcrEngine):
    engine_id = "paddleocr"
    name = "PaddleOCR"

    def recognize(self, image_path: str, options: dict | None = None) -> tuple[str, list[dict]]:
        global _ocr
        try:
            import paddleocr
        except ImportError as exc:
            import sys

            ver = f"{sys.version_info.major}.{sys.version_info.minor}"
            raise ImportError(
                f"paddleocr 미설치 (Python {ver}). pip install -r requirements-paddle.txt"
            ) from exc

        major = _paddle_major_version()

        if major >= 3:
            raise ImportError(
                f"PaddleOCR {paddleocr.__version__} (3.x)는 이 Windows 환경에서 oneDNN 오류가 납니다. "
                "다음으로 2.7을 설치하세요: pip uninstall paddlepaddle paddleocr paddlex -y && "
                "pip install -r requirements-paddle.txt"
            )

        if _ocr is None:
            _ocr = _create_paddle_ocr_v2()

        raw = _ocr.ocr(image_path, cls=True)
        lines, blocks = _parse_v2_result(raw)
        return "\n".join(lines).strip(), blocks
