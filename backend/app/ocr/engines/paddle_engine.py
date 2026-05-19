from __future__ import annotations

import os
import sys

os.environ.setdefault("FLAGS_use_mkldnn", "0")
os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

from app.ocr.engines.base import OcrEngine
from app.utils.gpu_config import paddle_use_gpu
from app.utils.serialize_utils import make_json_safe

_ocr = None
_ocr_use_gpu: bool | None = None
_ocr_major: int | None = None


def _paddle_import_error(exc: BaseException) -> ImportError:
    ver = f"{sys.version_info.major}.{sys.version_info.minor}"
    detail = str(exc).strip() or type(exc).__name__
    dl = detail.lower()
    if "libgl.so" in dl:
        hint = (
            "Docker: OpenCV/Paddle용 libGL 누락. "
            "docker compose -f docker-compose.gpu.yml build --no-cache backend"
        )
    elif "nccl" in dl or "libtorch_cuda" in dl:
        hint = (
            "Docker GPU: EasyOCR용 CUDA PyTorch가 Paddle과 충돌할 수 있습니다. "
            "backend 이미지를 재빌드하세요: "
            "docker compose -f docker-compose.gpu.yml build --no-cache backend"
        )
    else:
        hint = (
            "Windows: pip install -r requirements-paddle.txt (2.7). "
            "Linux/Docker GPU: docker compose -f docker-compose.gpu.yml up --build"
        )
    return ImportError(f"paddleocr 로드 실패 (Python {ver}): {detail}. {hint}")


def _paddle_major_version() -> int:
    import paddleocr

    try:
        return int(str(paddleocr.__version__).split(".")[0])
    except (ValueError, AttributeError):
        return 2


def _windows_blocks_v3() -> bool:
    """Windows 네이티브 pip 3.x는 oneDNN 이슈로 차단. Docker(Linux)는 허용."""
    if sys.platform != "win32":
        return False
    return os.environ.get("ALLOW_PADDLE_V3", "").strip().lower() not in (
        "1",
        "true",
        "yes",
    )


def _create_paddle_ocr_v2(*, use_gpu: bool):
    from paddleocr import PaddleOCR

    return PaddleOCR(
        use_angle_cls=True,
        lang="korean",
        show_log=False,
        use_gpu=use_gpu,
    )


def _create_paddle_ocr_v3(*, use_gpu: bool):
    from paddleocr import PaddleOCR

    device = "gpu:0" if use_gpu else "cpu"
    return PaddleOCR(
        lang="korean",
        device=device,
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
    )


def _is_cuda_dll_error(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return any(
        k in msg
        for k in ("cublas", "cudnn", "cuda", "dynamic library", "error code is 126")
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

    items = raw if isinstance(raw, list) else [raw]
    for item in items:
        data = getattr(item, "json", None)
        if callable(data):
            data = data()
        if data is None:
            data = item if isinstance(item, dict) else {}
        if not isinstance(data, dict):
            continue

        texts = data.get("rec_texts") or []
        scores = data.get("rec_scores") or []
        polys = data.get("rec_polys") or data.get("dt_polys") or []

        for i, text in enumerate(texts):
            if not text:
                continue
            lines.append(str(text))
            conf = scores[i] if i < len(scores) else None
            block: dict = {"text": str(text), "confidence": make_json_safe(conf)}
            if i < len(polys):
                block["bbox"] = make_json_safe(polys[i])
            blocks.append(block)

    return lines, blocks


class PaddleOcrEngine(OcrEngine):
    engine_id = "paddleocr"
    name = "PaddleOCR"

    def recognize(self, image_path: str, options: dict | None = None) -> tuple[str, list[dict]]:
        global _ocr, _ocr_use_gpu, _ocr_major
        try:
            import paddleocr
        except ImportError as exc:
            raise _paddle_import_error(exc) from exc
        except Exception as exc:
            raise _paddle_import_error(exc) from exc

        major = _paddle_major_version()

        if major >= 3 and _windows_blocks_v3():
            raise ImportError(
                f"PaddleOCR {paddleocr.__version__} (3.x)는 Windows pip에서 oneDNN 오류가 날 수 있습니다. "
                "Docker GPU 스택을 쓰세요: docker compose -f docker-compose.gpu.yml up --build"
            )

        if _ocr is None:
            use_gpu = paddle_use_gpu()
            if major >= 3:
                _ocr = _create_paddle_ocr_v3(use_gpu=use_gpu)
            else:
                _ocr = _create_paddle_ocr_v2(use_gpu=use_gpu)
            _ocr_use_gpu = use_gpu
            _ocr_major = major

        try:
            if _ocr_major and _ocr_major >= 3:
                raw = _ocr.predict(image_path)
                lines, blocks = _parse_v3_result(raw)
            else:
                raw = _ocr.ocr(image_path, cls=True)
                lines, blocks = _parse_v2_result(raw)
        except Exception as exc:
            if _ocr_major and _ocr_major < 3 and _ocr_use_gpu and _is_cuda_dll_error(exc):
                from app.utils.gpu_config import reset_paddle_gpu_cache

                reset_paddle_gpu_cache()
                _ocr = _create_paddle_ocr_v2(use_gpu=False)
                _ocr_use_gpu = False
                raw = _ocr.ocr(image_path, cls=True)
                lines, blocks = _parse_v2_result(raw)
            else:
                raise

        return "\n".join(lines).strip(), blocks
