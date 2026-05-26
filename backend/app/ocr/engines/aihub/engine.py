"""
AI Hub CRAFT + Swin-Transformer OCR 엔진 어댑터.

2단계 파이프라인:
  1) CRAFT — 이미지에서 단어 영역(바운딩 박스) 검출
  2) Swin-Transformer — 각 영역 크롭 → 문자 인식

모델 가중치 파일 3개가 필요:
  - craft.ckpt           (텍스트 검출)
  - swin_transformer.ckpt (문자 인식)
  - token.pkl            (Swin 토크나이저)

기본 탐색 경로: $AIHUB_MODEL_DIR (없으면 <backend>/models/aihub/).
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

from app.ocr.engines.base import OcrEngine
from app.utils.serialize_utils import make_json_safe

logger = logging.getLogger(__name__)

_craft = None
_swin = None
_device = None
_load_error: str | None = None

_CRAFT_CKPT = "craft.ckpt"
_SWIN_CKPT = "swin_transformer.ckpt"
_TOKEN_PKL = "token.pkl"


def _default_model_dir() -> Path:
    """프로젝트 models/aihub/ 또는 Docker /app/models/aihub/."""
    env = os.environ.get("AIHUB_MODEL_DIR", "").strip()
    if env:
        return Path(env)
    app_dir = Path(__file__).resolve().parents[4]  # backend/
    return app_dir / "models" / "aihub"


def _ensure_models():
    """싱글톤 — 첫 호출 시 CRAFT + Swin 모델 로드. 실패 시 _load_error 설정."""
    global _craft, _swin, _device, _load_error

    if _craft is not None and _swin is not None:
        return
    if _load_error is not None:
        raise ImportError(_load_error)

    try:
        import torch
    except ImportError:
        _load_error = "PyTorch 미설치. pip install torch torchvision"
        raise ImportError(_load_error)
    except OSError as exc:
        if "already registered" in str(exc).lower() or "shm.dll" in str(exc).lower():
            _load_error = (
                "Windows: PaddlePaddle이 먼저 로드되어 PyTorch를 사용할 수 없습니다 "
                "(pybind11 타입 충돌). 서버를 재시작한 뒤 AI Hub 엔진을 먼저 사용하거나, "
                "Docker(Linux)에서는 이 문제가 발생하지 않습니다."
            )
        else:
            _load_error = f"PyTorch 로드 실패: {exc}"
        raise ImportError(_load_error) from exc

    model_dir = _default_model_dir()
    craft_path = model_dir / _CRAFT_CKPT
    swin_path = model_dir / _SWIN_CKPT
    token_path = model_dir / _TOKEN_PKL

    missing = [p for p in (craft_path, swin_path, token_path) if not p.is_file()]
    if missing:
        names = ", ".join(p.name for p in missing)
        _load_error = (
            f"AI Hub 모델 파일 누락: {names}. "
            f"'{model_dir}' 디렉터리에 craft.ckpt, swin_transformer.ckpt, "
            f"token.pkl 파일을 배치하세요. (환경변수 AIHUB_MODEL_DIR 로 경로 변경 가능)"
        )
        raise ImportError(_load_error)

    _device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info("AI Hub OCR 모델 로드 시작 (device=%s, dir=%s)", _device, model_dir)

    from app.ocr.engines.aihub.craft_model import CRAFTModel
    from app.ocr.engines.aihub.swin_model import SwinTransformerOCR
    from app.ocr.engines.aihub.tokenizer import load_tokenizer

    try:
        craft_model = CRAFTModel(
            img_size=1536,
            threshold_character=0.6,
            threshold_affinity=0.3,
            threshold_word=0.7,
        ).to(_device)
        saved = torch.load(str(craft_path), map_location=_device, weights_only=False)
        craft_model.load_state_dict(saved["state_dict"])
        craft_model.eval()
        _craft = craft_model
        logger.info("CRAFT 검출 모델 로드 완료")
    except Exception as exc:
        err = f"CRAFT 모델 로드 실패: {exc}"
        _load_error = err
        raise ImportError(err) from exc

    try:
        tokenizer = load_tokenizer(str(token_path))
        swin_model = SwinTransformerOCR(tokenizer).to(_device)
        saved = torch.load(str(swin_path), map_location=_device, weights_only=False)
        swin_model.load_state_dict(saved["state_dict"])
        swin_model.eval()
        _swin = swin_model
        logger.info("Swin-Transformer 인식 모델 로드 완료 (vocab=%d)", len(tokenizer))
    except Exception as exc:
        _craft = None
        err = f"Swin-Transformer 모델 로드 실패: {exc}"
        _load_error = err
        raise ImportError(err) from exc


class AihubSwinEngine(OcrEngine):
    engine_id = "aihub_swin"
    name = "AI Hub (CRAFT + Swin)"

    def recognize(
        self, image_path: str, options: dict | None = None
    ) -> tuple[str, list[dict]]:
        import cv2
        from PIL import Image

        _ensure_models()

        np_image = cv2.imread(image_path)
        if np_image is None:
            raise ValueError(f"이미지를 열 수 없습니다: {image_path}")
        pil_image = Image.open(image_path).convert("RGB")

        boxes = _craft.predict(np_image)

        cropped_images = []
        valid_boxes = []
        for box in boxes:
            lx, ly, rx, ry = box
            if lx >= rx or ly >= ry:
                continue
            lx = max(0, lx)
            ly = max(0, ly)
            rx = min(pil_image.width, rx)
            ry = min(pil_image.height, ry)
            if lx >= rx or ly >= ry:
                continue
            cropped = pil_image.crop((lx, ly, rx, ry))
            cropped_images.append(cropped)
            valid_boxes.append([lx, ly, rx, ry])

        if not cropped_images:
            return "", []

        batch_size = int((options or {}).get("batch_size", 64))
        texts = _swin.predict(cropped_images, batch_size=batch_size)

        lines: list[str] = []
        blocks: list[dict] = []
        for box, text in zip(valid_boxes, texts):
            text = text.strip()
            if not text:
                continue
            lines.append(text)
            blocks.append({
                "text": text,
                "bbox": make_json_safe(box),
            })

        return "\n".join(lines).strip(), blocks
