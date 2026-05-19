"""전처리 단계를 순서대로 적용. step_id → PREPROCESS_STEP_FUNCS (steps.py)."""
from __future__ import annotations

import tempfile
from pathlib import Path

from PIL import Image

from app.ocr.preprocess.steps import PREPROCESS_STEP_FUNCS
from app.utils.log_utils import log_item


def apply_preprocess(
    image_path: str,
    step_ids: list[str],
    options: dict | None = None,
) -> tuple[Image.Image, list]:
    opts = options or {}
    logs = []
    with Image.open(image_path) as img:
        current = img.convert("RGB")

    for step_id in step_ids:
        fn = PREPROCESS_STEP_FUNCS.get(step_id)
        if not fn:
            logs.append(log_item("WARN", f"알 수 없는 전처리 단계 무시: {step_id}"))
            continue
        current = fn(current, **opts.get(step_id, {}))
        logs.append(log_item("INFO", f"전처리 적용: {step_id}"))

    return current, logs


def preprocess_to_temp_file(
    image_path: str,
    step_ids: list[str],
    options: dict | None = None,
) -> tuple[str, list]:
    if not step_ids:
        return image_path, []

    image, logs = apply_preprocess(image_path, step_ids, options)
    suffix = Path(image_path).suffix or ".png"
    fd, temp_path = tempfile.mkstemp(suffix=suffix, prefix="ocr_pre_")
    import os

    os.close(fd)
    image.save(temp_path)
    return temp_path, logs
