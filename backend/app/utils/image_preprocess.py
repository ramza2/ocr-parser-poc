"""Tesseract용 이미지 전처리 (칠판·사진·저해상도 대응)."""

from __future__ import annotations

import numpy as np
from PIL import Image


def preprocess_for_tesseract(image_path: str, min_side: int = 1600) -> Image.Image:
    import cv2

    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"이미지를 읽을 수 없습니다: {image_path}")

    h, w = img.shape[:2]
    short = min(h, w)
    if short < min_side:
        scale = min_side / short
        img = cv2.resize(
            img,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_CUBIC,
        )

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 어두운 배경(칠판·야간 사진) → 흰 배경/검은 글자로 반전
    if float(np.mean(gray)) < 120:
        gray = cv2.bitwise_not(gray)

    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    gray = cv2.fastNlMeansDenoising(gray, h=8)

    binary = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        12,
    )

    return Image.fromarray(binary)
