from __future__ import annotations

from collections.abc import Callable

import numpy as np
from PIL import Image


def _to_cv2(img: Image.Image) -> np.ndarray:
    import cv2

    arr = np.array(img)
    if arr.ndim == 2:
        return arr
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def _to_pil(gray: np.ndarray) -> Image.Image:
    return Image.fromarray(gray)


def _as_gray(bgr: np.ndarray) -> np.ndarray:
    import cv2

    if bgr.ndim == 2:
        return bgr
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)


def step_deskew(img: Image.Image, max_angle: float = 15.0, **_kwargs) -> Image.Image:
    """회전 보정: 기울어진 문서를 수평에 가깝게 맞춥니다."""
    import cv2

    gray = _as_gray(_to_cv2(img))
    inv = cv2.bitwise_not(gray)
    coords = np.column_stack(np.where(inv > 0))
    if len(coords) < 100:
        return img

    rect = cv2.minAreaRect(coords)
    angle = rect[-1]
    if angle < -45:
        angle = 90 + angle
    if abs(angle) < 0.5 or abs(angle) > max_angle:
        return img

    h, w = gray.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(
        _to_cv2(img),
        matrix,
        (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )
    if rotated.ndim == 2:
        return _to_pil(rotated)
    return Image.fromarray(cv2.cvtColor(rotated, cv2.COLOR_BGR2RGB))


def step_binarize(img: Image.Image, **_kwargs) -> Image.Image:
    """이진화: 배경 노이즈·흐린 글씨 대비 강화."""
    import cv2

    gray = _as_gray(_to_cv2(img))
    binary = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        21,
        10,
    )
    return _to_pil(binary)


def step_crop_roi(img: Image.Image, padding: int = 12, **_kwargs) -> Image.Image:
    """크롭: 텍스트 영역(ROI)만 잘라 처리 부담을 줄입니다."""
    import cv2

    bgr = _to_cv2(img)
    gray = _as_gray(bgr)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    coords = cv2.findNonZero(binary)
    if coords is None:
        return img

    x, y, w, h = cv2.boundingRect(coords)
    pad = max(0, padding)
    x0 = max(0, x - pad)
    y0 = max(0, y - pad)
    x1 = min(bgr.shape[1], x + w + pad)
    y1 = min(bgr.shape[0], y + h + pad)
    cropped = bgr[y0:y1, x0:x1]
    if cropped.ndim == 2:
        return _to_pil(cropped)
    return Image.fromarray(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))


def step_enhance(
    img: Image.Image,
    scale: float = 1.5,
    clip_limit: float = 2.5,
    sharpen: bool = True,
    **_kwargs,
) -> Image.Image:
    """명암비·해상도: CLAHE 대비 향상 + 선택적 확대·선명화."""
    import cv2

    bgr = _to_cv2(img)
    gray = _as_gray(bgr)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    if sharpen:
        blur = cv2.GaussianBlur(enhanced, (0, 0), 1.0)
        enhanced = cv2.addWeighted(enhanced, 1.4, blur, -0.4, 0)

    if scale and scale > 1.0:
        enhanced = cv2.resize(
            enhanced,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_CUBIC,
        )

    return _to_pil(enhanced)


PREPROCESS_STEP_FUNCS: dict[str, Callable[..., Image.Image]] = {
    "deskew": step_deskew,
    "binarize": step_binarize,
    "crop_roi": step_crop_roi,
    "enhance": step_enhance,
}
