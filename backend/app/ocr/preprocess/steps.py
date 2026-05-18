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


def step_resize(img: Image.Image, scale: float = 2.0, **_kwargs) -> Image.Image:
    import cv2

    bgr = _to_cv2(img)
    out = cv2.resize(bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    if out.ndim == 2:
        return _to_pil(out)
    import cv2 as cv

    return Image.fromarray(cv.cvtColor(out, cv.COLOR_BGR2RGB))


def step_grayscale(img: Image.Image, **_kwargs) -> Image.Image:
    import cv2

    bgr = _to_cv2(img)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) if bgr.ndim == 3 else bgr
    return _to_pil(gray)


def step_clahe(img: Image.Image, **_kwargs) -> Image.Image:
    import cv2

    gray = _to_cv2(img)
    if gray.ndim == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    return _to_pil(clahe.apply(gray))


def step_denoise(img: Image.Image, **_kwargs) -> Image.Image:
    import cv2

    gray = _to_cv2(img)
    if gray.ndim == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    return _to_pil(cv2.fastNlMeansDenoising(gray, h=8))


def step_invert_dark(img: Image.Image, threshold: float = 120, **_kwargs) -> Image.Image:
    gray = _to_cv2(img)
    if gray.ndim == 3:
        import cv2

        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    if float(np.mean(gray)) < threshold:
        gray = 255 - gray
    return _to_pil(gray)


def step_binary(img: Image.Image, **_kwargs) -> Image.Image:
    import cv2

    gray = _to_cv2(img)
    if gray.ndim == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    binary = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        15,
        9,
    )
    return _to_pil(binary)


def step_erosion(img: Image.Image, **_kwargs) -> Image.Image:
    import cv2

    gray = _to_cv2(img)
    if gray.ndim == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    kernel = np.ones((3, 3), np.uint8)
    return _to_pil(cv2.erode(gray, kernel, iterations=1))


def step_dilation(img: Image.Image, **_kwargs) -> Image.Image:
    import cv2

    gray = _to_cv2(img)
    if gray.ndim == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    kernel = np.ones((3, 3), np.uint8)
    return _to_pil(cv2.dilate(gray, kernel, iterations=1))


PREPROCESS_STEP_FUNCS: dict[str, Callable[..., Image.Image]] = {
    "resize": step_resize,
    "grayscale": step_grayscale,
    "clahe": step_clahe,
    "denoise": step_denoise,
    "invert_dark": step_invert_dark,
    "binary": step_binary,
    "erosion": step_erosion,
    "dilation": step_dilation,
}
