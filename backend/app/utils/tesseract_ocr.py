from __future__ import annotations

from dataclasses import dataclass

from PIL import Image

from app.utils.image_preprocess import preprocess_for_tesseract


@dataclass
class TesseractOcrConfig:
    lang: str = "kor+eng"
    psm: int = 11
    oem: int = 3
    preprocess: bool = True


def run_tesseract_ocr(
    image_path: str,
    config: TesseractOcrConfig | None = None,
) -> tuple[str, str]:
    """
    Returns (text, settings_description for logs).
  """
    import pytesseract

    cfg = config or TesseractOcrConfig()

    if cfg.preprocess:
        image = preprocess_for_tesseract(image_path)
        source = "전처리+Tesseract"
    else:
        with Image.open(image_path) as img:
            image = img.convert("RGB") if img.mode not in ("RGB", "L") else img
        source = "Tesseract(원본)"

    tess_cfg = f"--oem {cfg.oem} --psm {cfg.psm}"
    text = pytesseract.image_to_string(image, lang=cfg.lang, config=tess_cfg)
    settings = f"{source} lang={cfg.lang} {tess_cfg}"
    return text.strip(), settings
