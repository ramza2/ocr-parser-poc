"""Tesseract 엔진 — 시스템 tesseract 바이너리 + kor+eng (options.lang/psm/oem)."""
from __future__ import annotations

from app.ocr.engines.base import OcrEngine


class TesseractEngine(OcrEngine):
    engine_id = "tesseract"
    name = "Tesseract"

    def recognize(self, image_path: str, options: dict | None = None) -> tuple[str, list[dict]]:
        import pytesseract
        from PIL import Image

        opts = options or {}
        lang = opts.get("lang", "kor+eng")
        psm = int(opts.get("psm", 6))
        oem = int(opts.get("oem", 3))
        tess_cfg = f"--oem {oem} --psm {psm}"

        with Image.open(image_path) as img:
            if img.mode not in ("RGB", "L"):
                img = img.convert("RGB")
            text = pytesseract.image_to_string(img, lang=lang, config=tess_cfg)

        return text.strip(), [{"engine": self.engine_id, "lang": lang, "psm": psm}]
