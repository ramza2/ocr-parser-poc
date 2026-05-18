from __future__ import annotations

_reader = None


def get_easyocr_reader():
    global _reader
    if _reader is None:
        import easyocr

        _reader = easyocr.Reader(["ko", "en"], gpu=False, verbose=False)
    return _reader


def run_easyocr(image_path: str, min_confidence: float = 0.25) -> tuple[str, list[dict]]:
    reader = get_easyocr_reader()
    raw = reader.readtext(image_path)

    lines: list[str] = []
    blocks: list[dict] = []
    for bbox, text, conf in raw:
        if conf < min_confidence:
            continue
        lines.append(text)
        blocks.append({"text": text, "confidence": round(float(conf), 4), "bbox": bbox})

    return "\n".join(lines).strip(), blocks
