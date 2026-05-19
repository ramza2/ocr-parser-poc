"""
파서 ID ↔ 구현체 매핑 (UI·API 의 단일 진실 공급원).

- 이미지: TESSERACT_OCR / EASYOCR / PADDLEOCR → ocr_image_parser (엔진만 다름)
- 스캔 PDF: PDF_* → pdf_ocr_parser (페이지별 이미지 렌더 후 동일 OCR 파이프라인)
- EXTENSION_PARSER_MAP: 파일 확장자에 따라 UI에 노출할 파서 ID 목록

새 파서 추가 시: ParserAdapter 구현 → PARSERS 등록 → EXTENSION_PARSER_MAP 에 ID 추가.
"""
from app.parsers.base import ParserAdapter
from app.parsers.ocr_image_parser import EasyOcrParser, PaddleOcrParser, TesseractOcrParser
from app.parsers.pdf_ocr_parser import (
    PdfEasyOcrParser,
    PdfPaddleOcrParser,
    PdfTesseractOcrParser,
)
from app.schemas.parser import ParserInfo

# parser_id → ParserAdapter 인스턴스(또는 동적 생성 클래스)
PARSERS: dict[str, ParserAdapter] = {
    "TESSERACT_OCR": TesseractOcrParser,
    "EASYOCR": EasyOcrParser,
    "PADDLEOCR": PaddleOcrParser,
    "PDF_TESSERACT_OCR": PdfTesseractOcrParser,
    "PDF_EASYOCR": PdfEasyOcrParser,
    "PDF_PADDLEOCR": PdfPaddleOcrParser,
}

# 확장자별 UI 노출 순서 (docx 등 문서 파서 없음 — OCR 이미지·스캔 PDF 전용)
EXTENSION_PARSER_MAP: dict[str, list[str]] = {
    "pdf": [
        "PDF_TESSERACT_OCR",
        "PDF_EASYOCR",
        "PDF_PADDLEOCR",
    ],
    "jpg": ["TESSERACT_OCR", "EASYOCR", "PADDLEOCR"],
    "jpeg": ["TESSERACT_OCR", "EASYOCR", "PADDLEOCR"],
    "png": ["TESSERACT_OCR", "EASYOCR", "PADDLEOCR"],
    "tif": ["TESSERACT_OCR", "EASYOCR", "PADDLEOCR"],
    "tiff": ["TESSERACT_OCR", "EASYOCR", "PADDLEOCR"],
}


def list_all_parsers() -> list[ParserInfo]:
    return [
        ParserInfo(
            parser_id=p.parser_id,
            name=p.name,
            description=p.description,
            supported_extensions=p.supported_extensions,
        )
        for p in PARSERS.values()
    ]


def get_parsers_for_extension(extension: str) -> list[ParserInfo]:
    ext = extension.lower().lstrip(".")
    ids = EXTENSION_PARSER_MAP.get(ext, [])
    return [
        ParserInfo(
            parser_id=PARSERS[pid].parser_id,
            name=PARSERS[pid].name,
            description=PARSERS[pid].description,
            supported_extensions=PARSERS[pid].supported_extensions,
        )
        for pid in ids
        if pid in PARSERS
    ]


def get_parser(parser_id: str) -> ParserAdapter | None:
    return PARSERS.get(parser_id)


def is_parser_available(parser_id: str, extension: str) -> bool:
    ext = extension.lower().lstrip(".")
    allowed = EXTENSION_PARSER_MAP.get(ext, [])
    return parser_id in allowed
