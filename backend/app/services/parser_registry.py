from app.parsers.auto_parser import AutoParser
from app.parsers.base import ParserAdapter
from app.parsers.image_ocr_parser import ImageOcrParser
from app.parsers.pdf_image_ocr_parser import PdfImageOcrParser
from app.parsers.pdf_text_parser import PdfTextParser
from app.parsers.table_ocr_parser import TableOcrParser
from app.schemas.parser import ParserInfo

PARSERS: dict[str, ParserAdapter] = {
    "PDF_TEXT": PdfTextParser(),
    "PDF_IMAGE_OCR": PdfImageOcrParser(),
    "IMAGE_OCR": ImageOcrParser(),
    "TABLE_OCR": TableOcrParser(),
    "AUTO": AutoParser(),
}

EXTENSION_PARSER_MAP: dict[str, list[str]] = {
    "pdf": ["PDF_TEXT", "PDF_IMAGE_OCR", "TABLE_OCR", "AUTO"],
    "jpg": ["IMAGE_OCR", "TABLE_OCR", "AUTO"],
    "jpeg": ["IMAGE_OCR", "TABLE_OCR", "AUTO"],
    "png": ["IMAGE_OCR", "TABLE_OCR", "AUTO"],
    "tif": ["IMAGE_OCR", "TABLE_OCR", "AUTO"],
    "tiff": ["IMAGE_OCR", "TABLE_OCR", "AUTO"],
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
