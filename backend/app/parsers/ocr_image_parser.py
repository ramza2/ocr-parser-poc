"""
이미지 파일용 파서 — 엔진 ID만 다르고 처리는 ocr_pipeline.run_image_ocr 공통.

_create_image_parser 로 Tesseract/Easy/Paddle 파서 클래스를 동적 생성한다.
"""
from app.parsers.base import ParseResult, ParserAdapter
from app.services.ocr_pipeline import run_image_ocr


class _ImageOcrParserBase(ParserAdapter):
    _engine_id: str

    def parse(
        self, file_path: str, file_name: str, options: dict | None = None
    ) -> ParseResult:
        return run_image_ocr(
            file_path,
            file_name,
            self.parser_id,
            self._engine_id,
            options,
        )


def _create_image_parser(
    parser_id: str,
    name: str,
    description: str,
    engine_id: str,
) -> ParserAdapter:
    return type(
        f"_{parser_id}",
        (_ImageOcrParserBase,),
        {
            "parser_id": parser_id,
            "name": name,
            "description": description,
            "supported_extensions": ["jpg", "jpeg", "png", "tif", "tiff"],
            "_engine_id": engine_id,
        },
    )()


TesseractOcrParser = _create_image_parser(
    "TESSERACT_OCR",
    "Tesseract OCR (이미지)",
    "이미지 파일에서 문자를 인식합니다. 전·후처리 옵션 적용 가능.",
    "tesseract",
)

EasyOcrParser = _create_image_parser(
    "EASYOCR",
    "EasyOCR (이미지)",
    "이미지 파일에서 문자를 인식합니다. 사진·간판 등에 유리한 경우가 많습니다.",
    "easyocr",
)

PaddleOcrParser = _create_image_parser(
    "PADDLEOCR",
    "PaddleOCR (이미지)",
    "이미지 파일에서 문자를 인식합니다.",
    "paddleocr",
)
