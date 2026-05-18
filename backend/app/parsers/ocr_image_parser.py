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
    "Tesseract OCR",
    "Tesseract 엔진. 전처리·후처리 파이프라인을 선택 적용할 수 있습니다.",
    "tesseract",
)

EasyOcrParser = _create_image_parser(
    "EASYOCR",
    "EasyOCR",
    "딥러닝 EasyOCR. 사진·간판·손글씨에 유리한 경우가 많습니다.",
    "easyocr",
)

PaddleOcrParser = _create_image_parser(
    "PADDLEOCR",
    "PaddleOCR",
    "PaddleOCR 한글 모델. 설치 용량이 큽니다.",
    "paddleocr",
)
