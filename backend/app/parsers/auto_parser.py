from app.parsers.base import ParseResult, ParserAdapter
from app.parsers.image_ocr_parser import ImageOcrParser
from app.parsers.pdf_image_ocr_parser import PdfImageOcrParser
from app.parsers.pdf_text_parser import PdfTextParser
from app.utils.log_utils import log_item


class AutoParser(ParserAdapter):
    parser_id = "AUTO"
    name = "자동 선택 파서"
    description = "확장자와 파일 특성에 따라 기본 파서를 자동 실행합니다."
    supported_extensions = ["pdf", "jpg", "jpeg", "png", "tif", "tiff"]

    def parse(self, file_path: str, file_name: str, options: dict | None = None) -> ParseResult:
        ext = file_name.rsplit(".", 1)[-1].lower()
        logs = [log_item("INFO", "AUTO 파서가 파일 유형을 분석합니다.")]

        if ext == "pdf":
            text_result = PdfTextParser().parse(file_path, file_name, options)
            if text_result.text.strip():
                text_result.parser_id = self.parser_id
                text_result.logs = logs + text_result.logs + [
                    log_item("INFO", "PDF_TEXT 결과를 사용합니다.")
                ]
                return text_result

            logs.append(log_item("INFO", "텍스트 레이어가 없어 PDF_IMAGE_OCR로 전환합니다."))
            ocr_result = PdfImageOcrParser().parse(file_path, file_name, options)
            ocr_result.parser_id = self.parser_id
            ocr_result.logs = logs + ocr_result.logs
            return ocr_result

        image_result = ImageOcrParser().parse(file_path, file_name, options)
        image_result.parser_id = self.parser_id
        image_result.logs = logs + image_result.logs + [
            log_item("INFO", "이미지 파일에 IMAGE_OCR을 적용했습니다.")
        ]
        return image_result
