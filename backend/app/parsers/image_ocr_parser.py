import time

from PIL import Image

from app.parsers.base import ParseResult, ParserAdapter
from app.schemas.parser import ErrorItem, PageResult
from app.utils.log_utils import log_item


def _run_ocr(image_path: str) -> str:
    import pytesseract

    with Image.open(image_path) as img:
        if img.mode not in ("RGB", "L"):
            img = img.convert("RGB")
        return pytesseract.image_to_string(img, lang="kor+eng")


class ImageOcrParser(ParserAdapter):
    parser_id = "IMAGE_OCR"
    name = "이미지 OCR 파서"
    description = "이미지 파일에서 텍스트를 추출합니다."
    supported_extensions = ["jpg", "jpeg", "png", "tif", "tiff"]

    def parse(self, file_path: str, file_name: str, options: dict | None = None) -> ParseResult:
        start = time.perf_counter()
        logs = [log_item("INFO", "이미지 OCR을 시작합니다.")]

        try:
            text = _run_ocr(file_path).strip()
            elapsed = int((time.perf_counter() - start) * 1000)
            logs.append(log_item("INFO", "이미지 OCR이 완료되었습니다."))

            pages = [PageResult(page_no=1, text=text, blocks=[])]
            formatted = f"[페이지 1]\n{text}" if text else ""

            if not text:
                logs.append(log_item("WARN", "추출된 텍스트가 없습니다."))
                return ParseResult(
                    success=False,
                    parser_id=self.parser_id,
                    file_name=file_name,
                    elapsed_ms=elapsed,
                    page_count=1,
                    text="",
                    pages=pages,
                    logs=logs,
                    errors=[
                        ErrorItem(
                            code="EMPTY_RESULT",
                            message="추출된 텍스트가 없습니다.",
                        )
                    ],
                )

            return ParseResult(
                success=True,
                parser_id=self.parser_id,
                file_name=file_name,
                elapsed_ms=elapsed,
                page_count=1,
                text=formatted,
                pages=pages,
                logs=logs,
            )
        except Exception as exc:
            elapsed = int((time.perf_counter() - start) * 1000)
            detail = str(exc)
            if "tesseract" in detail.lower() or "TesseractNotFoundError" in type(exc).__name__:
                code, message = "OCR_FAILED", "OCR 처리 중 오류가 발생했습니다."
                detail = "Tesseract OCR 엔진이 설치되어 있지 않거나 PATH에 없습니다."
            else:
                code, message = "OCR_FAILED", "OCR 처리 중 오류가 발생했습니다."

            return ParseResult(
                success=False,
                parser_id=self.parser_id,
                file_name=file_name,
                elapsed_ms=elapsed,
                logs=logs + [log_item("ERROR", detail)],
                errors=[ErrorItem(code=code, message=message, detail=detail)],
            )
