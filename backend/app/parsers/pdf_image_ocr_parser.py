import time

from app.parsers.base import ParseResult, ParserAdapter
from app.parsers.image_ocr_parser import _run_ocr
from app.schemas.parser import ErrorItem, PageResult
from app.utils.log_utils import log_item


class PdfImageOcrParser(ParserAdapter):
    parser_id = "PDF_IMAGE_OCR"
    name = "PDF 이미지 OCR 파서"
    description = "PDF 페이지를 이미지로 변환한 뒤 OCR을 수행합니다."
    supported_extensions = ["pdf"]

    def parse(self, file_path: str, file_name: str, options: dict | None = None) -> ParseResult:
        start = time.perf_counter()
        logs = [log_item("INFO", "PDF 이미지 OCR을 시작합니다.")]
        pages: list[PageResult] = []

        try:
            from pdf2image import convert_from_path

            images = convert_from_path(file_path, dpi=200)
            logs.append(log_item("INFO", f"PDF를 {len(images)}개 이미지로 변환했습니다."))

            for idx, image in enumerate(images, start=1):
                temp_path = f"{file_path}_page_{idx}.png"
                image.save(temp_path, "PNG")
                try:
                    text = _run_ocr(temp_path).strip()
                finally:
                    import os

                    if os.path.isfile(temp_path):
                        os.remove(temp_path)

                pages.append(PageResult(page_no=idx, text=text, blocks=[]))
                logs.append(log_item("INFO", f"페이지 {idx} OCR 완료"))

            full_text = "\n\n".join(
                f"[페이지 {p.page_no}]\n{p.text}" for p in pages if p.text
            )
            elapsed = int((time.perf_counter() - start) * 1000)
            logs.append(log_item("INFO", "PDF 이미지 OCR이 완료되었습니다."))

            if not full_text.strip():
                return ParseResult(
                    success=False,
                    parser_id=self.parser_id,
                    file_name=file_name,
                    elapsed_ms=elapsed,
                    page_count=len(pages),
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
                page_count=len(pages),
                text=full_text,
                pages=pages,
                logs=logs,
            )
        except Exception as exc:
            elapsed = int((time.perf_counter() - start) * 1000)
            detail = str(exc)
            if "poppler" in detail.lower() or "Unable to get page count" in detail:
                code = "PDF_CONVERT_FAILED"
                message = "PDF 이미지 변환에 실패했습니다."
                detail = "Poppler가 설치되어 있지 않을 수 있습니다."
            elif "tesseract" in detail.lower():
                code = "OCR_FAILED"
                message = "OCR 처리 중 오류가 발생했습니다."
                detail = "Tesseract OCR 엔진이 필요합니다."
            else:
                code = "PDF_CONVERT_FAILED"
                message = "PDF 이미지 변환에 실패했습니다."

            return ParseResult(
                success=False,
                parser_id=self.parser_id,
                file_name=file_name,
                elapsed_ms=elapsed,
                logs=logs + [log_item("ERROR", detail)],
                errors=[ErrorItem(code=code, message=message, detail=detail)],
            )
