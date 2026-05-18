import os
import time

from app.parsers.base import ParseResult, ParserAdapter
from app.schemas.parser import ErrorItem, PageResult
from app.services.ocr_pipeline import run_image_ocr
from app.utils.log_utils import log_item


class _PdfOcrParserBase(ParserAdapter):
    _engine_id: str

    def parse(
        self, file_path: str, file_name: str, options: dict | None = None
    ) -> ParseResult:
        start = time.perf_counter()
        logs = [log_item("INFO", f"PDF → 이미지 변환 후 {self._engine_id} OCR")]

        try:
            from pdf2image import convert_from_path
        except ImportError as exc:
            return ParseResult(
                success=False,
                parser_id=self.parser_id,
                file_name=file_name,
                elapsed_ms=0,
                logs=logs,
                errors=[
                    ErrorItem(
                        code="PDF_CONVERT_FAILED",
                        message="PDF 이미지 변환에 실패했습니다.",
                        detail=str(exc),
                    )
                ],
            )

        try:
            images = convert_from_path(file_path, dpi=200)
            logs.append(log_item("INFO", f"{len(images)}페이지 변환 완료"))
        except Exception as exc:
            elapsed = int((time.perf_counter() - start) * 1000)
            return ParseResult(
                success=False,
                parser_id=self.parser_id,
                file_name=file_name,
                elapsed_ms=elapsed,
                logs=logs + [log_item("ERROR", str(exc))],
                errors=[
                    ErrorItem(
                        code="PDF_CONVERT_FAILED",
                        message="PDF 이미지 변환에 실패했습니다.",
                        detail="Poppler(pdftoppm) 설치 및 PATH 확인이 필요할 수 있습니다.",
                    )
                ],
            )

        all_pages: list[PageResult] = []
        all_logs = list(logs)
        combined: list[str] = []

        for idx, image in enumerate(images, start=1):
            temp_path = f"{file_path}_page_{idx}.png"
            image.save(temp_path, "PNG")
            try:
                page_result = run_image_ocr(
                    temp_path,
                    file_name,
                    self.parser_id,
                    self._engine_id,
                    options,
                )
                all_logs.extend(page_result.logs)
                if page_result.errors:
                    elapsed = int((time.perf_counter() - start) * 1000)
                    return ParseResult(
                        success=False,
                        parser_id=self.parser_id,
                        file_name=file_name,
                        elapsed_ms=elapsed,
                        page_count=len(images),
                        logs=all_logs,
                        errors=page_result.errors,
                    )
                page_text = page_result.pages[0].text if page_result.pages else ""
                all_pages.append(
                    PageResult(page_no=idx, text=page_text, blocks=[])
                )
                if page_text:
                    combined.append(f"[페이지 {idx}]\n{page_text}")
            finally:
                if os.path.isfile(temp_path):
                    os.remove(temp_path)

        full_text = "\n\n".join(combined)
        elapsed = int((time.perf_counter() - start) * 1000)

        if not full_text.strip():
            return ParseResult(
                success=False,
                parser_id=self.parser_id,
                file_name=file_name,
                elapsed_ms=elapsed,
                page_count=len(images),
                pages=all_pages,
                logs=all_logs,
                errors=[
                    ErrorItem(
                        code="EMPTY_RESULT",
                        message="추출된 텍스트가 없습니다.",
                    )
                ],
            )

        all_logs.append(log_item("INFO", "PDF OCR 완료"))
        return ParseResult(
            success=True,
            parser_id=self.parser_id,
            file_name=file_name,
            elapsed_ms=elapsed,
            page_count=len(images),
            text=full_text,
            pages=all_pages,
            logs=all_logs,
        )


def _create_pdf_ocr_parser(
    parser_id: str,
    name: str,
    description: str,
    engine_id: str,
) -> ParserAdapter:
    return type(
        f"_{parser_id}",
        (_PdfOcrParserBase,),
        {
            "parser_id": parser_id,
            "name": name,
            "description": description,
            "supported_extensions": ["pdf"],
            "_engine_id": engine_id,
        },
    )()


PdfTesseractOcrParser = _create_pdf_ocr_parser(
    "PDF_TESSERACT_OCR",
    "PDF + Tesseract OCR",
    "PDF 페이지를 이미지로 변환한 뒤 Tesseract OCR을 수행합니다.",
    "tesseract",
)

PdfEasyOcrParser = _create_pdf_ocr_parser(
    "PDF_EASYOCR",
    "PDF + EasyOCR",
    "PDF 페이지를 이미지로 변환한 뒤 EasyOCR을 수행합니다.",
    "easyocr",
)

PdfPaddleOcrParser = _create_pdf_ocr_parser(
    "PDF_PADDLEOCR",
    "PDF + PaddleOCR",
    "PDF 페이지를 이미지로 변환한 뒤 PaddleOCR을 수행합니다.",
    "paddleocr",
)
