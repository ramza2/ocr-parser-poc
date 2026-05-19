"""
스캔 PDF용 파서 — 텍스트 레이어 추출 없음.

pdf2image(convert_from_path) 로 페이지를 PNG 렌더한 뒤,
페이지마다 run_image_ocr() 호출. Poppler(pdftoppm) 필요.
"""
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
        logs = [log_item("INFO", f"스캔 PDF 페이지 OCR ({self._engine_id})")]

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
            logs.append(log_item("INFO", f"스캔 PDF {len(images)}페이지 렌더링 완료"))
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

        # 업로드 임시 파일명 기준 옆에 페이지 PNG 생성 (요청 종료 시 upload temp 만 삭제됨)
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

        all_logs.append(log_item("INFO", "스캔 PDF OCR 완료"))
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
    "Tesseract OCR (스캔 PDF)",
    "스캔 PDF 각 페이지를 이미지로 렌더링한 뒤 OCR합니다. (텍스트 레이어 PDF는 대상 아님)",
    "tesseract",
)

PdfEasyOcrParser = _create_pdf_ocr_parser(
    "PDF_EASYOCR",
    "EasyOCR (스캔 PDF)",
    "스캔 PDF 각 페이지를 이미지로 렌더링한 뒤 OCR합니다. (텍스트 레이어 PDF는 대상 아님)",
    "easyocr",
)

PdfPaddleOcrParser = _create_pdf_ocr_parser(
    "PDF_PADDLEOCR",
    "PaddleOCR (스캔 PDF)",
    "스캔 PDF 각 페이지를 이미지로 렌더링한 뒤 OCR합니다. (텍스트 레이어 PDF는 대상 아님)",
    "paddleocr",
)
