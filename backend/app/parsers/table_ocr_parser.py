import time
from pathlib import Path

import pdfplumber

from app.parsers.base import ParseResult, ParserAdapter
from app.schemas.parser import ErrorItem, PageResult, TableResult
from app.utils.log_utils import log_item


class TableOcrParser(ParserAdapter):
    parser_id = "TABLE_OCR"
    name = "표 OCR 테스트 파서"
    description = "표 영역 및 셀 데이터 추출 가능성을 검토합니다."
    supported_extensions = ["pdf", "jpg", "jpeg", "png", "tif", "tiff"]

    def parse(self, file_path: str, file_name: str, options: dict | None = None) -> ParseResult:
        start = time.perf_counter()
        logs = [log_item("INFO", "표 추출 테스트를 시작합니다.")]
        ext = Path(file_name).suffix.lower()

        if ext == ".pdf":
            return self._parse_pdf(file_path, file_name, start, logs)

        logs.append(log_item("WARN", "이미지 파일은 PDF 대비 표 추출 정확도가 제한됩니다."))
        from app.parsers.ocr_image_parser import EasyOcrParser

        ocr_result = EasyOcrParser.parse(file_path, file_name, options)
        ocr_result.parser_id = self.parser_id
        ocr_result.logs = logs + ocr_result.logs
        if ocr_result.success:
            ocr_result.logs.append(
                log_item("INFO", "이미지 표 OCR은 텍스트 추출 결과만 제공합니다.")
            )
        return ocr_result

    def _parse_pdf(
        self, file_path: str, file_name: str, start: float, logs: list
    ) -> ParseResult:
        pages: list[PageResult] = []
        tables: list[TableResult] = []
        table_idx = 0

        try:
            with pdfplumber.open(file_path) as pdf:
                for page_no, page in enumerate(pdf.pages, start=1):
                    page_text = (page.extract_text() or "").strip()
                    pages.append(PageResult(page_no=page_no, text=page_text, blocks=[]))

                    extracted = page.extract_tables() or []
                    for raw in extracted:
                        if not raw:
                            continue
                        rows = [
                            [str(cell or "").strip() for cell in row] for row in raw if row
                        ]
                        if not rows:
                            continue
                        table_idx += 1
                        tables.append(
                            TableResult(
                                table_id=f"table_{table_idx}",
                                page_no=page_no,
                                rows=rows,
                            )
                        )

            full_text = "\n\n".join(
                f"[페이지 {p.page_no}]\n{p.text}" for p in pages if p.text
            )
            elapsed = int((time.perf_counter() - start) * 1000)
            logs.append(log_item("INFO", f"표 {len(tables)}개를 감지했습니다."))

            if not tables:
                logs.append(log_item("WARN", "감지된 표가 없습니다."))

            return ParseResult(
                success=True,
                parser_id=self.parser_id,
                file_name=file_name,
                elapsed_ms=elapsed,
                page_count=len(pages),
                text=full_text,
                pages=pages,
                tables=tables,
                logs=logs,
            )
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
                        code="INTERNAL_ERROR",
                        message="처리 중 알 수 없는 오류가 발생했습니다.",
                        detail=str(exc),
                    )
                ],
            )
