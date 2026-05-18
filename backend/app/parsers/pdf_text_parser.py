import time

from pypdf import PdfReader

from app.parsers.base import ParseResult, ParserAdapter
from app.schemas.parser import PageResult
from app.utils.log_utils import log_item


class PdfTextParser(ParserAdapter):
    parser_id = "PDF_TEXT"
    name = "PDF 텍스트 추출 파서"
    description = "PDF 내부 텍스트 레이어를 추출합니다. 스캔 PDF에서는 결과가 없을 수 있습니다."
    supported_extensions = ["pdf"]

    def parse(self, file_path: str, file_name: str, options: dict | None = None) -> ParseResult:
        start = time.perf_counter()
        logs = [log_item("INFO", "PDF 텍스트 추출을 시작합니다.")]
        pages: list[PageResult] = []

        try:
            reader = PdfReader(file_path)
            for idx, page in enumerate(reader.pages, start=1):
                text = (page.extract_text() or "").strip()
                pages.append(PageResult(page_no=idx, text=text, blocks=[]))
                logs.append(log_item("INFO", f"페이지 {idx} 텍스트 추출 완료"))

            full_text = "\n\n".join(
                f"[페이지 {p.page_no}]\n{p.text}" for p in pages if p.text
            )
            elapsed = int((time.perf_counter() - start) * 1000)
            logs.append(log_item("INFO", "PDF 텍스트 추출이 완료되었습니다."))

            if not full_text.strip():
                logs.append(log_item("WARN", "추출된 텍스트가 없습니다. 스캔 PDF일 수 있습니다."))

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
            from app.schemas.parser import ErrorItem

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
