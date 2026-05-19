"""
업로드 검증 + 파서 실행 + API 응답(ParseResponse) 변환.

흐름: validate_parse_request → parser.parse → to_response
에러 코드는 schemas.parser.ErrorItem.code (UNSUPPORTED_EXTENSION, EMPTY_RESULT 등).
"""
from app.parsers.base import ParseResult
from app.schemas.parser import PageResult, ParseResponse
from app.services import parser_registry
from app.utils.file_utils import get_extension, is_supported_extension
from app.utils.serialize_utils import make_json_safe


def _safe_pages(pages: list) -> list[PageResult]:
    safe: list[PageResult] = []
    for p in pages:
        safe.append(
            PageResult(
                page_no=p.page_no,
                text=p.text,
                blocks=make_json_safe(p.blocks),
            )
        )
    return safe


def to_response(result: ParseResult, extension: str) -> ParseResponse:
    text_length = len(result.text or "")
    error_count = len(result.errors)
    return ParseResponse(
        success=result.success and error_count == 0,
        parser_id=result.parser_id,
        file_name=result.file_name,
        extension=extension,
        elapsed_ms=result.elapsed_ms,
        page_count=result.page_count,
        text_length=text_length,
        table_count=len(result.tables),
        error_count=error_count,
        text=result.text,
        pages=_safe_pages(result.pages),
        tables=result.tables,
        logs=result.logs,
        errors=result.errors,
    )


def validate_parse_request(
    filename: str, parser_id: str | None, file_size: int
) -> ParseResponse | None:
    extension = get_extension(filename)

    if not is_supported_extension(extension):
        from app.schemas.parser import ErrorItem

        return ParseResponse(
            success=False,
            parser_id=parser_id or "",
            file_name=filename,
            extension=extension,
            elapsed_ms=0,
            errors=[
                ErrorItem(
                    code="UNSUPPORTED_EXTENSION",
                    message="지원하지 않는 파일 형식입니다.",
                )
            ],
        )

    from app.utils.file_utils import MAX_FILE_SIZE_BYTES

    if file_size > MAX_FILE_SIZE_BYTES:
        from app.schemas.parser import ErrorItem

        return ParseResponse(
            success=False,
            parser_id=parser_id or "",
            file_name=filename,
            extension=extension,
            elapsed_ms=0,
            errors=[
                ErrorItem(
                    code="FILE_TOO_LARGE",
                    message="파일 크기가 너무 큽니다.",
                )
            ],
        )

    if not parser_id:
        from app.schemas.parser import ErrorItem

        return ParseResponse(
            success=False,
            parser_id="",
            file_name=filename,
            extension=extension,
            elapsed_ms=0,
            errors=[
                ErrorItem(
                    code="PARSER_NOT_SELECTED",
                    message="실행할 파서를 선택해 주세요.",
                )
            ],
        )

    if not parser_registry.is_parser_available(parser_id, extension):
        from app.schemas.parser import ErrorItem

        return ParseResponse(
            success=False,
            parser_id=parser_id,
            file_name=filename,
            extension=extension,
            elapsed_ms=0,
            errors=[
                ErrorItem(
                    code="PARSER_NOT_AVAILABLE",
                    message="선택한 파서를 사용할 수 없습니다.",
                )
            ],
        )

    parser = parser_registry.get_parser(parser_id)
    if not parser:
        from app.schemas.parser import ErrorItem

        return ParseResponse(
            success=False,
            parser_id=parser_id,
            file_name=filename,
            extension=extension,
            elapsed_ms=0,
            errors=[
                ErrorItem(
                    code="PARSER_NOT_AVAILABLE",
                    message="선택한 파서를 사용할 수 없습니다.",
                )
            ],
        )

    return None


def run_parse(
    file_path: str,
    filename: str,
    parser_id: str,
    parse_options: dict | None = None,
) -> ParseResponse:
    extension = get_extension(filename)
    parser = parser_registry.get_parser(parser_id)
    assert parser is not None
    result = parser.parse(file_path, filename, parse_options)
    return to_response(result, extension)
