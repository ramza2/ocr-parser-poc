"""
파서 계층 공통 타입.

ParserAdapter: 파일 경로를 받아 ParseResult 반환 (텍스트·페이지·로그·에러).
API 레이어는 ParseResult → ParseResponse 로 직렬화(parse_service.to_response).
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from app.schemas.parser import ErrorItem, LogItem, PageResult, TableResult


@dataclass
class ParseResult:
    success: bool
    parser_id: str
    file_name: str
    elapsed_ms: int = 0
    page_count: int | None = None
    text: str = ""
    pages: list[PageResult] = field(default_factory=list)
    tables: list[TableResult] = field(default_factory=list)
    logs: list[LogItem] = field(default_factory=list)
    errors: list[ErrorItem] = field(default_factory=list)


class ParserAdapter(ABC):
    parser_id: str
    name: str
    description: str
    supported_extensions: list[str]

    @abstractmethod
    def parse(self, file_path: str, file_name: str, options: dict | None = None) -> ParseResult:
        ...
