from typing import Any

from pydantic import BaseModel, Field


class ParserInfo(BaseModel):
    parser_id: str
    name: str
    description: str
    supported_extensions: list[str]


class LogItem(BaseModel):
    level: str
    message: str
    timestamp: str | None = None


class ErrorItem(BaseModel):
    code: str
    message: str
    detail: str | None = None


class PageResult(BaseModel):
    page_no: int
    text: str
    blocks: list[dict[str, Any]] = Field(default_factory=list)


class TableResult(BaseModel):
    table_id: str
    page_no: int
    rows: list[list[str]]


class ParseResponse(BaseModel):
    success: bool
    parser_id: str
    file_name: str
    extension: str
    elapsed_ms: int
    page_count: int | None = None
    text_length: int = 0
    table_count: int = 0
    error_count: int = 0
    text: str = ""
    pages: list[PageResult] = Field(default_factory=list)
    tables: list[TableResult] = Field(default_factory=list)
    logs: list[LogItem] = Field(default_factory=list)
    errors: list[ErrorItem] = Field(default_factory=list)
