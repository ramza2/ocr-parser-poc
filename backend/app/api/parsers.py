"""GET /api/parsers?extension=pdf — 확장자별 파서 목록 (parser_registry)."""
from fastapi import APIRouter, Query

from app.services import parser_registry

router = APIRouter()


@router.get("/parsers")
def list_parsers(extension: str | None = Query(default=None)):
    if extension:
        parsers = parser_registry.get_parsers_for_extension(extension)
    else:
        parsers = parser_registry.list_all_parsers()
    return {"parsers": parsers}
