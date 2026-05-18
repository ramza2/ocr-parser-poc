from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api import health, parse, parsers, pipeline
from app.schemas.parser import ErrorItem, ParseResponse

app = FastAPI(title="OCR Parser PoC API", version="0.1.0")


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=200,
        content=ParseResponse(
            success=False,
            parser_id="",
            file_name="",
            extension="",
            elapsed_ms=0,
            error_count=1,
            errors=[
                ErrorItem(
                    code="INTERNAL_ERROR",
                    message="처리 중 알 수 없는 오류가 발생했습니다.",
                    detail=str(exc),
                )
            ],
        ).model_dump(),
    )

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:8080",
        "http://127.0.0.1:8080",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router, prefix="/api", tags=["health"])
app.include_router(parsers.router, prefix="/api", tags=["parsers"])
app.include_router(parse.router, prefix="/api", tags=["parse"])
app.include_router(pipeline.router, prefix="/api", tags=["pipeline"])
