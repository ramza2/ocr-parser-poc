from __future__ import annotations

import os
import time

from app.ocr.engines.registry import get_engine
from app.ocr.postprocess.pipeline import apply_postprocess
from app.ocr.preprocess.pipeline import preprocess_to_temp_file
from app.parsers.base import ParseResult
from app.schemas.parser import ErrorItem, PageResult
from app.utils.log_utils import log_item


def _parse_options(options: dict | None) -> tuple[list[str], list[str], dict]:
    opts = options or {}
    preprocess = opts.get("preprocess_steps") or []
    postprocess = opts.get("postprocess_steps") or []
    ocr_opts = opts.get("ocr_options") or {}
    if isinstance(preprocess, str):
        import json

        preprocess = json.loads(preprocess)
    if isinstance(postprocess, str):
        import json

        postprocess = json.loads(postprocess)
    return list(preprocess), list(postprocess), ocr_opts


def run_image_ocr(
    image_path: str,
    file_name: str,
    parser_id: str,
    engine_id: str,
    options: dict | None = None,
) -> ParseResult:
    start = time.perf_counter()
    preprocess_steps, postprocess_steps, ocr_opts = _parse_options(options)
    logs = [
        log_item("INFO", f"OCR 엔진: {engine_id}"),
        log_item(
            "INFO",
            f"전처리: {preprocess_steps if preprocess_steps else '(없음)'}",
        ),
        log_item(
            "INFO",
            f"후처리: {postprocess_steps if postprocess_steps else '(없음)'}",
        ),
    ]

    engine = get_engine(engine_id)
    if not engine:
        return ParseResult(
            success=False,
            parser_id=parser_id,
            file_name=file_name,
            elapsed_ms=0,
            logs=logs,
            errors=[
                ErrorItem(
                    code="OCR_FAILED",
                    message="OCR 처리 중 오류가 발생했습니다.",
                    detail=f"알 수 없는 엔진: {engine_id}",
                )
            ],
        )

    temp_pre: str | None = None
    try:
        work_path, pre_logs = preprocess_to_temp_file(
            image_path, preprocess_steps, options
        )
        logs.extend(pre_logs)
        if work_path != image_path:
            temp_pre = work_path

        text, blocks = engine.recognize(work_path, ocr_opts)
        text, post_logs = apply_postprocess(text, postprocess_steps, options)
        logs.extend(post_logs)

        elapsed = int((time.perf_counter() - start) * 1000)
        pages = [PageResult(page_no=1, text=text, blocks=blocks)]
        formatted = f"[페이지 1]\n{text}" if text else ""

        if not text.strip():
            return ParseResult(
                success=False,
                parser_id=parser_id,
                file_name=file_name,
                elapsed_ms=elapsed,
                page_count=1,
                pages=pages,
                logs=logs,
                errors=[
                    ErrorItem(
                        code="EMPTY_RESULT",
                        message="추출된 텍스트가 없습니다.",
                    )
                ],
            )

        logs.append(log_item("INFO", "OCR 처리 완료"))
        return ParseResult(
            success=True,
            parser_id=parser_id,
            file_name=file_name,
            elapsed_ms=elapsed,
            page_count=1,
            text=formatted,
            pages=pages,
            logs=logs,
        )
    except ImportError as exc:
        elapsed = int((time.perf_counter() - start) * 1000)
        return ParseResult(
            success=False,
            parser_id=parser_id,
            file_name=file_name,
            elapsed_ms=elapsed,
            logs=logs + [log_item("ERROR", str(exc))],
            errors=[
                ErrorItem(
                    code="OCR_FAILED",
                    message="OCR 처리 중 오류가 발생했습니다.",
                    detail=str(exc),
                )
            ],
        )
    except Exception as exc:
        elapsed = int((time.perf_counter() - start) * 1000)
        detail = str(exc)
        if "tesseract" in detail.lower():
            detail = "Tesseract가 PATH에 없거나 언어팩이 없을 수 있습니다."
        return ParseResult(
            success=False,
            parser_id=parser_id,
            file_name=file_name,
            elapsed_ms=elapsed,
            logs=logs + [log_item("ERROR", detail)],
            errors=[
                ErrorItem(
                    code="OCR_FAILED",
                    message="OCR 처리 중 오류가 발생했습니다.",
                    detail=detail,
                )
            ],
        )
    finally:
        if temp_pre and os.path.isfile(temp_pre):
            try:
                os.remove(temp_pre)
            except OSError:
                pass
