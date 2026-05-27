"""
VLM 전용 API 라우터.

엔드포인트:
  GET  /api/vlm/models           — 사용 가능 VLM 모델 목록 + 현재 로드 상태
  POST /api/vlm/load             — 모델 로드/전환
  POST /api/vlm/ocr              — 전체 텍스트 추출
  POST /api/vlm/extract          — Schema 기반 구조화 추출
  POST /api/vlm/ask              — 문서 Q&A
"""
from __future__ import annotations

import json
import logging

from fastapi import APIRouter, File, Form, UploadFile

from app.schemas.vlm import (
    QaResponse,
    SchemaExtractResponse,
    SchemaField,
    VlmModelInfo,
    VlmModelsResponse,
    VlmOcrResponse,
)
from app.utils.file_utils import remove_file, save_upload_to_temp

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/vlm")


def _get_registry():
    from app.ocr.engines.vlm_registry import VLM_ENGINES
    return VLM_ENGINES


def _get_manager():
    from app.ocr.engines import vlm_model_manager as mgr
    return mgr


# ── 모델 목록 ─────────────────────────────────────────

@router.get("/models")
async def list_models() -> VlmModelsResponse:
    registry = _get_registry()
    mgr = _get_manager()
    models = [
        VlmModelInfo(
            model_id=eng.engine_id,
            name=eng.name,
            description=f"HuggingFace: {eng.model_id} (약 {eng.vram_gb}GB VRAM)",
            vram_gb=eng.vram_gb,
            loaded=eng.is_loaded(),
        )
        for eng in registry.values()
    ]
    return VlmModelsResponse(
        models=models,
        current_model=mgr.get_current_model_id(),
    )


# ── 모델 로드 ─────────────────────────────────────────

@router.post("/load")
async def load_model(model_id: str = Form(...)):
    mgr = _get_manager()
    try:
        mgr.switch_model(model_id)
        return {"success": True, "model_id": model_id}
    except Exception as exc:
        logger.exception("모델 로드 실패: %s", model_id)
        return {"success": False, "error": str(exc)}


# ── OCR ───────────────────────────────────────────────

@router.post("/ocr")
async def vlm_ocr(
    file: UploadFile = File(...),
    model_id: str = Form(...),
) -> VlmOcrResponse:
    registry = _get_registry()
    mgr = _get_manager()

    if model_id not in registry:
        return VlmOcrResponse(
            success=False, model_id=model_id,
            error=f"알 수 없는 모델: {model_id}",
        )

    try:
        mgr.switch_model(model_id)
    except Exception as exc:
        logger.exception("모델 로드 실패: %s", model_id)
        return VlmOcrResponse(
            success=False, model_id=model_id,
            error=f"모델 로드 실패: {exc}",
        )

    engine = registry[model_id]
    content = await file.read()
    tmp = save_upload_to_temp(content, file.filename or "upload.png")
    try:
        return engine.ocr(tmp)
    finally:
        remove_file(tmp)


# ── Schema 추출 ───────────────────────────────────────

@router.post("/extract")
async def vlm_extract(
    file: UploadFile = File(...),
    model_id: str = Form(...),
    schema_fields_json: str = Form(...),
) -> SchemaExtractResponse:
    registry = _get_registry()
    mgr = _get_manager()

    if model_id not in registry:
        return SchemaExtractResponse(
            success=False, model_id=model_id,
            error=f"알 수 없는 모델: {model_id}",
        )

    try:
        raw = json.loads(schema_fields_json)
        schema = [SchemaField(**f) for f in raw]
    except Exception as exc:
        return SchemaExtractResponse(
            success=False, model_id=model_id,
            error=f"Schema 파싱 오류: {exc}",
        )

    try:
        mgr.switch_model(model_id)
    except Exception as exc:
        logger.exception("모델 로드 실패: %s", model_id)
        return SchemaExtractResponse(
            success=False, model_id=model_id,
            error=f"모델 로드 실패: {exc}",
        )

    engine = registry[model_id]
    content = await file.read()
    tmp = save_upload_to_temp(content, file.filename or "upload.png")
    try:
        return engine.extract_schema(tmp, schema)
    finally:
        remove_file(tmp)


# ── Q&A ───────────────────────────────────────────────

@router.post("/ask")
async def vlm_ask(
    file: UploadFile = File(...),
    model_id: str = Form(...),
    question: str = Form(...),
) -> QaResponse:
    registry = _get_registry()
    mgr = _get_manager()

    if model_id not in registry:
        return QaResponse(
            success=False, model_id=model_id,
            error=f"알 수 없는 모델: {model_id}",
        )

    try:
        mgr.switch_model(model_id)
    except Exception as exc:
        logger.exception("모델 로드 실패: %s", model_id)
        return QaResponse(
            success=False, model_id=model_id,
            error=f"모델 로드 실패: {exc}",
        )

    engine = registry[model_id]
    content = await file.read()
    tmp = save_upload_to_temp(content, file.filename or "upload.png")
    try:
        return engine.ask(tmp, question)
    finally:
        remove_file(tmp)
