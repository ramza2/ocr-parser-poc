"""
VLM 엔진 전용 요청·응답 스키마.

- VlmOcrResponse: 전체 텍스트 추출 (기존 OCR 호환 + confidence/bbox 추가)
- SchemaExtractResponse: Schema 기반 Key-Value 구조화 추출
- QaResponse: 문서 Q&A
"""
from __future__ import annotations

from pydantic import BaseModel, Field


# ── 공통 ──────────────────────────────────────────────

class BoundingBox(BaseModel):
    x: float
    y: float
    width: float
    height: float


# ── OCR ───────────────────────────────────────────────

class VlmOcrItem(BaseModel):
    text: str
    confidence: float | None = None
    bbox: BoundingBox | None = None


class VlmOcrResponse(BaseModel):
    success: bool = True
    model_id: str
    elapsed_ms: int = 0
    items: list[VlmOcrItem] = Field(default_factory=list)
    full_text: str = ""
    error: str | None = None
    prompt_mode: str | None = None  # auto | bbox | custom
    prompt_label: str | None = None  # 실제 사용 단계 (spotting, custom 등)
    raw_response_preview: str | None = None  # 모델 원본 응답 앞부분 (디버그)


# ── Schema 추출 ───────────────────────────────────────

class SchemaField(BaseModel):
    key: str
    description: str = ""
    type: str = "text"


class SchemaExtractItem(BaseModel):
    key: str
    value: str
    confidence: float | None = None
    bbox: BoundingBox | None = None


class SchemaExtractRequest(BaseModel):
    schema_fields: list[SchemaField]


class SchemaExtractResponse(BaseModel):
    success: bool = True
    model_id: str
    elapsed_ms: int = 0
    items: list[SchemaExtractItem] = Field(default_factory=list)
    error: str | None = None


# ── Q&A ───────────────────────────────────────────────

class QaResponse(BaseModel):
    success: bool = True
    model_id: str
    elapsed_ms: int = 0
    answer: str = ""
    confidence: float | None = None
    error: str | None = None


# ── 모델 목록 ─────────────────────────────────────────

class VlmModelInfo(BaseModel):
    model_id: str
    name: str
    description: str
    vram_gb: float
    loaded: bool = False


class VlmModelsResponse(BaseModel):
    models: list[VlmModelInfo] = Field(default_factory=list)
    current_model: str | None = None
