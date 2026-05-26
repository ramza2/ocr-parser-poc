"""
VLM 엔진 공통 인터페이스.

OcrEngine 을 확장하여 Schema 추출·Q&A 기능을 추가한다.
각 VLM 구현체(Qwen, GOT-OCR, Florence)는 이 클래스를 상속.
"""
from __future__ import annotations

from abc import abstractmethod

from app.ocr.engines.base import OcrEngine
from app.schemas.vlm import (
    QaResponse,
    SchemaExtractResponse,
    SchemaField,
    VlmOcrResponse,
)


class VlmEngine(OcrEngine):
    """VLM 엔진 공통 베이스 — 세 가지 모드를 모두 지원."""

    model_id: str  # HuggingFace repo or unique ID
    vram_gb: float = 0.0  # 예상 VRAM 사용량 (4-bit 기준)

    # ── 기존 OCR 호환 ─────────────────────────────────
    def recognize(
        self, image_path: str, options: dict | None = None
    ) -> tuple[str, list[dict]]:
        """기존 OCR 파이프라인 호환. full_text + 빈 블록 반환."""
        resp = self.ocr(image_path, options)
        blocks = [
            {"text": item.text, "confidence": item.confidence}
            for item in resp.items
        ]
        return resp.full_text, blocks

    # ── VLM 전용 메서드 ───────────────────────────────
    @abstractmethod
    def ocr(
        self, image_path: str, options: dict | None = None
    ) -> VlmOcrResponse:
        """이미지에서 전체 텍스트를 추출."""

    @abstractmethod
    def extract_schema(
        self,
        image_path: str,
        schema: list[SchemaField],
        options: dict | None = None,
    ) -> SchemaExtractResponse:
        """사용자 정의 Schema 에 따라 Key-Value 추출."""

    @abstractmethod
    def ask(
        self,
        image_path: str,
        question: str,
        options: dict | None = None,
    ) -> QaResponse:
        """이미지에 대한 자유 질의응답."""

    @abstractmethod
    def is_loaded(self) -> bool:
        """모델이 GPU/CPU 에 로드되어 있는지."""

    @abstractmethod
    def load(self) -> None:
        """모델을 메모리에 로드."""

    @abstractmethod
    def unload(self) -> None:
        """모델을 메모리에서 해제."""
