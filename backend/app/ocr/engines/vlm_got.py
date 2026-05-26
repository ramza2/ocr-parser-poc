"""
GOT-OCR2.0 엔진.

OCR 특화 VLM — 수식, 표, 다국어 텍스트를 잘 처리한다.
약 6GB VRAM 사용.

모델: stepfun-ai/GOT-OCR-2.0-hf
"""
from __future__ import annotations

import json
import logging
import time

from app.ocr.engines.vlm_base import VlmEngine
from app.schemas.vlm import (
    QaResponse,
    SchemaExtractItem,
    SchemaExtractResponse,
    SchemaField,
    VlmOcrItem,
    VlmOcrResponse,
)

logger = logging.getLogger(__name__)

_HF_MODEL = "stepfun-ai/GOT-OCR-2.0-hf"


class GotOcrEngine(VlmEngine):
    engine_id = "got_ocr"
    name = "GOT-OCR 2.0"
    model_id = _HF_MODEL
    vram_gb = 6.0

    def __init__(self) -> None:
        self._model = None
        self._processor = None
        self._device = "cpu"

    # ── 라이프사이클 ─────────────────────────────────

    def is_loaded(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if self._device == "cuda" else torch.float32

        logger.info("GOT-OCR2.0 로드 시작: %s", _HF_MODEL)
        self._processor = AutoProcessor.from_pretrained(_HF_MODEL)
        self._model = AutoModelForImageTextToText.from_pretrained(
            _HF_MODEL,
            dtype=dtype,
            device_map=self._device,
        )
        self._model.eval()
        logger.info("GOT-OCR2.0 로드 완료")

    def unload(self) -> None:
        self._model = None
        self._processor = None

    # ── 내부 헬퍼 ─────────────────────────────────────

    def _run_ocr(self, image_path: str) -> str:
        """이미지 → OCR 텍스트 추출."""
        import torch
        from PIL import Image

        image = Image.open(image_path).convert("RGB")
        inputs = self._processor(image, return_tensors="pt").to(self._device)

        with torch.no_grad():
            generate_ids = self._model.generate(
                **inputs,
                do_sample=False,
                tokenizer=self._processor.tokenizer,
                stop_strings="<|im_end|>",
                max_new_tokens=4096,
            )

        text = self._processor.decode(
            generate_ids[0, inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )
        return text.strip()

    # ── VLM 메서드 ────────────────────────────────────

    def ocr(self, image_path: str, options: dict | None = None) -> VlmOcrResponse:
        t0 = time.time()
        try:
            text = self._run_ocr(image_path)
            elapsed = int((time.time() - t0) * 1000)
            items = [VlmOcrItem(text=line) for line in text.split("\n") if line.strip()]
            return VlmOcrResponse(
                model_id=self.engine_id,
                elapsed_ms=elapsed,
                items=items,
                full_text=text,
            )
        except Exception as exc:
            elapsed = int((time.time() - t0) * 1000)
            logger.exception("GOT-OCR 실패")
            return VlmOcrResponse(
                success=False,
                model_id=self.engine_id,
                elapsed_ms=elapsed,
                error=str(exc),
            )

    def extract_schema(
        self,
        image_path: str,
        schema: list[SchemaField],
        options: dict | None = None,
    ) -> SchemaExtractResponse:
        """GOT-OCR 은 범용 chat 미지원. OCR 결과에서 키워드 매칭으로 추출."""
        t0 = time.time()
        try:
            text = self._run_ocr(image_path)
            elapsed = int((time.time() - t0) * 1000)
            items = self._match_schema_from_text(text, schema)
            return SchemaExtractResponse(
                model_id=self.engine_id,
                elapsed_ms=elapsed,
                items=items,
            )
        except Exception as exc:
            elapsed = int((time.time() - t0) * 1000)
            logger.exception("GOT-OCR Schema 추출 실패")
            return SchemaExtractResponse(
                success=False,
                model_id=self.engine_id,
                elapsed_ms=elapsed,
                error=str(exc),
            )

    def ask(
        self, image_path: str, question: str, options: dict | None = None
    ) -> QaResponse:
        """GOT-OCR 은 Q&A 미지원 — OCR 텍스트 전체 반환."""
        t0 = time.time()
        try:
            text = self._run_ocr(image_path)
            elapsed = int((time.time() - t0) * 1000)
            return QaResponse(
                model_id=self.engine_id,
                elapsed_ms=elapsed,
                answer=f"[GOT-OCR은 Q&A를 직접 지원하지 않습니다. OCR 결과:]\n{text}",
            )
        except Exception as exc:
            elapsed = int((time.time() - t0) * 1000)
            logger.exception("GOT-OCR Q&A 실패")
            return QaResponse(
                success=False,
                model_id=self.engine_id,
                elapsed_ms=elapsed,
                error=str(exc),
            )

    # ── 유틸 ──────────────────────────────────────────

    @staticmethod
    def _match_schema_from_text(
        text: str, schema: list[SchemaField]
    ) -> list[SchemaExtractItem]:
        """OCR 텍스트에서 키워드 기반으로 Schema 필드 값을 추출."""
        lines = text.split("\n")
        results: list[SchemaExtractItem] = []
        for field in schema:
            value = ""
            for line in lines:
                if field.key in line:
                    parts = line.split(field.key, 1)
                    if len(parts) > 1:
                        candidate = parts[1].strip().lstrip(":：").strip()
                        if candidate:
                            value = candidate
                            break
            results.append(SchemaExtractItem(key=field.key, value=value))
        return results
