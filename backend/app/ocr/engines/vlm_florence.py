"""
Florence-2-large 엔진.

Microsoft 의 경량 VLM — OCR + Region + Caption 통합.
약 2GB VRAM (float16) 으로 매우 가볍다.

모델: microsoft/Florence-2-large
"""
from __future__ import annotations

import json
import logging
import re
import time

from app.ocr.engines.vlm_base import VlmEngine
from app.schemas.vlm import (
    BoundingBox,
    QaResponse,
    SchemaExtractItem,
    SchemaExtractResponse,
    SchemaField,
    VlmOcrItem,
    VlmOcrResponse,
)

logger = logging.getLogger(__name__)

_HF_MODEL = "microsoft/Florence-2-large"


class FlorenceVlmEngine(VlmEngine):
    engine_id = "florence"
    name = "Florence-2 (Large)"
    model_id = _HF_MODEL
    vram_gb = 2.0

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
        from transformers import AutoModelForCausalLM, AutoProcessor

        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if self._device == "cuda" else torch.float32

        logger.info("Florence-2 로드 시작: %s (FP16=%s)", _HF_MODEL, self._device == "cuda")
        self._processor = AutoProcessor.from_pretrained(
            _HF_MODEL, trust_remote_code=True
        )
        self._model = AutoModelForCausalLM.from_pretrained(
            _HF_MODEL,
            trust_remote_code=True,
            torch_dtype=dtype,
        ).to(self._device)
        self._model.eval()
        logger.info("Florence-2 로드 완료")

    def unload(self) -> None:
        self._model = None
        self._processor = None

    # ── 내부 헬퍼 ─────────────────────────────────────

    @staticmethod
    def _load_and_resize(image_path: str, max_side: int = 1024):
        from PIL import Image
        img = Image.open(image_path).convert("RGB")
        w, h = img.size
        if max(w, h) <= max_side:
            return img
        scale = max_side / max(w, h)
        return img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

    def _run_task(self, image_path: str, task: str, text_input: str = "") -> dict:
        """Florence task 실행 → parsed result dict 반환."""
        import torch

        image = self._load_and_resize(image_path, max_side=1024)
        prompt = task if not text_input else task + text_input

        inputs = self._processor(
            text=prompt, images=image, return_tensors="pt"
        ).to(self._device)

        with torch.no_grad():
            ids = self._model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=1024,
                num_beams=3,
            )
        decoded = self._processor.batch_decode(ids, skip_special_tokens=False)[0]
        parsed = self._processor.post_process_generation(
            decoded, task=task, image_size=(image.width, image.height)
        )
        return parsed

    # ── VLM 메서드 ────────────────────────────────────

    def ocr(self, image_path: str, options: dict | None = None) -> VlmOcrResponse:
        t0 = time.time()
        try:
            result = self._run_task(image_path, "<OCR_WITH_REGION>")
            elapsed = int((time.time() - t0) * 1000)

            ocr_data = result.get("<OCR_WITH_REGION>", {})
            labels = ocr_data.get("labels", [])
            quads = ocr_data.get("quad_boxes", [])

            img = self._load_and_resize(image_path, max_side=1024)
            w, h = img.size

            items: list[VlmOcrItem] = []
            for i, label in enumerate(labels):
                text = label.strip()
                if not text:
                    continue
                bbox = None
                if i < len(quads):
                    q = quads[i]  # [x1,y1,x2,y2,x3,y3,x4,y4]
                    if len(q) >= 8:
                        x_min = min(q[0], q[6]) / w
                        y_min = min(q[1], q[3]) / h
                        x_max = max(q[2], q[4]) / w
                        y_max = max(q[5], q[7]) / h
                        bbox = BoundingBox(
                            x=x_min, y=y_min,
                            width=x_max - x_min, height=y_max - y_min,
                        )
                items.append(VlmOcrItem(text=text, bbox=bbox))

            full_text = "\n".join(item.text for item in items)
            return VlmOcrResponse(
                model_id=self.engine_id,
                elapsed_ms=elapsed,
                items=items,
                full_text=full_text,
            )
        except Exception as exc:
            elapsed = int((time.time() - t0) * 1000)
            logger.exception("Florence-2 OCR 실패")
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
        """Florence-2 는 범용 chat 미지원 — OCR 결과에서 키워드 매칭."""
        t0 = time.time()
        try:
            ocr_resp = self.ocr(image_path)
            elapsed = int((time.time() - t0) * 1000)
            items = self._match_schema(ocr_resp.full_text, schema)
            return SchemaExtractResponse(
                model_id=self.engine_id,
                elapsed_ms=elapsed,
                items=items,
            )
        except Exception as exc:
            elapsed = int((time.time() - t0) * 1000)
            logger.exception("Florence-2 Schema 추출 실패")
            return SchemaExtractResponse(
                success=False,
                model_id=self.engine_id,
                elapsed_ms=elapsed,
                error=str(exc),
            )

    def ask(
        self, image_path: str, question: str, options: dict | None = None
    ) -> QaResponse:
        """Florence-2 의 VQA task 활용."""
        t0 = time.time()
        try:
            result = self._run_task(image_path, "<VQA>", question)
            answer = result.get("<VQA>", str(result))
            elapsed = int((time.time() - t0) * 1000)
            return QaResponse(
                model_id=self.engine_id,
                elapsed_ms=elapsed,
                answer=answer if isinstance(answer, str) else str(answer),
            )
        except Exception as exc:
            elapsed = int((time.time() - t0) * 1000)
            logger.exception("Florence-2 Q&A 실패")
            return QaResponse(
                success=False,
                model_id=self.engine_id,
                elapsed_ms=elapsed,
                error=str(exc),
            )

    # ── 유틸 ──────────────────────────────────────────

    @staticmethod
    def _match_schema(
        text: str, schema: list[SchemaField]
    ) -> list[SchemaExtractItem]:
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
