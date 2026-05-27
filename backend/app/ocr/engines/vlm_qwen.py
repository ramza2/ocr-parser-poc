"""
Qwen2.5-VL 엔진 (기본: 2B / 환경변수로 7B 선택 가능).

한국어/영어/중국어 문서 이해에 강한 VLM.
- 3B: ~6GB VRAM (RTX 1080 Ti 등 11GB 이하 호환)
- 7B: ~14GB VRAM (FP16, 24GB+ GPU 권장)
"""
from __future__ import annotations

import json
import logging
import time

import re

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

import os

_HF_MODEL_DEFAULT = "Qwen/Qwen2.5-VL-3B-Instruct"
_HF_MODEL_LARGE = "Qwen/Qwen2.5-VL-7B-Instruct"

_VRAM_MAP = {
    _HF_MODEL_DEFAULT: 6.0,
    _HF_MODEL_LARGE: 14.0,
}

# processor·smart_resize 공통 (UI 스크린샷·문서용 해상도 상향)
_MIN_PIXELS = 256 * 28 * 28
_MAX_PIXELS = 1280 * 28 * 28


def _pick_model() -> str:
    """환경변수 QWEN_VL_MODEL 로 모델 지정 가능. 기본: 3B (11GB VRAM 이하 호환)."""
    env = os.environ.get("QWEN_VL_MODEL", "").strip()
    if env:
        return env
    return _HF_MODEL_DEFAULT


class QwenVlmEngine(VlmEngine):
    engine_id = "qwen_vl"
    name = "Qwen2.5-VL"
    model_id = _pick_model()
    vram_gb = _VRAM_MAP.get(model_id, 4.0)

    def __init__(self) -> None:
        self._model = None
        self._processor = None

    # ── 라이프사이클 ─────────────────────────────────

    def is_loaded(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

        hf_model = self.model_id
        has_gpu = torch.cuda.is_available()

        load_kwargs: dict = {
            "torch_dtype": torch.float16 if has_gpu else torch.float32,
            "device_map": "auto" if has_gpu else "cpu",
        }

        logger.info("Qwen2.5-VL 로드 시작: %s", hf_model)
        self._processor = AutoProcessor.from_pretrained(
            hf_model,
            min_pixels=_MIN_PIXELS,
            max_pixels=_MAX_PIXELS,
        )
        self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            hf_model, **load_kwargs
        )
        self._model.eval()
        logger.info("Qwen2.5-VL 로드 완료 (%s)", hf_model)

    def unload(self) -> None:
        self._model = None
        self._processor = None

    # ── 내부 헬퍼 ─────────────────────────────────────

    @staticmethod
    def _get_processed_size(image_path: str) -> tuple[int, int]:
        """processor가 실제 사용하는 리사이즈 크기 (width, height)."""
        from PIL import Image
        from qwen_vl_utils import smart_resize

        with Image.open(image_path) as img:
            w, h = img.size
        rh, rw = smart_resize(
            h, w, factor=28, min_pixels=_MIN_PIXELS, max_pixels=_MAX_PIXELS
        )
        return rw, rh

    def _chat(self, image_path: str, prompt: str) -> tuple[str, tuple[int, int]]:
        """이미지 + 프롬프트 → (텍스트 응답, processor 리사이즈 크기)."""
        import torch
        from qwen_vl_utils import process_vision_info

        processed_size = self._get_processed_size(image_path)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        text_input = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self._processor(
            text=[text_input],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self._model.device)

        with torch.no_grad():
            ids = self._model.generate(**inputs, max_new_tokens=2048)
        trimmed = ids[:, inputs.input_ids.shape[1]:]
        text = self._processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        return text, processed_size

    @staticmethod
    def _bbox_ocr_prompts() -> list[str]:
        """Qwen2.5-VL: processor 리사이즈 이미지 기준 절대 픽셀 bbox."""
        return [
            (
                "이 이미지의 모든 텍스트를 줄/블록 단위로 읽고 위치를 표시하세요.\n"
                "반드시 JSON 배열만 출력하세요. 형식:\n"
                '[{"text": "텍스트", "bbox": [x1, y1, x2, y2]}]\n'
                "bbox는 모델이 보는 이미지 기준 픽셀 좌표입니다 "
                "(좌상단 x,y → 우하단 x,y).\n"
                "화면에 보이는 텍스트가 있으면 빈 배열 []을 반환하지 마세요."
            ),
            (
                "Read every visible text line in this image with its bounding box.\n"
                "Output ONLY a JSON array like:\n"
                '[{"text": "line text", "bbox": [x1, y1, x2, y2]}]\n'
                "Use pixel coordinates on the input image (top-left to bottom-right).\n"
                "Example: [{\"text\": \"Hello\", \"bbox\": [12, 40, 180, 72]}]\n"
                "Never return [] if any text is visible."
            ),
        ]

    # ── VLM 메서드 ────────────────────────────────────

    def ocr(self, image_path: str, options: dict | None = None) -> VlmOcrResponse:
        t0 = time.time()
        try:
            items: list[VlmOcrItem] = []
            processed_size: tuple[int, int] | None = None

            for attempt, prompt in enumerate(self._bbox_ocr_prompts(), start=1):
                raw, processed_size = self._chat(image_path, prompt)
                items = self._parse_ocr_with_bbox(raw, processed_size)
                has_bbox = any(it.bbox for it in items)
                logger.info(
                    "bbox OCR 시도 %d: 항목 %d개, bbox %d개",
                    attempt, len(items), sum(1 for it in items if it.bbox),
                )
                if items and has_bbox:
                    break

            if not items or not any(it.bbox for it in items):
                logger.info("bbox OCR 실패, 일반 OCR 재시도 (bbox 없음)")
                plain_prompt = (
                    "이 이미지에 있는 모든 텍스트를 빠짐없이 읽어주세요. "
                    "원본 레이아웃(줄바꿈, 들여쓰기)을 최대한 유지하세요."
                )
                raw_plain, _ = self._chat(image_path, plain_prompt)
                items = self._parse_plain_ocr(raw_plain)

            elapsed = int((time.time() - t0) * 1000)
            logger.info("파싱된 항목 수: %d, bbox 있는 항목: %d",
                        len(items), sum(1 for it in items if it.bbox))
            for it in items:
                logger.info("  text=%s, bbox=%s", it.text, it.bbox)
            full_text = "\n".join(it.text for it in items)
            return VlmOcrResponse(
                model_id=self.engine_id,
                elapsed_ms=elapsed,
                items=items,
                full_text=full_text,
            )
        except Exception as exc:
            elapsed = int((time.time() - t0) * 1000)
            logger.exception("Qwen OCR 실패")
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
        t0 = time.time()
        try:
            fields_desc = "\n".join(
                f'- "{f.key}": {f.description or f.key} (타입: {f.type})'
                for f in schema
            )
            prompt = (
                "이 이미지에서 아래 항목들의 값과 위치를 추출하세요.\n\n"
                f"{fields_desc}\n\n"
                "반드시 JSON 배열로만 응답하세요. 각 원소는 다음 형태입니다:\n"
                '{"key": "...", "value": "...", "bbox": [x1, y1, x2, y2]}\n'
                "bbox는 모델이 보는 이미지 기준 픽셀 좌표입니다 "
                "(좌상단 x, 좌상단 y, 우하단 x, 우하단 y).\n"
                "값을 찾을 수 없으면 value를 빈 문자열로, bbox를 null로 하세요."
            )
            raw, processed_size = self._chat(image_path, prompt)
            elapsed = int((time.time() - t0) * 1000)
            items = self._parse_schema_with_bbox(raw, schema, processed_size)
            return SchemaExtractResponse(
                model_id=self.engine_id,
                elapsed_ms=elapsed,
                items=items,
            )
        except Exception as exc:
            elapsed = int((time.time() - t0) * 1000)
            logger.exception("Qwen Schema 추출 실패")
            return SchemaExtractResponse(
                success=False,
                model_id=self.engine_id,
                elapsed_ms=elapsed,
                error=str(exc),
            )

    def ask(
        self, image_path: str, question: str, options: dict | None = None
    ) -> QaResponse:
        t0 = time.time()
        try:
            answer, _ = self._chat(image_path, question)
            elapsed = int((time.time() - t0) * 1000)
            return QaResponse(
                model_id=self.engine_id,
                elapsed_ms=elapsed,
                answer=answer,
            )
        except Exception as exc:
            elapsed = int((time.time() - t0) * 1000)
            logger.exception("Qwen Q&A 실패")
            return QaResponse(
                success=False,
                model_id=self.engine_id,
                elapsed_ms=elapsed,
                error=str(exc),
            )

    # ── 유틸 ──────────────────────────────────────────

    @staticmethod
    def _to_bbox(
        raw_bbox, img_size: tuple[int, int] | None = None
    ) -> BoundingBox | None:
        """[x1,y1,x2,y2] → 0~1 비율 BoundingBox.
        Qwen2.5-VL은 processor 리사이즈 이미지 기준 절대 픽셀 좌표를 출력.
        """
        if not isinstance(raw_bbox, (list, tuple)) or len(raw_bbox) < 4:
            return None
        try:
            x1, y1, x2, y2 = [float(v) for v in raw_bbox[:4]]
            if x1 > 1 or y1 > 1 or x2 > 1 or y2 > 1:
                if img_size:
                    w, h = img_size
                    x1, y1, x2, y2 = x1 / w, y1 / h, x2 / w, y2 / h
                else:
                    x1, y1, x2, y2 = x1 / 1000, y1 / 1000, x2 / 1000, y2 / 1000
            if x2 <= x1 or y2 <= y1:
                return None
            return BoundingBox(x=x1, y=y1, width=x2 - x1, height=y2 - y1)
        except (ValueError, TypeError):
            return None

    @staticmethod
    def _extract_json_array(raw: str) -> list[dict]:
        """LLM 응답에서 JSON 배열 추출."""
        start = raw.find("[")
        end = raw.rfind("]")
        if start == -1 or end == -1:
            return []
        try:
            data = json.loads(raw[start:end + 1])
            return data if isinstance(data, list) else []
        except json.JSONDecodeError:
            return []

    @staticmethod
    def _parse_plain_ocr(raw: str) -> list[VlmOcrItem]:
        """일반 텍스트 OCR 응답 → VlmOcrItem 리스트."""
        lines = [ln.strip() for ln in raw.split("\n") if ln.strip()]
        if lines:
            return [VlmOcrItem(text=ln) for ln in lines]
        text = raw.strip()
        return [VlmOcrItem(text=text)] if text else []

    @staticmethod
    def _entry_bbox(entry: dict) -> list | None:
        """JSON 항목에서 bbox 좌표 추출 (bbox / bbox_2d 키 지원)."""
        raw = entry.get("bbox") or entry.get("bbox_2d")
        return raw if isinstance(raw, (list, tuple)) else None

    def _parse_ocr_with_bbox(
        self, raw: str, img_size: tuple[int, int] | None = None
    ) -> list[VlmOcrItem]:
        """OCR JSON 응답 → VlmOcrItem 리스트 (bbox 포함)."""
        logger.info("Qwen OCR 원본 응답:\n%s", raw[:1000])
        if img_size:
            logger.info("processor 리사이즈 크기: %s", img_size)
        data = self._extract_json_array(raw)
        if not data:
            return []

        items: list[VlmOcrItem] = []
        for entry in data:
            if not isinstance(entry, dict):
                continue
            text = str(entry.get("text", "")).strip()
            if not text:
                continue
            bbox = self._to_bbox(self._entry_bbox(entry), img_size)
            items.append(VlmOcrItem(text=text, bbox=bbox))
        return items

    def _parse_schema_with_bbox(
        self, raw: str, schema: list[SchemaField],
        img_size: tuple[int, int] | None = None,
    ) -> list[SchemaExtractItem]:
        """Schema JSON 응답 → SchemaExtractItem 리스트 (bbox 포함)."""
        data = self._extract_json_array(raw)

        result_map: dict[str, dict] = {}
        for item in data:
            if isinstance(item, dict) and "key" in item:
                result_map[item["key"]] = item

        items: list[SchemaExtractItem] = []
        for f in schema:
            entry = result_map.get(f.key, {})
            value = str(entry.get("value", "")) if entry else ""
            bbox = (
                self._to_bbox(self._entry_bbox(entry), img_size) if entry else None
            )
            items.append(SchemaExtractItem(key=f.key, value=value, bbox=bbox))
        return items
