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

import os

_HF_MODEL_DEFAULT = "Qwen/Qwen2.5-VL-3B-Instruct"
_HF_MODEL_LARGE = "Qwen/Qwen2.5-VL-7B-Instruct"

_VRAM_MAP = {
    _HF_MODEL_DEFAULT: 6.0,
    _HF_MODEL_LARGE: 14.0,
}


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
        load_kwargs: dict = {"torch_dtype": torch.float16}

        if torch.cuda.is_available():
            load_kwargs["device_map"] = "auto"
        else:
            load_kwargs["device_map"] = "cpu"
            load_kwargs["torch_dtype"] = torch.float32

        logger.info("Qwen2.5-VL 로드 시작: %s", hf_model)
        self._processor = AutoProcessor.from_pretrained(hf_model)
        self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            hf_model, **load_kwargs
        )
        self._model.eval()
        logger.info("Qwen2.5-VL 로드 완료 (%s)", hf_model)

    def unload(self) -> None:
        self._model = None
        self._processor = None

    # ── 내부 헬퍼 ─────────────────────────────────────

    def _chat(self, image_path: str, prompt: str) -> str:
        """이미지 + 프롬프트 → 텍스트 응답."""
        import torch
        from qwen_vl_utils import process_vision_info

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": f"file://{image_path}"},
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
        return self._processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

    # ── VLM 메서드 ────────────────────────────────────

    def ocr(self, image_path: str, options: dict | None = None) -> VlmOcrResponse:
        t0 = time.time()
        try:
            prompt = (
                "이 이미지에 있는 모든 텍스트를 빠짐없이 읽어주세요. "
                "원본 레이아웃(줄바꿈, 들여쓰기)을 최대한 유지하세요."
            )
            text = self._chat(image_path, prompt)
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
                "이 이미지에서 아래 항목들의 값을 추출하세요.\n\n"
                f"{fields_desc}\n\n"
                '반드시 JSON 배열로만 응답하세요. 각 원소는 {"key": "...", "value": "..."} 형태입니다.\n'
                "값을 찾을 수 없으면 value를 빈 문자열로 하세요."
            )
            raw = self._chat(image_path, prompt)
            elapsed = int((time.time() - t0) * 1000)
            items = self._parse_schema_response(raw, schema)
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
            answer = self._chat(image_path, question)
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
    def _parse_schema_response(
        raw: str, schema: list[SchemaField]
    ) -> list[SchemaExtractItem]:
        """LLM 응답에서 JSON 배열을 파싱하여 SchemaExtractItem 리스트로 변환."""
        start = raw.find("[")
        end = raw.rfind("]")
        if start == -1 or end == -1:
            return [
                SchemaExtractItem(key=f.key, value="") for f in schema
            ]
        try:
            data = json.loads(raw[start:end + 1])
        except json.JSONDecodeError:
            return [SchemaExtractItem(key=f.key, value="") for f in schema]

        result_map: dict[str, str] = {}
        for item in data:
            if isinstance(item, dict) and "key" in item:
                result_map[item["key"]] = str(item.get("value", ""))

        return [
            SchemaExtractItem(
                key=f.key,
                value=result_map.get(f.key, ""),
            )
            for f in schema
        ]
