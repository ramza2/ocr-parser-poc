"""
Qwen2.5-VL 엔진 (기본: 2B / 환경변수로 7B 선택 가능).

한국어/영어/중국어 문서 이해에 강한 VLM.
- 3B: ~6GB VRAM (RTX 1080 Ti 등 11GB 이하 호환)
- 7B: ~14GB VRAM (FP16, 24GB+ GPU 권장)
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
            ids = self._model.generate(
                **inputs,
                max_new_tokens=4096,
                do_sample=False,
            )
        trimmed = ids[:, inputs.input_ids.shape[1]:]
        text = self._processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        return text, processed_size

    # Scene-text OCR + grounding (bbox_2d + text_content)
    _SPOTTING_OCR_PROMPT = (
        "You are a scene-text OCR and text-grounding engine.\n\n"
        "Find and transcribe ALL visible text in this image, whether it appears in:\n\n"
        "* a document or mobile screen,\n"
        "* a signboard, road sign, direction board, label, poster, or product,\n"
        "* an outdoor or natural scene.\n\n"
        "Pay special attention to:\n\n"
        "* Korean, English, Chinese characters, numbers, and punctuation,\n"
        "* stylized, embossed, shadowed, engraved, painted, low-contrast, or "
        "angled text,\n"
        "* text on colored boards or textured backgrounds.\n\n"
        "Before returning an empty result, carefully inspect the entire image "
        "for any object or region containing readable characters.\n\n"
        "Return ONLY a valid JSON array in reading order, from top to bottom "
        "and left to right.\n\n"
        "Use exactly this schema:\n"
        '[{"bbox_2d": [x1, y1, x2, y2], "text_content": "recognized text"}]\n\n'
        "Rules:\n\n"
        "* Return one object per visible text line.\n"
        "* Preserve text exactly as it appears in the image.\n"
        "* Each bbox_2d must tightly enclose its corresponding text line.\n"
        "* Do not omit text because it is decorative, on a signboard, photographed "
        "outdoors, shadowed, embossed, or low contrast.\n"
        "* Do not output markdown, explanations, comments, confidence scores, or "
        "additional keys.\n"
        "* The JSON must be strict RFC 8259: no trailing commas.\n"
        "* Return [] only when there are truly no visible characters or words "
        "anywhere in the image."
    )

    def _run_single_vlm(
        self, image_path: str, prompt: str, label: str
    ) -> tuple[list[VlmOcrItem], str, str]:
        """VLM 1회 호출 → JSON 파싱 (재시도 없음)."""
        raw, processed_size = self._chat(image_path, prompt)
        items = self._parse_ocr_with_bbox(raw, processed_size)
        logger.info(
            "%s: 항목 %d개, bbox %d개",
            label,
            len(items),
            sum(1 for it in items if it.bbox),
        )
        return items, label, raw

    # ── VLM 메서드 ────────────────────────────────────

    def ocr(self, image_path: str, options: dict | None = None) -> VlmOcrResponse:
        opts = options or {}
        raw_mode = str(opts.get("prompt_mode", "spotting")).strip().lower() or "spotting"
        # 구 API 호환 (auto, bbox, plain → spotting)
        prompt_mode = (
            "custom" if raw_mode == "custom" else "spotting"
        )
        custom_prompt = str(opts.get("custom_prompt", "")).strip()

        t0 = time.time()
        try:
            if prompt_mode == "custom":
                if not custom_prompt:
                    raise ValueError("커스텀 프롬프트가 비어 있습니다.")
                items, label, raw = self._run_single_vlm(
                    image_path, custom_prompt, "custom"
                )
            else:
                items, label, raw = self._run_single_vlm(
                    image_path, self._SPOTTING_OCR_PROMPT, "spotting"
                )

            elapsed = int((time.time() - t0) * 1000)
            logger.info(
                "OCR 완료 mode=%s label=%s 항목=%d bbox=%d",
                prompt_mode,
                label,
                len(items),
                sum(1 for it in items if it.bbox),
            )
            full_text = "\n".join(it.text for it in items)
            return VlmOcrResponse(
                model_id=self.engine_id,
                elapsed_ms=elapsed,
                items=items,
                full_text=full_text,
                prompt_mode=prompt_mode,
                prompt_label=label,
                raw_response_preview=raw[:2000] if raw else None,
            )
        except Exception as exc:
            elapsed = int((time.time() - t0) * 1000)
            logger.exception("Qwen OCR 실패")
            return VlmOcrResponse(
                success=False,
                model_id=self.engine_id,
                elapsed_ms=elapsed,
                prompt_mode=prompt_mode,
                error=str(exc),
            )

    @staticmethod
    def _build_schema_prompt(schema: list[SchemaField]) -> str:
        """Schema 필드 추출 — scene-text grounding + key/value 언어 규칙."""
        lines: list[str] = []
        for i, f in enumerate(schema, start=1):
            desc = (f.description or f.key).strip()
            lines.append(
                f'{i}. key="{f.key}" (copy this key exactly in output) — '
                f"locate: {desc} (type: {f.type})"
            )

        rules = [
            "Return exactly one JSON object per schema key; array length must equal "
            f"the number of fields ({len(schema)}).",
            'The "key" in each object must exactly match the schema key string '
            "(including Chinese or other scripts — never rename or translate keys).",
            'The "value" must be transcribed verbatim from the image for that field. '
            "Never translate, summarize, or rewrite into another language.",
            "The language of the field description does NOT control the output "
            "language — only visible image text does.",
            "If the image shows Chinese, use Chinese; if Korean, use Korean; if "
            "English, use English. Mixed scripts are allowed.",
            "Search the whole image: documents, screens, signboards, labels, posters, "
            "embossed, shadowed, low-contrast, or angled text.",
            "bbox_2d must tightly enclose the text region for that value.",
            "If not found after careful inspection, use value=\"\" and bbox_2d=null.",
            "Do not output markdown, explanations, comments, or extra keys.",
            "The JSON must be strict RFC 8259: no trailing commas.",
        ]

        return (
            "You are a scene-text OCR and structured field extraction engine.\n\n"
            "Extract each schema field below from ALL visible text in this image "
            "(documents, screens, signboards, labels, outdoor scenes).\n\n"
            "Return ONLY a valid JSON array.\n\n"
            "[Fields]\n"
            + "\n".join(lines)
            + "\n\n[Rules]\n"
            + "\n".join(f"* {r}" for r in rules)
            + "\n\nUse exactly this schema for every field:\n"
            '[{"key": "schema_key", "value": "verbatim text from image", '
            '"bbox_2d": [x1, y1, x2, y2]}]\n'
        )

    def extract_schema(
        self,
        image_path: str,
        schema: list[SchemaField],
        options: dict | None = None,
    ) -> SchemaExtractResponse:
        t0 = time.time()
        try:
            prompt = self._build_schema_prompt(schema)
            raw, processed_size = self._chat(image_path, prompt)
            logger.info("Qwen Schema 원본 응답:\n%s", raw[:2000])
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
    def _strip_code_fences(raw: str) -> str:
        text = raw.strip()
        if not text.startswith("```"):
            return text
        lines = text.splitlines()
        if len(lines) >= 2 and lines[-1].strip().startswith("```"):
            return "\n".join(lines[1:-1]).strip()
        return "\n".join(lines[1:]).strip()

    @staticmethod
    def _sanitize_json(text: str) -> str:
        """LLM이 자주 내는 trailing comma 등 보정."""
        return re.sub(r",(\s*[\]}])", r"\1", text)

    @staticmethod
    def _extract_json_objects_fallback(raw: str) -> list[dict]:
        """배열 파싱 실패 시 개별 {...} 객체를 순차 파싱."""
        objects: list[dict] = []
        i = 0
        while i < len(raw):
            start = raw.find("{", i)
            if start == -1:
                break
            depth = 0
            for j in range(start, len(raw)):
                ch = raw[j]
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        snippet = raw[start : j + 1]
                        for candidate in (snippet, QwenVlmEngine._sanitize_json(snippet)):
                            try:
                                obj = json.loads(candidate)
                                if isinstance(obj, dict):
                                    objects.append(obj)
                                break
                            except json.JSONDecodeError:
                                continue
                        i = j + 1
                        break
            else:
                break
        return objects

    @classmethod
    def _extract_json_array(cls, raw: str) -> list[dict]:
        """LLM 응답에서 JSON 배열 추출 (trailing comma·코드펜스 tolerant)."""
        cleaned = cls._strip_code_fences(raw)
        start = cleaned.find("[")
        end = cleaned.rfind("]")
        if start != -1 and end != -1 and end > start:
            chunk = cleaned[start : end + 1]
            for label, candidate in (
                ("strict", chunk),
                ("sanitized", cls._sanitize_json(chunk)),
            ):
                try:
                    data = json.loads(candidate)
                    if isinstance(data, list):
                        items = [x for x in data if isinstance(x, dict)]
                        if items:
                            if label == "sanitized":
                                logger.info("JSON 배열: trailing comma 보정 후 파싱 성공")
                            return items
                except json.JSONDecodeError as exc:
                    logger.debug("JSON 배열 %s 파싱 실패: %s", label, exc)

        fallback = cls._extract_json_objects_fallback(cleaned)
        if fallback:
            logger.info("JSON 배열 파싱 실패 → 개별 객체 %d개 추출", len(fallback))
        return fallback

    @staticmethod
    def _entry_bbox(entry: dict) -> list | None:
        """JSON 항목에서 bbox 좌표 추출 (bbox_2d / bbox 키 지원)."""
        raw = entry.get("bbox_2d") or entry.get("bbox")
        return raw if isinstance(raw, (list, tuple)) else None

    @staticmethod
    def _entry_text(entry: dict) -> str:
        """JSON 항목에서 텍스트 추출 (text_content / text 키 지원)."""
        for key in ("text_content", "text", "content"):
            val = entry.get(key)
            if val is not None and str(val).strip():
                return str(val).strip()
        return ""

    @staticmethod
    def _entry_schema_value(entry: dict) -> str:
        """Schema JSON 항목에서 추출값 (value / text_content / text)."""
        for key in ("value", "text_content", "text"):
            val = entry.get(key)
            if val is not None and str(val).strip():
                return str(val).strip()
        return ""

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
            text = self._entry_text(entry)
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
            value = self._entry_schema_value(entry) if entry else ""
            bbox = (
                self._to_bbox(self._entry_bbox(entry), img_size) if entry else None
            )
            items.append(SchemaExtractItem(key=f.key, value=value, bbox=bbox))
        return items
