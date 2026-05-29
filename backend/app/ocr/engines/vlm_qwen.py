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

    _SCHEMA_MATCH_MODES = frozenset(
        {"exact_text", "script_filter", "semantic_field"}
    )
    _SCRIPT_LABELS: dict[str, str] = {
        "han": "Han (CJK ideographs)",
        "hangul": "Hangul (Korean syllables)",
        "latin": "Latin letters",
        "english": "English letters",
        "digit": "digits and numbers",
    }

    @classmethod
    def _normalize_match_mode(cls, mode: str | None) -> str:
        m = (mode or "exact_text").strip().lower()
        return m if m in cls._SCHEMA_MATCH_MODES else "exact_text"

    @classmethod
    def _script_label(cls, script: str | None) -> str:
        s = (script or "han").strip().lower()
        return cls._SCRIPT_LABELS.get(s, cls._SCRIPT_LABELS["han"])

    @classmethod
    def _format_schema_target(cls, field: SchemaField) -> str:
        key = field.key.strip()
        hint = (field.description or "").strip()
        mode = cls._normalize_match_mode(field.match_mode)
        hint_part = f'; location_hint="{hint}"' if hint else ""

        if mode == "exact_text":
            return (
                f'key="{key}" — match_mode=exact_text; '
                f'target_text="{key}" (find this literal string only){hint_part}'
            )
        if mode == "script_filter":
            script = cls._script_label(field.script)
            return (
                f'key="{key}" — match_mode=script_filter; '
                f"extract only visible {script} from the image{hint_part}"
            )
        locate = hint or key
        return (
            f'key="{key}" — match_mode=semantic_field; '
            f'locate="{locate}" (semantic field; transcribe visible text){hint_part}'
        )

    @classmethod
    def _build_schema_prompt(cls, schema: list[SchemaField]) -> str:
        """Schema 추출 — match_mode별 규칙 (exact / script / semantic)."""
        targets = "\n".join(cls._format_schema_target(f) for f in schema)
        n = len(schema)
        return (
            "You are a scene-text localization and extraction engine.\n\n"
            "Each target below has a match_mode. Follow only the rules for that "
            "target's mode.\n\n"
            "[Targets]\n"
            f"{targets}\n\n"
            "[Global rules]\n\n"
            f"Return exactly one JSON object per target. "
            f"Array length must equal {n}.\n"
            'The output "key" must exactly copy the requested key string.\n'
            "Search documents, screens, signboards, labels, posters, outdoor scenes, "
            "embossed, shadowed, low-contrast, or angled text.\n"
            'bbox_2d must tightly enclose the text returned in "value".\n'
            "Do not output markdown, code fences, explanations, comments, "
            "confidence scores, or additional keys.\n"
            "Output strict RFC 8259 JSON only, with no trailing commas.\n\n"
            "[exact_text rules]\n\n"
            "target_text is the literal visible string to find, not a concept.\n"
            '"value" must contain only that exact visible string.\n'
            "Never concatenate neighboring lines or other languages, even on the "
            "same signboard or with the same meaning.\n"
            "Never include translations or equivalent wording.\n"
            "location_hint must never appear in value.\n"
            'If target_text is not found, use value="" and bbox_2d=null.\n\n'
            "[script_filter rules]\n\n"
            '"value" must contain only characters of the requested script, '
            "transcribed exactly as visible.\n"
            "Do not include characters from other scripts in the same value.\n"
            "If multiple regions match, prefer the region indicated by location_hint; "
            "otherwise use the most prominent matching region.\n"
            'If none found, use value="" and bbox_2d=null.\n\n'
            "[semantic_field rules]\n\n"
            "Find the visible text that matches the locate description.\n"
            '"value" must be verbatim visible text from the image (no translation).\n'
            "location_hint helps position only; do not copy it into value.\n"
            'If not found, use value="" and bbox_2d=null.\n\n'
            "Use exactly this output schema:\n"
            '[{"key": "requested_key", "value": "extracted visible text", '
            '"bbox_2d": [x1, y1, x2, y2]}]\n'
        )

    @staticmethod
    def _normalize_match_text(text: str) -> str:
        return " ".join(text.split())

    @classmethod
    def is_valid_exact_match(cls, key: str, value: str) -> bool:
        """exact-match: value가 비어 있지 않으면 key와 동일 문자열만 허용."""
        if not value.strip():
            return True
        return cls._normalize_match_text(key) == cls._normalize_match_text(value)

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

    @staticmethod
    def _build_qa_prompt(question: str) -> str:
        """범용 이미지 Q&A 프롬프트."""
        q = question.strip()
        return (
            "You are a visual question-answering assistant for images containing "
            "text, signs, documents, screens, objects, and natural scenes.\n\n"
            "Answer the user's question using only what is visibly supported by "
            "the image.\n"
            "Reply in the same language as the user's question.\n"
            "Return plain text only unless the user explicitly requests another "
            "format.\n\n"
            "[Reasoning Policy]\n\n"
            "* First identify the specific visual target, text span, object, "
            "region, or relationship referred to by the user's question.\n"
            "* Then answer only the requested attribute of that target, such as "
            "its visible text, language, color, location, direction, count, shape, "
            "or relationship.\n"
            "* Keep the target as narrow as the question requires. Do not expand "
            "it to surrounding lines, nearby labels, translations, or the whole "
            "object unless the user asks for them.\n"
            "* If the user refers to part of a visible text line or part of an "
            "object, answer about that part only.\n"
            "* If the user uses approximate or conversational wording, infer the "
            "most likely visible referent from the image and answer naturally when "
            "the referent is clear.\n"
            "* If two or more visible referents are genuinely plausible and would "
            "produce different answers, ask one brief clarification question instead "
            "of guessing.\n\n"
            "[Evidence Policy]\n\n"
            "* Use only visual evidence present in the image.\n"
            "* Do not invent, translate, paraphrase, combine, or complete text "
            "unless the user explicitly asks for translation, interpretation, or "
            "summarization.\n"
            "* When asked for visible text, preserve the text as shown in the image.\n"
            "* When asked for a visual property such as color, position, "
            "orientation, or size, answer only that property for the identified "
            "target.\n"
            "* When asked for an exact count or exact measurement, provide a "
            "result only when it can be determined reliably from the visible "
            "content; otherwise state briefly that precise confirmation is "
            "difficult from the image.\n\n"
            "[Answer Policy]\n\n"
            "* Answer directly and concisely.\n"
            "* Do not provide the full OCR transcription unless the user explicitly "
            "asks for it.\n"
            "* Do not explain your internal reasoning.\n"
            "* Do not mention unrelated visible content.\n"
            "* If the answer cannot be confirmed from the image, say so briefly.\n\n"
            f"User question:\n{q}"
        )

    def ask(
        self, image_path: str, question: str, options: dict | None = None
    ) -> QaResponse:
        t0 = time.time()
        try:
            prompt = self._build_qa_prompt(question)
            answer, _ = self._chat(image_path, prompt)
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
                self._to_bbox(self._entry_bbox(entry), img_size)
                if entry and value
                else None
            )
            if (
                self._normalize_match_mode(f.match_mode) == "exact_text"
                and value
                and not self.is_valid_exact_match(f.key, value)
            ):
                logger.warning(
                    "Schema exact_text 거부 key=%r value=%r (병합/번역 의심)",
                    f.key,
                    value,
                )
                value = ""
                bbox = None
            items.append(SchemaExtractItem(key=f.key, value=value, bbox=bbox))
        return items
