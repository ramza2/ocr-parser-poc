"""
Qwen2.5-VL 엔진 (기본: 2B / 환경변수로 7B 선택 가능).

한국어/영어/중국어 문서 이해에 강한 VLM.
- 3B: ~6GB VRAM (RTX 1080 Ti 등 11GB 이하 호환)
- 7B: ~14GB VRAM (FP16, 24GB+ GPU 권장)
"""
from __future__ import annotations

import json
import logging
import os
import re
import time
from pathlib import Path

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


def _pick_device_map(has_gpu: bool):
    """
    Qwen 로드 디바이스 맵.
    기본은 단일 GPU 고정(single)으로 latency 우선.
    - QWEN_DEVICE_MAP=auto  : 이전 방식
    - QWEN_DEVICE_MAP=single: 단일 GPU(cuda:0)
    """
    if not has_gpu:
        return "cpu"
    mode = os.environ.get("QWEN_DEVICE_MAP", "single").strip().lower()
    if mode == "auto":
        return "auto"
    return "cuda:0"


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

        device_map = _pick_device_map(has_gpu)
        load_kwargs: dict = {
            "torch_dtype": "auto",
        }
        if has_gpu:
            load_kwargs["device_map"] = device_map

        logger.info("Qwen2.5-VL 로드 시작: %s (device_map=%s)", hf_model, device_map)
        self._processor = AutoProcessor.from_pretrained(
            hf_model,
            min_pixels=_MIN_PIXELS,
            max_pixels=_MAX_PIXELS,
        )
        self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            hf_model, **load_kwargs
        )
        self._model.eval()
        if hasattr(self._model, "hf_device_map"):
            logger.info("Qwen2.5-VL hf_device_map=%s", getattr(self._model, "hf_device_map"))
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

    @staticmethod
    def _resolve_image_ref(image_path: str) -> str:
        """Qwen process_vision_info 호환 file URI."""
        path = Path(image_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"이미지 파일 없음: {image_path}")
        return path.as_uri()

    def _inference_device(self) -> str:
        import torch

        if torch.cuda.is_available():
            return "cuda:0"
        return "cpu"

    def _prepare_vlm_inputs(self, image_path: str, prompt: str):
        """이미지 + 프롬프트 → (processor inputs, 리사이즈 크기)."""
        from qwen_vl_utils import process_vision_info

        processed_size = self._get_processed_size(image_path)
        image_ref = self._resolve_image_ref(image_path)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_ref},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        text_input = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        if not image_inputs:
            raise RuntimeError(
                f"비전 입력 로드 실패 (path={image_ref}). "
                "워커 컨테이너에서 이미지 경로·qwen-vl-utils를 확인하세요."
            )

        inputs = self._processor(
            text=[text_input],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        device = self._inference_device()
        inputs = inputs.to(device)
        if getattr(inputs, "pixel_values", None) is None:
            raise RuntimeError(
                "pixel_values가 생성되지 않았습니다. "
                "transformers/qwen-vl-utils 버전 또는 이미지 입력을 확인하세요."
            )
        logger.debug(
            "VLM 입력 준비: device=%s pixel_values=%s",
            device,
            tuple(inputs.pixel_values.shape),
        )
        return inputs, processed_size

    def _generate_from_inputs(
        self,
        inputs,
        *,
        max_new_tokens: int,
        do_sample: bool = False,
        temperature: float | None = None,
        repetition_penalty: float = 1.05,
    ) -> str:
        import torch

        gen_kwargs: dict = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "repetition_penalty": repetition_penalty,
        }
        if do_sample and temperature is not None:
            gen_kwargs["temperature"] = temperature
        tokenizer = getattr(self._processor, "tokenizer", None)
        if tokenizer is not None:
            if tokenizer.pad_token_id is not None:
                gen_kwargs["pad_token_id"] = tokenizer.pad_token_id
            if tokenizer.eos_token_id is not None:
                gen_kwargs["eos_token_id"] = tokenizer.eos_token_id

        with torch.no_grad():
            ids = self._model.generate(**inputs, **gen_kwargs)
        trimmed = ids[:, inputs.input_ids.shape[1]:]
        return self._processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

    def _chat(
        self,
        image_path: str,
        prompt: str,
        max_new_tokens: int = 2048,
        *,
        repetition_penalty: float | None = None,
    ) -> tuple[str, tuple[int, int]]:
        """이미지 + 프롬프트 → (텍스트 응답, processor 리사이즈 크기)."""
        inputs, processed_size = self._prepare_vlm_inputs(image_path, prompt)
        rep = repetition_penalty if repetition_penalty is not None else 1.05

        text = self._generate_from_inputs(
            inputs,
            max_new_tokens=max_new_tokens,
            repetition_penalty=rep,
        )
        if self._is_degenerate_output(text):
            logger.warning(
                "VLM 퇴화 출력 감지(! 반복 등) → 샘플링 재시도 (max_tokens=%d)",
                max_new_tokens,
            )
            text = self._generate_from_inputs(
                inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.2,
                repetition_penalty=max(rep, 1.12),
            )
        if self._is_degenerate_output(text):
            raise RuntimeError(
                "모델이 비정상 출력(같은 문자 반복)을 생성했습니다. "
                "vlm-worker 재시작, transformers 버전(4.49~4.51), GPU 상태를 확인하세요."
            )
        return text, processed_size

    # bbox + JSON 전용 OCR 프롬프트
    _BBOX_OCR_PROMPT = (
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

    # 텍스트만 OCR — 가독성 있는 plain text (표·열 구조 유지)
    _TEXT_ONLY_OCR_PROMPT = (
        "You are a document and scene-text OCR assistant.\n\n"
        "Transcribe ALL visible text from this image as readable plain text.\n"
        "Preserve the visual layout: tables, columns, headings, and line groupings.\n\n"
        "[Layout rules]\n\n"
        "* Use line breaks and spacing so a human can read the result like the original.\n"
        "* For tables: put column headers on one row; align each data row with its columns.\n"
        "* Merge hierarchical row labels into one line when appropriate "
        '(e.g. "일반권 / 개인 Private").\n'
        "* For bilingual headers (어른 Adult), keep both languages as shown.\n"
        "* In table body cells, when Korean and English repeat the same value "
        "(e.g. 3,000원 with 3,000won directly below), include only the Korean "
        "amount in the table. Do not duplicate identical translations.\n"
        "* For multi-language titles, list each language line as in the image.\n"
        "* Do not split one logical table row into many separate single-word lines.\n\n"
        "[Output rules]\n\n"
        "* Return plain text only.\n"
        "* Do not output JSON, coordinates, bbox, markdown code fences, or explanations.\n"
        "* Preserve text exactly as visible in the image."
    )

    _OUTPUT_FORMATS = frozenset({"bbox", "text_only"})

    @classmethod
    def _ocr_prompt(cls, output_format: str) -> str:
        if output_format == "text_only":
            return cls._TEXT_ONLY_OCR_PROMPT
        return cls._BBOX_OCR_PROMPT

    @classmethod
    def _normalize_output_format(cls, opts: dict | None) -> str:
        fmt = str((opts or {}).get("output_format", "text_only")).strip().lower()
        return fmt if fmt in cls._OUTPUT_FORMATS else "text_only"

    @classmethod
    def _max_new_tokens(cls, output_format: str, task: str) -> int:
        if task == "qa":
            return 512
        if output_format == "text_only":
            if task == "schema":
                return 512
            return 1024
        return 1024

    def _run_ocr(
        self,
        image_path: str,
        output_format: str = "bbox",
    ) -> tuple[list[VlmOcrItem], str, str]:
        """output_format별 전용 프롬프트로 OCR 1회 실행."""
        max_tokens = self._max_new_tokens(output_format, "ocr")

        if output_format == "bbox":
            prompt = self._BBOX_OCR_PROMPT
            raw, processed_size = self._chat(
                image_path, prompt, max_new_tokens=max_tokens
            )
            items = self._parse_ocr_with_bbox(raw, processed_size)
            logger.info(
                "bbox OCR: 항목=%d bbox=%d max_tokens=%d",
                len(items),
                sum(1 for it in items if it.bbox),
                max_tokens,
            )
            return items, "bbox", raw

        prompt = self._TEXT_ONLY_OCR_PROMPT
        raw, processed_size = self._chat(
            image_path, prompt, max_new_tokens=max_tokens
        )
        label = "text_only"

        if self._is_degenerate_output(raw):
            logger.warning("text_only OCR 퇴화 → 샘플링 재시도")
            inputs, processed_size = self._prepare_vlm_inputs(image_path, prompt)
            raw = self._generate_from_inputs(
                inputs,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=0.2,
                repetition_penalty=1.12,
            )

        if self._is_degenerate_output(raw):
            logger.warning("text_only OCR 실패 → bbox 프롬프트 fallback")
            fallback_prompt = self._BBOX_OCR_PROMPT
            raw, processed_size = self._chat(
                image_path, fallback_prompt, max_new_tokens=max_tokens
            )
            items = self._items_without_bbox(
                self._parse_ocr_with_bbox(raw, processed_size)
            )
            label = "text_only→bbox_fallback"
            logger.info(
                "%s: 항목=%d max_tokens=%d", label, len(items), max_tokens
            )
            return items, label, raw

        full_text = self._strip_code_fences(raw.strip())
        items = [VlmOcrItem(text=full_text)] if full_text else []
        logger.info(
            "%s: chars=%d max_tokens=%d", label, len(full_text), max_tokens
        )
        return items, label, raw

    @staticmethod
    def _items_without_bbox(items: list[VlmOcrItem]) -> list[VlmOcrItem]:
        return [VlmOcrItem(text=it.text, confidence=it.confidence) for it in items]

    # ── VLM 메서드 ────────────────────────────────────

    def ocr(self, image_path: str, options: dict | None = None) -> VlmOcrResponse:
        opts = options or {}
        output_format = self._normalize_output_format(opts)

        t0 = time.time()
        try:
            items, label, raw = self._run_ocr(image_path, output_format=output_format)

            elapsed = int((time.time() - t0) * 1000)
            logger.info(
                "OCR 완료 label=%s format=%s 항목=%d bbox=%d",
                label,
                output_format,
                len(items),
                sum(1 for it in items if it.bbox),
            )
            if output_format == "text_only" and label == "text_only":
                full_text = self._strip_code_fences(raw.strip())
            else:
                full_text = "\n".join(it.text for it in items)
            raw_preview = raw[:4000] if raw else None
            return VlmOcrResponse(
                model_id=self.engine_id,
                elapsed_ms=elapsed,
                items=items,
                full_text=full_text,
                prompt_label=label,
                output_format=output_format,
                raw_response_preview=raw_preview,
            )
        except Exception as exc:
            elapsed = int((time.time() - t0) * 1000)
            logger.exception("Qwen OCR 실패")
            return VlmOcrResponse(
                success=False,
                model_id=self.engine_id,
                elapsed_ms=elapsed,
                output_format=output_format,
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
    def _build_schema_prompt(
        cls, schema: list[SchemaField], *, with_bbox: bool = True
    ) -> str:
        """Schema 추출 — match_mode별 규칙 (exact / script / semantic)."""
        targets = "\n".join(cls._format_schema_target(f) for f in schema)
        n = len(schema)
        bbox_rules = (
            'bbox_2d must tightly enclose the text returned in "value".\n'
            if with_bbox
            else ""
        )
        not_found_rule = (
            'If target_text is not found, use value="" and bbox_2d=null.\n\n'
            if with_bbox
            else 'If target_text is not found, use value="".\n\n'
        )
        script_not_found = (
            'If none found, use value="" and bbox_2d=null.\n\n'
            if with_bbox
            else 'If none found, use value="".\n\n'
        )
        semantic_not_found = (
            'If not found, use value="" and bbox_2d=null.\n\n'
            if with_bbox
            else 'If not found, use value="".\n\n'
        )
        if with_bbox:
            output_schema = (
                '[{"key": "requested_key", "value": "extracted visible text", '
                '"bbox_2d": [x1, y1, x2, y2]}]\n'
            )
        else:
            output_schema = (
                '[{"key": "requested_key", "value": "extracted visible text"}]\n'
            )
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
            f"{bbox_rules}"
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
            f"{not_found_rule}"
            "[script_filter rules]\n\n"
            '"value" must contain only characters of the requested script, '
            "transcribed exactly as visible.\n"
            "Do not include characters from other scripts in the same value.\n"
            "If multiple regions match, prefer the region indicated by location_hint; "
            "otherwise use the most prominent matching region.\n"
            f"{script_not_found}"
            "[semantic_field rules]\n\n"
            "Find the visible text that matches the locate description.\n"
            '"value" must be verbatim visible text from the image (no translation).\n'
            "location_hint helps position only; do not copy it into value.\n"
            f"{semantic_not_found}"
            "Use exactly this output schema:\n"
            f"{output_schema}"
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
        opts = options or {}
        output_format = self._normalize_output_format(opts)
        with_bbox = output_format == "bbox"
        t0 = time.time()
        try:
            prompt = self._build_schema_prompt(schema, with_bbox=with_bbox)
            max_tokens = self._max_new_tokens(output_format, "schema")
            raw, processed_size = self._chat(
                image_path, prompt, max_new_tokens=max_tokens
            )
            logger.info(
                "Qwen Schema 원본 응답 (format=%s, max_tokens=%d):\n%s",
                output_format,
                max_tokens,
                raw[:2000],
            )
            elapsed = int((time.time() - t0) * 1000)
            if with_bbox:
                items = self._parse_schema_with_bbox(raw, schema, processed_size)
            else:
                items = self._parse_schema_text_only(raw, schema)
            return SchemaExtractResponse(
                model_id=self.engine_id,
                elapsed_ms=elapsed,
                items=items,
                output_format=output_format,
            )
        except Exception as exc:
            elapsed = int((time.time() - t0) * 1000)
            logger.exception("Qwen Schema 추출 실패")
            return SchemaExtractResponse(
                success=False,
                model_id=self.engine_id,
                elapsed_ms=elapsed,
                output_format=output_format,
                error=str(exc),
            )

    @staticmethod
    def _build_qa_prompt(question: str, *, text_only: bool = False) -> str:
        """범용 이미지 Q&A 프롬프트 (표·계산형 질문 강화)."""
        q = question.strip()
        return (
            "You are a visual question-answering assistant for images containing "
            "text, signs, documents, screens, objects, and natural scenes.\n\n"
            "Answer the user's question using only:\n\n"
            "* information visibly supported by the image, and\n"
            "* explicit conditions provided by the user in the question.\n\n"
            "Reply in the same language as the user's question.\n"
            "Return plain text only unless the user explicitly requests another "
            "format.\n\n"
            "[Target Identification Policy]\n\n"
            "* First identify the exact visual target, text span, object, table, "
            "row, column, or relationship referred to by the user's question.\n"
            "* Keep the target as narrow as the question requires.\n"
            "* If the question refers to a table, identify the relevant row labels, "
            "column labels, and cell values before answering.\n"
            "* If two or more visible referents are genuinely plausible and would "
            "produce different answers, ask one brief clarification question "
            "instead of guessing.\n\n"
            "[Table Policy]\n\n"
            "* For table images, preserve the table structure mentally before "
            "answering.\n"
            "* Distinguish row headers, column headers, merged headers, category "
            "labels, and price/value cells.\n"
            "* Do not mix values from different rows or columns.\n"
            "* When a question mentions a category such as adult, student, child, "
            "private, group, discount, or general ticket, select the cell at the "
            "intersection of the matching row and column.\n"
            "* If the user provides a condition such as \"3명 이상이면 단체\", "
            "apply that condition to choose the matching visible table row, even "
            "if the condition itself is not written in the image.\n"
            "* If the user asks about multiple people or multiple categories, "
            "calculate each category separately and then sum them.\n\n"
            "[Calculation Policy]\n\n"
            "* When arithmetic is required, first extract the relevant visible "
            "values, then apply the user's condition, then calculate.\n"
            "* For fee, price, count, total, discount, or comparison questions, "
            "show a short calculation in the final answer.\n"
            "* Do not silently skip any quantity mentioned by the user.\n"
            "* If the question includes \"어른 넷 아이 셋\", interpret it as "
            "4 adults and 3 children.\n"
            "* If the question says to use group fees, use the Group row for all "
            "applicable categories unless the image or question clearly says "
            "otherwise.\n"
            "* If the required value cannot be read reliably from the image, say "
            "that it cannot be confirmed clearly.\n\n"
            "[Evidence Policy]\n\n"
            "* Do not invent values that are not visible in the image.\n"
            "* Do not translate, paraphrase, combine, or complete visible text "
            "unless the user asks for interpretation, calculation, or "
            "summarization.\n"
            "* When asked for visible text, preserve the text as shown in the "
            "image.\n"
            "* When asked for color, position, direction, count, or shape, "
            "answer only that attribute.\n\n"
            "[Answer Policy]\n\n"
            "* Answer directly and concisely.\n"
            "* For table calculation questions, include the selected unit prices "
            "and the final total.\n"
            "* Do not provide the full OCR transcription unless the user "
            "explicitly asks for it.\n"
            "* Do not mention unrelated visible content.\n"
            "* Do not include JSON, coordinates, bbox, or markdown.\n\n"
            f"User question:\n{q}"
        )

    def ask(
        self, image_path: str, question: str, options: dict | None = None
    ) -> QaResponse:
        opts = options or {}
        output_format = self._normalize_output_format(opts)
        text_only = output_format == "text_only"
        t0 = time.time()
        try:
            prompt = self._build_qa_prompt(question, text_only=text_only)
            max_tokens = self._max_new_tokens(output_format, "qa")
            answer, _ = self._chat(image_path, prompt, max_new_tokens=max_tokens)
            elapsed = int((time.time() - t0) * 1000)
            return QaResponse(
                model_id=self.engine_id,
                elapsed_ms=elapsed,
                answer=answer,
                output_format=output_format,
            )
        except Exception as exc:
            elapsed = int((time.time() - t0) * 1000)
            logger.exception("Qwen Q&A 실패")
            return QaResponse(
                success=False,
                model_id=self.engine_id,
                elapsed_ms=elapsed,
                output_format=output_format,
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

    def _parse_schema_text_only(
        self, raw: str, schema: list[SchemaField]
    ) -> list[SchemaExtractItem]:
        """Schema JSON 응답 → SchemaExtractItem (bbox 없음)."""
        data = self._extract_json_array(raw)

        result_map: dict[str, dict] = {}
        for item in data:
            if isinstance(item, dict) and "key" in item:
                result_map[item["key"]] = item

        items: list[SchemaExtractItem] = []
        for f in schema:
            entry = result_map.get(f.key, {})
            value = self._entry_schema_value(entry) if entry else ""
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
            items.append(SchemaExtractItem(key=f.key, value=value))
        return items

    @staticmethod
    def _is_degenerate_output(raw: str) -> bool:
        """동일 문자 반복 등 비정상 생성 감지."""
        text = raw.strip()
        if len(text) < 12:
            return False
        chars = [c for c in text if not c.isspace()]
        if not chars:
            return True
        from collections import Counter

        _char, count = Counter(chars).most_common(1)[0]
        return count / len(chars) > 0.75
