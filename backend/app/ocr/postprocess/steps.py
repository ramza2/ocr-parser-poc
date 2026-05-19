"""
후처리 step 구현체. step_id → POSTPROCESS_STEP_FUNCS.

strip_normalize / format_rules / char_correct / layout_order(bbox 정렬)
"""
from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

from app.utils.log_utils import log_item

# 제어 문자·깨진 기호 등 노이즈
_NOISE_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\ufffd]")
_MULTI_SPACE_RE = re.compile(r"[ \t]+")
_BLANK_LINES_RE = re.compile(r"\n{3,}")

# 한국 휴대폰 / 주민번호(마스킹 포함) 포맷
_PHONE_RE = re.compile(
    r"(?<!\d)(01[016789])[\s\-.]?\d{3,4}[\s\-.]?\d{4}(?!\d)"
)
_RRN_RE = re.compile(
    r"(?<!\d)(\d{6})[\s\-]?([1-4]\d{6})(?!\d)"
)


def postprocess_strip_normalize(text: str, **_kwargs) -> tuple[str, list]:
    lines = [ln.strip() for ln in text.splitlines()]
    joined = "\n".join(lines).strip()
    joined = _NOISE_CHARS_RE.sub("", joined)
    joined = _MULTI_SPACE_RE.sub(" ", joined)
    joined = _BLANK_LINES_RE.sub("\n\n", joined)
    return joined, [log_item("INFO", "공백·특수문자 정규화 적용")]


def _format_phone(match: re.Match[str]) -> str:
    digits = match.group(0)
    nums = re.sub(r"\D", "", digits)
    if len(nums) == 11:
        return f"{nums[:3]}-{nums[3:7]}-{nums[7:]}"
    if len(nums) == 10:
        return f"{nums[:3]}-{nums[3:6]}-{nums[6:]}"
    return match.group(0)


def _format_rrn(match: re.Match[str]) -> str:
    return f"{match.group(1)}-{match.group(2)}"


def postprocess_format_rules(text: str, **_kwargs) -> tuple[str, list]:
    out = _PHONE_RE.sub(_format_phone, text)
    out = _RRN_RE.sub(_format_rrn, out)
    return out, [log_item("INFO", "도메인 포맷 교정(전화·주민번호) 적용")]


def _fix_digit_context(text: str) -> str:
    """숫자 구간에서 O→0, l/I→1 등 흔한 OCR 혼동 교정."""
    chars = list(text)
    n = len(chars)
    for i, ch in enumerate(chars):
        if ch not in ("O", "o", "l", "I", "|"):
            continue
        prev_d = i > 0 and chars[i - 1].isdigit()
        next_d = i + 1 < n and chars[i + 1].isdigit()
        if not (prev_d or next_d):
            continue
        if ch in ("O", "o"):
            chars[i] = "0"
        elif ch in ("l", "I", "|"):
            chars[i] = "1"
    return "".join(chars)


def postprocess_char_correct(text: str, **_kwargs) -> tuple[str, list]:
    return _fix_digit_context(text), [
        log_item("INFO", "문맥 기반 문자 교정(O/0, l/1) 적용")
    ]


def _bbox_sort_key(block: dict[str, Any]) -> tuple[float, float]:
    bbox = block.get("bbox")
    if not bbox:
        return (0.0, 0.0)
    try:
        if isinstance(bbox[0], (list, tuple)):
            xs = [float(p[0]) for p in bbox]
            ys = [float(p[1]) for p in bbox]
        else:
            xs = [float(bbox[0]), float(bbox[2])]
            ys = [float(bbox[1]), float(bbox[3])]
        return (min(ys), min(xs))
    except (TypeError, ValueError, IndexError):
        return (0.0, 0.0)


def postprocess_layout_order(
    text: str,
    blocks: list[dict[str, Any]] | None = None,
    line_threshold: float = 20.0,
    **_kwargs,
) -> tuple[str, list]:
    """bbox 기준 위→아래·좌→우 정렬 (PP-Structure 대용 PoC)."""
    if not blocks:
        return text, [
            log_item(
                "WARN",
                "bbox 정보 없음 — 레이아웃 정렬 스킵 (Paddle/EasyOCR bbox 사용 권장)",
            )
        ]

    sorted_blocks = sorted(blocks, key=_bbox_sort_key)
    lines: list[str] = []
    current_y: float | None = None
    row: list[str] = []

    for block in sorted_blocks:
        t = str(block.get("text", "")).strip()
        if not t:
            continue
        y, _ = _bbox_sort_key(block)
        if current_y is None or abs(y - current_y) <= line_threshold:
            row.append(t)
            current_y = y if current_y is None else (current_y + y) / 2
        else:
            if row:
                lines.append(" ".join(row))
            row = [t]
            current_y = y
    if row:
        lines.append(" ".join(row))

    if not lines:
        return text, [log_item("WARN", "레이아웃 정렬 결과 없음 — 원문 유지")]

    return "\n".join(lines), [
        log_item("INFO", f"문단·레이아웃 정렬 적용 ({len(sorted_blocks)} 블록)")
    ]


POSTPROCESS_STEP_FUNCS: dict[str, Callable[..., tuple[str, list]]] = {
    "strip_normalize": postprocess_strip_normalize,
    "format_rules": postprocess_format_rules,
    "char_correct": postprocess_char_correct,
    "layout_order": postprocess_layout_order,
}
