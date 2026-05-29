"""VLM Q&A 보조 — 글자 수 질문 감지·deterministic counting."""
from __future__ import annotations

import re
from typing import Callable, Literal

CountPolicy = Literal["no_whitespace", "spaces_no_newlines", "all_chars"]

COUNT_CLARIFICATION_KO = (
    "공백과 줄바꿈을 포함해서 셀까요, 제외하고 셀까요?"
)

_COUNT_PATTERNS = re.compile(
    r"(몇\s*글자|글자\s*수|문자\s*수|몇\s*자|전부\s*몇|총\s*몇\s*글자|"
    r"character\s*count|how\s+many\s+characters|count\s+(the\s+)?characters)",
    re.IGNORECASE,
)


def is_count_question(question: str) -> bool:
    return bool(_COUNT_PATTERNS.search(question.strip()))


def parse_count_policy(question: str) -> CountPolicy | None:
    """질문에 공백/줄바꿈 규칙이 명시됐으면 정책 반환, 없으면 None."""
    q = question.strip().lower()
    compact = re.sub(r"\s+", "", q)

    no_space_markers = (
        "공백제외",
        "공백없",
        "띄어쓰기제외",
        "공백미포함",
        "withoutspace",
        "excludespace",
        "no space",
    )
    if any(m in compact or m in q for m in no_space_markers):
        return "no_whitespace"

    all_markers = ("줄바꿈포함", "줄바꿈 포함", "withnewline", "including line break")
    if any(m in compact or m in q for m in all_markers):
        return "all_chars"

    space_markers = (
        "공백포함",
        "공백 포함",
        "띄어쓰기포함",
        "withspace",
        "include space",
    )
    newline_exclude = ("줄바꿈제외", "줄바꿈 제외", "without newline")
    if any(m in compact or m in q for m in space_markers):
        return "spaces_no_newlines"
    if any(m in compact or m in q for m in newline_exclude):
        return "spaces_no_newlines"

    # "공백만 포함" 등 단독 명시 없이 줄바꿈만 제외한 경우
    if "줄바꿈제외" in compact:
        return "spaces_no_newlines"

    return None


def count_visible_chars(text: str, policy: CountPolicy) -> int:
    if policy == "no_whitespace":
        return len("".join(text.split()))
    if policy == "spaces_no_newlines":
        collapsed = text.replace("\r\n", "\n").replace("\r", "\n")
        return len(collapsed.replace("\n", ""))
    return len(text)


def count_policy_label(policy: CountPolicy, *, lang_ko: bool = True) -> str:
    if lang_ko:
        labels = {
            "no_whitespace": "공백·줄바꿈 제외",
            "spaces_no_newlines": "줄바꿈 제외, 단어 사이 공백 포함",
            "all_chars": "공백·줄바꿈 포함",
        }
    else:
        labels = {
            "no_whitespace": "excluding all whitespace",
            "spaces_no_newlines": "including spaces, excluding line breaks",
            "all_chars": "including spaces and line breaks",
        }
    return labels[policy]


def format_count_answer(count: int, policy: CountPolicy, *, lang_ko: bool = True) -> str:
    rule = count_policy_label(policy, lang_ko=lang_ko)
    if lang_ko:
        return f"총 {count}자 ({rule})"
    return f"Total {count} characters ({rule})"


def _question_lang_ko(question: str) -> bool:
    return bool(re.search(r"[가-힣]", question))


def try_answer_count_question(
    question: str,
    ocr_text_getter: Callable[[], str],
) -> str | None:
    """
    글자 수 질문 처리.
    - 규칙 미지정 → 확인 질문 (VLM 호출 없음)
    - 규칙 지정 → OCR 텍스트 + deterministic count
    - 글자 수 질문이 아니면 None
    """
    if not is_count_question(question):
        return None

    policy = parse_count_policy(question)
    lang_ko = _question_lang_ko(question)

    if policy is None:
        return COUNT_CLARIFICATION_KO if lang_ko else (
            "Should spaces and line breaks be included in the count?"
        )

    text = ocr_text_getter().strip()
    if not text:
        return (
            "이미지에서 글자를 읽지 못했습니다. OCR 후 다시 시도해 주세요."
            if lang_ko
            else "Could not read text from the image."
        )

    count = count_visible_chars(text, policy)
    return format_count_answer(count, policy, lang_ko=lang_ko)
