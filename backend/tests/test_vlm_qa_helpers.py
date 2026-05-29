"""VLM Q&A helper 단위 테스트."""
from app.ocr.engines.vlm_qa_helpers import (
    COUNT_CLARIFICATION_KO,
    count_visible_chars,
    is_count_question,
    parse_count_policy,
    try_answer_count_question,
)
from app.ocr.engines.vlm_qwen import QwenVlmEngine

SIGN_TEXT = (
    "안동놋다리밟기전수교육관\n"
    "安東놋다리밟기傳修敎育館\n"
    "Andong Nottaribalggi Training Center"
)


def test_build_qa_prompt_partial_span_rules():
    prompt = QwenVlmEngine._build_qa_prompt("놋다리밟기에 해당하는 영문 표기는?")
    assert "smallest visible text span" in prompt
    assert "Nottaribalggi" in prompt
    assert "Do not answer character-count questions yourself" in prompt


def test_count_question_ambiguous_returns_clarification():
    answer = try_answer_count_question("전부 몇글자인가?", lambda: SIGN_TEXT)
    assert answer == COUNT_CLARIFICATION_KO


def test_count_question_with_no_whitespace_policy():
    answer = try_answer_count_question(
        "전부 몇 글자인가? 공백 제외", lambda: SIGN_TEXT
    )
    assert answer is not None
    assert "57" in answer
    assert count_visible_chars(SIGN_TEXT, "no_whitespace") == 57


def test_count_question_with_spaces_policy():
    answer = try_answer_count_question(
        "글자 수 알려줘. 공백 포함 줄바꿈 제외", lambda: SIGN_TEXT
    )
    assert answer is not None
    assert "60" in answer
    assert count_visible_chars(SIGN_TEXT, "spaces_no_newlines") == 60


def test_is_count_question():
    assert is_count_question("전부 몇글자인가?")
    assert is_count_question("how many characters are there?")
    assert not is_count_question("놋다리밟기 영문 표기는?")


def test_parse_count_policy():
    assert parse_count_policy("공백 제외로 세어줘") == "no_whitespace"
    assert parse_count_policy("공백 포함") == "spaces_no_newlines"
    assert parse_count_policy("전부 몇글자") is None


def test_non_count_returns_none():
    assert try_answer_count_question("영문 표기는?", lambda: SIGN_TEXT) is None
