"""VLM Q&A 프롬프트 단위 테스트."""
from app.ocr.engines.vlm_qwen import QWEN_VL_VARIANTS, QwenVlmEngine


def test_build_qa_prompt_general_policy():
    engine = QwenVlmEngine(QWEN_VL_VARIANTS[1])
    prompt = engine._build_qa_prompt("놋다리밟기에 해당하는 영문 표기는?")
    assert "[Calculation Policy]" in prompt
    assert "[Evidence Policy]" in prompt
    assert "[Answer Policy]" in prompt
    assert "놋다리밟기에 해당하는 영문 표기는?" in prompt
    assert "Do not provide the full OCR transcription" in prompt
