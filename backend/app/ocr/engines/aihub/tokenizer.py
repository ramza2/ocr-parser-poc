"""
AI Hub Swin-Transformer 토크나이저.

원본: AI Hub 공공 OCR 데이터 — swin_transformer/dataset.py Tokenizer 클래스 (MIT License)

token.pkl 파일에서 pickle 로 로드된 dict(token2id, id2token) 를 감싸는 래퍼.
원본 pickle 이 'dataset.Tokenizer' 모듈 경로를 참조하므로
_TokenizerUnpickler 로 우리 Tokenizer 클래스로 리다이렉트한다.
"""
from __future__ import annotations

import io
import pickle
from pathlib import Path


class Tokenizer:
    """문자 ↔ 정수 ID 매핑. 특수 토큰: PAD(0) BOS(1) EOS(2) OOV(3)."""

    def __init__(self, d: dict):
        self.token2id: dict[str, int] = d["token2id"]
        self.id2token: dict[int, str] = d["id2token"]

    def __len__(self) -> int:
        return len(self.token2id)

    def decode(self, labels) -> list[str]:
        pad = self.token2id["[PAD]"]
        bos = self.token2id["[BOS]"]
        eos = self.token2id["[EOS]"]

        texts = []
        for label in labels.tolist():
            text = ""
            for token_id in label:
                if token_id == bos:
                    continue
                if token_id in (pad, eos):
                    break
                text += self.id2token.get(token_id, "")
            texts.append(text)
        return texts


# pickle 이 참조할 수 있는 원본 모듈 경로들
_REDIRECT_MODULES = {
    ("dataset", "Tokenizer"),
    ("swin_transformer.dataset", "Tokenizer"),
}


class _TokenizerUnpickler(pickle.Unpickler):
    """원본 pickle 의 모듈 경로(dataset.Tokenizer 등)를 우리 Tokenizer 로 치환."""

    def find_class(self, module: str, name: str):
        if (module, name) in _REDIRECT_MODULES:
            return Tokenizer
        return super().find_class(module, name)


def load_tokenizer(path: str | Path) -> Tokenizer:
    """pickle 파일에서 Tokenizer 로드. 원본 모듈 경로를 자동 리다이렉트."""
    path = Path(path)
    with open(path, "rb") as f:
        obj = _TokenizerUnpickler(f).load()

    if isinstance(obj, Tokenizer):
        return obj
    if isinstance(obj, dict) and "token2id" in obj:
        return Tokenizer(obj)
    return Tokenizer({"token2id": obj.token2id, "id2token": obj.id2token})
