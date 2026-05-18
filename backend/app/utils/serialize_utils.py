from __future__ import annotations

import json
from typing import Any


def make_json_safe(value: Any) -> Any:
    """numpy 등 JSON 비호환 타입을 Python 기본 타입으로 변환합니다."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value

    try:
        import numpy as np

        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
    except ImportError:
        pass

    if isinstance(value, dict):
        return {str(k): make_json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [make_json_safe(v) for v in value]

    return str(value)


def sanitize_blocks(blocks: list[dict]) -> list[dict]:
    return make_json_safe(blocks)
