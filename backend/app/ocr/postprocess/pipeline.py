from __future__ import annotations

from app.ocr.postprocess.steps import POSTPROCESS_STEP_FUNCS
from app.utils.log_utils import log_item


def apply_postprocess(
    text: str,
    step_ids: list[str],
    options: dict | None = None,
) -> tuple[str, list]:
    if not step_ids or not text.strip():
        return text, []

    opts = options or {}
    logs: list = []
    current = text

    for step_id in step_ids:
        fn = POSTPROCESS_STEP_FUNCS.get(step_id)
        if not fn:
            logs.append(log_item("WARN", f"알 수 없는 후처리 단계 무시: {step_id}"))
            continue
        current, step_logs = fn(current, **opts.get(step_id, {}))
        logs.extend(step_logs)

    return current, logs
