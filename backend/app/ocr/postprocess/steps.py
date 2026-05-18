from __future__ import annotations

from collections.abc import Callable

from app.utils.log_utils import log_item


def postprocess_hanspell(text: str, **_kwargs) -> tuple[str, list]:
    logs = []
    try:
        from hanspell import spell_checker

        checked = spell_checker.check(text)
        result = checked.checked if checked else text
        logs.append(log_item("INFO", "Hanspell 맞춤법 교정 적용"))
        return result, logs
    except ImportError:
        logs.append(
            log_item("WARN", "hanspell 미설치 — pip install py-hanspell")
        )
        return text, logs
    except Exception as exc:
        logs.append(log_item("WARN", f"Hanspell 실패: {exc}"))
        return text, logs


POSTPROCESS_STEP_FUNCS: dict[str, Callable[..., tuple[str, list]]] = {
    "hanspell": postprocess_hanspell,
}
