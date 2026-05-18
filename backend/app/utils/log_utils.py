from datetime import datetime, timezone

from app.schemas.parser import LogItem


def log_item(level: str, message: str) -> LogItem:
    return LogItem(
        level=level,
        message=message,
        timestamp=datetime.now(timezone.utc).strftime("%H:%M:%S"),
    )
