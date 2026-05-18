import mimetypes
import os
import tempfile
import uuid
from pathlib import Path

SUPPORTED_EXTENSIONS = {".pdf", ".jpg", ".jpeg", ".png", ".tif", ".tiff"}
MAX_FILE_SIZE_BYTES = 50 * 1024 * 1024


def get_extension(filename: str) -> str:
    return Path(filename).suffix.lower().lstrip(".")


def is_supported_extension(extension: str) -> bool:
    return f".{extension.lower()}" in SUPPORTED_EXTENSIONS


def guess_mime_type(filename: str) -> str:
    mime, _ = mimetypes.guess_type(filename)
    return mime or "application/octet-stream"


def save_upload_to_temp(content: bytes, filename: str) -> str:
    suffix = Path(filename).suffix or ""
    temp_dir = tempfile.gettempdir()
    path = os.path.join(temp_dir, f"ocr_poc_{uuid.uuid4().hex}{suffix}")
    with open(path, "wb") as f:
        f.write(content)
    return path


def remove_file(path: str | None) -> None:
    if path and os.path.isfile(path):
        try:
            os.remove(path)
        except OSError:
            pass
