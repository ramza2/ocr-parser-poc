import json

from fastapi import APIRouter, File, Form, UploadFile

from app.services.parse_service import run_parse, validate_parse_request
from app.utils.file_utils import get_extension, remove_file, save_upload_to_temp

router = APIRouter()


def _parse_json_list(raw: str | None) -> list[str]:
    if not raw or raw.strip() in ("", "[]"):
        return []
    try:
        data = json.loads(raw)
        return list(data) if isinstance(data, list) else []
    except json.JSONDecodeError:
        return []


@router.post("/parse")
async def parse_file(
    file: UploadFile = File(...),
    parser_id: str = Form(...),
    preprocess_steps: str = Form(default="[]"),
    postprocess_steps: str = Form(default="[]"),
):
    filename = file.filename or "unknown"
    content = await file.read()
    extension = get_extension(filename)

    validation = validate_parse_request(filename, parser_id, len(content))
    if validation:
        return validation

    parse_options = {
        "preprocess_steps": _parse_json_list(preprocess_steps),
        "postprocess_steps": _parse_json_list(postprocess_steps),
    }

    temp_path = save_upload_to_temp(content, filename)
    try:
        return run_parse(temp_path, filename, parser_id, parse_options)
    finally:
        remove_file(temp_path)
