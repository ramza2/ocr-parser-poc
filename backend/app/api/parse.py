from fastapi import APIRouter, File, Form, UploadFile

from app.services.parse_service import run_parse, validate_parse_request
from app.utils.file_utils import get_extension, remove_file, save_upload_to_temp

router = APIRouter()


@router.post("/parse")
async def parse_file(
    file: UploadFile = File(...),
    parser_id: str = Form(...),
):
    filename = file.filename or "unknown"
    content = await file.read()
    extension = get_extension(filename)

    validation = validate_parse_request(filename, parser_id, len(content))
    if validation:
        return validation

    temp_path = save_upload_to_temp(content, filename)
    try:
        return run_parse(temp_path, filename, parser_id)
    finally:
        remove_file(temp_path)
