# OCR Parser PoC Backend

## 실행

```bash
cd backend
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

## 선택 의존성

- **Tesseract OCR**: `IMAGE_OCR`, `PDF_IMAGE_OCR`, `AUTO`(fallback)에 필요
- **Poppler**: `PDF_IMAGE_OCR`의 PDF→이미지 변환에 필요

## API

- `GET /api/health`
- `GET /api/parsers?extension=pdf`
- `POST /api/parse` (multipart: `file`, `parser_id`)
