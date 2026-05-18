# OCR Parser PoC Backend

## 실행

```bash
cd backend
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

## Docker

프로젝트 루트에서:

```bash
docker compose up --build
```

- API: http://localhost:8000
- 웹 UI: http://localhost:8080 (Nginx가 `/api`를 backend로 프록시)

Docker 이미지에는 Tesseract(`kor` 포함)와 Poppler가 포함됩니다.

## 로컬 실행 시 선택 의존성 (Windows)

- **Tesseract OCR**: `IMAGE_OCR`, `PDF_IMAGE_OCR`, `AUTO`(fallback)에 필요
- **Poppler**: `PDF_IMAGE_OCR`의 PDF→이미지 변환에 필요

## API

- `GET /api/health`
- `GET /api/parsers?extension=pdf`
- `POST /api/parse` (multipart: `file`, `parser_id`)
