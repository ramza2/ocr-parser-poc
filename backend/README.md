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

프로젝트 루트: `docker compose up --build`

## API

| Method | URL | 설명 |
|--------|-----|------|
| GET | `/api/health` | 상태 |
| GET | `/api/parsers?extension=png` | 확장자별 파서 목록 |
| GET | `/api/pipeline-steps?extension=png` | 전처리/후처리 단계 목록 |
| POST | `/api/parse` | 파싱 (`file`, `parser_id`, `preprocess_steps`, `postprocess_steps`) |

`preprocess_steps` / `postprocess_steps`는 JSON 배열 문자열입니다.

예: `["resize","grayscale","binary","erosion","dilation"]`

## 파서 (확장자별 동적 노출)

**이미지** (jpg/png/…): `TESSERACT_OCR`, `EASYOCR`, `PADDLEOCR`, `TABLE_OCR`, `AUTO`

**PDF**: `PDF_TEXT`, `PDF_TESSERACT_OCR`, `PDF_EASYOCR`, `PDF_PADDLEOCR`, `TABLE_OCR`, `AUTO`

## 전처리 단계

| step_id | 설명 |
|---------|------|
| resize | 2배 확대 |
| grayscale | 그레이스케일 |
| clahe | 대비 enhancement |
| denoise | 노이즈 제거 |
| invert_dark | 어두운 배경 반전 |
| binary | 적응형 이진화 |
| erosion | 침식 |
| dilation | 팽창 |

프리셋 `tutorial_full`: resize → grayscale → binary → erosion → dilation  
([참고 튜토리얼](https://timemash.tistory.com/entry/%EB%A8%B8%EC%8B%A0%EB%9F%AC%EB%8B%9D-%EC%9D%B4%EB%AF%B8%EC%A7%80%EC%97%90%EC%84%9C-%ED%95%9C%EA%B8%80-%EC%9D%B8%EC%8B%9D%ED%95%98%EA%B8%B0-OpenCV-Tesseract-Hanspell))

## 후처리 단계

| step_id | 설명 |
|---------|------|
| hanspell | py-hanspell 맞춤법 교정 |

## 선택 설치

- **Tesseract** + `kor`/`eng`: `TESSERACT_OCR`, `PDF_TESSERACT_OCR`
- **Poppler**: PDF → 이미지
- **EasyOCR**: `EASYOCR`, `PDF_EASYOCR` (PyTorch 포함)
- **PaddleOCR**: `PADDLEOCR`, `PDF_PADDLEOCR` (`paddlepaddle` 포함, 용량 큼)
