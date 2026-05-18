# OCR Parser PoC Backend

## 실행 (Python 3.12 venv 권장 — PaddleOCR 포함)

```powershell
cd backend
.\setup-venv.ps1
.\.venv\Scripts\activate
uvicorn app.main:app --reload --port 8000
```

수동 설정:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt -r requirements-paddle.txt
uvicorn app.main:app --reload --port 8000
```

> **주의:** 시스템 기본 Python이 3.14이면 `python -m venv` 대신 반드시 `py -3.12 -m venv` 를 사용하세요.

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

**이미지** (jpg/png/…): `TESSERACT_OCR`, `EASYOCR`, `PADDLEOCR`, `AUTO`

**PDF**: `PDF_TEXT`, `PDF_TESSERACT_OCR`, `PDF_EASYOCR`, `PDF_PADDLEOCR`, `AUTO`

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
| hanspell | py-hanspell 맞춤법 교정 (`pip install -r requirements-hanspell.txt`) |

## 선택 설치

- **Tesseract** + `kor`/`eng`: `TESSERACT_OCR`, `PDF_TESSERACT_OCR`
- **Poppler**: PDF → 이미지
- **EasyOCR**: `EASYOCR`, `PDF_EASYOCR` (PyTorch 포함, Python 3.14에서도 대체로 동작)
- **PaddleOCR**: `PADDLEOCR`, `PDF_PADDLEOCR` — **Python 3.10~3.12만 지원** (3.14에서는 wheel 없음)

### PaddleOCR 설치 (Python 3.12 가상환경 권장)

현재 PC가 **Python 3.14**이면 `pip install paddlepaddle`이 실패합니다. PoC에서 Paddle만 쓰려면 3.12 venv를 따로 만드세요.

```bash
py -3.12 -m venv .venv312
.venv312\Scripts\activate
pip install -r requirements.txt
pip install -r requirements-paddle.txt
```

Windows CPU 공식 미러 예시:

```bash
pip install paddlepaddle==3.0.0 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/
pip install paddleocr>=2.7.0
```

Paddle 없이도 **Tesseract / EasyOCR** 비교는 `pip install -r requirements.txt` 만으로 가능합니다.
