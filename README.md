# OCR 파서 검증 PoC

PDF·이미지 파일을 업로드하고 OCR/문서 파서를 선택·실행하여 결과를 확인하는 내부 검토용 웹 도구입니다.

## 구조

```
ocr-parser-poc/
├── backend/     # FastAPI
├── frontend/    # React + TypeScript + Tailwind
└── docs/        # 기획서 추출본
```

## 빠른 시작

### 1. 백엔드

```bash
cd backend
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

### 2. 프론트엔드

```bash
cd frontend
npm install
npm run dev
```

브라우저: http://localhost:5173

## Docker (Linux 배포 / 환경 통일)

프로젝트 루트에서 한 번에 실행:

```bash
docker compose up --build
```

| 서비스 | URL |
|--------|-----|
| 웹 UI | http://localhost:8080 |
| API (직접) | http://localhost:8000 |

- 프론트 Nginx가 `/api` 요청을 backend로 전달합니다.
- 이미지에 **Tesseract(kor)** + **Poppler**가 포함됩니다. PaddleOCR는 Docker(Python 3.12)에서 `requirements-paddle.txt` 설치를 시도합니다.

**로컬 Python 3.14**에서는 PaddlePaddle wheel이 없어 `requirements.txt`만 설치하세요. Paddle 비교는 Docker 또는 Python 3.12 venv를 사용하세요.

백그라운드 실행:

```bash
docker compose up --build -d
docker compose down
```

## API

| Method | URL | 설명 |
|--------|-----|------|
| GET | `/api/health` | 서버 상태 |
| GET | `/api/parsers?extension=pdf` | 파서 목록 |
| POST | `/api/parse` | 파싱 실행 (`file`, `parser_id`) |

## 파서 (파일 확장자에 따라 UI에 동적 표시)

**이미지:** `TESSERACT_OCR` · `EASYOCR` · `PADDLEOCR` · `AUTO`

**PDF:** `PDF_TEXT` · `PDF_TESSERACT_OCR` · `PDF_EASYOCR` · `PDF_PADDLEOCR` · `AUTO`

**전처리/후처리:** 체크박스로 단계 선택 (튜토리얼 프리셋: 확대→grayscale→binary→erosion→dilation). OCR 파서에만 적용.

## Mock 모드

백엔드 없이 UI만 확인하려면 헤더의 **Mock 모드**를 켠 뒤 파서를 실행하세요.

## 선택 설치 (OCR)

- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) + 한글 데이터 (`kor`)
- [Poppler](https://github.com/oschwartz10612/poppler-windows/releases) (PDF 이미지 변환, PATH 등록)
