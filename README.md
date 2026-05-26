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

### CPU 스택 (PaddleOCR 2.7, GPU 없음)

```bash
docker compose up --build
```

### GPU 스택 (PaddleOCR 3.x + CUDA) — 권장 (3.x 검증용)

**사전 설치:** [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)

```bash
docker compose -f docker-compose.gpu.yml up --build
```

첫 기동 시 Paddle 모델 다운로드로 **수 분** 걸릴 수 있습니다.

| 서비스 | URL |
|--------|-----|
| 웹 UI | http://localhost:8080 |
| API (직접) | http://localhost:8000 |

| 파일 | 용도 |
|------|------|
| `backend/Dockerfile` | CPU, Paddle 2.7 (`requirements-paddle.txt`) |
| `backend/Dockerfile.gpu` | GPU, Paddle 3.3 + `requirements-paddle-v3.txt` |
| `docker-compose.gpu.yml` | GPU 백엔드 + 프론트 |

- 프론트 Nginx가 `/api` 요청을 backend로 전달합니다.
- CPU 이미지: Tesseract(kor) + Poppler + Paddle 2.7.
- GPU 이미지: 동일 + **paddlepaddle-gpu 3.3** (cu126 wheel).

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

**이미지:** `TESSERACT_OCR` · `EASYOCR` · `PADDLEOCR`

**스캔 PDF:** `PDF_TESSERACT_OCR` · `PDF_EASYOCR` · `PDF_PADDLEOCR` (페이지를 이미지로 렌더링 후 OCR)

**전처리/후처리:** 체크박스로 단계 선택 (튜토리얼 프리셋: 확대→grayscale→binary→erosion→dilation). OCR 파서에만 적용.

## 선택 설치 (OCR)

- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) + 한글 데이터 (`kor`)
- [Poppler](https://github.com/oschwartz10612/poppler-windows/releases) (PDF 이미지 변환, PATH 등록)
