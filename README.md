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

**이미지:** `TESSERACT_OCR` · `EASYOCR` · `PADDLEOCR` · `AIHUB_SWIN_OCR`

**스캔 PDF:** `PDF_TESSERACT_OCR` · `PDF_EASYOCR` · `PDF_PADDLEOCR` · `PDF_AIHUB_SWIN_OCR` (페이지를 이미지로 렌더링 후 OCR)

## VLM 엔진 (UI 상단 "VLM" 탭)

Vision Language Model 기반 OCR — 문맥을 이해하는 차세대 문서 AI.

| 엔진 | 모델 | VRAM | 특성 |
|------|------|------|------|
| Qwen2.5-VL | Qwen/Qwen2.5-VL-7B-Instruct | ~8GB | 한국어 최강, Schema 추출/Q&A |
| GOT-OCR 2.0 | stepfun-ai/GOT-OCR-2.0-hf | ~6GB | OCR 특화, 수식/표 |
| Florence-2 | microsoft/Florence-2-large | ~2GB | 경량, Bounding Box 내장 |

**기능:** 전체 OCR · Schema 기반 Key-Value 추출 · 문서 Q&A · Confidence 시각화 · Bounding Box 오버레이

**요구사항:** GPU (최소 RTX 3060 12GB), 모델은 최초 실행 시 HuggingFace 에서 자동 다운로드.

### AI Hub CRAFT + Swin-Transformer OCR

AI Hub 공공 OCR 데이터에서 제공하는 2단계 모델입니다.

1. **CRAFT** — VGG16-BN 백본으로 문자 영역(바운딩 박스) 검출
2. **Swin-Transformer** — 각 영역을 크롭하여 문자 인식

모델 가중치 3개를 `backend/models/aihub/` 에 배치해야 합니다:

| 파일 | 설명 |
|------|------|
| `craft.ckpt` | CRAFT 텍스트 검출 가중치 |
| `swin_transformer.ckpt` | Swin 문자 인식 가중치 |
| `token.pkl` | Swin 토크나이저 (문자↔ID 매핑) |

환경변수 `AIHUB_MODEL_DIR` 로 경로를 변경할 수 있습니다.

**전처리/후처리:** 체크박스로 단계 선택 (튜토리얼 프리셋: 확대→grayscale→binary→erosion→dilation). OCR 파서에만 적용.

## 선택 설치 (OCR)

- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) + 한글 데이터 (`kor`)
- [Poppler](https://github.com/oschwartz10612/poppler-windows/releases) (PDF 이미지 변환, PATH 등록)
