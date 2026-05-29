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

| 명령 | 스택 |
|------|------|
| `docker compose up --build` | CPU, PaddleOCR **2.7** (`Dockerfile` + `requirements-paddle.txt`) |
| `docker compose -f docker-compose.gpu.yml up --build` | GPU, PaddleOCR **3.x** (`Dockerfile.gpu` + `requirements-paddle-v3.txt`) |

### PaddleOCR 3.x + GPU 테스트 (Docker)

**1. 사전 확인 (호스트)**

```bash
nvidia-smi
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

**2. 빌드·실행** (프로젝트 루트)

```bash
docker compose -f docker-compose.gpu.yml up --build
```

**3. GPU·버전 확인**

```bash
curl http://localhost:8000/api/health
```

`gpu.paddle_use_gpu: true`, `paddleocr` 3.x 버전이 보이면 정상입니다.

컨테이너 안에서 상세 확인:

```bash
docker exec -it ocr-parser-poc-backend-gpu python scripts/verify_paddle_gpu.py
```

**4. OCR API 테스트**

```bash
curl -X POST http://localhost:8000/api/parse \
  -F "file=@/path/to/test.png" \
  -F "parser_id=PADDLEOCR" \
  -F "preprocess_steps=[]" \
  -F "postprocess_steps=[]"
```

**5. UI**

브라우저 http://localhost:8080 → 파일 업로드 → **PaddleOCR** 선택 → 실행.

**Windows 로컬 venv** 는 2.7 유지, 3.x는 Docker(Linux)에서만 사용합니다 (`paddle_engine`이 Windows pip 3.x는 차단).

## API

| Method | URL | 설명 |
|--------|-----|------|
| GET | `/api/health` | 상태 |
| GET | `/api/parsers?extension=png` | 확장자별 파서 목록 |
| GET | `/api/pipeline-steps?extension=png` | 전처리/후처리 단계 목록 |
| POST | `/api/parse` | 파싱 (`file`, `parser_id`, `preprocess_steps`, `postprocess_steps`) |
| GET | `/api/vlm/models` | VLM 모델 목록 + 로드 상태 |
| POST | `/api/vlm/load` | VLM 모델 로드/전환 (`model_id`) |
| POST | `/api/vlm/ocr` | VLM 전체 텍스트 추출 (`file`, `model_id`) |
| POST | `/api/vlm/extract` | VLM Schema 추출 (`file`, `model_id`, `schema_json`) |
| POST | `/api/vlm/ask` | VLM Q&A (`file`, `model_id`, `question`) |

`preprocess_steps` / `postprocess_steps`는 JSON 배열 문자열입니다.

예: `["resize","grayscale","binary","erosion","dilation"]`

## 파서 (확장자별 동적 노출)

**이미지** (jpg/png/…): `TESSERACT_OCR`, `EASYOCR`, `PADDLEOCR`, `AIHUB_SWIN_OCR`

**스캔 PDF**: `PDF_TESSERACT_OCR`, `PDF_EASYOCR`, `PDF_PADDLEOCR`, `PDF_AIHUB_SWIN_OCR` (각 페이지를 이미지로 렌더링 후 OCR. 텍스트 레이어 PDF는 대상 아님)

### AI Hub CRAFT + Swin-Transformer

CRAFT(텍스트 검출) + Swin-Transformer(문자 인식) 2단계 파이프라인.
모델 가중치 3개(`craft.ckpt`, `swin_transformer.ckpt`, `token.pkl`)를 `models/aihub/` 에 배치 후 사용.
환경변수 `AIHUB_MODEL_DIR` 로 경로 변경 가능. 추가 의존성은 `requirements-aihub.txt`.

## 전처리 단계 (필요 시)

| step_id | 설명 |
|---------|------|
| deskew | 회전 보정 — 기울어진 문서를 수평에 맞춤 |
| enhance | 명암비(CLAHE)·선명화·해상도 확대 |
| binarize | 이진화 — 노이즈·흐린 글씨 대비 강화 |
| crop_roi | 텍스트 ROI만 크롭 |

프리셋 `scanned_doc`: deskew → enhance → binarize

## 후처리 단계 (권장)

| step_id | 설명 |
|---------|------|
| strip_normalize | 공백·제어문자·노이즈 정리 |
| format_rules | 전화번호·주민등록번호 포맷 교정 |
| char_correct | 숫자 구간 O/0, l/1 교정 |
| layout_order | bbox 기준 읽기 순서 정렬 (PP-Structure 대용 PoC) |

프리셋 `postprocess_essential`: strip_normalize → format_rules → char_correct

### VLM 엔진 (Qwen2.5-VL, GOT-OCR2.0, Florence-2)

Vision Language Model 기반 OCR/Schema 추출/Q&A 엔진.
의존성은 `requirements-vlm.txt`. 모델은 최초 실행 시 HuggingFace 에서 자동 다운로드.

| 엔진 | 모델 | VRAM (4-bit) | 특성 |
|------|------|-------------|------|
| `qwen_vl` | Qwen2.5-VL-7B-Instruct | ~8GB | 한국어 최강, Schema 추출 우수 |
| `got_ocr` | GOT-OCR2.0 | ~6GB | OCR 특화, 수식/표 지원 |
| `florence` | Florence-2-large | ~2GB | 경량, grounding(bbox) 내장 |

UI 상단의 **VLM** 탭에서 사용. 한 번에 하나의 VLM 모델만 GPU 에 로드됩니다.

#### VLM 원격 워커 (GPU PC + Ubuntu 서버 분리)

GPU가 없는 Ubuntu 서버에서는 OCR/API만 두고, VLM은 GPU PC에서 **전용 워커**로 실행합니다.

| 구성 | 역할 | Compose |
|------|------|---------|
| **GPU PC** | VLM 추론 전용 (`/api/vlm/*`) | `docker-compose.vlm-worker.yml` → `:8001` |
| **Ubuntu 서버** | OCR·API·프론트, VLM은 원격 프록시 | `docker-compose.yml` + `VLM_WORKER_URL` |

**1) GPU PC (개발 PC) — VLM 워커**

```bash
# NVIDIA Container Toolkit 설치 후
docker compose -f docker-compose.vlm-worker.yml up --build -d

# 확인
curl http://localhost:8001/api/health
curl http://localhost:8001/api/vlm/models
```

- HuggingFace 캐시: Docker volume `vlm_hf_cache` (또는 `HF_HOME=/cache/huggingface`)
- Windows PC IP 확인: `ipconfig` → Ubuntu에서 접근 가능한 LAN IP 사용

**2) Ubuntu 서버 — 메인 스택**

프로젝트 루트에 `.env` 파일:

```env
VLM_WORKER_URL=http://192.168.0.10:8001
```

(`192.168.0.10` → GPU PC의 실제 IP)

```bash
docker compose up --build -d

# VLM 원격 모드 확인
curl http://localhost:8000/api/health
# → "vlm_mode": "remote", "vlm_worker_url": "http://..."
```

**3) 동작**

- 프론트 `/api/vlm/*` → Ubuntu backend → GPU PC `:8001` 프록시
- Ubuntu backend 이미지에는 torch/transformers 미포함 (경량)
- GPU PC 워커 이미지에는 Paddle 미포함 (VRAM 절약)

**4) 로컬 개발 (워커 없이)**

`VLM_WORKER_URL`을 비우고 backend에서 `requirements-vlm.txt` 설치 후 기존처럼 로컬 VLM 사용.

## 선택 설치

- **Tesseract** + `kor`/`eng`: `TESSERACT_OCR`, `PDF_TESSERACT_OCR`
- **Poppler**: PDF → 이미지
- **EasyOCR**: `EASYOCR`, `PDF_EASYOCR` (PyTorch 포함, Python 3.14에서도 대체로 동작)
- **PaddleOCR**: `PADDLEOCR`, `PDF_PADDLEOCR` — **Python 3.10~3.12만 지원** (3.14에서는 wheel 없음)

### PaddleOCR 설치 (Python 3.12 venv)

**권장:** `requirements-paddle.txt`는 **PaddleOCR 2.7** 기준입니다 (Windows에서 3.x oneDNN 오류 회피).

```powershell
.\.venv\Scripts\activate
pip uninstall paddlepaddle paddleocr paddlex -y
pip install -r requirements-paddle.txt
```

`Unknown argument: use_gpu` / `oneDNN` 오류 → PaddleOCR **3.x**가 설치된 상태입니다. 위처럼 **2.7로 재설치**하세요.

`paddleocr 미설치` / `paddleocr 로드 실패` / `cv2 has no attribute INTER_NEAREST` → **paddleocr는 설치됐지만 import 실패**인 경우가 많습니다. `opencv-python-headless`(4.10+)와 PaddleOCR용 OpenCV(4.6)가 섞이면 발생합니다.

```powershell
pip uninstall opencv-python-headless -y
pip install opencv-contrib-python==4.6.0.66 opencv-python==4.6.0.66
python -c "import paddleocr; print(paddleocr.__version__)"
```

GPU는 `paddlepaddle-gpu`만 있으면 됩니다 (`paddleocr` 패키지 이름은 동일). 설치 후 **uvicorn 재시작**하세요.

### GPU (EasyOCR·PaddleOCR)

CUDA가 **실제로 동작할 때만** GPU를 씁니다 (`torch.cuda` / Paddle GPU 텐서 프로브).  
cuBLAS DLL이 없으면 Paddle은 **자동으로 CPU**로 폴백합니다.

`GET /api/health` → `gpu.paddle_use_gpu`, `gpu.paddle_gpu_note` 확인.

#### `cublas64_118.dll` / error code 126

`paddlepaddle-gpu`는 **pip 패키지**만으로 끝나지 않고, PC에 **CUDA Toolkit의 cuBLAS DLL**이 PATH에 있어야 합니다.  
`nvidia-smi`의 CUDA 12.6(드라이버)과 Paddle wheel이 요구하는 **런타임 CUDA 버전**은 다를 수 있습니다.

| 증상 | 의미 |
|------|------|
| `cublas64_118.dll` | Paddle wheel이 **CUDA 11.8** 런타임을 찾는 경우가 많음 |
| error code 126 | Windows에서 해당 DLL을 **PATH에서 못 찾음** |

**해결 (택 1)**

1. **CUDA Toolkit 11.8** 설치 후 PATH 추가 (Paddle 2.6 GPU wheel과 맞출 때)
   - [NVIDIA CUDA 11.8](https://developer.nvidia.com/cuda-11-8-0-download-archive) 설치
   - 시스템 PATH에 추가: `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin`
   - 터미널·uvicorn **재시작** 후 확인:
   ```powershell
   python -c "from app.utils.gpu_config import paddle_use_gpu; print(paddle_use_gpu())"
   ```

2. **당장 OCR만 필요** — CPU로 사용 (코드가 자동 폴백). 또는:
   ```powershell
   pip uninstall paddlepaddle-gpu -y
   pip install paddlepaddle==2.6.2
   ```

3. **EasyOCR만 GPU** — PyTorch CUDA는 드라이버만으로 동작하는 경우가 많음. Paddle만 CPU로 쓰면 됨.

**EasyOCR (PyTorch CUDA)**

```powershell
pip uninstall torch torchvision torchaudio -y
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
python -c "import torch; print(torch.cuda.is_available())"
```

**PaddleOCR (paddlepaddle-gpu)**

```powershell
pip uninstall paddlepaddle paddlepaddle-gpu -y
pip install paddlepaddle-gpu==2.6.2 -f https://www.paddlepaddle.org.cn/whl/windows/mkl/avx/stable.html
python -c "import paddle; paddle.device.set_device('gpu:0'); print(paddle.to_tensor([1.0]))"
```

마지막 명령이 실패하면 CUDA Toolkit/PATH를 맞추거나 CPU wheel을 쓰세요.

Paddle 없이도 **Tesseract / EasyOCR** 비교는 `pip install -r requirements.txt` 만으로 가능합니다.
