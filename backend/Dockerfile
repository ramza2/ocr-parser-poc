FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    tesseract-ocr-kor \
    poppler-utils \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt requirements-paddle.txt requirements-aihub.txt ./
# paddle 설치 실패 시 빌드 중단 (|| true 제거 — 런타임 No module named 'paddle' 방지)
# VLM은 VLM_WORKER_URL 원격 워커 사용 시 서버에 torch/transformers 불필요
RUN pip install -r requirements.txt \
    && pip install -r requirements-paddle.txt \
    && pip install -r requirements-aihub.txt

COPY app ./app

# AI Hub 모델 가중치 (빌드 시 models/aihub/ 에 파일이 있으면 복사)
COPY models/aihub/ ./models/aihub/

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
