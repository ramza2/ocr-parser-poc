#!/usr/bin/env bash
# Ubuntu GPU 서버 — NVIDIA + Docker GPU + (선택) VLM 워커 점검/설치
#
# 사용:
#   chmod +x scripts/setup-gpu-ubuntu.sh
#   ./scripts/setup-gpu-ubuntu.sh check          # 점검만
#   ./scripts/setup-gpu-ubuntu.sh install-toolkit # NVIDIA Container Toolkit 설치
#   ./scripts/setup-gpu-ubuntu.sh verify-docker   # Docker GPU 동작 확인
#   ./scripts/setup-gpu-ubuntu.sh vlm-worker     # 프로젝트 루트에서 VLM 워커 기동
#
set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

ok()   { echo -e "${GREEN}[OK]${NC} $*"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $*"; }
fail() { echo -e "${RED}[FAIL]${NC} $*"; exit 1; }

section() { echo ""; echo "======== $* ========"; }

check_host() {
  section "1. OS / GPU 드라이버"
  if [[ "$(uname -s)" != "Linux" ]]; then
    fail "Linux(Ubuntu)에서 실행하세요."
  fi
  . /etc/os-release 2>/dev/null || true
  echo "OS: ${PRETTY_NAME:-unknown}"

  if command -v nvidia-smi &>/dev/null; then
    ok "nvidia-smi 사용 가능"
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
    nvidia-smi
  else
    fail "nvidia-smi 없음. NVIDIA 드라이버 설치 필요: sudo ubuntu-drivers install && sudo reboot"
  fi
}

check_docker() {
  section "2. Docker"
  if ! command -v docker &>/dev/null; then
    fail "Docker 미설치. 설치: https://docs.docker.com/engine/install/ubuntu/"
  fi
  ok "docker $(docker --version)"
  if docker compose version &>/dev/null; then
    ok "docker compose $(docker compose version)"
  elif command -v docker-compose &>/dev/null; then
    ok "docker-compose $(docker-compose --version)"
  else
    fail "docker compose 플러그인 없음"
  fi
  if groups "$USER" | grep -q docker; then
    ok "사용자 $USER 가 docker 그룹에 포함됨"
  else
    warn "docker 그룹 없음 → sudo docker ... 또는: sudo usermod -aG docker $USER && newgrp docker"
  fi
}

check_nvidia_ctk() {
  section "3. NVIDIA Container Toolkit"
  if command -v nvidia-ctk &>/dev/null; then
    ok "nvidia-ctk $(nvidia-ctk --version 2>/dev/null | head -1)"
  else
    fail "nvidia-ctk 없음. 설치: ./scripts/setup-gpu-ubuntu.sh install-toolkit"
  fi
  if docker info 2>/dev/null | grep -qi nvidia; then
    ok "docker info에 NVIDIA 런타임 등록됨"
  else
    warn "docker info에 nvidia 미표시 — install-toolkit 후 docker 재시작 필요"
  fi
}

verify_docker_gpu() {
  section "4. Docker 컨테이너 GPU 테스트"
  docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi \
    && ok "컨테이너에서 GPU 인식 성공" \
    || fail "docker --gpus all 실패 (Toolkit/드라이버 확인)"
}

install_toolkit() {
  section "NVIDIA Container Toolkit 설치"
  if [[ $EUID -ne 0 ]]; then
    fail "root 필요: sudo ./scripts/setup-gpu-ubuntu.sh install-toolkit"
  fi
  apt-get update
  apt-get install -y curl ca-certificates gnupg
  curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
    | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
  curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
    | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
    | tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
  apt-get update
  apt-get install -y nvidia-container-toolkit
  nvidia-ctk runtime configure --runtime=docker
  systemctl restart docker
  ok "설치 완료 — verify-docker 로 확인하세요"
}

start_vlm_worker() {
  section "5. VLM 워커 (ocr-parser-poc)"
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
  if [[ ! -f "$ROOT/docker-compose.vlm-worker.yml" ]]; then
    fail "프로젝트 루트를 찾을 수 없습니다: $ROOT"
  fi
  cd "$ROOT"
  docker compose -f docker-compose.vlm-worker.yml up --build -d
  ok "기동 요청 완료 (첫 빌드는 수십 분 소요 가능)"
  echo "대기 후 확인:"
  echo "  curl -s http://127.0.0.1:8001/api/health"
  echo "  curl -s http://127.0.0.1:8001/api/vlm/models"
  echo "Ubuntu 메인 서버 .env: VLM_WORKER_URL=http://<이서버LAN_IP>:8001"
}

verify_vlm_torch() {
  section "6. VLM 워커 내부 CUDA (컨테이너 실행 중일 때)"
  if ! docker ps --format '{{.Names}}' | grep -q '^ocr-vlm-worker$'; then
    warn "ocr-vlm-worker 컨테이너 없음 — vlm-worker 먼저 기동"
    return 0
  fi
  docker exec ocr-vlm-worker python -c \
    "import torch; print('cuda_available=', torch.cuda.is_available()); print('device=', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"
  curl -sf http://127.0.0.1:8001/api/health && echo "" && ok "health OK" || warn "health 실패 (모델 로드 대기 중일 수 있음)"
}

cmd="${1:-check}"
case "$cmd" in
  check)
    check_host
    check_docker
    check_nvidia_ctk || true
    verify_docker_gpu 2>/dev/null || warn "Docker GPU 테스트 스킵/실패 — install-toolkit 후 verify-docker"
    verify_vlm_torch || true
    ;;
  install-toolkit) install_toolkit ;;
  verify-docker)
    check_host
    check_docker
    check_nvidia_ctk
    verify_docker_gpu
    ;;
  vlm-worker) start_vlm_worker ;;
  *)
    echo "Usage: $0 {check|install-toolkit|verify-docker|vlm-worker}"
    exit 1
    ;;
esac

echo ""
ok "완료 ($cmd)"
