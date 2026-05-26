"""
GPU 사용 여부 판별 (엔진별로 독립).

- Paddle: paddlepaddle-gpu 설치 + cuBLAS 등 CUDA DLL 실제 동작 여부 프로브
- EasyOCR: torch.cuda.is_available() (Docker GPU 이미지는 EASYOCR_FORCE_CPU=1 권장)

Windows 로컬: Paddle 2.7 + CUDA Toolkit PATH, Docker: Dockerfile.gpu (Paddle 3.3 cu126).
"""
from __future__ import annotations

import os

# 프로브 결과 캐시 (요청마다 CUDA 초기화 비용 절감)
_paddle_gpu_ok: bool | None = None


def easyocr_use_gpu() -> bool:
    if os.environ.get("EASYOCR_FORCE_CPU", "").strip().lower() in (
        "1",
        "true",
        "yes",
    ):
        return False
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:
        return False


def _probe_paddle_gpu() -> bool:
    """paddlepaddle-gpu 설치 + CUDA/cuBLAS DLL이 실제로 동작하는지 확인."""
    try:
        import paddle

        if not paddle.device.is_compiled_with_cuda():
            return False
        paddle.device.set_device("gpu:0")
        _ = paddle.to_tensor([1.0], place="gpu:0")
        return True
    except Exception:
        return False


def paddle_use_gpu() -> bool:
    global _paddle_gpu_ok
    if _paddle_gpu_ok is None:
        _paddle_gpu_ok = _probe_paddle_gpu()
    return _paddle_gpu_ok


def reset_paddle_gpu_cache() -> None:
    global _paddle_gpu_ok
    _paddle_gpu_ok = None


def aihub_use_gpu() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def engine_device_label(engine_id: str) -> str | None:
    if engine_id == "easyocr":
        return f"EasyOCR: {'GPU' if easyocr_use_gpu() else 'CPU'}"
    if engine_id == "paddleocr":
        return f"PaddleOCR: {'GPU' if paddle_use_gpu() else 'CPU'}"
    if engine_id == "aihub_swin":
        return f"AI Hub CRAFT+Swin: {'GPU' if aihub_use_gpu() else 'CPU'}"
    return None


def gpu_runtime_status() -> dict:
    status: dict = {
        "easyocr_use_gpu": easyocr_use_gpu(),
        "paddle_use_gpu": paddle_use_gpu(),
    }
    try:
        import torch

        status["easyocr_torch_cuda"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            status["easyocr_device"] = torch.cuda.get_device_name(0)
    except ImportError:
        status["easyocr_torch_cuda"] = None

    try:
        import paddle

        status["paddle_cuda_compiled"] = paddle.device.is_compiled_with_cuda()
        if status["paddle_cuda_compiled"]:
            try:
                status["paddle_cuda_device_count"] = paddle.device.cuda.device_count()
            except Exception:
                status["paddle_cuda_device_count"] = None
    except Exception as exc:
        status["paddle_cuda_compiled"] = None
        status["paddle_import_error"] = str(exc)[:200]

    if status.get("paddle_cuda_compiled") and not status["paddle_use_gpu"]:
        status["paddle_gpu_note"] = (
            "GPU 빌드이나 cuBLAS/CUDA DLL 미설치·PATH 미설정 — CPU로 동작"
        )

    return status
