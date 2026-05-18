from __future__ import annotations

from pydantic import BaseModel


class PipelineStepInfo(BaseModel):
    step_id: str
    name: str
    description: str
    applicable: list[str]
    default_order: int


PREPROCESS_CATALOG: list[PipelineStepInfo] = [
    PipelineStepInfo(
        step_id="resize",
        name="확대",
        description="이미지를 2배 확대합니다 (OpenCV resize).",
        applicable=["image", "pdf"],
        default_order=1,
    ),
    PipelineStepInfo(
        step_id="grayscale",
        name="Grayscale",
        description="그레이스케일로 변환합니다.",
        applicable=["image", "pdf"],
        default_order=2,
    ),
    PipelineStepInfo(
        step_id="clahe",
        name="CLAHE 대비",
        description="국소 대비를 enhancement 합니다.",
        applicable=["image", "pdf"],
        default_order=3,
    ),
    PipelineStepInfo(
        step_id="denoise",
        name="노이즈 제거",
        description="fastNlMeansDenoising을 적용합니다.",
        applicable=["image", "pdf"],
        default_order=4,
    ),
    PipelineStepInfo(
        step_id="invert_dark",
        name="어두운 배경 반전",
        description="칠판·야간 사진 등 어두운 배경을 반전합니다.",
        applicable=["image", "pdf"],
        default_order=5,
    ),
    PipelineStepInfo(
        step_id="binary",
        name="Binary (이진화)",
        description="적응형 이진화(adaptiveThreshold)를 적용합니다.",
        applicable=["image", "pdf"],
        default_order=6,
    ),
    PipelineStepInfo(
        step_id="erosion",
        name="Erosion (침식)",
        description="형태학적 침식 연산을 적용합니다.",
        applicable=["image", "pdf"],
        default_order=7,
    ),
    PipelineStepInfo(
        step_id="dilation",
        name="Dilation (팽창)",
        description="형태학적 팽창 연산을 적용합니다.",
        applicable=["image", "pdf"],
        default_order=8,
    ),
]

POSTPROCESS_CATALOG: list[PipelineStepInfo] = [
    PipelineStepInfo(
        step_id="hanspell",
        name="Hanspell 맞춤법",
        description="네이버 맞춤법 검사기 기반 한글 교정 (py-hanspell).",
        applicable=["image", "pdf"],
        default_order=1,
    ),
]

# 블로그 권장 순서 프리셋
PRESET_FULL_TUTORIAL = [
    "resize",
    "grayscale",
    "binary",
    "erosion",
    "dilation",
]


def filter_steps(
    catalog: list[PipelineStepInfo],
    file_kind: str,
) -> list[PipelineStepInfo]:
    return [s for s in catalog if file_kind in s.applicable]


def file_kind_from_extension(ext: str) -> str:
    return "pdf" if ext.lower() in ("pdf",) else "image"
