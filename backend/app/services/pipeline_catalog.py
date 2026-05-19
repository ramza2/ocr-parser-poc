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
        step_id="deskew",
        name="회전 보정 (Deskew)",
        description="기울어진 문서를 수평에 가깝게 맞춥니다.",
        applicable=["image", "pdf"],
        default_order=1,
    ),
    PipelineStepInfo(
        step_id="enhance",
        name="명암비·해상도",
        description="CLAHE 대비 향상, 선명화, 필요 시 확대합니다.",
        applicable=["image", "pdf"],
        default_order=2,
    ),
    PipelineStepInfo(
        step_id="binarize",
        name="이진화 (Binarization)",
        description="배경 노이즈·흐린 글씨 시 흑백 대비를 뚜렷하게 합니다.",
        applicable=["image", "pdf"],
        default_order=3,
    ),
    PipelineStepInfo(
        step_id="crop_roi",
        name="크롭 (ROI)",
        description="텍스트가 있는 영역만 잘라 OCR 부담을 줄입니다.",
        applicable=["image", "pdf"],
        default_order=4,
    ),
]

POSTPROCESS_CATALOG: list[PipelineStepInfo] = [
    PipelineStepInfo(
        step_id="strip_normalize",
        name="공백·특수문자 정리",
        description="strip(), 제어문자·노이즈 제거, 공백 정규화.",
        applicable=["image", "pdf"],
        default_order=1,
    ),
    PipelineStepInfo(
        step_id="format_rules",
        name="도메인 포맷 교정",
        description="전화번호(010-0000-0000), 주민등록번호 형식 정규화.",
        applicable=["image", "pdf"],
        default_order=2,
    ),
    PipelineStepInfo(
        step_id="char_correct",
        name="문자 혼동 교정",
        description="숫자 구간에서 O/0, l/1 등 흔한 OCR 오인식 교정.",
        applicable=["image", "pdf"],
        default_order=3,
    ),
    PipelineStepInfo(
        step_id="layout_order",
        name="문단·레이아웃 정렬",
        description="bbox 기준 위→아래·좌→우 재배열 (PP-Structure 대용 PoC).",
        applicable=["image", "pdf"],
        default_order=4,
    ),
]

# 스캔·사진 문서 권장 전처리 순서
PRESET_SCANNED_DOC = ["deskew", "enhance", "binarize"]

# 후처리 기본 권장 (거의 필수)
PRESET_POSTPROCESS_ESSENTIAL = [
    "strip_normalize",
    "format_rules",
    "char_correct",
]


def filter_steps(
    catalog: list[PipelineStepInfo],
    file_kind: str,
) -> list[PipelineStepInfo]:
    return [s for s in catalog if file_kind in s.applicable]


def file_kind_from_extension(ext: str) -> str:
    return "pdf" if ext.lower() in ("pdf",) else "image"
