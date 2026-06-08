import type { VlmModelInfo } from "../types/vlm";

/** UI에서 숨길 구/미지원 모델 (워커 미배포 시 API에 남아 있어도 필터) */
export const DEPRECATED_VLM_MODEL_IDS = new Set([
  "qwen3_vl_8b",
  "got_ocr",
  "qwen_vl",
]);

/** 모델 선택 목록 표시 순서 */
export const VLM_MODEL_ORDER = [
  "qwen3_vl_2b",
  "qwen3_vl_4b",
  "qwen3_vl_4b_thinking",
] as const;

export function isThinkingVlmModel(modelId: string): boolean {
  return modelId.includes("thinking");
}

export function isDeprecatedVlmModel(modelId: string): boolean {
  return DEPRECATED_VLM_MODEL_IDS.has(modelId);
}

export function filterVlmModels(models: VlmModelInfo[]): VlmModelInfo[] {
  const filtered = models.filter((m) => !isDeprecatedVlmModel(m.model_id));
  return filtered.sort((a, b) => {
    const ia = VLM_MODEL_ORDER.indexOf(
      a.model_id as (typeof VLM_MODEL_ORDER)[number]
    );
    const ib = VLM_MODEL_ORDER.indexOf(
      b.model_id as (typeof VLM_MODEL_ORDER)[number]
    );
    const ao = ia === -1 ? 999 : ia;
    const bo = ib === -1 ? 999 : ib;
    return ao - bo;
  });
}

export function defaultVlmModelId(models: VlmModelInfo[]): string | null {
  if (models.length === 0) return null;
  const preferred = models.find((m) => m.model_id === "qwen3_vl_4b");
  return preferred?.model_id ?? models[0].model_id;
}
