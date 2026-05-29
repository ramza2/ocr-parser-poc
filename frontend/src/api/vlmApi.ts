/* VLM 전용 API 클라이언트 */
import type {
  QaResponse,
  SchemaExtractResponse,
  SchemaField,
  VlmModelsResponse,
  VlmOcrOptions,
  VlmOcrResponse,
} from "../types/vlm";

async function parseJsonResponse<T>(res: Response, label: string): Promise<T> {
  const contentType = res.headers.get("content-type") ?? "";
  if (!contentType.includes("application/json")) {
    const text = await res.text();
    const snippet = text.replace(/\s+/g, " ").slice(0, 120);
    throw new Error(
      `${label} 실패 (HTTP ${res.status}). JSON 대신 HTML/텍스트가 반환되었습니다. ` +
        `프록시·VLM_WORKER_URL·워커 연결을 확인하세요. 응답: ${snippet}`
    );
  }
  return res.json() as Promise<T>;
}

export async function fetchVlmModels(): Promise<VlmModelsResponse> {
  const res = await fetch("/api/vlm/models");
  if (!res.ok) throw new Error(`VLM 모델 목록 로드 실패 (HTTP ${res.status})`);
  return parseJsonResponse<VlmModelsResponse>(res, "VLM 모델 목록");
}

export async function loadVlmModel(
  modelId: string
): Promise<{ success: boolean; error?: string }> {
  const form = new FormData();
  form.append("model_id", modelId);
  const res = await fetch("/api/vlm/load", { method: "POST", body: form });
  return parseJsonResponse(res, "VLM 모델 로드");
}

export async function vlmOcr(
  file: File,
  modelId: string,
  options?: VlmOcrOptions
): Promise<VlmOcrResponse> {
  const form = new FormData();
  form.append("file", file);
  form.append("model_id", modelId);
  form.append("ocr_prompt_mode", options?.promptMode ?? "spotting");
  if (options?.customPrompt?.trim()) {
    form.append("custom_prompt", options.customPrompt.trim());
  }
  const res = await fetch("/api/vlm/ocr", { method: "POST", body: form });
  return parseJsonResponse<VlmOcrResponse>(res, "VLM OCR");
}

export async function vlmExtract(
  file: File,
  modelId: string,
  schema: SchemaField[]
): Promise<SchemaExtractResponse> {
  const form = new FormData();
  form.append("file", file);
  form.append("model_id", modelId);
  form.append("schema_fields_json", JSON.stringify(schema));
  const res = await fetch("/api/vlm/extract", { method: "POST", body: form });
  return parseJsonResponse<SchemaExtractResponse>(res, "VLM Schema 추출");
}

export async function vlmAsk(
  file: File,
  modelId: string,
  question: string
): Promise<QaResponse> {
  const form = new FormData();
  form.append("file", file);
  form.append("model_id", modelId);
  form.append("question", question);
  const res = await fetch("/api/vlm/ask", { method: "POST", body: form });
  return parseJsonResponse<QaResponse>(res, "VLM Q&A");
}
