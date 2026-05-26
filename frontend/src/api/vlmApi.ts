/* VLM 전용 API 클라이언트 */
import type {
  QaResponse,
  SchemaExtractResponse,
  SchemaField,
  VlmModelsResponse,
  VlmOcrResponse,
} from "../types/vlm";

export async function fetchVlmModels(): Promise<VlmModelsResponse> {
  const res = await fetch("/api/vlm/models");
  if (!res.ok) throw new Error("VLM 모델 목록 로드 실패");
  return res.json();
}

export async function loadVlmModel(
  modelId: string
): Promise<{ success: boolean; error?: string }> {
  const form = new FormData();
  form.append("model_id", modelId);
  const res = await fetch("/api/vlm/load", { method: "POST", body: form });
  return res.json();
}

export async function vlmOcr(
  file: File,
  modelId: string
): Promise<VlmOcrResponse> {
  const form = new FormData();
  form.append("file", file);
  form.append("model_id", modelId);
  const res = await fetch("/api/vlm/ocr", { method: "POST", body: form });
  return res.json();
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
  return res.json();
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
  return res.json();
}
