/* VLM 전용 타입 정의 */

export interface BoundingBox {
  x: number;
  y: number;
  width: number;
  height: number;
}

export interface VlmOcrItem {
  text: string;
  confidence?: number | null;
  bbox?: BoundingBox | null;
}

export type VlmOcrPromptMode = "auto" | "bbox" | "custom";

export interface VlmOcrOptions {
  promptMode?: VlmOcrPromptMode;
  customPrompt?: string;
}

export interface VlmOcrResponse {
  success: boolean;
  model_id: string;
  elapsed_ms: number;
  items: VlmOcrItem[];
  full_text: string;
  error?: string | null;
  prompt_mode?: string | null;
  prompt_label?: string | null;
  raw_response_preview?: string | null;
}

export interface SchemaField {
  key: string;
  description: string;
  type: string;
}

export interface SchemaExtractItem {
  key: string;
  value: string;
  confidence?: number | null;
  bbox?: BoundingBox | null;
}

export interface SchemaExtractResponse {
  success: boolean;
  model_id: string;
  elapsed_ms: number;
  items: SchemaExtractItem[];
  error?: string | null;
}

export interface QaResponse {
  success: boolean;
  model_id: string;
  elapsed_ms: number;
  answer: string;
  confidence?: number | null;
  error?: string | null;
}

export interface VlmModelInfo {
  model_id: string;
  name: string;
  description: string;
  vram_gb: number;
  loaded: boolean;
}

export interface VlmModelsResponse {
  models: VlmModelInfo[];
  current_model: string | null;
}

export type VlmMode = "ocr" | "schema" | "qa";
