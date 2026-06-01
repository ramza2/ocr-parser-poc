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

export type VlmOutputFormat = "bbox" | "text_only";

export interface VlmOcrOptions {
  outputFormat?: VlmOutputFormat;
}

export interface VlmRequestOptions {
  outputFormat?: VlmOutputFormat;
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
  output_format?: string | null;
  raw_response_preview?: string | null;
  model_raw_preview?: string | null;
}

/** exact_text: key = 이미지에 있는 문자열 그대로 | 그 외: key = 결과 필드 ID */
export type SchemaMatchMode = "exact_text" | "script_filter" | "semantic_field";

export type SchemaScriptFilter =
  | "han"
  | "hangul"
  | "latin"
  | "english"
  | "digit";

export interface SchemaField {
  key: string;
  /** location_hint (exact/script) 또는 의미 설명 (semantic) */
  description: string;
  type: string;
  match_mode?: SchemaMatchMode;
  script?: SchemaScriptFilter;
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
  output_format?: string | null;
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

export interface QaResponse {
  success: boolean;
  model_id: string;
  elapsed_ms: number;
  answer: string;
  confidence?: number | null;
  output_format?: string | null;
  error?: string | null;
}
