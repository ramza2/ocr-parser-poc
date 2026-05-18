export interface ParserInfo {
  parser_id: string;
  name: string;
  description: string;
  supported_extensions: string[];
}

export interface PipelineStepInfo {
  step_id: string;
  name: string;
  description: string;
  applicable: string[];
  default_order: number;
}

export interface PipelineStepsResponse {
  file_kind: string;
  preprocess: PipelineStepInfo[];
  postprocess: PipelineStepInfo[];
  presets: {
    tutorial_full: string[];
    none: string[];
  };
}

export interface LogItem {
  level: string;
  message: string;
  timestamp?: string | null;
}

export interface ErrorItem {
  code: string;
  message: string;
  detail?: string | null;
}

export interface PageResult {
  page_no: number;
  text: string;
  blocks: Record<string, unknown>[];
}

export interface TableResult {
  table_id: string;
  page_no: number;
  rows: string[][];
}

export interface ParseResponse {
  success: boolean;
  parser_id: string;
  file_name: string;
  extension: string;
  elapsed_ms: number;
  page_count?: number | null;
  text_length: number;
  table_count: number;
  error_count: number;
  text: string;
  pages: PageResult[];
  tables: TableResult[];
  logs: LogItem[];
  errors: ErrorItem[];
}

export interface SelectedFileInfo {
  file: File;
  name: string;
  extension: string;
  size: number;
  mimeType: string;
}

export type ResultTab =
  | "text"
  | "json"
  | "log"
  | "error"
  | "table"
  | "compare";

export type RunStatus = "idle" | "ready" | "running" | "success" | "failed";
