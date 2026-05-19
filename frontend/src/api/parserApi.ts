/**
 * 백엔드 API 클라이언트.
 *
 * 개발: vite proxy → localhost:8000
 * Docker: nginx 가 /api/* 를 backend:8000 으로 프록시 (상대 경로 /api 사용)
 */
import type {
  ParseResponse,
  ParserInfo,
  PipelineStepsResponse,
} from "../types/parser";

export async function fetchHealth(): Promise<{ status: string }> {
  const res = await fetch("/api/health");
  if (!res.ok) throw new Error("Health check failed");
  return res.json();
}

export async function fetchParsers(extension?: string): Promise<ParserInfo[]> {
  const query = extension ? `?extension=${encodeURIComponent(extension)}` : "";
  const res = await fetch(`/api/parsers${query}`);
  if (!res.ok) throw new Error("Failed to load parsers");
  const data = await res.json();
  return data.parsers as ParserInfo[];
}

export async function fetchPipelineSteps(
  extension?: string
): Promise<PipelineStepsResponse> {
  const query = extension ? `?extension=${encodeURIComponent(extension)}` : "";
  const res = await fetch(`/api/pipeline-steps${query}`);
  if (!res.ok) throw new Error("Failed to load pipeline steps");
  return res.json() as Promise<PipelineStepsResponse>;
}

export async function parseFile(
  file: File,
  parserId: string,
  preprocessSteps: string[] = [],
  postprocessSteps: string[] = []
): Promise<ParseResponse> {
  const form = new FormData();
  form.append("file", file);
  form.append("parser_id", parserId);
  form.append("preprocess_steps", JSON.stringify(preprocessSteps));
  form.append("postprocess_steps", JSON.stringify(postprocessSteps));

  let res: Response;
  try {
    res = await fetch("/api/parse", {
      method: "POST",
      body: form,
    });
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    throw new Error(
      `API 서버에 연결할 수 없습니다. 백엔드(uvicorn)가 실행 중인지, .venv(3.12)로 실행했는지 확인하세요. (${msg})`
    );
  }

  let data: ParseResponse;
  try {
    data = (await res.json()) as ParseResponse;
  } catch {
    throw new Error(`서버 응답을 읽을 수 없습니다. (HTTP ${res.status})`);
  }

  if (!res.ok) {
    const detail =
      data.errors?.[0]?.detail ||
      data.errors?.[0]?.message ||
      `HTTP ${res.status}`;
    throw new Error(detail);
  }

  return data;
}
