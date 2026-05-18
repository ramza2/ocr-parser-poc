import type { ParseResponse, ParserInfo } from "../types/parser";

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

export async function parseFile(
  file: File,
  parserId: string
): Promise<ParseResponse> {
  const form = new FormData();
  form.append("file", file);
  form.append("parser_id", parserId);

  const res = await fetch("/api/parse", {
    method: "POST",
    body: form,
  });

  if (!res.ok) {
    throw new Error("Parse request failed");
  }

  return res.json() as Promise<ParseResponse>;
}
