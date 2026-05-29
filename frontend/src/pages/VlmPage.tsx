/**
 * VLM 엔진 PoC 페이지.
 *
 * 흐름: 모델 선택 → 이미지 업로드 → 모드(OCR/Schema/QA) 선택 → 실행 → 결과 표시
 */
import { useEffect, useRef, useState } from "react";
import {
  fetchVlmModels,
  vlmAsk,
  vlmExtract,
  vlmOcr,
} from "../api/vlmApi";
import OcrResultView from "../components/vlm/OcrResultView";
import QaInput from "../components/vlm/QaInput";
import QaResultView from "../components/vlm/QaResultView";
import SchemaEditor from "../components/vlm/SchemaEditor";
import SchemaResultView from "../components/vlm/SchemaResultView";
import VlmModelSelector from "../components/vlm/VlmModelSelector";
import VlmModeSelector from "../components/vlm/VlmModeSelector";
import VlmOcrPromptOptions from "../components/vlm/VlmOcrPromptOptions";
import type {
  QaResponse,
  SchemaExtractResponse,
  SchemaField,
  VlmMode,
  VlmModelInfo,
  VlmOcrPromptMode,
  VlmOcrResponse,
} from "../types/vlm";

type Status = "idle" | "ready" | "loading" | "running" | "done" | "error";

const SUPPORTED = ["jpg", "jpeg", "png", "tif", "tiff", "bmp", "webp"];

function getExt(name: string) {
  return name.split(".").pop()?.toLowerCase() ?? "";
}

interface QaEntry {
  question: string;
  response: QaResponse;
}

export default function VlmPage() {
  const [models, setModels] = useState<VlmModelInfo[]>([]);
  const [currentLoaded, setCurrentLoaded] = useState<string | null>(null);
  const [selectedModelId, setSelectedModelId] = useState<string | null>(null);
  const [file, setFile] = useState<File | null>(null);
  const [imageUrl, setImageUrl] = useState<string | null>(null);
  const [mode, setMode] = useState<VlmMode>("ocr");
  const [status, setStatus] = useState<Status>("idle");
  const [statusMsg, setStatusMsg] = useState("");

  // Schema mode
  const [schema, setSchema] = useState<SchemaField[]>([
    { key: "", description: "", type: "text" },
  ]);

  // Results
  const [ocrResult, setOcrResult] = useState<VlmOcrResponse | null>(null);
  const [schemaResult, setSchemaResult] = useState<SchemaExtractResponse | null>(null);
  const [qaHistory, setQaHistory] = useState<QaEntry[]>([]);
  const [ocrPromptMode, setOcrPromptMode] = useState<VlmOcrPromptMode>("auto");
  const [customPrompt, setCustomPrompt] = useState("");

  const fileInputRef = useRef<HTMLInputElement>(null);

  // ── 초기 모델 목록 로드 ──────────────────────────
  useEffect(() => {
    fetchVlmModels()
      .then((res) => {
        setModels(res.models);
        setCurrentLoaded(res.current_model);
      })
      .catch(() => setModels([]));
  }, []);

  // ── 파일 선택 ────────────────────────────────────
  const handleFile = (f: File) => {
    const ext = getExt(f.name);
    if (!SUPPORTED.includes(ext)) {
      setStatusMsg("지원하지 않는 파일 형식입니다.");
      return;
    }
    setFile(f);
    setImageUrl(URL.createObjectURL(f));
    setOcrResult(null);
    setSchemaResult(null);
    setQaHistory([]);
    setStatus("ready");
    setStatusMsg("");
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    const f = e.dataTransfer.files[0];
    if (f) handleFile(f);
  };

  // ── 실행 ─────────────────────────────────────────
  const canRun =
    !!file &&
    !!selectedModelId &&
    status !== "running" &&
    status !== "loading" &&
    (mode !== "ocr" || ocrPromptMode !== "custom" || customPrompt.trim().length > 0);

  const handleRun = async () => {
    if (!file || !selectedModelId) return;

    setStatus("running");
    setStatusMsg("모델 추론 중...");
    try {
      if (mode === "ocr") {
        const res = await vlmOcr(file, selectedModelId, {
          promptMode: ocrPromptMode,
          customPrompt: ocrPromptMode === "custom" ? customPrompt : undefined,
        });
        setOcrResult(res);
        setCurrentLoaded(selectedModelId);
        if (!res.success) {
          setStatus("error");
          setStatusMsg(res.error ?? "OCR 실패");
        } else {
          setStatus("done");
          setStatusMsg(`완료 — ${res.elapsed_ms}ms`);
        }
      } else if (mode === "schema") {
        const validSchema = schema.filter((f) => f.key.trim());
        if (validSchema.length === 0) {
          setStatus("error");
          setStatusMsg("최소 1개의 Key를 입력하세요.");
          return;
        }
        const res = await vlmExtract(file, selectedModelId, validSchema);
        setSchemaResult(res);
        setCurrentLoaded(selectedModelId);
        if (!res.success) {
          setStatus("error");
          setStatusMsg(res.error ?? "Schema 추출 실패");
        } else {
          setStatus("done");
          setStatusMsg(`완료 — ${res.elapsed_ms}ms`);
        }
      }
    } catch (err) {
      setStatus("error");
      setStatusMsg(err instanceof Error ? err.message : "알 수 없는 오류");
    }
  };

  const handleQa = async (question: string) => {
    if (!file || !selectedModelId) return;
    setStatus("running");
    setStatusMsg("답변 생성 중...");
    try {
      const res = await vlmAsk(file, selectedModelId, question);
      setCurrentLoaded(selectedModelId);
      setQaHistory((prev) => [...prev, { question, response: res }]);
      setStatus("done");
      setStatusMsg(`답변 완료 — ${res.elapsed_ms}ms`);
    } catch (err) {
      setStatus("error");
      setStatusMsg(err instanceof Error ? err.message : "알 수 없는 오류");
    }
  };

  const handleReset = () => {
    setFile(null);
    if (imageUrl) URL.revokeObjectURL(imageUrl);
    setImageUrl(null);
    setOcrResult(null);
    setSchemaResult(null);
    setQaHistory([]);
    setStatus("idle");
    setStatusMsg("");
  };

  // ── 렌더 ─────────────────────────────────────────
  return (
    <div className="mx-auto flex max-w-[1680px] gap-0 min-h-0 flex-1">
      {/* ── 사이드바 ── */}
      <aside className="w-[400px] shrink-0 overflow-y-auto border-r border-slate-200 bg-white p-5 space-y-4">
        <VlmModelSelector
          models={models}
          selectedModelId={selectedModelId}
          currentLoaded={currentLoaded}
          onSelect={setSelectedModelId}
          disabled={status === "running"}
        />

        <hr className="border-slate-100" />

        <VlmModeSelector
          mode={mode}
          onChange={setMode}
          disabled={status === "running"}
        />

        <hr className="border-slate-100" />

        {/* 이미지 업로드 */}
        <div className="space-y-1.5">
          <h3 className="text-sm font-semibold text-slate-700">3. 이미지 업로드</h3>
          <div
            onDragOver={(e) => e.preventDefault()}
            onDrop={handleDrop}
            onClick={() => fileInputRef.current?.click()}
            className={`cursor-pointer rounded-lg border-2 border-dashed p-4 text-center transition
              ${file ? "border-brand-300 bg-brand-50" : "border-slate-200 hover:border-slate-300"}`}
          >
            <input
              ref={fileInputRef}
              type="file"
              accept={SUPPORTED.map((e) => `.${e}`).join(",")}
              className="hidden"
              onChange={(e) => {
                const f = e.target.files?.[0];
                if (f) handleFile(f);
              }}
            />
            {file ? (
              <div className="space-y-1">
                <p className="text-sm font-medium text-brand-700 truncate">{file.name}</p>
                <p className="text-xs text-slate-500">{(file.size / 1024).toFixed(1)} KB</p>
              </div>
            ) : (
              <p className="text-sm text-slate-500">
                클릭 또는 드래그하여 이미지 업로드
              </p>
            )}
          </div>
        </div>

        {/* 모드별 옵션 */}
        {mode === "ocr" && (
          <>
            <hr className="border-slate-100" />
            <VlmOcrPromptOptions
              promptMode={ocrPromptMode}
              customPrompt={customPrompt}
              onPromptModeChange={setOcrPromptMode}
              onCustomPromptChange={setCustomPrompt}
              disabled={status === "running"}
            />
          </>
        )}

        {mode === "schema" && (
          <>
            <hr className="border-slate-100" />
            <SchemaEditor
              schema={schema}
              onChange={setSchema}
              disabled={status === "running"}
            />
          </>
        )}

        <hr className="border-slate-100" />

        {/* 실행 / 초기화 */}
        <div className="flex gap-2">
          {mode !== "qa" && (
            <button
              onClick={handleRun}
              disabled={!canRun}
              className="flex-1 rounded-lg bg-brand-600 py-2.5 text-sm font-semibold text-white hover:bg-brand-700 disabled:opacity-50 transition"
            >
              {status === "running" ? (
                <span className="inline-flex items-center gap-2">
                  <span className="h-4 w-4 animate-spin rounded-full border-2 border-white border-t-transparent" />
                  실행 중...
                </span>
              ) : (
                "실행"
              )}
            </button>
          )}
          <button
            onClick={handleReset}
            disabled={status === "running"}
            className="rounded-lg border border-slate-200 px-4 py-2.5 text-sm text-slate-600 hover:bg-slate-50 disabled:opacity-50"
          >
            초기화
          </button>
        </div>

        {statusMsg && (
          <p className={`text-xs ${status === "error" ? "text-red-500" : "text-slate-500"}`}>
            {statusMsg}
          </p>
        )}
      </aside>

      {/* ── 메인 영역 ── */}
      <main className="min-w-0 flex-1 space-y-4 overflow-y-auto p-5">
        {/* 이미지 미리보기 (결과 없을 때) */}
        {imageUrl && !ocrResult && !schemaResult && mode !== "qa" && status !== "running" && (
          <div className="rounded-xl border border-slate-200 bg-white p-4 shadow-sm">
            <img
              src={imageUrl}
              alt="미리보기"
              className="max-h-[500px] max-w-full rounded-lg mx-auto"
            />
          </div>
        )}

        {/* 실행 중 스피너 */}
        {status === "running" && (
          <div className="rounded-xl border border-slate-200 bg-white p-12 text-center">
            <span className="inline-block h-8 w-8 animate-spin rounded-full border-2 border-brand-600 border-t-transparent" />
            <p className="mt-4 text-sm text-slate-600">{statusMsg}</p>
          </div>
        )}

        {/* OCR 결과 */}
        {mode === "ocr" && ocrResult && imageUrl && status !== "running" && (
          <div className="rounded-xl border border-slate-200 bg-white p-4 shadow-sm">
            <OcrResultView result={ocrResult} imageUrl={imageUrl} />
          </div>
        )}

        {/* Schema 결과 */}
        {mode === "schema" && schemaResult && imageUrl && status !== "running" && (
          <div className="rounded-xl border border-slate-200 bg-white p-4 shadow-sm">
            <SchemaResultView result={schemaResult} imageUrl={imageUrl} />
          </div>
        )}

        {/* Q&A 모드 */}
        {mode === "qa" && (
          <div className="rounded-xl border border-slate-200 bg-white p-4 shadow-sm space-y-4">
            {imageUrl && (
              <img
                src={imageUrl}
                alt="문서"
                className="max-h-[300px] max-w-full rounded-lg mx-auto"
              />
            )}
            <QaResultView history={qaHistory} />
            <QaInput
              onSubmit={handleQa}
              disabled={!file || !selectedModelId}
              loading={status === "running"}
            />
          </div>
        )}

        {/* 초기 안내 */}
        {!file && status === "idle" && (
          <div className="rounded-xl border border-dashed border-slate-200 bg-white p-12 text-center">
            <p className="text-sm text-slate-500">
              좌측에서 VLM 모델과 모드를 선택하고 이미지를 업로드하세요.
            </p>
          </div>
        )}
      </main>
    </div>
  );
}
