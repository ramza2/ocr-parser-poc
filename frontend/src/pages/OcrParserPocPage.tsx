import { useCallback, useEffect, useState } from "react";
import {
  fetchParsers,
  fetchPipelineSteps,
  parseFile,
} from "../api/parserApi";
import ErrorResultView from "../components/ErrorResultView";
import FileSelectPanel from "../components/FileSelectPanel";
import JsonResultView from "../components/JsonResultView";
import LogResultView from "../components/LogResultView";
import ParserSelectPanel from "../components/ParserSelectPanel";
import PipelineOptionsPanel from "../components/PipelineOptionsPanel";
import ResultSummaryCards from "../components/ResultSummaryCards";
import ResultTabs from "../components/ResultTabs";
import RunControlPanel from "../components/RunControlPanel";
import TextResultView from "../components/TextResultView";
import type {
  ParseResponse,
  ParserInfo,
  PipelineStepInfo,
  ResultTab,
  RunStatus,
  SelectedFileInfo,
} from "../types/parser";
import { getExtension, isSupportedExtension } from "../utils/fileUtils";
import { MOCK_PARSE_RESULT } from "../utils/mockData";

const NO_PIPELINE_PARSERS = new Set(["PDF_TEXT"]);

function supportsPipeline(parserId: string | null): boolean {
  return !!parserId && !NO_PIPELINE_PARSERS.has(parserId);
}

export default function OcrParserPocPage() {
  const [allParsers, setAllParsers] = useState<ParserInfo[]>([]);
  const [availableParsers, setAvailableParsers] = useState<ParserInfo[]>([]);
  const [preprocessCatalog, setPreprocessCatalog] = useState<PipelineStepInfo[]>(
    []
  );
  const [postprocessCatalog, setPostprocessCatalog] = useState<
    PipelineStepInfo[]
  >([]);
  const [tutorialPreset, setTutorialPreset] = useState<string[]>([]);
  const [selectedPreprocess, setSelectedPreprocess] = useState<string[]>([]);
  const [selectedPostprocess, setSelectedPostprocess] = useState<string[]>([]);
  const [selectedFile, setSelectedFile] = useState<SelectedFileInfo | null>(null);
  const [selectedParserId, setSelectedParserId] = useState<string | null>(null);
  const [result, setResult] = useState<ParseResponse | null>(null);
  const [status, setStatus] = useState<RunStatus>("idle");
  const [activeTab, setActiveTab] = useState<ResultTab>("text");
  const [useMock, setUseMock] = useState(false);
  const [apiOnline, setApiOnline] = useState<boolean | null>(null);

  useEffect(() => {
    fetchParsers()
      .then((parsers) => {
        setAllParsers(parsers);
        setApiOnline(true);
      })
      .catch(() => {
        setApiOnline(false);
        setAllParsers([]);
      });
  }, []);

  const loadPipelineSteps = useCallback(async (extension: string) => {
    try {
      const data = await fetchPipelineSteps(extension);
      setPreprocessCatalog(data.preprocess);
      setPostprocessCatalog(data.postprocess);
      setTutorialPreset(data.presets.tutorial_full ?? []);
    } catch {
      setPreprocessCatalog([]);
      setPostprocessCatalog([]);
      setTutorialPreset([]);
    }
  }, []);

  const loadParsersForFile = useCallback(
    async (extension: string) => {
      if (!isSupportedExtension(extension)) {
        setAvailableParsers([]);
        return;
      }
      try {
        const parsers = await fetchParsers(extension);
        setAvailableParsers(parsers);
        setApiOnline(true);
        await loadPipelineSteps(extension);
      } catch {
        setApiOnline(false);
        setAvailableParsers([]);
      }
    },
    [loadPipelineSteps]
  );

  const handleFileSelect = async (file: File) => {
    const extension = getExtension(file.name);
    const info: SelectedFileInfo = {
      file,
      name: file.name,
      extension,
      size: file.size,
      mimeType: file.type || "application/octet-stream",
    };
    setSelectedFile(info);
    setSelectedParserId(null);
    setSelectedPreprocess([]);
    setSelectedPostprocess([]);
    setResult(null);
    setStatus(isSupportedExtension(extension) ? "ready" : "idle");
    await loadParsersForFile(extension);
  };

  const handleRun = async () => {
    if (!selectedFile || !selectedParserId) return;

    setStatus("running");
    setActiveTab("text");

    const preprocess = supportsPipeline(selectedParserId)
      ? selectedPreprocess
      : [];
    const postprocess = supportsPipeline(selectedParserId)
      ? selectedPostprocess
      : [];

    if (useMock) {
      await new Promise((r) => setTimeout(r, 800));
      const mock = {
        ...MOCK_PARSE_RESULT,
        parser_id: selectedParserId,
        file_name: selectedFile.name,
        extension: selectedFile.extension,
      };
      setResult(mock);
      setStatus(mock.success ? "success" : "failed");
      if (!mock.success || mock.error_count > 0) setActiveTab("error");
      return;
    }

    try {
      const response = await parseFile(
        selectedFile.file,
        selectedParserId,
        preprocess,
        postprocess
      );
      setResult(response);
      setStatus(response.success ? "success" : "failed");
      if (!response.success || response.error_count > 0) {
        setActiveTab("error");
      }
    } catch (err) {
      const detail =
        err instanceof Error ? err.message : "알 수 없는 오류가 발생했습니다.";
      setResult({
        success: false,
        parser_id: selectedParserId,
        file_name: selectedFile.name,
        extension: selectedFile.extension,
        elapsed_ms: 0,
        text_length: 0,
        table_count: 0,
        error_count: 1,
        text: "",
        pages: [],
        tables: [],
        logs: [],
        errors: [
          {
            code: "INTERNAL_ERROR",
            message: "처리 중 알 수 없는 오류가 발생했습니다.",
            detail,
          },
        ],
      });
      setStatus("failed");
      setActiveTab("error");
    }
  };

  const handleReset = () => {
    setSelectedFile(null);
    setSelectedParserId(null);
    setSelectedPreprocess([]);
    setSelectedPostprocess([]);
    setResult(null);
    setStatus("idle");
    setAvailableParsers([]);
    setPreprocessCatalog([]);
    setPostprocessCatalog([]);
    setActiveTab("text");
  };

  const canRun =
    !!selectedFile &&
    !!selectedParserId &&
    isSupportedExtension(selectedFile.extension) &&
    status !== "running";

  const pipelineEnabled =
    !!selectedFile && supportsPipeline(selectedParserId) && status !== "running";

  const renderTabContent = () => {
    if (!result && status !== "running") {
      return (
        <div className="rounded-xl border border-dashed border-slate-200 bg-white p-12 text-center">
          <p className="text-sm text-slate-500">
            파일을 선택하고 파서를 실행하면 결과가 표시됩니다.
          </p>
        </div>
      );
    }

    if (status === "running") {
      return (
        <div className="rounded-xl border border-slate-200 bg-white p-12 text-center">
          <span className="inline-block h-8 w-8 animate-spin rounded-full border-2 border-brand-600 border-t-transparent" />
          <p className="mt-4 text-sm text-slate-600">파서를 실행하는 중입니다...</p>
        </div>
      );
    }

    if (!result) return null;

    switch (activeTab) {
      case "text":
        return <TextResultView text={result.text} />;
      case "json":
        return <JsonResultView data={result} />;
      case "log":
        return <LogResultView logs={result.logs} />;
      case "error":
        return <ErrorResultView errors={result.errors} />;
      case "compare":
        return (
          <section className="rounded-xl border border-amber-200 bg-amber-50 p-6">
            <h3 className="text-sm font-semibold text-amber-900">파서 비교</h3>
            <p className="mt-2 text-sm text-amber-800">
              동일 파일로 Tesseract / EasyOCR / PaddleOCR을 바꿔 실행하고,
              전처리·후처리 조합을 달리해 결과를 비교해 주세요.
            </p>
          </section>
        );
      default:
        return null;
    }
  };

  return (
    <div className="min-h-screen">
      <header className="border-b border-slate-200 bg-white px-6 py-4 shadow-sm">
        <div className="mx-auto flex max-w-[1680px] items-start justify-between">
          <div>
            <h1 className="text-xl font-bold text-slate-900">OCR 파서 검증 PoC</h1>
            <p className="mt-0.5 text-sm text-slate-500">
              OCR 엔진·전처리·후처리 조합 비교
            </p>
          </div>
          <div className="flex items-center gap-3">
            <label className="flex cursor-pointer items-center gap-2 text-xs text-slate-600">
              <input
                type="checkbox"
                checked={useMock}
                onChange={(e) => setUseMock(e.target.checked)}
                className="rounded border-slate-300"
              />
              Mock 모드
            </label>
            <span className="rounded-full bg-brand-100 px-3 py-1 text-xs font-semibold text-brand-700">
              PoC Mode
            </span>
            {apiOnline === false && (
              <span className="rounded-full bg-amber-100 px-3 py-1 text-xs font-medium text-amber-700">
                API 오프라인
              </span>
            )}
          </div>
        </div>
      </header>

      <div className="mx-auto flex max-w-[1680px] gap-0">
        <aside className="w-[400px] shrink-0 overflow-y-auto border-r border-slate-200 bg-white p-5">
          <FileSelectPanel
            selectedFile={selectedFile}
            onSelect={handleFileSelect}
            disabled={status === "running"}
          />
          <hr className="my-4 border-slate-100" />
          <ParserSelectPanel
            parsers={availableParsers}
            selectedParserId={selectedParserId}
            onSelect={setSelectedParserId}
            disabled={!selectedFile || status === "running"}
          />

          {selectedFile && (
            <>
              <hr className="my-4 border-slate-100" />
              <h2 className="mb-2 text-sm font-semibold text-slate-800">
                3. 전처리 · 후처리
              </h2>
              {!supportsPipeline(selectedParserId) && selectedParserId && (
                <p className="mb-2 text-xs text-slate-500">
                  선택한 파서는 OCR 파이프라인 전처리/후처리를 사용하지 않습니다.
                </p>
              )}
              <PipelineOptionsPanel
                title="전처리"
                steps={preprocessCatalog}
                selected={selectedPreprocess}
                onChange={setSelectedPreprocess}
                disabled={!pipelineEnabled}
                presetLabel="튜토리얼 전체"
                onPreset={() => setSelectedPreprocess(tutorialPreset)}
                onClear={() => setSelectedPreprocess([])}
              />
              <div className="mt-3">
                <PipelineOptionsPanel
                  title="후처리"
                  steps={postprocessCatalog}
                  selected={selectedPostprocess}
                  onChange={setSelectedPostprocess}
                  disabled={!pipelineEnabled}
                  onClear={() => setSelectedPostprocess([])}
                />
              </div>
            </>
          )}

          <div className="mt-4">
            <RunControlPanel
              onRun={handleRun}
              onReset={handleReset}
              canRun={canRun}
              isRunning={status === "running"}
            />
          </div>
        </aside>

        <main className="min-w-0 flex-1 space-y-4 p-5">
          <ResultSummaryCards
            status={status}
            result={result}
            parsers={allParsers.length ? allParsers : availableParsers}
            selectedParserId={selectedParserId}
          />
          <div className="rounded-xl border border-slate-200 bg-white shadow-sm">
            <ResultTabs activeTab={activeTab} onChange={setActiveTab} />
            <div className="p-4">{renderTabContent()}</div>
          </div>
        </main>
      </div>
    </div>
  );
}
