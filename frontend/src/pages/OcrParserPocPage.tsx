import { useCallback, useEffect, useState } from "react";
import { fetchParsers, parseFile } from "../api/parserApi";
import ErrorResultView from "../components/ErrorResultView";
import FileSelectPanel from "../components/FileSelectPanel";
import JsonResultView from "../components/JsonResultView";
import LogResultView from "../components/LogResultView";
import ParserSelectPanel from "../components/ParserSelectPanel";
import ResultSummaryCards from "../components/ResultSummaryCards";
import ResultTabs from "../components/ResultTabs";
import RunControlPanel from "../components/RunControlPanel";
import TableResultView from "../components/TableResultView";
import TextResultView from "../components/TextResultView";
import type {
  ParseResponse,
  ParserInfo,
  ResultTab,
  RunStatus,
  SelectedFileInfo,
} from "../types/parser";
import { getExtension, isSupportedExtension } from "../utils/fileUtils";
import { MOCK_PARSE_RESULT } from "../utils/mockData";

export default function OcrParserPocPage() {
  const [allParsers, setAllParsers] = useState<ParserInfo[]>([]);
  const [availableParsers, setAvailableParsers] = useState<ParserInfo[]>([]);
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
      } catch {
        setApiOnline(false);
        setAvailableParsers([]);
      }
    },
    []
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
    setResult(null);
    setStatus(isSupportedExtension(extension) ? "ready" : "idle");
    await loadParsersForFile(extension);
  };

  const handleRun = async () => {
    if (!selectedFile || !selectedParserId) return;

    setStatus("running");
    setActiveTab("text");

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
      const response = await parseFile(selectedFile.file, selectedParserId);
      setResult(response);
      setStatus(response.success ? "success" : "failed");
      if (!response.success || response.error_count > 0) {
        setActiveTab("error");
      }
    } catch {
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
            detail: "API 서버에 연결할 수 없습니다.",
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
    setResult(null);
    setStatus("idle");
    setAvailableParsers([]);
    setActiveTab("text");
  };

  const canRun =
    !!selectedFile &&
    !!selectedParserId &&
    isSupportedExtension(selectedFile.extension) &&
    status !== "running";

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
      case "table":
        return <TableResultView tables={result.tables} />;
      case "compare":
        return (
          <section className="rounded-xl border border-amber-200 bg-amber-50 p-6">
            <h3 className="text-sm font-semibold text-amber-900">파서 비교 (후순위)</h3>
            <p className="mt-2 text-sm text-amber-800">
              여러 파서 결과를 한 화면에서 비교하는 기능은 1차 PoC 범위에서 제외되었습니다.
              동일 파일로 파서를 바꿔 실행하여 결과를 비교해 주세요.
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
        <div className="mx-auto flex max-w-[1600px] items-start justify-between">
          <div>
            <h1 className="text-xl font-bold text-slate-900">OCR 파서 검증 PoC</h1>
            <p className="mt-0.5 text-sm text-slate-500">
              PDF·이미지 기반 OCR/문서 파서 자체개발 가능성 검토용
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

      <div className="mx-auto flex max-w-[1600px] gap-0">
        <aside className="w-[360px] shrink-0 border-r border-slate-200 bg-white p-5">
          <FileSelectPanel
            selectedFile={selectedFile}
            onSelect={handleFileSelect}
            disabled={status === "running"}
          />
          <hr className="my-5 border-slate-100" />
          <ParserSelectPanel
            parsers={availableParsers}
            selectedParserId={selectedParserId}
            onSelect={setSelectedParserId}
            disabled={!selectedFile || status === "running"}
          />
          <RunControlPanel
            onRun={handleRun}
            onReset={handleReset}
            canRun={canRun}
            isRunning={status === "running"}
          />
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
