import { useState } from "react";
import OcrParserPocPage from "./pages/OcrParserPocPage";
import VlmPage from "./pages/VlmPage";

type AppTab = "ocr" | "vlm";

export default function App() {
  const [tab, setTab] = useState<AppTab>("ocr");

  return (
    <div className="flex min-h-screen flex-col">
      <header className="border-b border-slate-200 bg-white px-6 py-3 shadow-sm">
        <div className="mx-auto flex max-w-[1680px] items-center justify-between">
          <div className="flex items-center gap-6">
            <div>
              <h1 className="text-xl font-bold text-slate-900">OCR 파서 검증 PoC</h1>
              <p className="text-xs text-slate-500">
                이미지·스캔 PDF OCR 엔진 · VLM 비교
              </p>
            </div>
            <nav className="flex rounded-lg border border-slate-200 p-0.5">
              <button
                onClick={() => setTab("ocr")}
                className={`rounded-md px-4 py-1.5 text-sm font-medium transition
                  ${tab === "ocr" ? "bg-brand-600 text-white shadow-sm" : "text-slate-600 hover:text-slate-900"}`}
              >
                기존 OCR
              </button>
              <button
                onClick={() => setTab("vlm")}
                className={`rounded-md px-4 py-1.5 text-sm font-medium transition
                  ${tab === "vlm" ? "bg-brand-600 text-white shadow-sm" : "text-slate-600 hover:text-slate-900"}`}
              >
                VLM
              </button>
            </nav>
          </div>
          <span className="rounded-full bg-brand-100 px-3 py-1 text-xs font-semibold text-brand-700">
            PoC Mode
          </span>
        </div>
      </header>

      {tab === "ocr" ? <OcrInner /> : <VlmPage />}
    </div>
  );
}

function OcrInner() {
  return <OcrParserPocPage />;
}
