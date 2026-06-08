import type { VlmOcrResponse } from "../../types/vlm";

interface Props {
  result: VlmOcrResponse;
  imageUrl: string;
}

export default function OcrResultView({ result, imageUrl }: Props) {
  const rawText =
    result.raw_response_preview?.trim() ||
    result.model_raw_preview?.trim() ||
    "";

  return (
    <div className="space-y-4">
      <div className="flex items-center gap-3 text-xs text-slate-500">
        <span className="rounded bg-slate-100 px-2 py-0.5 font-medium text-slate-600">
          {result.model_id}
        </span>
        <span>{result.elapsed_ms.toLocaleString()}ms</span>
      </div>

      <div className="rounded-lg border border-slate-200 bg-slate-50 p-3">
        <img
          src={imageUrl}
          alt="검사 이미지"
          className="max-h-[480px] max-w-full rounded-md mx-auto"
        />
      </div>

      <div>
        <h4 className="mb-1.5 text-xs font-semibold text-slate-600">모델 raw 응답</h4>
        <pre className="max-h-64 overflow-auto rounded-lg border border-slate-200 bg-white p-3 text-xs text-slate-700 whitespace-pre-wrap">
          {rawText || "(응답 없음)"}
        </pre>
      </div>

      <div>
        <h4 className="mb-1.5 text-xs font-semibold text-slate-600">전체 텍스트</h4>
        <pre className="max-h-96 overflow-auto rounded-lg border border-slate-200 bg-slate-50 p-3 text-sm text-slate-800 whitespace-pre-wrap leading-relaxed">
          {result.full_text || "(추출된 텍스트 없음)"}
        </pre>
      </div>
    </div>
  );
}
