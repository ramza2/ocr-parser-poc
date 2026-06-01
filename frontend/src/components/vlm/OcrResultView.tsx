import { useState } from "react";
import type { VlmOcrResponse } from "../../types/vlm";
import ConfidenceBadge from "./ConfidenceBadge";
import ImageWithBbox from "./ImageWithBbox";
import { formatOutputFormatLabel } from "./VlmOutputFormatSelector";

interface Props {
  result: VlmOcrResponse;
  imageUrl: string;
}

export default function OcrResultView({ result, imageUrl }: Props) {
  const [hoverIdx, setHoverIdx] = useState<number | null>(null);
  const hasBbox = result.items.some((it) => it.bbox);

  return (
    <div className="space-y-4">
      <div className="flex items-center gap-3">
        <span className="rounded bg-slate-100 px-2 py-0.5 text-xs font-medium text-slate-600">
          {result.model_id}
        </span>
        <span className="text-xs text-slate-500">
          {result.elapsed_ms.toLocaleString()}ms
        </span>
        <span className="text-xs text-slate-500">
          {result.items.length}개 항목
        </span>
        {result.prompt_label && (
          <span className="rounded bg-amber-50 px-2 py-0.5 text-xs text-amber-800">
            {result.prompt_label}
          </span>
        )}
        {result.output_format && (
          <span className="rounded bg-violet-50 px-2 py-0.5 text-xs text-violet-800">
            {formatOutputFormatLabel(result.output_format)}
          </span>
        )}
      </div>

      {result.output_format === "text_only" && result.prompt_label === "text_only→bbox_fallback" && (
        <p className="text-[11px] text-amber-700 leading-snug">
          텍스트 전용 프롬프트 실패 → bbox 프롬프트로 자동 fallback 되었습니다.
        </p>
      )}

      {result.output_format === "text_only" && result.prompt_label === "text_only" && (
        <p className="text-[11px] text-slate-500 leading-snug">
          텍스트 전용 프롬프트로 추론했습니다 (bbox 없음).
        </p>
      )}

      {result.raw_response_preview && (
        <details className="rounded-lg border border-slate-200 bg-slate-50 text-xs">
          <summary className="cursor-pointer px-3 py-2 font-medium text-slate-600">
            {result.output_format === "text_only"
              ? "추출 텍스트 (미리보기)"
              : "모델 원본 응답 (미리보기)"}
          </summary>
          <pre className="max-h-40 overflow-auto whitespace-pre-wrap border-t border-slate-200 p-3 text-slate-700">
            {result.raw_response_preview}
          </pre>
        </details>
      )}

      {result.output_format === "text_only" && result.model_raw_preview && (
        <details className="rounded-lg border border-dashed border-slate-200 bg-white text-xs">
          <summary className="cursor-pointer px-3 py-2 font-medium text-slate-500">
            모델 raw 응답 (디버그)
          </summary>
          <pre className="max-h-40 overflow-auto whitespace-pre-wrap border-t border-slate-100 p-3 text-slate-600">
            {result.model_raw_preview}
          </pre>
        </details>
      )}

      <div className={hasBbox ? "grid grid-cols-2 gap-4" : "space-y-4"}>
        <div>
          <ImageWithBbox
            imageUrl={imageUrl}
            items={result.items}
            highlightIndex={hoverIdx}
          />
        </div>

        <div className="space-y-3">
          <div className="max-h-[400px] overflow-y-auto rounded-lg border border-slate-200">
            {result.items.map((item, i) => (
              <div
                key={i}
                onMouseEnter={() => setHoverIdx(i)}
                onMouseLeave={() => setHoverIdx(null)}
                className={`flex items-center gap-2 border-b border-slate-100 px-3 py-1.5 text-sm transition last:border-b-0
                  ${hoverIdx === i ? "bg-blue-50" : "hover:bg-slate-50"}`}
              >
                <span className="min-w-0 flex-1 text-slate-800">{item.text}</span>
                <ConfidenceBadge value={item.confidence} />
              </div>
            ))}
          </div>

          <div>
            <h4 className="mb-1 text-xs font-semibold text-slate-600">전체 텍스트</h4>
            <pre className="max-h-48 overflow-auto rounded-lg bg-slate-50 p-3 text-xs text-slate-700 whitespace-pre-wrap">
              {result.full_text || "(추출된 텍스트 없음)"}
            </pre>
          </div>
        </div>
      </div>
    </div>
  );
}
