import { useState } from "react";
import type { VlmOcrResponse } from "../../types/vlm";
import ConfidenceBadge from "./ConfidenceBadge";
import ImageWithBbox from "./ImageWithBbox";

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
      </div>

      <div className={hasBbox ? "grid grid-cols-2 gap-4" : ""}>
        {hasBbox && (
          <div>
            <ImageWithBbox
              imageUrl={imageUrl}
              items={result.items}
              highlightIndex={hoverIdx}
            />
          </div>
        )}

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
