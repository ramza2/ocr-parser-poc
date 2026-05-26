import { useState } from "react";
import type { SchemaExtractResponse } from "../../types/vlm";
import ConfidenceBadge from "./ConfidenceBadge";
import ImageWithBbox from "./ImageWithBbox";

interface Props {
  result: SchemaExtractResponse;
  imageUrl: string;
}

export default function SchemaResultView({ result, imageUrl }: Props) {
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
      </div>

      <div className={hasBbox ? "grid grid-cols-2 gap-4" : ""}>
        {hasBbox && (
          <div>
            <ImageWithBbox
              imageUrl={imageUrl}
              items={result.items.map((it) => ({
                text: it.value,
                bbox: it.bbox,
                confidence: it.confidence,
              }))}
              highlightIndex={hoverIdx}
            />
          </div>
        )}

        <div className="rounded-lg border border-slate-200 overflow-hidden">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-slate-200 bg-slate-50">
                <th className="px-3 py-2 text-left text-xs font-semibold text-slate-600">Key</th>
                <th className="px-3 py-2 text-left text-xs font-semibold text-slate-600">Value</th>
                <th className="px-3 py-2 text-right text-xs font-semibold text-slate-600 w-16">신뢰도</th>
              </tr>
            </thead>
            <tbody>
              {result.items.map((item, i) => (
                <tr
                  key={i}
                  onMouseEnter={() => setHoverIdx(i)}
                  onMouseLeave={() => setHoverIdx(null)}
                  className={`border-b border-slate-100 last:border-b-0 transition
                    ${hoverIdx === i ? "bg-blue-50" : "hover:bg-slate-50"}`}
                >
                  <td className="px-3 py-2 font-medium text-slate-700">{item.key}</td>
                  <td className="px-3 py-2 text-slate-800">
                    {item.value || <span className="text-slate-400 italic">-</span>}
                  </td>
                  <td className="px-3 py-2 text-right">
                    <ConfidenceBadge value={item.confidence} />
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
