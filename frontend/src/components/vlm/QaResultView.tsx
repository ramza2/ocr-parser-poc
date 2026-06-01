import type { QaResponse } from "../../types/vlm";
import ConfidenceBadge from "./ConfidenceBadge";
import { formatOutputFormatLabel } from "./VlmOutputFormatSelector";

interface QaEntry {
  question: string;
  response: QaResponse;
}

interface Props {
  history: QaEntry[];
}

export default function QaResultView({ history }: Props) {
  if (history.length === 0) {
    return (
      <div className="rounded-lg border border-dashed border-slate-200 p-8 text-center text-sm text-slate-400">
        질문을 입력하면 답변이 표시됩니다.
      </div>
    );
  }

  return (
    <div className="space-y-4 max-h-[500px] overflow-y-auto">
      {history.map((entry, i) => (
        <div key={i} className="space-y-2">
          <div className="flex justify-end">
            <div className="max-w-[80%] rounded-lg bg-brand-50 px-3 py-2 text-sm text-brand-800">
              {entry.question}
            </div>
          </div>
          <div className="flex justify-start">
            <div className="max-w-[80%] rounded-lg border border-slate-200 bg-white px-3 py-2 text-sm text-slate-800">
              {entry.response.error ? (
                <span className="text-red-500">{entry.response.error}</span>
              ) : (
                <>
                  <p className="whitespace-pre-wrap">{entry.response.answer}</p>
                  <div className="mt-1 flex items-center gap-2">
                    <span className="text-[10px] text-slate-400">
                      {entry.response.model_id} · {entry.response.elapsed_ms}ms
                    </span>
                    {entry.response.output_format && (
                      <span className="rounded bg-violet-50 px-1.5 py-0.5 text-[10px] text-violet-800">
                        {formatOutputFormatLabel(entry.response.output_format)}
                      </span>
                    )}
                    <ConfidenceBadge value={entry.response.confidence} />
                  </div>
                </>
              )}
            </div>
          </div>
        </div>
      ))}
    </div>
  );
}
