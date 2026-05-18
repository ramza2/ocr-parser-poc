import type { ParseResponse, ParserInfo, RunStatus } from "../types/parser";

interface Props {
  status: RunStatus;
  result: ParseResponse | null;
  parsers: ParserInfo[];
  selectedParserId: string | null;
}

function parserLabel(
  parserId: string | null,
  parsers: ParserInfo[]
): string {
  if (!parserId) return "-";
  return parsers.find((p) => p.parser_id === parserId)?.name ?? parserId;
}

export default function ResultSummaryCards({
  status,
  result,
  parsers,
  selectedParserId,
}: Props) {
  const isSuccess = status === "success" && result?.success;
  const isFailed = status === "failed" || (result && !result.success);
  const isRunning = status === "running";

  const badge = isRunning ? (
    <span className="rounded-full bg-amber-100 px-2.5 py-0.5 text-xs font-semibold text-amber-700">
      처리 중
    </span>
  ) : isSuccess ? (
    <span className="rounded-full bg-emerald-100 px-2.5 py-0.5 text-xs font-semibold text-emerald-700">
      성공
    </span>
  ) : isFailed ? (
    <span className="rounded-full bg-red-100 px-2.5 py-0.5 text-xs font-semibold text-red-700">
      실패
    </span>
  ) : (
    <span className="rounded-full bg-slate-100 px-2.5 py-0.5 text-xs font-semibold text-slate-600">
      대기
    </span>
  );

  const cards = [
    {
      label: "사용 파서",
      value: result
        ? parserLabel(result.parser_id, parsers)
        : parserLabel(selectedParserId, parsers),
    },
    {
      label: "실행 시간",
      value: result ? `${(result.elapsed_ms / 1000).toFixed(2)}s` : "-",
    },
    {
      label: "페이지 수",
      value: result?.page_count != null ? String(result.page_count) : "-",
    },
    {
      label: "추출 문자 수",
      value: result ? result.text_length.toLocaleString() : "-",
    },
    {
      label: "표 개수",
      value: result ? String(result.table_count) : "-",
    },
    {
      label: "오류 개수",
      value: result ? String(result.error_count) : "-",
      highlight: result && result.error_count > 0,
    },
  ];

  return (
    <section className="rounded-xl border border-slate-200 bg-white p-4 shadow-sm">
      <div className="mb-3 flex items-center justify-between">
        <h2 className="text-sm font-semibold text-slate-800">처리 결과 요약</h2>
        {badge}
      </div>
      <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-6">
        {cards.map((card) => (
          <div
            key={card.label}
            className="rounded-lg border border-slate-100 bg-slate-50 px-3 py-2"
          >
            <p className="text-xs text-slate-500">{card.label}</p>
            <p
              className={`mt-0.5 truncate text-sm font-semibold ${
                card.highlight ? "text-red-600" : "text-slate-800"
              }`}
            >
              {card.value}
            </p>
          </div>
        ))}
      </div>
    </section>
  );
}
