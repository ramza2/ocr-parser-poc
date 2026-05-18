import type { LogItem } from "../types/parser";

interface Props {
  logs: LogItem[];
}

const levelColor: Record<string, string> = {
  INFO: "text-blue-600",
  WARN: "text-amber-600",
  ERROR: "text-red-600",
};

export default function LogResultView({ logs }: Props) {
  return (
    <section className="rounded-xl border border-slate-200 bg-white p-4 shadow-sm">
      <h3 className="mb-3 text-sm font-semibold text-slate-800">상세 로그</h3>
      {logs.length === 0 ? (
        <p className="text-sm text-slate-500">로그가 없습니다.</p>
      ) : (
        <ul className="max-h-64 space-y-2 overflow-auto rounded-lg border border-slate-100 bg-slate-50 p-3 font-mono text-xs">
          {logs.map((log, i) => (
            <li key={i} className="flex gap-2">
              <span className="shrink-0 text-slate-400">
                [{log.timestamp ?? "--:--:--"}]
              </span>
              <span className={`shrink-0 font-semibold ${levelColor[log.level] ?? "text-slate-600"}`}>
                {log.level}
              </span>
              <span className="text-slate-700">{log.message}</span>
            </li>
          ))}
        </ul>
      )}
    </section>
  );
}
