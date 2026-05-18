import type { ErrorItem } from "../types/parser";

interface Props {
  errors: ErrorItem[];
}

export default function ErrorResultView({ errors }: Props) {
  if (errors.length === 0) {
    return (
      <section className="rounded-xl border border-emerald-200 bg-emerald-50 p-4">
        <p className="text-sm text-emerald-800">오류가 없습니다.</p>
      </section>
    );
  }

  return (
    <section className="space-y-3">
      {errors.map((err, i) => (
        <div
          key={i}
          className="rounded-xl border border-red-200 bg-red-50 p-4 shadow-sm"
        >
          <p className="text-xs font-semibold uppercase text-red-600">
            {err.code}
          </p>
          <p className="mt-1 text-sm font-medium text-red-800">{err.message}</p>
          {err.detail && (
            <p className="mt-2 text-xs text-red-700">{err.detail}</p>
          )}
        </div>
      ))}
    </section>
  );
}
