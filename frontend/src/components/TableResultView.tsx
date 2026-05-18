import type { TableResult } from "../types/parser";

interface Props {
  tables: TableResult[];
}

export default function TableResultView({ tables }: Props) {
  if (tables.length === 0) {
    return (
      <section className="rounded-xl border border-slate-200 bg-white p-4 shadow-sm">
        <p className="text-sm text-slate-500">추출된 표 데이터가 없습니다.</p>
      </section>
    );
  }

  return (
    <section className="space-y-6">
      {tables.map((table) => {
        const [header, ...rows] = table.rows;
        return (
          <div
            key={table.table_id}
            className="rounded-xl border border-slate-200 bg-white p-4 shadow-sm"
          >
            <h3 className="mb-3 text-sm font-semibold text-slate-800">
              추출된 표 데이터 ({table.table_id} · 페이지 {table.page_no})
            </h3>
            <div className="overflow-x-auto">
              <table className="min-w-full border-collapse text-sm">
                {header && (
                  <thead>
                    <tr className="border-b border-slate-200 bg-slate-50">
                      {header.map((cell, i) => (
                        <th
                          key={i}
                          className="px-3 py-2 text-left text-xs font-semibold uppercase text-slate-600"
                        >
                          {cell}
                        </th>
                      ))}
                    </tr>
                  </thead>
                )}
                <tbody>
                  {rows.map((row, ri) => (
                    <tr key={ri} className="border-b border-slate-100">
                      {row.map((cell, ci) => {
                        const isConfidence =
                          header &&
                          (header[ci]?.includes("신뢰도") ||
                            header[ci]?.toLowerCase().includes("confidence"));
                        const conf = parseFloat(cell);
                        const highConf =
                          isConfidence && !Number.isNaN(conf) && conf >= 95;

                        return (
                          <td
                            key={ci}
                            className={`px-3 py-2 text-slate-700 ${
                              highConf ? "font-semibold text-emerald-600" : ""
                            }`}
                          >
                            {cell}
                          </td>
                        );
                      })}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        );
      })}
    </section>
  );
}
