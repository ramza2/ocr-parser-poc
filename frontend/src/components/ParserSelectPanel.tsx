import type { ParserInfo } from "../types/parser";

interface Props {
  parsers: ParserInfo[];
  selectedParserId: string | null;
  onSelect: (parserId: string) => void;
  disabled?: boolean;
}

export default function ParserSelectPanel({
  parsers,
  selectedParserId,
  onSelect,
  disabled,
}: Props) {
  if (parsers.length === 0) {
    return (
      <section className="space-y-3">
        <h2 className="text-sm font-semibold text-slate-800">2. 파서 선택</h2>
        <p className="text-xs text-amber-600">
          지원하지 않는 확장자이거나 사용 가능한 파서가 없습니다.
        </p>
      </section>
    );
  }

  return (
    <section className="space-y-3">
      <h2 className="text-sm font-semibold text-slate-800">2. 파서 선택</h2>
      <div className="space-y-2">
        {parsers.map((parser) => {
          const selected = selectedParserId === parser.parser_id;
          return (
            <label
              key={parser.parser_id}
              className={`block cursor-pointer rounded-lg border p-3 transition ${
                disabled ? "cursor-not-allowed opacity-60" : ""
              } ${
                selected
                  ? "border-brand-500 bg-brand-50 ring-1 ring-brand-500"
                  : "border-slate-200 bg-white hover:border-slate-300"
              }`}
            >
              <div className="flex items-start gap-2">
                <input
                  type="radio"
                  name="parser"
                  value={parser.parser_id}
                  checked={selected}
                  disabled={disabled}
                  onChange={() => onSelect(parser.parser_id)}
                  className="mt-1 text-brand-600"
                />
                <div>
                  <p className="text-sm font-medium text-slate-800">
                    {parser.name}
                  </p>
                  <p className="mt-0.5 text-xs text-slate-500">
                    {parser.description}
                  </p>
                </div>
              </div>
            </label>
          );
        })}
      </div>
    </section>
  );
}
