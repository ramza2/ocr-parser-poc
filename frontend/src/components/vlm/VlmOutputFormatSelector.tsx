import type { VlmOutputFormat } from "../../types/vlm";

const FORMATS: { id: VlmOutputFormat; label: string; desc: string }[] = [
  {
    id: "bbox",
    label: "bbox + JSON",
    desc: "좌표 포함 (느림, 오버레이 가능)",
  },
  {
    id: "text_only",
    label: "텍스트만",
    desc: "좌표 없음 (빠름, 속도 비교용)",
  },
];

interface Props {
  outputFormat: VlmOutputFormat;
  onChange: (format: VlmOutputFormat) => void;
  disabled?: boolean;
}

export default function VlmOutputFormatSelector({
  outputFormat,
  onChange,
  disabled,
}: Props) {
  return (
    <div className="space-y-2">
      <h3 className="text-sm font-semibold text-slate-700">출력 형식</h3>
      <div className="grid grid-cols-2 gap-1.5">
        {FORMATS.map((f) => (
          <button
            key={f.id}
            type="button"
            disabled={disabled}
            onClick={() => onChange(f.id)}
            className={`rounded-lg border px-2 py-2 text-left transition disabled:opacity-50
              ${
                outputFormat === f.id
                  ? "border-brand-500 bg-brand-50 ring-1 ring-brand-500"
                  : "border-slate-200 hover:border-slate-300"
              }`}
          >
            <div className="text-xs font-semibold text-slate-800">{f.label}</div>
            <div className="text-[10px] text-slate-500 leading-tight mt-0.5">
              {f.desc}
            </div>
          </button>
        ))}
      </div>
    </div>
  );
}

export function formatOutputFormatLabel(format: string | null | undefined): string {
  if (format === "text_only") return "텍스트만";
  if (format === "bbox") return "bbox + JSON";
  return format ?? "";
}
