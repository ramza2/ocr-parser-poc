import type { VlmMode } from "../../types/vlm";

interface Props {
  mode: VlmMode;
  onChange: (mode: VlmMode) => void;
  disabled?: boolean;
}

const MODES: { id: VlmMode; label: string; desc: string }[] = [
  { id: "ocr", label: "전체 OCR", desc: "이미지의 모든 텍스트 추출" },
  { id: "schema", label: "Schema 추출", desc: "Key-Value 구조화 추출" },
  { id: "qa", label: "문서 Q&A", desc: "이미지에 대해 질문" },
];

export default function VlmModeSelector({ mode, onChange, disabled }: Props) {
  return (
    <div className="space-y-1.5">
      <h3 className="text-sm font-semibold text-slate-700">2. 모드 선택</h3>
      <div className="flex gap-1">
        {MODES.map((m) => (
          <button
            key={m.id}
            onClick={() => onChange(m.id)}
            disabled={disabled}
            className={`flex-1 rounded-lg border px-2 py-2 text-center text-xs transition
              ${mode === m.id ? "border-brand-500 bg-brand-50 font-semibold text-brand-700" : "border-slate-200 text-slate-600 hover:border-slate-300"}
              ${disabled ? "cursor-not-allowed opacity-50" : "cursor-pointer"}`}
            title={m.desc}
          >
            {m.label}
          </button>
        ))}
      </div>
    </div>
  );
}
