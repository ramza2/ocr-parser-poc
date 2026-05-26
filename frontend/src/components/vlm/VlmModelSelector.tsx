import type { VlmModelInfo } from "../../types/vlm";

interface Props {
  models: VlmModelInfo[];
  selectedModelId: string | null;
  currentLoaded: string | null;
  onSelect: (id: string) => void;
  disabled?: boolean;
}

const VRAM_COLORS: Record<string, string> = {
  low: "bg-emerald-100 text-emerald-700",
  mid: "bg-amber-100 text-amber-700",
  high: "bg-rose-100 text-rose-700",
};

function vramLevel(gb: number) {
  if (gb <= 3) return "low";
  if (gb <= 8) return "mid";
  return "high";
}

export default function VlmModelSelector({
  models,
  selectedModelId,
  currentLoaded,
  onSelect,
  disabled,
}: Props) {
  if (models.length === 0) {
    return (
      <div className="rounded-lg border border-dashed border-slate-200 p-4 text-center text-sm text-slate-400">
        사용 가능한 VLM 모델이 없습니다.
      </div>
    );
  }

  return (
    <div className="space-y-2">
      <h3 className="text-sm font-semibold text-slate-700">1. 모델 선택</h3>
      <div className="grid grid-cols-1 gap-2">
        {models.map((m) => {
          const selected = m.model_id === selectedModelId;
          const loaded = m.model_id === currentLoaded;
          const level = vramLevel(m.vram_gb);
          return (
            <button
              key={m.model_id}
              onClick={() => onSelect(m.model_id)}
              disabled={disabled}
              className={`flex items-center justify-between rounded-lg border px-3 py-2.5 text-left text-sm transition
                ${selected ? "border-brand-500 bg-brand-50 ring-1 ring-brand-300" : "border-slate-200 bg-white hover:border-slate-300"}
                ${disabled ? "cursor-not-allowed opacity-50" : "cursor-pointer"}`}
            >
              <div className="min-w-0">
                <p className="font-medium text-slate-800">{m.name}</p>
                <p className="mt-0.5 truncate text-xs text-slate-500">
                  {m.description}
                </p>
              </div>
              <div className="ml-2 flex shrink-0 flex-col items-end gap-1">
                <span className={`rounded px-1.5 py-0.5 text-[10px] font-semibold ${VRAM_COLORS[level]}`}>
                  {m.vram_gb}GB
                </span>
                {loaded && (
                  <span className="rounded bg-emerald-500 px-1.5 py-0.5 text-[10px] font-semibold text-white">
                    로드됨
                  </span>
                )}
              </div>
            </button>
          );
        })}
      </div>
    </div>
  );
}
