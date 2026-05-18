import type { PipelineStepInfo } from "../types/parser";

interface Props {
  title: string;
  steps: PipelineStepInfo[];
  selected: string[];
  onChange: (ids: string[]) => void;
  disabled?: boolean;
  presetLabel?: string;
  onPreset?: () => void;
  onClear?: () => void;
}

export default function PipelineOptionsPanel({
  title,
  steps,
  selected,
  onChange,
  disabled,
  presetLabel,
  onPreset,
  onClear,
}: Props) {
  const toggle = (stepId: string) => {
    if (disabled) return;
    if (selected.includes(stepId)) {
      onChange(selected.filter((id) => id !== stepId));
    } else {
      const ordered = [...selected, stepId].sort((a, b) => {
        const oa = steps.find((s) => s.step_id === a)?.default_order ?? 99;
        const ob = steps.find((s) => s.step_id === b)?.default_order ?? 99;
        return oa - ob;
      });
      onChange(ordered);
    }
  };

  return (
    <section className="space-y-2">
      <div className="flex items-center justify-between gap-2">
        <h3 className="text-xs font-semibold uppercase tracking-wide text-slate-600">
          {title}
        </h3>
        <div className="flex gap-1">
          {onPreset && presetLabel && (
            <button
              type="button"
              disabled={disabled}
              onClick={onPreset}
              className="rounded border border-slate-200 px-2 py-0.5 text-[10px] text-slate-600 hover:bg-slate-50 disabled:opacity-50"
            >
              {presetLabel}
            </button>
          )}
          {onClear && (
            <button
              type="button"
              disabled={disabled}
              onClick={onClear}
              className="rounded border border-slate-200 px-2 py-0.5 text-[10px] text-slate-600 hover:bg-slate-50 disabled:opacity-50"
            >
              초기화
            </button>
          )}
        </div>
      </div>

      {steps.length === 0 ? (
        <p className="text-xs text-slate-500">사용 가능한 단계가 없습니다.</p>
      ) : (
        <ul className="max-h-40 space-y-1 overflow-y-auto rounded-lg border border-slate-100 bg-slate-50 p-2">
          {steps.map((step) => (
            <li key={step.step_id}>
              <label
                className={`flex cursor-pointer items-start gap-2 rounded p-1.5 text-xs hover:bg-white ${
                  disabled ? "cursor-not-allowed opacity-60" : ""
                }`}
              >
                <input
                  type="checkbox"
                  checked={selected.includes(step.step_id)}
                  disabled={disabled}
                  onChange={() => toggle(step.step_id)}
                  className="mt-0.5 rounded border-slate-300"
                />
                <span>
                  <span className="font-medium text-slate-800">{step.name}</span>
                  <span className="mt-0.5 block text-slate-500">
                    {step.description}
                  </span>
                </span>
              </label>
            </li>
          ))}
        </ul>
      )}

      {selected.length > 0 && (
        <p className="text-[10px] text-slate-500">
          적용 순서: {selected.join(" → ")}
        </p>
      )}
    </section>
  );
}
