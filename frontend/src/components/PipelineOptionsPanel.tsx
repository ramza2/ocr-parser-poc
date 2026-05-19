import type { PipelineStepInfo } from "../types/parser";

type Props = {
  title: string;
  steps: PipelineStepInfo[];
  selected: string[];
  onChange: (ids: string[]) => void;
  disabled?: boolean;
  presetLabel?: string;
  onPreset?: () => void;
  onClear?: () => void;
};

export default function PipelineOptionsPanel({
  title,
  steps,
  selected,
  onChange,
  disabled = false,
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
    <div className="rounded-lg border border-slate-100 bg-slate-50/50 p-3">
      <div className="mb-2 flex items-center justify-between gap-2">
        <h3 className="text-xs font-semibold text-slate-700">{title}</h3>
        <div className="flex gap-1">
          {onPreset && presetLabel && (
            <button
              type="button"
              disabled={disabled}
              onClick={onPreset}
              className="rounded px-2 py-0.5 text-[10px] font-medium text-brand-700 hover:bg-brand-50 disabled:opacity-40"
            >
              {presetLabel}
            </button>
          )}
          {onClear && (
            <button
              type="button"
              disabled={disabled}
              onClick={onClear}
              className="rounded px-2 py-0.5 text-[10px] text-slate-500 hover:bg-slate-100 disabled:opacity-40"
            >
              초기화
            </button>
          )}
        </div>
      </div>
      <ul className="space-y-1.5">
        {steps.map((step) => (
          <li key={step.step_id}>
            <label
              className={`flex cursor-pointer gap-2 rounded-md px-2 py-1.5 text-xs ${
                disabled ? "cursor-not-allowed opacity-50" : "hover:bg-white"
              }`}
            >
              <input
                type="checkbox"
                className="mt-0.5 rounded border-slate-300"
                checked={selected.includes(step.step_id)}
                disabled={disabled}
                onChange={() => toggle(step.step_id)}
              />
              <span>
                <span className="font-medium text-slate-800">{step.name}</span>
                <span className="mt-0.5 block text-[10px] leading-snug text-slate-500">
                  {step.description}
                </span>
              </span>
            </label>
          </li>
        ))}
      </ul>
    </div>
  );
}
