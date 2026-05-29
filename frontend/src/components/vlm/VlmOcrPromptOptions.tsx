import type { VlmOcrPromptMode } from "../../types/vlm";

const MODES: { id: VlmOcrPromptMode; label: string; desc: string }[] = [
  {
    id: "auto",
    label: "기본 (spotting)",
    desc: "Qwen bbox_2d 1회",
  },
  {
    id: "bbox",
    label: "spotting",
    desc: "기본과 동일 (1회)",
  },
  {
    id: "custom",
    label: "커스텀",
    desc: "직접 입력 프롬프트 1회",
  },
];

export const DEFAULT_SPOTTING_PROMPT = `Spot all visible text in this image at line level.

Return ONLY a valid JSON array in reading order from top to bottom and left to right.

Use exactly this schema for every detected text line:
[{"bbox_2d": [x1, y1, x2, y2], "text_content": "recognized text"}]

Requirements:

* Detect every visible text line, including Korean, English, numbers, punctuation, table cell text, headers, labels, and rotated text.
* Preserve the recognized text exactly as it appears in the image.
* Use one JSON object per visual text line. Do not merge separate lines.
* Each bbox_2d must tightly enclose only its corresponding text line.
* Do not output markdown, code fences, explanations, comments, confidence scores, or any additional keys.
* Return [] only when there is truly no visible text in the image.`;

interface Props {
  promptMode: VlmOcrPromptMode;
  customPrompt: string;
  onPromptModeChange: (mode: VlmOcrPromptMode) => void;
  onCustomPromptChange: (text: string) => void;
  disabled?: boolean;
}

export default function VlmOcrPromptOptions({
  promptMode,
  customPrompt,
  onPromptModeChange,
  onCustomPromptChange,
  disabled,
}: Props) {
  const fillSpottingPreset = () => {
    onCustomPromptChange(DEFAULT_SPOTTING_PROMPT);
    onPromptModeChange("custom");
  };

  return (
    <div className="space-y-2">
      <h3 className="text-sm font-semibold text-slate-700">OCR 프롬프트</h3>
      <div className="grid grid-cols-3 gap-1.5">
        {MODES.map((m) => (
          <button
            key={m.id}
            type="button"
            disabled={disabled}
            onClick={() => onPromptModeChange(m.id)}
            className={`rounded-lg border px-2 py-2 text-left transition disabled:opacity-50
              ${
                promptMode === m.id
                  ? "border-brand-500 bg-brand-50 ring-1 ring-brand-500"
                  : "border-slate-200 hover:border-slate-300"
              }`}
          >
            <div className="text-xs font-semibold text-slate-800">{m.label}</div>
            <div className="text-[10px] text-slate-500 leading-tight mt-0.5">
              {m.desc}
            </div>
          </button>
        ))}
      </div>

      {promptMode === "custom" && (
        <div className="space-y-1.5">
          <button
            type="button"
            disabled={disabled}
            onClick={fillSpottingPreset}
            className="rounded border border-slate-200 px-2 py-0.5 text-[10px] text-slate-600 hover:bg-slate-50 disabled:opacity-50"
          >
            Qwen spotting 예시
          </button>
          <textarea
            value={customPrompt}
            onChange={(e) => onCustomPromptChange(e.target.value)}
            disabled={disabled}
            rows={8}
            placeholder="모델에 보낼 프롬프트를 입력하세요..."
            className="w-full rounded-lg border border-slate-200 px-2 py-1.5 text-xs text-slate-800 focus:border-brand-400 focus:outline-none disabled:opacity-50"
          />
        </div>
      )}
    </div>
  );
}
