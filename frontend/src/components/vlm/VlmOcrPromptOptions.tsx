import type { VlmOcrPromptMode } from "../../types/vlm";

const MODES: { id: VlmOcrPromptMode; label: string; desc: string }[] = [
  {
    id: "spotting",
    label: "전체 OCR",
    desc: "간판·문서·scene-text + bbox (1회)",
  },
  {
    id: "custom",
    label: "커스텀",
    desc: "직접 입력 프롬프트 (1회)",
  },
];

export const DEFAULT_SPOTTING_PROMPT = `You are a scene-text OCR and text-grounding engine.

Find and transcribe ALL visible text in this image, whether it appears in:

* a document or mobile screen,
* a signboard, road sign, direction board, label, poster, or product,
* an outdoor or natural scene.

Pay special attention to:

* Korean, English, Chinese characters, numbers, and punctuation,
* stylized, embossed, shadowed, engraved, painted, low-contrast, or angled text,
* text on colored boards or textured backgrounds.

Before returning an empty result, carefully inspect the entire image for any object or region containing readable characters.

Return ONLY a valid JSON array in reading order, from top to bottom and left to right.

Use exactly this schema:
[{"bbox_2d": [x1, y1, x2, y2], "text_content": "recognized text"}]

Rules:

* Return one object per visible text line.
* Preserve text exactly as it appears in the image.
* Each bbox_2d must tightly enclose its corresponding text line.
* Do not omit text because it is decorative, on a signboard, photographed outdoors, shadowed, embossed, or low contrast.
* Do not output markdown, explanations, comments, confidence scores, or additional keys.
* The JSON must be strict RFC 8259: no trailing commas.
* Return [] only when there are truly no visible characters or words anywhere in the image.`;

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
      <div className="grid grid-cols-2 gap-1.5">
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
            전체 OCR 프롬프트 예시
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
