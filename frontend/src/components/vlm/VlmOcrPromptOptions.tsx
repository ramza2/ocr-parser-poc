import type { VlmOcrPromptMode } from "../../types/vlm";

const MODES: { id: VlmOcrPromptMode; label: string; desc: string }[] = [
  {
    id: "auto",
    label: "자동",
    desc: "1차 bbox → 실패 시 2차 텍스트",
  },
  {
    id: "bbox",
    label: "1차 bbox",
    desc: "JSON+bbox만 (재시도 없음)",
  },
  {
    id: "plain",
    label: "2차 텍스트",
    desc: "일반 OCR 프롬프트만",
  },
  {
    id: "custom",
    label: "커스텀",
    desc: "직접 입력한 프롬프트",
  },
];

export const DEFAULT_PLAIN_PROMPT =
  "이 이미지에 있는 모든 텍스트를 빠짐없이 읽어주세요. 원본 레이아웃(줄바꿈, 들여쓰기)을 최대한 유지하세요.";

export const DEFAULT_BBOX_PROMPT_KO = `이 이미지의 모든 텍스트를 줄/블록 단위로 읽고 위치를 표시하세요.
반드시 JSON 배열만 출력하세요. 형식:
[{"text": "텍스트", "bbox": [x1, y1, x2, y2]}]
bbox는 모델이 보는 이미지 기준 픽셀 좌표입니다 (좌상단 x,y → 우하단 x,y).
화면에 보이는 텍스트가 있으면 빈 배열 []을 반환하지 마세요.`;

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
  const fillPreset = (preset: "plain" | "bbox_ko") => {
    onCustomPromptChange(
      preset === "plain" ? DEFAULT_PLAIN_PROMPT : DEFAULT_BBOX_PROMPT_KO
    );
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
          <div className="flex gap-1">
            <button
              type="button"
              disabled={disabled}
              onClick={() => fillPreset("bbox_ko")}
              className="rounded border border-slate-200 px-2 py-0.5 text-[10px] text-slate-600 hover:bg-slate-50 disabled:opacity-50"
            >
              1차 bbox 예시
            </button>
            <button
              type="button"
              disabled={disabled}
              onClick={() => fillPreset("plain")}
              className="rounded border border-slate-200 px-2 py-0.5 text-[10px] text-slate-600 hover:bg-slate-50 disabled:opacity-50"
            >
              2차 텍스트 예시
            </button>
          </div>
          <textarea
            value={customPrompt}
            onChange={(e) => onCustomPromptChange(e.target.value)}
            disabled={disabled}
            rows={6}
            placeholder="모델에 보낼 프롬프트를 입력하세요..."
            className="w-full rounded-lg border border-slate-200 px-2 py-1.5 text-xs text-slate-800 focus:border-brand-400 focus:outline-none disabled:opacity-50"
          />
        </div>
      )}
    </div>
  );
}
