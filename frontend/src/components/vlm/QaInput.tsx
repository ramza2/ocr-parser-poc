import { useState } from "react";

interface Props {
  onSubmit: (question: string) => void;
  disabled?: boolean;
  loading?: boolean;
}

export default function QaInput({ onSubmit, disabled, loading }: Props) {
  const [question, setQuestion] = useState("");

  const handleSubmit = () => {
    const q = question.trim();
    if (!q || disabled || loading) return;
    onSubmit(q);
    setQuestion("");
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key !== "Enter" || e.shiftKey || disabled) return;
    e.preventDefault();
    handleSubmit();
  };

  return (
    <div className="space-y-2">
      <h3 className="text-sm font-semibold text-slate-700">질문 입력</h3>
      <p className="text-[11px] text-slate-500">
        Enter: 전송 · Shift+Enter: 줄바꿈
      </p>
      <div className="flex items-end gap-2">
        <textarea
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="예: 3명 이상이면 단체 요금입니다. 어른 4명, 아이 3명의 총 관람료는?"
          disabled={disabled}
          rows={3}
          className="min-w-0 flex-1 resize-y rounded-lg border border-slate-200 px-3 py-2 text-sm leading-relaxed focus:border-brand-400 focus:outline-none disabled:opacity-50"
        />
        <button
          onClick={handleSubmit}
          disabled={disabled || !question.trim() || loading}
          className="shrink-0 rounded-lg bg-brand-600 px-4 py-2 text-sm font-medium text-white hover:bg-brand-700 disabled:opacity-50"
        >
          {loading ? "처리 중..." : "질문"}
        </button>
      </div>
    </div>
  );
}
