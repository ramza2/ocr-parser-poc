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
    if (q) {
      onSubmit(q);
    }
  };

  return (
    <div className="space-y-2">
      <h3 className="text-sm font-semibold text-slate-700">질문 입력</h3>
      <div className="flex gap-2">
        <input
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && !disabled && handleSubmit()}
          placeholder="예: 총액이 얼마인가요?"
          disabled={disabled}
          className="min-w-0 flex-1 rounded-lg border border-slate-200 px-3 py-2 text-sm focus:border-brand-400 focus:outline-none"
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
