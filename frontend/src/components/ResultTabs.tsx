import type { ResultTab } from "../types/parser";

const TABS: { id: ResultTab; label: string }[] = [
  { id: "text", label: "텍스트 결과" },
  { id: "json", label: "JSON 결과" },
  { id: "log", label: "로그" },
  { id: "error", label: "오류" },
];

interface Props {
  activeTab: ResultTab;
  onChange: (tab: ResultTab) => void;
}

export default function ResultTabs({ activeTab, onChange }: Props) {
  return (
    <div className="flex flex-wrap gap-1 border-b border-slate-200">
      {TABS.map((tab) => (
        <button
          key={tab.id}
          type="button"
          onClick={() => onChange(tab.id)}
          className={`px-4 py-2.5 text-sm font-medium transition ${
            activeTab === tab.id
              ? "border-b-2 border-brand-600 text-brand-600"
              : "text-slate-500 hover:text-slate-700"
          }`}
        >
          {tab.label}
        </button>
      ))}
    </div>
  );
}
