import type {
  SchemaField,
  SchemaMatchMode,
  SchemaScriptFilter,
} from "../../types/vlm";

interface Props {
  schema: SchemaField[];
  onChange: (schema: SchemaField[]) => void;
  disabled?: boolean;
}

const TYPES = ["text", "number", "date", "phone", "email", "address"];

const MATCH_MODES: {
  id: SchemaMatchMode;
  label: string;
  keyPlaceholder: string;
  hintPlaceholder: string;
}[] = [
  {
    id: "exact_text",
    label: "정확 문자열",
    keyPlaceholder: "찾을 문자열 (이미지와 동일)",
    hintPlaceholder: "위치 힌트 (선택, 의미 설명 X)",
  },
  {
    id: "script_filter",
    label: "문자 종류",
    keyPlaceholder: "필드 ID (예: cjk_line)",
    hintPlaceholder: "위치 힌트 (선택)",
  },
  {
    id: "semantic_field",
    label: "의미 기반",
    keyPlaceholder: "필드 ID (예: address)",
    hintPlaceholder: "찾을 내용 설명 (예: 상단 파란 카드 2번째 줄)",
  },
];

const SCRIPTS: { id: SchemaScriptFilter; label: string }[] = [
  { id: "han", label: "한자 (CJK)" },
  { id: "hangul", label: "한글" },
  { id: "latin", label: "라틴" },
  { id: "english", label: "영문" },
  { id: "digit", label: "숫자" },
];

const defaultField = (): SchemaField => ({
  key: "",
  description: "",
  type: "text",
  match_mode: "exact_text",
  script: "han",
});

export default function SchemaEditor({ schema, onChange, disabled }: Props) {
  const addField = () => {
    onChange([...schema, defaultField()]);
  };

  const removeField = (idx: number) => {
    onChange(schema.filter((_, i) => i !== idx));
  };

  const updateField = (idx: number, patch: Partial<SchemaField>) => {
    onChange(schema.map((f, i) => (i === idx ? { ...f, ...patch } : f)));
  };

  const modeMeta = (mode: SchemaMatchMode | undefined) =>
    MATCH_MODES.find((m) => m.id === (mode ?? "exact_text")) ?? MATCH_MODES[0];

  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-semibold text-slate-700">Schema 추출</h3>
        <button
          type="button"
          onClick={addField}
          disabled={disabled}
          className="rounded bg-brand-600 px-2.5 py-1 text-xs font-medium text-white hover:bg-brand-700 disabled:opacity-50"
        >
          + 추가
        </button>
      </div>

      <p className="text-[10px] text-slate-500 leading-snug">
        <strong>정확 문자열</strong>: Key에 적은 글자만 위치 탐색 ·{" "}
        <strong>문자 종류</strong>: 한자/한글 등 계열만 ·{" "}
        <strong>의미 기반</strong>: 설명에 맞는 화면 텍스트
      </p>

      {schema.length === 0 && (
        <p className="text-center text-xs text-slate-400 py-3">
          추출할 항목을 추가하세요.
        </p>
      )}

      <div className="space-y-2 max-h-64 overflow-y-auto">
        {schema.map((field, idx) => {
          const mode = field.match_mode ?? "exact_text";
          const meta = modeMeta(mode);
          return (
            <div
              key={idx}
              className="rounded-lg border border-slate-200 bg-slate-50/50 p-2 space-y-1.5"
            >
              <div className="flex flex-wrap gap-1.5 items-center">
                <select
                  value={mode}
                  onChange={(e) =>
                    updateField(idx, {
                      match_mode: e.target.value as SchemaMatchMode,
                    })
                  }
                  disabled={disabled}
                  className="w-[5.5rem] shrink-0 rounded border border-slate-200 bg-white px-1 py-1.5 text-[10px] focus:border-brand-400 focus:outline-none"
                  title="검색 방식"
                >
                  {MATCH_MODES.map((m) => (
                    <option key={m.id} value={m.id}>
                      {m.label}
                    </option>
                  ))}
                </select>
                <input
                  value={field.key}
                  onChange={(e) => updateField(idx, { key: e.target.value })}
                  placeholder={meta.keyPlaceholder}
                  disabled={disabled}
                  className="min-w-[7rem] flex-1 rounded border border-slate-200 bg-white px-2 py-1.5 text-xs focus:border-brand-400 focus:outline-none"
                />
                {mode === "script_filter" && (
                  <select
                    value={field.script ?? "han"}
                    onChange={(e) =>
                      updateField(idx, {
                        script: e.target.value as SchemaScriptFilter,
                      })
                    }
                    disabled={disabled}
                    className="w-[4.5rem] shrink-0 rounded border border-slate-200 bg-white px-1 py-1.5 text-[10px] focus:border-brand-400 focus:outline-none"
                    title="문자 종류"
                  >
                    {SCRIPTS.map((s) => (
                      <option key={s.id} value={s.id}>
                        {s.label}
                      </option>
                    ))}
                  </select>
                )}
                <select
                  value={field.type}
                  onChange={(e) => updateField(idx, { type: e.target.value })}
                  disabled={disabled}
                  className="w-16 shrink-0 rounded border border-slate-200 bg-white px-1 py-1.5 text-[10px] focus:border-brand-400 focus:outline-none"
                >
                  {TYPES.map((t) => (
                    <option key={t} value={t}>
                      {t}
                    </option>
                  ))}
                </select>
                <button
                  type="button"
                  onClick={() => removeField(idx)}
                  disabled={disabled}
                  className="shrink-0 rounded p-1 text-slate-400 hover:bg-red-50 hover:text-red-500 disabled:opacity-50"
                  title="삭제"
                >
                  <svg
                    className="h-3.5 w-3.5"
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M6 18L18 6M6 6l12 12"
                    />
                  </svg>
                </button>
              </div>
              <input
                value={field.description}
                onChange={(e) =>
                  updateField(idx, { description: e.target.value })
                }
                placeholder={meta.hintPlaceholder}
                disabled={disabled}
                className="w-full rounded border border-slate-200 bg-white px-2 py-1.5 text-xs focus:border-brand-400 focus:outline-none"
              />
            </div>
          );
        })}
      </div>
    </div>
  );
}
